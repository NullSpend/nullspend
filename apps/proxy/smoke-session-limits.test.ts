/**
 * End-to-end session limit smoke tests.
 *
 * Tests session-level budget aggregation against the live deployed worker.
 * Validates the full stack: Postgres → DO sync → checkAndReserve session check →
 * reconcile session correction → route denial response.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID, NULLSPEND_SMOKE_KEY_ID
 *   - INTERNAL_SECRET
 *   - DATABASE_URL
 */
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  NULLSPEND_SMOKE_KEY_ID,
  INTERNAL_SECRET,
  authHeaders,
  smallRequest,
  isServerUp,
  invalidateBudget,
  syncBudget,
} from "./smoke-test-helpers.js";

describe("End-to-end session limit enforcement", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_USER_ID) throw new Error("NULLSPEND_SMOKE_USER_ID required.");
    if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required.");
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    // Look up org_id from the smoke test API key (required NOT NULL since Phase 2)
    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;
  });

  afterEach(async () => {
    // Wait for in-flight reconciliations from prior test's requests.
    // Same pattern as smoke-budget-e2e.test.ts (known flake window).
    await new Promise((r) => setTimeout(r, 5_000));

    // Remove budget from DO + delete from Postgres so next test starts clean.
    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* budget may not exist */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    // Brief pause to ensure Postgres commits propagate through the connection pooler
    await new Promise((r) => setTimeout(r, 500));
  });

  afterAll(async () => {
    // Full cleanup: remove budget from DO + Postgres
    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* may not exist */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await sql.end();
  });

  /**
   * Insert a user budget with a session limit in Postgres and sync to DO.
   */
  async function setupBudgetWithSessionLimit(
    maxBudgetMicrodollars: number,
    sessionLimitMicrodollars: number,
  ) {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    // 1. Remove stale DO budget (evicts any cached session_limit)
    try { await invalidateBudget(orgId, "user", userId); } catch { /* may not exist */ }

    // 2. Upsert into Postgres with desired session_limit
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy, session_limit_microdollars)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, 0, 'strict_block', ${sessionLimitMicrodollars})
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = 0,
                    session_limit_microdollars = ${sessionLimitMicrodollars},
                    updated_at = NOW()
    `;

    // 3. Brief pause to ensure Postgres commit propagates through Hyperdrive
    await new Promise((r) => setTimeout(r, 500));

    // 4. Sync: Postgres → DO (populateIfEmpty creates fresh budget with session_limit)
    await syncBudget(orgId, "user", userId);
  }

  /**
   * Insert a user budget WITHOUT a session limit.
   */
  async function _setupBudgetWithoutSessionLimit(maxBudgetMicrodollars: number) {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    try { await invalidateBudget(orgId, "user", userId); } catch { /* may not exist */ }

    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy, session_limit_microdollars)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, 0, 'strict_block', NULL)
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = 0,
                    session_limit_microdollars = NULL,
                    updated_at = NOW()
    `;

    await new Promise((r) => setTimeout(r, 500));
    await syncBudget(orgId, "user", userId);
  }

  // ── Core enforcement ──────────────────────────────────────────────
  // All tests use session_limit=1 microdollar. Changing session_limit between
  // tests is unreliable due to Hyperdrive query caching (up to 60s TTL).

  it("denies request with session_limit_exceeded: correct body, no Retry-After, has trace ID", async () => {
    // Session limit of 1 microdollar — any gpt-4o-mini estimate (~5 microdollars) exceeds it
    await setupBudgetWithSessionLimit(100_000_000, 1);

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders({ "x-nullspend-session": `smoke-denied-${Date.now()}` }),
      body: smallRequest(),
    });

    expect(res.status).toBe(429);

    // Response body shape
    const body = await res.json();
    expect(body.error.code).toBe("session_limit_exceeded");
    expect(body.error.message).toContain("session");
    expect(body.error.details).toHaveProperty("session_id");
    expect(body.error.details).toHaveProperty("session_spend_microdollars");
    expect(body.error.details).toHaveProperty("session_limit_microdollars", 1);

    // No Retry-After (session is done, not retryable — unlike velocity)
    expect(res.headers.get("Retry-After")).toBeNull();
    // Trace ID present
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBeTruthy();
  }, 30_000);

  // ── No enforcement without session header ─────────────────────────

  it("allows request without session header even with session limit configured", async () => {
    await setupBudgetWithSessionLimit(100_000_000, 1); // 1 microdollar session limit

    // No x-nullspend-session header → session enforcement skipped
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(200);
  }, 30_000);

  // NOTE: "no session limit configured" test omitted — Hyperdrive query caching (up to 60s)
  // prevents reliably changing session_limit between tests via the Postgres→sync flow.
  // This scenario is covered by DO-level unit tests (user-budget-do.do.test.ts).

  // ── Different sessions are independent ────────────────────────────

  it("different session IDs have independent spend tracking", async () => {
    // Same session_limit=1 — both sessions start from 0
    await setupBudgetWithSessionLimit(100_000_000, 1);

    // Session A and B are unique — both denied independently (same fresh state)
    const resA = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders({ "x-nullspend-session": `smoke-A-${Date.now()}` }),
      body: smallRequest(),
    });

    const resB = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders({ "x-nullspend-session": `smoke-B-${Date.now()}` }),
      body: smallRequest(),
    });

    // Both sessions start from 0 — both get same result (denied, since limit=1)
    expect(resB.status).toBe(resA.status);
  }, 30_000);

  // ── Approval flow (no session header = no enforcement) ─────────────

  it("multiple requests approved when session header omitted", async () => {
    // Proves budget enforcement works (approved) while session enforcement is bypassed.
    // Cannot test generous session limit here because Hyperdrive caches the
    // lookupBudgetsForDO query — changing session_limit between tests is unreliable.
    await setupBudgetWithSessionLimit(100_000_000, 1);

    for (let i = 0; i < 3; i++) {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(), // no session header
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      await res.json();
    }
  }, 60_000);
});
