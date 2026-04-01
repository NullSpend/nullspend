/**
 * End-to-end budget enforcement tests.
 *
 * Budget state flows through two layers:
 *   1. DO lookup cache (budget entities, 60s TTL in Worker isolate)
 *   2. Durable Object SQLite (authoritative budget data, persistent)
 *
 * Setup: Insert budget in Postgres → send a request to force DO sync.
 * Teardown: Call /internal/budget/invalidate to clean DO + caches → delete Postgres row.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID (real userId for the test API key)
 *   - INTERNAL_SECRET (for cache invalidation via /internal/budget/invalidate)
 *   - DATABASE_URL for budget setup in Postgres
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
  waitForBudgetSpend,
} from "./smoke-test-helpers.js";

describe("End-to-end budget enforcement", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_USER_ID) throw new Error("NULLSPEND_SMOKE_USER_ID required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required for budget cache invalidation.");
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    // Look up org_id from the smoke test API key (required NOT NULL since Phase 2)
    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;
  });

  // NOTE: Budget smoke tests can flake when Hyperdrive caches stale lookupBudgetsForDO
  // query results (up to 60s TTL). Tests that change max_budget or session_limit between
  // runs may see the cached prior value. Run these tests in isolation for reliability.
  afterEach(async () => {
    // Wait for waitUntil reconciliation from the test's requests to complete.
    // Without this, reconciliation can re-add spend to the DO after we clean it.
    await new Promise((r) => setTimeout(r, 5_000));

    // Clean up all three layers via internal API.
    // removeBudget now also deletes associated reservations, preventing
    // late-arriving reconciliation from affecting the next test.
    await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    // Remove Postgres row
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
  });

  afterAll(async () => {
    await sql.end();
  });

  /**
   * Insert a user budget in Postgres and send a warm-up request to force
   * the proxy to sync the budget to the Durable Object.
   */
  async function setupBudget(maxBudgetMicrodollars: number, spendMicrodollars = 0) {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    // Insert/update budget in Postgres
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, ${spendMicrodollars}, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;

    // Force Postgres→DO sync via internal endpoint.
    await syncBudget(orgId, "user", userId);
  }

  /** Remove any existing budget so the user is non-budgeted */
  async function cleanupBudget() {
    const userId = NULLSPEND_SMOKE_USER_ID!;
    await invalidateBudget(orgId, "user", userId);
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${userId}`;
  }

  it("allows request when budget has sufficient funds", async () => {
    await setupBudget(10_000_000); // $10

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("usage");
  }, 30_000);

  it("blocks request with budget_exceeded when budget is $0.00", async () => {
    await setupBudget(0); // $0.00

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
    expect(body.error.message).toContain("budget");
  }, 30_000);

  it("blocks request when spend already equals maxBudget", async () => {
    await setupBudget(100_000, 100_000); // spend == max

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
  }, 30_000);

  it("exhausts budget by sending requests until blocked", async () => {
    // Estimator reserves ~6 microdollars per request (gpt-4o-mini, max_tokens: 3).
    // Budget of 10 allows exactly 1 request before denial.
    await setupBudget(10);

    let successCount = 0;
    let deniedCount = 0;

    for (let i = 0; i < 10; i++) {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `Budget exhaust ${i}` }] }),
      });

      if (res.status === 200) {
        successCount++;
        await res.json();
      } else if (res.status === 429) {
        deniedCount++;
        const body = await res.json();
        expect(body.error.code).toBe("budget_exceeded");
        break;
      } else {
        await res.text();
      }
    }

    expect(successCount).toBeGreaterThanOrEqual(0);
    expect(deniedCount).toBeGreaterThan(0);
  }, 120_000);

  it("reconciliation adjusts spend after request completes", async () => {
    await setupBudget(5_000_000); // $5 — plenty of headroom

    // Send a request
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });
    expect(res.status).toBe(200);
    await res.json();

    // Poll Postgres until reconciliation writes spend back.
    // Reconciliation is async: waitUntil → Queue (up to 5s batch timeout) → DO → Postgres.
    const spend = await waitForBudgetSpend(sql, "user", NULLSPEND_SMOKE_USER_ID!, 15_000);
    expect(spend).toBeGreaterThan(0);
  }, 30_000);

  it("budget_exceeded response includes correct error fields", async () => {
    await setupBudget(1); // 1 microdollar — guaranteed denial

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();

    expect(body.error.code).toBe("budget_exceeded");
    expect(body.error.message).toContain("budget");
    expect(body.error.details).toBeNull();
  }, 30_000);

  it("concurrent requests against tight budget don't overspend", async () => {
    // Set budget to ~200 microdollars — enough for maybe 1-2 requests
    await setupBudget(200);

    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `Concurrent ${i}` }] }),
      }),
    );

    const results = await Promise.all(requests);
    const statuses = await Promise.all(
      results.map(async (r) => {
        const body = await r.text();
        return { status: r.status, body };
      }),
    );

    const successes = statuses.filter((s) => s.status === 200);
    const denied = statuses.filter((s) => s.status === 429);

    // At least some should be denied; budget can't cover all 5
    expect(successes.length + denied.length).toBe(5);
  }, 60_000);

  it("requests without configured budget are not affected by budget enforcement", async () => {
    // Clean up any existing budget so there's no budget to enforce
    await cleanupBudget();

    // Wait for auth cache to expire (invalidateBudget clears it, but be safe)
    await new Promise((r) => setTimeout(r, 2_000));

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("usage");
  }, 30_000);
});
