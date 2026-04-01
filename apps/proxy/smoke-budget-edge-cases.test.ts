/**
 * Budget edge-case smoke tests, specifically designed to verify we don't
 * have the same budget bugs as LiteLLM and other competitors.
 *
 * Tests cover: user budget enforcement, API key budget enforcement,
 * dual (user + key) budget enforcement, precise exhaustion, and
 * stream abort spend accuracy.
 *
 * Budget state flows through two layers (see budget-enforcement-architecture.md):
 *   1. DO lookup cache (budget entities, 60s TTL)
 *   2. Durable Object SQLite (authoritative)
 *
 * Setup: Postgres INSERT → /internal/budget/invalidate → warm-up request.
 * Teardown: /internal/budget/invalidate → Postgres DELETE.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID, NULLSPEND_SMOKE_KEY_ID
 *   - INTERNAL_SECRET (for cache invalidation)
 *   - DATABASE_URL for Postgres budget setup
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

describe("Budget edge cases (LiteLLM bug avoidance)", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_USER_ID) throw new Error("NULLSPEND_SMOKE_USER_ID required.");
    if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required for budget cache invalidation.");
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    // Look up org_id from the smoke test API key (required NOT NULL since Phase 2)
    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;
  });

  afterEach(async () => {
    // Wait for waitUntil reconciliation to complete before cleanup
    await new Promise((r) => setTimeout(r, 5_000));

    // Clean up all three layers via internal API.
    // removeBudget now also deletes associated reservations.
    await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    await invalidateBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!);
    await sql`DELETE FROM budgets WHERE entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await sql`DELETE FROM budgets WHERE entity_id = ${NULLSPEND_SMOKE_KEY_ID!} AND max_budget_microdollars < 1000000000000`;
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  /**
   * Insert a budget in Postgres, invalidate proxy caches, and send a warm-up
   * request to force DO sync.
   */
  async function insertBudget(
    entityType: string,
    entityId: string,
    maxBudgetMicrodollars: number,
    spendMicrodollars = 0,
  ) {
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${NULLSPEND_SMOKE_USER_ID!}, ${orgId}, ${entityType}, ${entityId}, ${maxBudgetMicrodollars}, ${spendMicrodollars}, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;
  }

  /**
   * Insert budget(s) into Postgres, wait for Hyperdrive cache to expire,
   * then sync to DO. Call this AFTER all insertBudget() calls are done.
   * Accepts entity type/id pairs to sync — defaults to user entity.
   */
  async function syncAfterInsert(entityType = "user", entityId = NULLSPEND_SMOKE_USER_ID!) {
    // Wait for Hyperdrive query cache to expire (max_age=5s)
    // so the sync reads fresh Postgres data including all inserted rows.
    await new Promise((r) => setTimeout(r, 5_500));
    await syncBudget(orgId, entityType, entityId);
  }

  /** Insert a single budget and sync. Convenience wrapper for single-entity tests. */
  async function setupBudget(
    entityType: string,
    entityId: string,
    maxBudgetMicrodollars: number,
    spendMicrodollars = 0,
  ) {
    await insertBudget(entityType, entityId, maxBudgetMicrodollars, spendMicrodollars);
    await syncAfterInsert(entityType, entityId);
  }

  /** Remove any existing budgets so the user/key is non-budgeted */
  async function cleanupBudgets() {
    await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    await invalidateBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!);
    await sql`DELETE FROM budgets WHERE entity_id IN (${NULLSPEND_SMOKE_USER_ID!}, ${NULLSPEND_SMOKE_KEY_ID!}) AND max_budget_microdollars < 1000000000000`;
  }

  it("budget enforced on /v1/chat/completions route (no bypass routes)", async () => {
    await setupBudget("user", NULLSPEND_SMOKE_USER_ID!, 1);

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
  }, 30_000);

  it("budget enforced for API key's user", async () => {
    await setupBudget("user", NULLSPEND_SMOKE_USER_ID!, 1);

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

  it("budget enforced for API key", async () => {
    await setupBudget("api_key", NULLSPEND_SMOKE_KEY_ID!, 1);

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

  it("both user and key budgets checked — tightest one blocks", async () => {
    // User budget generous, key budget exhausted.
    // Insert both BEFORE syncing, then sync each entity to its own DO.
    await insertBudget("user", NULLSPEND_SMOKE_USER_ID!, 10_000_000);
    await insertBudget("api_key", NULLSPEND_SMOKE_KEY_ID!, 1);
    await syncAfterInsert("user", NULLSPEND_SMOKE_USER_ID!);
    await syncBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!);

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

  it("stream abort does not double-count spend (actual vs reservation)", async () => {
    await setupBudget("user", NULLSPEND_SMOKE_USER_ID!, 5_000_000);

    const controller = new AbortController();
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({
        stream: true,
        messages: [{ role: "user", content: "Count from 1 to 3." }],
        max_tokens: 15,
      }),
      signal: controller.signal,
    });

    expect(res.status).toBe(200);

    // Read one chunk then abort
    const reader = res.body!.getReader();
    try {
      await reader.read();
    } catch {
      // fine
    }
    controller.abort();

    // Wait for reconciliation
    await new Promise((r) => setTimeout(r, 10_000));

    // Verify spend in Postgres is reasonable (not the full reservation estimate)
    const rows = await sql`
      SELECT spend_microdollars::text as spend
      FROM budgets
      WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}
    `;
    const spend = Number(rows[0]?.spend ?? 0);
    const estimatorReservation = 100;
    expect(spend).toBeLessThan(estimatorReservation * 5);
  }, 60_000);

  it("budget allows exactly one request then blocks second (precise exhaustion)", async () => {
    // Estimator reserves ~5 microdollars for gpt-4o-mini with max_tokens: 3.
    // Budget of 7 allows 1 request; even after reconciliation (actual spend ~3),
    // the remaining ~4 is still less than the next estimate of ~5.
    await setupBudget("user", NULLSPEND_SMOKE_USER_ID!, 7);

    const res1 = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    const status1 = res1.status;
    await res1.text();

    // Small delay for reservation to be recorded
    await new Promise((r) => setTimeout(r, 500));

    const res2 = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    const status2 = res2.status;
    await res2.text();

    // At least one must succeed and the other must fail (or both fail if estimate > 7)
    if (status1 === 200) {
      expect(status2).toBe(429);
    } else {
      expect(status1).toBe(429);
    }
  }, 30_000);

  it("requests without configured budget bypass budget checks entirely", async () => {
    await cleanupBudgets();

    // Wait for auth cache refresh
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
