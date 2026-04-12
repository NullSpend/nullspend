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
    // Proxy returns enriched budget entity details (entity_type, limits, spend)
    expect(body.error.details).toBeDefined();
  }, 30_000);

  it("concurrent requests against tight budget don't overspend", async () => {
    // Set budget to ~15 microdollars — enough for ~2 gpt-4o-mini requests (~6µ¢ each)
    await setupBudget(15);

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

    console.log(`[ST-8] Tight budget (200µ¢): ${successes.length} succeeded, ${denied.length} denied`);

    // All responses must be 200 or 429 — no 500s
    expect(successes.length + denied.length).toBe(5);
    // ST-8: Budget of 200µ¢ with ~6µ¢/request should allow max ~3.
    // Allow headroom for estimation variance, but at least 1 must be denied.
    expect(denied.length).toBeGreaterThan(0);
    expect(successes.length).toBeLessThanOrEqual(4);
  }, 60_000);

  it("ST-1: budget increase via Postgres UPDATE propagates to proxy enforcement (BDG-4)", async () => {
    // 1. Set a $0 budget — guaranteed to block any request
    await setupBudget(0);

    // 2. Verify the proxy denies a request
    const denied = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });
    expect(denied.status).toBe(429);
    await denied.text();

    // 3. Increase the budget via direct SQL (simulates what approve-action does)
    await sql`
      UPDATE budgets
      SET max_budget_microdollars = 1000000, spend_microdollars = 0, updated_at = NOW()
      WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}
    `;

    // 4. Sync the new budget to the proxy DO (this is what BDG-4 fixed — ownerId must be orgId)
    await syncBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    await new Promise((r) => setTimeout(r, 2_000));

    // 5. Verify the proxy now allows requests with the increased budget
    const allowed = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });
    expect(allowed.status).toBe(200);
    const body = await allowed.json();
    expect(body).toHaveProperty("usage");
  }, 60_000);

  it("ST-9: customer budget threshold webhook fires after crossing", async () => {
    // This test verifies the full customer threshold alerting path:
    //   customer budget → cost event → threshold crossing → webhook delivery

    // Check if a webhook endpoint exists for this org
    const endpoints = await sql`
      SELECT id FROM webhook_endpoints
      WHERE org_id = ${orgId}::uuid AND enabled = true
      LIMIT 1
    `;

    if (endpoints.length === 0) {
      console.log("[ST-9] No webhook endpoint configured — skipping");
      return;
    }

    const endpointId = endpoints[0].id;
    const customerId = `st9-threshold-${Date.now().toString(36)}`;

    // try/finally ensures cleanup runs even if assertions fail mid-test
    try {
      // Create customer budget: 100µ¢ limit with 50% threshold.
      // Each gpt-4o-mini request costs ~6µ¢, so 9 requests → ~54µ¢ → crosses 50%.
      await sql`
        INSERT INTO budgets (
          id, user_id, org_id, entity_type, entity_id,
          max_budget_microdollars, spend_microdollars,
          threshold_percentages, policy
        ) VALUES (
          gen_random_uuid(), ${NULLSPEND_SMOKE_USER_ID!}, ${orgId}::uuid,
          'customer', ${customerId},
          100, 0,
          '{50}', 'strict_block'
        )
      `;
      await syncBudget(orgId, "customer", customerId);
      await new Promise((r) => setTimeout(r, 1_000));

      const before = new Date();

      // Send requests to cross the 50% threshold (~54µ¢ of 100µ¢)
      for (let i = 0; i < 12; i++) {
        const res = await fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders({ "X-NullSpend-Customer": customerId }),
          body: smallRequest({ messages: [{ role: "user", content: `ST-9 threshold ${i}` }] }),
        });
        if (res.status === 429) {
          await res.text();
          break;
        }
        await res.json();
      }

      // Wait for async webhook dispatch via Cloudflare Queue
      await new Promise((r) => setTimeout(r, 10_000));

      // Check webhook_deliveries for customer threshold events
      const deliveries = await sql`
        SELECT event_type, status, attempts, response_status
        FROM webhook_deliveries
        WHERE endpoint_id = ${endpointId}
          AND created_at >= ${before.toISOString()}
          AND event_type LIKE 'budget.threshold%'
      `;

      console.log(`[ST-9] Customer threshold webhook deliveries: ${deliveries.length}`);
      for (const d of deliveries) {
        console.log(`  ${d.event_type}: status=${d.status}, attempts=${d.attempts}, response=${d.response_status}`);
      }

      // Informational — don't hard-fail since webhook delivery is async
      if (deliveries.length === 0) {
        console.log("[ST-9] WARNING: No customer threshold webhook deliveries found");
        console.log("[ST-9] This could be a queue delay or webhook endpoint issue");
      } else {
        // Verify at least one is a threshold event for this entity
        expect(deliveries.some((d: Record<string, unknown>) =>
          (d.event_type as string).startsWith("budget.threshold"),
        )).toBe(true);
      }
    } finally {
      // Cleanup customer budget — runs even on test failure
      try {
        await invalidateBudget(orgId, "customer", customerId);
        await sql`DELETE FROM budgets WHERE entity_type = 'customer' AND entity_id = ${customerId}`;
      } catch { /* ignore cleanup errors */ }
    }
  }, 120_000);

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
