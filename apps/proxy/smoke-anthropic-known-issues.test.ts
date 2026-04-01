/**
 * Anthropic known-issues smoke tests.
 * Validates cost logging reliability and budget enforcement —
 * targeting specific bugs discovered through research across
 * Cloudflare Workers, Hyperdrive, and the Anthropic API.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL for direct Supabase queries
 */
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import postgres from "postgres";
import {
  BASE,
  ANTHROPIC_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  NULLSPEND_SMOKE_KEY_ID,
  DATABASE_URL,
  anthropicAuthHeaders,
  smallAnthropicRequest,
  isServerUp,
  waitForCostEvent,
  syncBudget,
} from "./smoke-test-helpers.js";

describe("Anthropic known issues: cost logging & budget edge cases", () => {
  let sql: postgres.Sql;
  let orgId: string;
  const usersToCleanup: string[] = [];

  function trackUser(id: string) {
    if (!usersToCleanup.includes(id)) usersToCleanup.push(id);
  }

  async function setupBudget(
    userId: string,
    maxBudgetMicrodollars: number,
    spendMicrodollars = 0,
  ) {
    trackUser(userId);

    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, ${spendMicrodollars}, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;
  }

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(DATABASE_URL, { max: 5, idle_timeout: 10 });

    // Look up org_id from the smoke test API key (required NOT NULL since Phase 2)
    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;
  });

  afterEach(async () => {
    // Clean up budgets between tests
    for (const id of usersToCleanup) {
      await sql`DELETE FROM budgets WHERE entity_id = ${id}`;
    }
    usersToCleanup.length = 0;
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  // --- Cost logging reliability ---

  it("5 concurrent Anthropic requests all produce cost events (connection concurrency guard)", async () => {
    const requestIds: string[] = [];

    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Connection guard ${i}` }],
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      requestIds.push(res.headers.get("x-request-id") ?? body.id);
    }

    await new Promise((r) => setTimeout(r, 10_000));

    let foundCount = 0;
    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 15_000, "anthropic");
      if (row) foundCount++;
    }

    expect(foundCount).toBe(5);
  }, 120_000);

  it("10 requests in 2 batches all produce cost events (waitUntil reliability)", async () => {
    const allRequestIds: string[] = [];
    const BATCH_SIZE = 5;
    const BATCHES = 2;

    for (let batch = 0; batch < BATCHES; batch++) {
      const requests = Array.from({ length: BATCH_SIZE }, (_, i) =>
        fetch(`${BASE}/v1/messages`, {
          method: "POST",
          headers: anthropicAuthHeaders(),
          body: JSON.stringify({
            model: "claude-3-haiku-20240307",
            max_tokens: 5,
            messages: [
              {
                role: "user",
                content: `waitUntil batch ${batch} req ${i}`,
              },
            ],
          }),
        }),
      );

      const responses = await Promise.all(requests);
      for (const res of responses) {
        expect(res.status).toBe(200);
        const body = await res.json();
        allRequestIds.push(res.headers.get("x-request-id") ?? body.id);
      }

      await new Promise((r) => setTimeout(r, 1_500));
    }

    await new Promise((r) => setTimeout(r, 15_000));

    let foundCount = 0;
    for (const id of allRequestIds) {
      const row = await waitForCostEvent(sql, id, 10_000, "anthropic");
      if (row) foundCount++;
    }

    expect(foundCount).toBe(BATCH_SIZE * BATCHES);
  }, 180_000);

  it("3 sequential Anthropic requests with 1s delays all produce cost events (Hyperdrive reuse)", async () => {
    const requestIds: string[] = [];

    for (let i = 0; i < 3; i++) {
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [
            {
              role: "user",
              content: `Sequential ${i} at ${Date.now()}`,
            },
          ],
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      requestIds.push(res.headers.get("x-request-id") ?? body.id);

      await new Promise((r) => setTimeout(r, 1_000));
    }

    await new Promise((r) => setTimeout(r, 8_000));

    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 10_000, "anthropic");
      expect(row).not.toBeNull();
      expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    }
  }, 60_000);

  it("two distinct Anthropic requests produce exactly 2 distinct cost events (no duplicates)", async () => {
    const requestIds: string[] = [];

    for (let i = 0; i < 2; i++) {
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [
            {
              role: "user",
              content: `Distinct event check ${i} ${Date.now()}`,
            },
          ],
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      requestIds.push(res.headers.get("x-request-id") ?? body.id);
    }

    await new Promise((r) => setTimeout(r, 8_000));

    for (const id of requestIds) {
      const rows = await sql`
        SELECT COUNT(*)::int as count FROM cost_events
        WHERE request_id = ${id} AND provider = 'anthropic'
      `;
      expect(rows[0].count).toBe(1);
    }

    expect(requestIds[0]).not.toBe(requestIds[1]);
  }, 60_000);

  // --- Budget edge cases ---

  it("budget enforced on /v1/messages route (no bypass)", async () => {
    const userId = NULLSPEND_SMOKE_USER_ID!;
    await setupBudget(userId, 1); // 1 microdollar — guaranteed denial
    await syncBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    await new Promise((r) => setTimeout(r, 1_000)); // settle

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
  }, 15_000);
});
