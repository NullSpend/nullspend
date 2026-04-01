/**
 * End-to-end budget enforcement tests for Anthropic.
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
 *   - ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID (real userId for the test API key)
 *   - INTERNAL_SECRET (for cache invalidation)
 *   - DATABASE_URL for budget setup in Postgres
 */
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import postgres from "postgres";
import {
  BASE,
  ANTHROPIC_API_KEY,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  NULLSPEND_SMOKE_KEY_ID,
  INTERNAL_SECRET,
  anthropicAuthHeaders,
  smallAnthropicRequest,
  isServerUp,
  invalidateBudget,
  syncBudget,
} from "./smoke-test-helpers.js";

describe("Anthropic end-to-end budget enforcement", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
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

  afterEach(async () => {
    // Wait for waitUntil reconciliation to complete before cleanup
    await new Promise((r) => setTimeout(r, 5_000));

    await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
  });

  afterAll(async () => {
    await sql.end();
  });

  async function setupBudget(maxBudgetMicrodollars: number, spendMicrodollars = 0) {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, ${spendMicrodollars}, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;

    // Force Postgres→DO sync via internal endpoint.
    // Under DO-first architecture, no Worker-level cache to invalidate.
    await syncBudget(orgId, "user", userId);
  }

  async function cleanupBudget() {
    const userId = NULLSPEND_SMOKE_USER_ID!;
    await invalidateBudget(orgId, "user", userId);
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${userId}`;
  }

  it("allows request when budget has sufficient funds", async () => {
    await setupBudget(10_000_000); // $10

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("usage");
  }, 30_000);

  it("blocks request with budget_exceeded when budget is $0", async () => {
    await setupBudget(0);

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
    expect(body.error.message).toContain("budget");
  }, 30_000);

  it("blocks when spend already equals maxBudget", async () => {
    await setupBudget(100_000, 100_000);

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
  }, 30_000);

  it("reconciliation adjusts spend after request completes", async () => {
    await setupBudget(5_000_000); // $5

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(200);
    await res.json();

    // Wait for reconciliation
    await new Promise((r) => setTimeout(r, 10_000));

    const rows = await sql`
      SELECT spend_microdollars::text as spend
      FROM budgets
      WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}
    `;
    expect(rows.length).toBe(1);
    expect(Number(rows[0].spend)).toBeGreaterThan(0);
  }, 30_000);

  it("budget_exceeded response includes correct error fields", async () => {
    await setupBudget(1); // 1 microdollar

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(429);
    const body = await res.json();

    expect(body.error.code).toBe("budget_exceeded");
    expect(body.error.message).toContain("budget");
    expect(body.error.details).toBeNull();
  }, 30_000);

  it("requests without configured budget are not affected by enforcement", async () => {
    await cleanupBudget();

    await new Promise((r) => setTimeout(r, 2_000));

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("usage");
  }, 30_000);
});
