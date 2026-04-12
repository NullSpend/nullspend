/**
 * End-to-end smoke tests for the customer primitive feature.
 * Tests the full stack: proxy → DB → budget enforcement → margins queries.
 *
 * Requires:
 *   - Deployed proxy at PROXY_URL
 *   - NULLSPEND_API_KEY valid
 *   - OPENAI_API_KEY valid
 *   - DATABASE_URL pointing to Supabase
 *   - Migration 0054 applied (customer_id column exists)
 *
 * Run: cd apps/proxy && npx vitest run --config vitest.smoke.config.ts smoke-customer-primitive.test.ts
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  DATABASE_URL,
  OPENAI_API_KEY,
  authHeaders,
  smallRequest,
  waitForCostEvent,
  invalidateBudget,
  syncBudget,
  isServerUp,
} from "./smoke-test-helpers.js";

let sql: postgres.Sql;
let SMOKE_ORG_ID: string;

// Unique customer IDs for this test run to avoid conflicts
const TEST_RUN_ID = Date.now().toString(36);
const CUSTOMER_ACME = `smoke-acme-${TEST_RUN_ID}`;
const CUSTOMER_BETA = `smoke-beta-${TEST_RUN_ID}`;
const CUSTOMER_INVALID = "acme corp with spaces"; // fails validation

/** Parse tags column — postgres.js returns JSONB as string on direct connection. */
function parseTags(raw: unknown): Record<string, string> {
  if (typeof raw === "string") return JSON.parse(raw);
  if (raw && typeof raw === "object") return raw as Record<string, string>;
  return {};
}

describe("customer primitive — end-to-end", () => {
  beforeAll(async () => {
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required");
    if (!(await isServerUp())) throw new Error(`Proxy not reachable at ${BASE}`);
    sql = postgres(DATABASE_URL, { max: 2, prepare: false });

    // Verify customer_id column exists
    const cols = await sql`
      SELECT column_name FROM information_schema.columns
      WHERE table_name = 'cost_events' AND column_name = 'customer_id'
    `;
    if (cols.length === 0) {
      throw new Error("customer_id column missing — apply migration 0054 first");
    }

    // Look up the real UUID org_id for the smoke user (required for budgets.org_id which is UUID type)
    const orgRows = await sql`
      SELECT DISTINCT org_id FROM budgets WHERE user_id = ${NULLSPEND_SMOKE_USER_ID} LIMIT 1
    `;
    if (orgRows.length === 0) {
      throw new Error(`No org_id found for smoke user ${NULLSPEND_SMOKE_USER_ID}`);
    }
    SMOKE_ORG_ID = String(orgRows[0].org_id);
  });

  afterAll(async () => {
    if (sql) {
      // Clean up any test budgets we created
      try {
        await sql`DELETE FROM budgets WHERE entity_type = 'customer' AND entity_id LIKE ${`smoke-%-${TEST_RUN_ID}`}`;
      } catch { /* ignore cleanup errors */ }
      await sql.end();
    }
  });

  // ── Layer 1: Header → customer_id column ────────────────────────

  describe("proxy header → customer_id column", () => {
    it("writes customer_id when X-NullSpend-Customer header is sent", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "X-NullSpend-Customer": CUSTOMER_ACME }),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-request-id") ?? (await res.json()).id;
      expect(requestId).toBeTruthy();

      const costEvent = await waitForCostEvent(sql, requestId, 20_000, "openai");
      expect(costEvent).not.toBeNull();
      expect(costEvent!.customer_id).toBe(CUSTOMER_ACME);
    });

    it("falls back to tags[customer] when header is absent", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({
          "X-NullSpend-Tags": JSON.stringify({ customer: CUSTOMER_BETA }),
        }),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-request-id") ?? (await res.json()).id;

      const costEvent = await waitForCostEvent(sql, requestId, 20_000, "openai");
      expect(costEvent).not.toBeNull();
      expect(costEvent!.customer_id).toBe(CUSTOMER_BETA);
    });

    it("header takes precedence over tag on conflict", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({
          "X-NullSpend-Customer": CUSTOMER_ACME,
          "X-NullSpend-Tags": JSON.stringify({ customer: "from-tag" }),
        }),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-request-id") ?? (await res.json()).id;

      const costEvent = await waitForCostEvent(sql, requestId, 20_000, "openai");
      expect(costEvent).not.toBeNull();
      expect(costEvent!.customer_id).toBe(CUSTOMER_ACME);
      // Tag should be auto-injected to match the header value
      const tags = parseTags(costEvent!.tags);
      expect(tags.customer).toBe(CUSTOMER_ACME);
    });

    it("invalid customer header sets X-NullSpend-Warning and skips customer_id", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "X-NullSpend-Customer": CUSTOMER_INVALID }),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get("X-NullSpend-Warning")).toBe("invalid_customer");
      const requestId = res.headers.get("x-request-id") ?? (await res.json()).id;

      const costEvent = await waitForCostEvent(sql, requestId, 20_000, "openai");
      expect(costEvent).not.toBeNull();
      expect(costEvent!.customer_id).toBeNull();
    });

    it("no customer header or tag results in null customer_id", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-request-id") ?? (await res.json()).id;

      const costEvent = await waitForCostEvent(sql, requestId, 20_000, "openai");
      expect(costEvent).not.toBeNull();
      expect(costEvent!.customer_id).toBeNull();
    });
  });

  // ── Layer 2: Customer budget enforcement (the DO fix) ───────────

  describe("customer budget enforcement via Durable Object", () => {
    // Two separate budget entities so we never need to mutate spend after creation.
    // The DO's populateIfEmpty is upsert-on-conflict for metadata only — it does NOT
    // overwrite spend on sync (production uses reconcile for that). Tests must use
    // tight initial budgets rather than UPDATE spend after the fact.
    const BUDGET_CUSTOMER_TIGHT = `smoke-budget-tight-${TEST_RUN_ID}`;
    const BUDGET_CUSTOMER_GENEROUS = `smoke-budget-generous-${TEST_RUN_ID}`;

    beforeAll(async () => {
      // Tight budget: 1 microdollar max → any real request exceeds it.
      await sql`
        INSERT INTO budgets (
          id, entity_type, entity_id, max_budget_microdollars, spend_microdollars,
          policy, threshold_percentages, user_id, org_id
        ) VALUES (
          gen_random_uuid(), 'customer', ${BUDGET_CUSTOMER_TIGHT}, 1, 0,
          'strict_block', ARRAY[50, 80, 90, 95], ${NULLSPEND_SMOKE_USER_ID}, ${SMOKE_ORG_ID}::uuid
        )
        ON CONFLICT DO NOTHING
      `;
      // Generous budget: $10 max → small requests pass cleanly.
      await sql`
        INSERT INTO budgets (
          id, entity_type, entity_id, max_budget_microdollars, spend_microdollars,
          policy, threshold_percentages, user_id, org_id
        ) VALUES (
          gen_random_uuid(), 'customer', ${BUDGET_CUSTOMER_GENEROUS}, 10000000, 0,
          'strict_block', ARRAY[50, 80, 90, 95], ${NULLSPEND_SMOKE_USER_ID}, ${SMOKE_ORG_ID}::uuid
        )
        ON CONFLICT DO NOTHING
      `;
      await syncBudget(SMOKE_ORG_ID, "customer", BUDGET_CUSTOMER_TIGHT);
      await syncBudget(SMOKE_ORG_ID, "customer", BUDGET_CUSTOMER_GENEROUS);
    });

    afterAll(async () => {
      try {
        await invalidateBudget(SMOKE_ORG_ID, "customer", BUDGET_CUSTOMER_TIGHT);
        await invalidateBudget(SMOKE_ORG_ID, "customer", BUDGET_CUSTOMER_GENEROUS);
        await sql`
          DELETE FROM budgets
          WHERE entity_type = 'customer'
            AND entity_id IN (${BUDGET_CUSTOMER_TIGHT}, ${BUDGET_CUSTOMER_GENEROUS})
        `;
      } catch { /* ignore */ }
    });

    it("denies request with 429 when customer budget is exhausted", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "X-NullSpend-Customer": BUDGET_CUSTOMER_TIGHT }),
        body: smallRequest(),
      });
      expect(res.status).toBe(429);
      const body = await res.json();
      expect(body.error?.code).toBe("customer_budget_exceeded");
      expect(body.error?.details?.customer_id).toBe(BUDGET_CUSTOMER_TIGHT);
    });

    it("allows request when customer budget has remaining headroom", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "X-NullSpend-Customer": BUDGET_CUSTOMER_GENEROUS }),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
    });
  });

  // ── Layer 3: Margins query with coalesce fallback ────────────────

  describe("margins queries handle both customer_id and tag fallback", () => {
    it("queries can read both customer_id and tag-derived customer IDs", async () => {
      // The actual validation is that the earlier tests populated both columns
      // and the queries below work without errors.
      const rows = await sql`
        SELECT coalesce(customer_id, tags->>'customer') as cid, count(*)::int as cnt
        FROM cost_events
        WHERE (customer_id IS NOT NULL OR tags ? 'customer')
          AND created_at > now() - interval '10 minutes'
        GROUP BY coalesce(customer_id, tags->>'customer')
      `;
      // Should have at least the test customers
      const customerIds = rows.map((r) => r.cid as string);
      expect(customerIds).toContain(CUSTOMER_ACME);
      expect(customerIds).toContain(CUSTOMER_BETA);
    });
  });
});
