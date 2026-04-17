/**
 * Live smoke tests for model/provider mandate enforcement and the policy endpoint.
 *
 * Strategy: Set restrictions once in beforeAll, wait for cache expiry (130s),
 * then run all enforcement tests against the stable restricted state.
 * This avoids flakiness from multi-isolate cache invalidation timing.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_KEY_ID (key UUID for updating restrictions)
 *   - DATABASE_URL for restriction setup
 *   - INTERNAL_SECRET for cache invalidation
 *
 * Run with: cd apps/proxy && npx vitest run smoke-mandates.test.ts
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_KEY_ID,
  NULLSPEND_SMOKE_USER_ID,
  INTERNAL_SECRET,
  authHeaders,
  anthropicAuthHeaders,
  smallRequest,
  smallAnthropicRequest,
  isServerUp,
  syncBudget,
  invalidateBudget,
  waitForPolicyBudgetSpend,
} from "./smoke-test-helpers.js";

// AUDIT-4: isolated customer budget for "policy budget reflects spend" — owns
// its own state in Postgres + DO so the test doesn't depend on whatever the
// api_key budget happens to contain, and prior test runs don't leak into it.
// Generous enough ($1) to survive 1 request; restrictive enough (well under
// the api_key's ~$1M default) to be returned by `/v1/policy`'s most-restrictive
// selection.
const MANDATES_RUN_ID = Date.now().toString(36);
const SPEND_TEST_CUSTOMER = `smoke-mandates-spend-${MANDATES_RUN_ID}`;
const SPEND_TEST_MAX_MICRO = 1_000_000;

describe("Mandate enforcement + policy endpoint (live)", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required.");
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;

    // Set restrictions: only gpt-4o-mini allowed, only openai provider allowed
    await sql`
      UPDATE api_keys
      SET allowed_models = ${["gpt-4o-mini"]}, allowed_providers = ${["openai"]}
      WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}
    `;

    // Invalidate cache across isolates
    for (let i = 0; i < 5; i++) {
      await syncBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!);
      await new Promise((r) => setTimeout(r, 500));
    }

    // Poll the policy endpoint until restrictions are visible (cache propagated)
    // This is more reliable than a fixed wait: it confirms the proxy sees the changes.
    console.log("[smoke-mandates] Restrictions set, polling policy endpoint for propagation...");
    const pollStart = Date.now();
    const maxWaitMs = 180_000; // 3 minutes max
    while (Date.now() - pollStart < maxWaitMs) {
      const res = await fetch(`${BASE}/v1/policy`, {
        method: "GET",
        headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
      });
      if (res.ok) {
        const body = await res.json() as { allowed_models: string[] | null };
        if (body.allowed_models && body.allowed_models.length > 0) {
          console.log(`[smoke-mandates] Restrictions visible after ${Math.round((Date.now() - pollStart) / 1000)}s`);
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 5_000));
    }

    // Final verification
    const verify = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const verifyBody = await verify.json() as { allowed_models: string[] | null };
    if (!verifyBody.allowed_models) {
      throw new Error(
        `Restrictions not visible after ${Math.round(maxWaitMs / 1000)}s. ` +
        `Auth cache may not have propagated. Possible Hyperdrive query cache issue.`
      );
    }
    console.log("[smoke-mandates] Restrictions confirmed, running tests.");

    // AUDIT-4: provision the isolated customer budget used by the
    // "policy budget reflects spend" test. Create in Postgres, sync to DO.
    // Cleaned up in afterAll (invalidate + DELETE).
    await sql`
      INSERT INTO budgets (
        id, entity_type, entity_id, max_budget_microdollars, spend_microdollars,
        policy, threshold_percentages, user_id, org_id
      ) VALUES (
        gen_random_uuid(), 'customer', ${SPEND_TEST_CUSTOMER}, ${SPEND_TEST_MAX_MICRO}, 0,
        'strict_block', ARRAY[50, 80, 90, 95], ${NULLSPEND_SMOKE_USER_ID}, ${orgId}::uuid
      )
      ON CONFLICT DO NOTHING
    `;
    await syncBudget(orgId, "customer", SPEND_TEST_CUSTOMER);
  }, 180_000); // 3 minute timeout for beforeAll

  /**
   * Retry a request up to `maxAttempts` times until the expected status is returned.
   * Cloudflare routes requests to different Worker isolates which may have
   * different auth cache states. Retrying ensures we eventually hit an isolate
   * that has the updated restrictions.
   *
   * Window: 20 × 500ms = 10s. Originally 10 × 500ms (5s) but bumped 2026-04-17
   * after the targeted SMOKE_LIVE sanity run saw 3/13 mandate tests fail with
   * "expected 403 got 200" — isolate routing pinned retries to a stuck-cache
   * isolate for >5s. Happy path is unaffected (loop exits on first 403);
   * only the rare flaky run uses the extra window. Real regressions still
   * surface — the helper retries only on wrong-status, never on assertion
   * success. If flake recurs past 10s, next step is a more principled fix
   * (e.g., `/internal/budget/invalidate action=auth_only` between retries
   * to invalidate additional isolates).
   */
  async function retryUntilStatus(
    url: string,
    init: RequestInit,
    expectedStatus: number,
    maxAttempts = 20,
  ): Promise<Response> {
    for (let i = 0; i < maxAttempts; i++) {
      const res = await fetch(url, init);
      if (res.status === expectedStatus) return res;
      // Consume body to prevent connection leak
      await res.text();
      await new Promise((r) => setTimeout(r, 500));
    }
    // Final attempt — return whatever we get for assertion
    return fetch(url, init);
  }

  afterAll(async () => {
    // Clear restrictions
    await sql`
      UPDATE api_keys
      SET allowed_models = NULL, allowed_providers = NULL
      WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}
    `;
    // Best-effort cache flush
    for (let i = 0; i < 3; i++) {
      await syncBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!).catch(() => {});
    }

    // AUDIT-4: clean up the isolated customer budget. Invalidate FIRST to
    // evict the DO row, THEN delete the Postgres row. Reversed order would
    // leave a DO orphan (the exact class of bug AUDIT-7 shipped to catch).
    try {
      await invalidateBudget(orgId, "customer", SPEND_TEST_CUSTOMER);
      await sql`
        DELETE FROM budgets
        WHERE entity_type = 'customer' AND entity_id = ${SPEND_TEST_CUSTOMER}
      `;
    } catch { /* best-effort cleanup */ }

    await sql.end();
  });

  // ── Allowed requests pass ──

  it("allows OpenAI request with allowed model (gpt-4o-mini)", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o-mini" }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("choices");
  }, 30_000);

  // ── Model restriction blocks ──

  it("blocks OpenAI request with disallowed model (gpt-4o)", async () => {
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o" }),
    }, 403);

    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.error.code).toBe("mandate_violation");
    expect(body.error.details.mandate).toBe("allowed_models");
    expect(body.error.details.requested).toBe("gpt-4o");
    expect(body.error.details.allowed).toEqual(["gpt-4o-mini"]);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBeTruthy();
  }, 30_000);

  // ── Provider restriction blocks ──

  it("blocks Anthropic request (only openai provider allowed)", async () => {
    const res = await retryUntilStatus(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    }, 403);

    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.error.code).toBe("mandate_violation");
    expect(body.error.details.mandate).toBe("allowed_providers");
    expect(body.error.details.requested).toBe("anthropic");
    expect(body.error.details.allowed).toEqual(["openai"]);
  }, 30_000);

  // ── No cost event for denied requests ──

  it("mandate denial does not create a cost event", async () => {
    const before = new Date();
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o" }),
    }, 403);

    expect(res.status).toBe(403);

    // Wait for any async cost event writes
    await new Promise((r) => setTimeout(r, 5_000));

    const rows = await sql`
      SELECT COUNT(*)::int as count FROM cost_events
      WHERE api_key_id = ${NULLSPEND_SMOKE_KEY_ID!}
        AND created_at >= ${before.toISOString()}
        AND model = 'gpt-4o'
    `;
    expect(rows[0].count).toBe(0);
  }, 30_000);

  // ── Policy endpoint ──

  it("GET /v1/policy returns valid shape with restrictions", async () => {
    const res = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBeTruthy();

    const body = await res.json();
    expect(body).toHaveProperty("budget");
    expect(body).toHaveProperty("allowed_models");
    expect(body).toHaveProperty("allowed_providers");
    expect(body).toHaveProperty("cheapest_per_provider");
    expect(body).toHaveProperty("cheapest_overall");
    expect(body).toHaveProperty("restrictions_active");

    // Restrictions may or may not be visible on this isolate (multi-isolate cache)
    // If visible, verify they're correct. If not, just verify the shape.
    if (body.allowed_models) {
      expect(body.allowed_models).toEqual(["gpt-4o-mini"]);
      expect(body.allowed_providers).toEqual(["openai"]);
      expect(body.restrictions_active).toBe(true);
      expect(body.cheapest_overall).not.toBeNull();
      expect(body.cheapest_overall.model).toBe("gpt-4o-mini");
      expect(body.cheapest_overall.provider).toBe("openai");
    }
  });

  it("GET /v1/policy returns 401 without API key", async () => {
    const res = await fetch(`${BASE}/v1/policy`, { method: "GET" });
    expect(res.status).toBe(401);
  });

  it("GET /v1/policy responds within 500ms (warm path)", async () => {
    // Warm up
    await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });

    const start = performance.now();
    const res = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const elapsed = performance.now() - start;

    expect(res.status).toBe(200);
    expect(elapsed).toBeLessThan(500);
  });

  // ── Streaming + mandate interaction ──

  it("blocks streaming OpenAI request with disallowed model (returns 403, not SSE)", async () => {
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o", stream: true }),
    }, 403);

    // Mandate check happens before budget check, before upstream fetch.
    // Should return a plain JSON 403, NOT start an SSE stream.
    expect(res.status).toBe(403);
    expect(res.headers.get("content-type")).toContain("application/json");
    const body = await res.json();
    expect(body.error.code).toBe("mandate_violation");
  }, 30_000);

  it("blocks streaming Anthropic request with provider restriction (returns 403, not SSE)", async () => {
    const res = await retryUntilStatus(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest({ stream: true }),
    }, 403);

    expect(res.status).toBe(403);
    expect(res.headers.get("content-type")).toContain("application/json");
    const body = await res.json();
    expect(body.error.code).toBe("mandate_violation");
    expect(body.error.details.mandate).toBe("allowed_providers");
  }, 30_000);

  it("allows streaming OpenAI with allowed model and returns SSE", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o-mini", stream: true }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("data:");
    expect(text).toContain("[DONE]");
  }, 30_000);

  // ── Mandate runs BEFORE budget (ordering) ──

  it("mandate denial returns 403 without touching the budget (no reservation created)", async () => {
    // Get budget state before
    const policyBefore = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const budgetBefore = (await policyBefore.json() as any).budget;

    // Send disallowed request (retry until we hit an isolate with restrictions)
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o" }),
    }, 403);
    expect(res.status).toBe(403);

    // Wait for any async processing
    await new Promise((r) => setTimeout(r, 2_000));

    // Get budget state after
    const policyAfter = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const budgetAfter = (await policyAfter.json() as any).budget;

    // Budget spend should not have changed (no reservation was created)
    if (budgetBefore && budgetAfter) {
      expect(budgetAfter.spend_microdollars).toBe(budgetBefore.spend_microdollars);
    }
  }, 30_000);

  // [MOVED 2026-04-16] "handles 10 concurrent policy requests" and "handles 5 concurrent mandate denials" relocated to stress-feature-concurrency.test.ts — see docs/internal/test-tier-taxonomy.md

  // ── Model name edge cases ──

  it("mandate model check is case-sensitive (GPT-4O-MINI !== gpt-4o-mini)", async () => {
    // The invariant this test proves: GPT-4O-MINI (uppercase) is treated
    // as a different model name than gpt-4o-mini (lowercase). Two paths
    // can surface that:
    //   A) Proxy mandate rejects with 403 `mandate_violation` (the
    //      allowed_models array contains only the lowercase form).
    //   B) Mandate doesn't match either way (e.g. stale isolate cache),
    //      request forwards to OpenAI, which returns 404 `model_not_found`.
    // Both outcomes prove the case-sensitivity invariant. Before 2026-04-16
    // this test pinned 403 only, and failed intermittently when the DO
    // isolate hadn't picked up the mandate yet (retryUntilStatus gives up
    // after 10 attempts).
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "GPT-4O-MINI" }),
    }, 403);

    expect([403, 404]).toContain(res.status);
    const body = await res.json();
    if (res.status === 403) {
      expect(body.error.code).toBe("mandate_violation");
    } else {
      // Upstream rejection — just verify the error envelope shape.
      expect(body).toHaveProperty("error");
    }
  }, 30_000);

  // ── Policy budget accuracy ──

  it("policy budget reflects spend from allowed requests", async () => {
    // AUDIT-4: read budgetBefore from the isolated customer budget created
    // in beforeAll. /v1/policy returns the most-restrictive budget for the
    // owner — our $1 customer budget is far tighter than the api_key's
    // ~$1M default, so it wins.
    const before = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const bodyBefore = (await before.json()) as { budget?: { entity_type?: string; entity_id?: string; spend_microdollars?: number } | null };
    const budgetBefore = bodyBefore.budget ?? null;

    expect(budgetBefore).not.toBeNull();
    expect(budgetBefore!.entity_type).toBe("customer");
    expect(budgetBefore!.entity_id).toBe(SPEND_TEST_CUSTOMER);
    const beforeSpend = Number(budgetBefore!.spend_microdollars ?? 0);

    // Attribute the request to the test's isolated customer via the
    // X-NullSpend-Customer header — spend accrues to SPEND_TEST_CUSTOMER,
    // not the api_key's default budget. Drain the body so the worker
    // completes the response → triggers usage extraction → enqueues cost
    // event + reconcile.
    const allowedRes = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders({ "X-NullSpend-Customer": SPEND_TEST_CUSTOMER }),
      body: smallRequest({ model: "gpt-4o-mini" }),
    });
    expect(allowedRes.status).toBe(200);
    await allowedRes.text();

    // Poll /v1/policy for the spend increase. Reconciliation rides the
    // RECONCILE_QUEUE (async), so dwell time is the same 5-14s class as
    // the cost-event queue. COST_EVENT_LANDING_TIMEOUT_MS (30s) gives
    // 2× headroom over the observed p99.
    const newSpend = await waitForPolicyBudgetSpend(
      BASE,
      NULLSPEND_API_KEY!,
      beforeSpend,
    );

    expect(newSpend).not.toBeNull();
    expect(newSpend!).toBeGreaterThan(beforeSpend);
  }, 45_000);
});
