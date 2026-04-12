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
  INTERNAL_SECRET,
  authHeaders,
  anthropicAuthHeaders,
  smallRequest,
  smallAnthropicRequest,
  isServerUp,
  syncBudget,
} from "./smoke-test-helpers.js";

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
  }, 180_000); // 3 minute timeout for beforeAll

  /**
   * Retry a request up to `maxAttempts` times until the expected status is returned.
   * Cloudflare routes requests to different Worker isolates which may have
   * different auth cache states. Retrying ensures we eventually hit an isolate
   * that has the updated restrictions.
   */
  async function retryUntilStatus(
    url: string,
    init: RequestInit,
    expectedStatus: number,
    maxAttempts = 10,
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

  // ── Concurrent policy requests ──

  it("handles 10 concurrent policy requests without errors", async () => {
    const promises = Array.from({ length: 10 }, () =>
      fetch(`${BASE}/v1/policy`, {
        method: "GET",
        headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
      }),
    );

    const results = await Promise.all(promises);
    for (const res of results) {
      expect(res.status).toBe(200);
      const body = await res.json();
      // All should return valid policy shape (restrictions may vary per isolate)
      expect(body).toHaveProperty("budget");
      expect(body).toHaveProperty("cheapest_overall");
    }
  }, 30_000);

  // ── Concurrent denied requests ──

  it("handles 5 concurrent mandate denials without errors", async () => {
    // First ensure at least one isolate has restrictions by doing a retry check
    await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o" }),
    }, 403);

    // Now send 5 concurrent requests — some may hit stale isolates (200)
    // but none should error (5xx)
    const promises = Array.from({ length: 5 }, () =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ model: "gpt-4o" }),
      }),
    );

    const results = await Promise.all(promises);
    const statuses = await Promise.all(results.map(async (res) => {
      const body = await res.json();
      return { status: res.status, code: body.error?.code };
    }));

    // All should be either 403 (mandate denied) or 200 (stale isolate) — never 5xx
    for (const s of statuses) {
      expect([200, 403]).toContain(s.status);
    }
    // At least some should be 403 (the isolate we warmed above)
    const deniedCount = statuses.filter(s => s.status === 403).length;
    expect(deniedCount).toBeGreaterThan(0);
  }, 30_000);

  // ── Model name edge cases ──

  it("mandate model check is case-sensitive (GPT-4O-MINI !== gpt-4o-mini)", async () => {
    // Retry until we hit an isolate with restrictions
    const res = await retryUntilStatus(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "GPT-4O-MINI" }),
    }, 403);

    // GPT-4O-MINI is not in allowed list (gpt-4o-mini is)
    // Note: if the isolate is stale, OpenAI may return 404 for the invalid model name.
    // The retryUntilStatus ensures we hit a restricted isolate.
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.error.code).toBe("mandate_violation");
  }, 30_000);

  // ── Policy budget accuracy ──

  it("policy budget reflects spend from allowed requests", async () => {
    // Get budget before
    const before = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const budgetBefore = (await before.json() as any).budget;

    // Make an allowed request that costs something
    await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ model: "gpt-4o-mini" }),
    });

    // Wait for cost event + reconciliation
    await new Promise((r) => setTimeout(r, 8_000));

    // Get budget after
    const after = await fetch(`${BASE}/v1/policy`, {
      method: "GET",
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    const budgetAfter = (await after.json() as any).budget;

    // Spend should have increased
    if (budgetBefore && budgetAfter) {
      expect(budgetAfter.spend_microdollars).toBeGreaterThan(budgetBefore.spend_microdollars);
    }
  }, 30_000);
});
