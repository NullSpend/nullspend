/**
 * Feature-specific concurrency / burst / latency stress tests.
 *
 * Consolidated destination for stress tests that previously lived inside
 * `smoke-*.test.ts` files. Live-stack, production-mutating, manual runs
 * only — picked up by `vitest.stress.config.ts` (include: "stress-*.test.ts").
 *
 * See docs/internal/test-tier-taxonomy.md for the tier model.
 *
 * Provenance (per-source-file origin):
 *   - `smoke-edge-cases.test.ts`       → "Concurrent request handling" + 2 "Proxy resilience" burst tests
 *   - `smoke-cloudflare.test.ts`       → "Connection management" concurrent/rapid-fire tests + "waitUntil" rapid-succession test
 *   - `smoke-anthropic-edge-cases.test.ts` → 3 concurrent streaming/non-streaming/mixed tests
 *   - `smoke-mandates.test.ts`         → 2 concurrent policy/mandate-denial tests
 *   - `smoke-session-limits.test.ts`   → ST-4 concurrent same-session denial test
 *   - `smoke-trace.test.ts`            → "generates unique trace-ids for concurrent requests"
 *   - `smoke-anthropic-known-issues.test.ts` → 5 concurrent + 10 batched cost-event tests
 *   - `smoke-advanced.test.ts`         → 3 latency/overhead measurement tests
 *
 * Requires the same `.env.smoke` config as the original smoke suite:
 *   - PROXY_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID, NULLSPEND_SMOKE_KEY_ID, INTERNAL_SECRET, DATABASE_URL
 */
import { describe, it, expect, beforeAll, beforeEach, afterAll, afterEach } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  NULLSPEND_SMOKE_KEY_ID,
  INTERNAL_SECRET,
  DATABASE_URL,
  authHeaders,
  anthropicAuthHeaders,
  anthropicRateLimitPace,
  smallRequest,
  isServerUp,
  waitForCostEvent,
  invalidateBudget,
  syncBudget,
} from "./smoke-test-helpers.js";

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-edge-cases — concurrent request handling + burst
// ────────────────────────────────────────────────────────────────────────

describe("Extracted from smoke-edge-cases — concurrent request handling", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  it("handles 5 concurrent streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: `Say '${i}' and nothing else.` }],
          stream: true,
          max_tokens: 5,
        }),
      }),
    );

    const responses = await Promise.all(requests);

    for (const res of responses) {
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");
    }
  }, 60_000);

  it("handles 5 concurrent non-streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: `Say '${i}' and nothing else.` }],
          stream: false,
          max_tokens: 5,
        }),
      }),
    );

    const responses = await Promise.all(requests);

    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
    }
  }, 60_000);

  it("handles mixed streaming and non-streaming requests concurrently", async () => {
    const streamReq = fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'stream'" }],
        stream: true,
        max_tokens: 5,
      }),
    });

    const nonStreamReq = fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'nostream'" }],
        stream: false,
        max_tokens: 5,
      }),
    });

    const errorReq = fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ model: "not-real", messages: [{ role: "user", content: "hi" }] }),
    });

    const [streamRes, nonStreamRes, errorRes] = await Promise.all([
      streamReq,
      nonStreamReq,
      errorReq,
    ]);

    expect(streamRes.status).toBe(200);
    const streamText = await streamRes.text();
    expect(streamText).toContain("[DONE]");

    expect(nonStreamRes.status).toBe(200);
    const nonStreamBody = await nonStreamRes.json();
    expect(nonStreamBody).toHaveProperty("usage");

    expect(errorRes.status).toBeGreaterThanOrEqual(400);
  }, 60_000);
});

describe("Extracted from smoke-edge-cases — proxy resilience under burst", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  it("proxy is healthy after a burst of error requests", async () => {
    const errorRequests = Array.from({ length: 10 }, () =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "not json!!!",
      }),
    );

    const responses = await Promise.all(errorRequests);
    for (const res of responses) {
      expect(res.status).toBe(400);
      await res.text();
    }

    // Proxy should still be healthy
    const healthRes = await fetch(`${BASE}/health`);
    expect(healthRes.status).toBe(200);
  });

  it("proxy is healthy after a burst of auth failures", async () => {
    const authFailRequests = Array.from({ length: 10 }, () =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "wrong-key",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      }),
    );

    const responses = await Promise.all(authFailRequests);
    for (const res of responses) {
      expect(res.status).toBe(401);
      await res.text();
    }

    const healthRes = await fetch(`${BASE}/health`);
    expect(healthRes.status).toBe(200);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-cloudflare — connection management + rapid-fire
// ────────────────────────────────────────────────────────────────────────

describe("Extracted from smoke-cloudflare — connection management", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  it("rapid sequential requests reuse connections correctly", async () => {
    for (let i = 0; i < 5; i++) {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
    }
  }, 120_000);

  it("handles 3 concurrent streaming + 3 concurrent non-streaming requests", async () => {
    const streamReqs = Array.from({ length: 3 }, (_, _i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      }),
    );

    const nonStreamReqs = Array.from({ length: 3 }, (_, _i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      }),
    );

    const results = await Promise.all([...streamReqs, ...nonStreamReqs]);

    for (let i = 0; i < 3; i++) {
      expect(results[i].status).toBe(200);
      const text = await results[i].text();
      expect(text).toContain("[DONE]");
    }

    for (let i = 3; i < 6; i++) {
      expect(results[i].status).toBe(200);
      const body = await results[i].json();
      expect(body).toHaveProperty("usage");
    }
  }, 90_000);

  it("10 rapid-fire auth failures don't exhaust connections", async () => {
    const reqs = Array.from({ length: 10 }, () =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-nullspend-key": "bad-key",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: smallRequest(),
      }),
    );

    const results = await Promise.all(reqs);
    for (const res of results) {
      expect(res.status).toBe(401);
      await res.text();
    }

    // Verify connections are freed up
    const healthRes = await fetch(`${BASE}/health`);
    expect(healthRes.status).toBe(200);
  }, 15_000);
});

describe("Extracted from smoke-cloudflare — waitUntil cost-logging burst", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  it("multiple requests in rapid succession all get cost-logged without error", async () => {
    const requests = Array.from({ length: 3 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: i % 2 === 0,
          messages: [{ role: "user", content: `Count: ${i}` }],
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      await res.text();
    }

    // Brief pause for waitUntil to complete
    await new Promise((r) => setTimeout(r, 1000));

    const healthRes = await fetch(`${BASE}/health`);
    expect(healthRes.status).toBe(200);
  }, 60_000);
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-anthropic-edge-cases — concurrent streaming/non
// ────────────────────────────────────────────────────────────────────────

describe("Extracted from smoke-anthropic-edge-cases — concurrent requests", () => {
  // Pace requests to stay under tier-1 Anthropic RPM during full-suite runs.
  beforeEach(anthropicRateLimitPace);

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
  });

  it("handles 5 concurrent streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: `Concurrent stream ${i}` }],
          stream: true,
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("event: message_stop");
    }
  }, 60_000);

  it("handles 5 concurrent non-streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Concurrent ns ${i}` }],
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.type).toBe("message");
    }
  }, 60_000);

  it("handles mixed streaming and non-streaming requests concurrently", async () => {
    const requests = [
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed streaming" }],
          stream: true,
        }),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: "Mixed non-streaming" }],
        }),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed streaming 2" }],
          stream: true,
        }),
      }),
    ];

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      await res.text();
    }
  }, 60_000);
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-mandates — concurrent policy + mandate denials
// ────────────────────────────────────────────────────────────────────────
// Note: mirrors the source-file's beforeAll (set restrictions, wait for
// cache propagation) and afterAll (clear restrictions). The original file
// executes a big chunk of setup — we keep the same contract here so the
// concurrency tests exercise the same restricted state.

describe("Extracted from smoke-mandates — concurrent policy + mandate denials", () => {
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
    console.log("[stress-feature-concurrency/mandates] Restrictions set, polling policy endpoint for propagation...");
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
          console.log(`[stress-feature-concurrency/mandates] Restrictions visible after ${Math.round((Date.now() - pollStart) / 1000)}s`);
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 5_000));
    }

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
    console.log("[stress-feature-concurrency/mandates] Restrictions confirmed, running tests.");
  }, 180_000);

  async function retryUntilStatus(
    url: string,
    init: RequestInit,
    expectedStatus: number,
    maxAttempts = 10,
  ): Promise<Response> {
    for (let i = 0; i < maxAttempts; i++) {
      const res = await fetch(url, init);
      if (res.status === expectedStatus) return res;
      await res.text();
      await new Promise((r) => setTimeout(r, 500));
    }
    return fetch(url, init);
  }

  afterAll(async () => {
    // Clear restrictions
    await sql`
      UPDATE api_keys
      SET allowed_models = NULL, allowed_providers = NULL
      WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}
    `;
    for (let i = 0; i < 3; i++) {
      await syncBudget(orgId, "api_key", NULLSPEND_SMOKE_KEY_ID!).catch(() => {});
    }
    await sql.end();
  });

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
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-session-limits — ST-4 concurrent same-session
// ────────────────────────────────────────────────────────────────────────
// Mirrors the original file's beforeAll/afterAll/afterEach budget plumbing,
// plus the `setupBudgetWithSessionLimit` helper the test depends on.

describe("Extracted from smoke-session-limits — ST-4 concurrent same-session", () => {
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

    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;
  });

  afterEach(async () => {
    // Wait for in-flight reconciliations from prior test's requests.
    await new Promise((r) => setTimeout(r, 5_000));

    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* budget may not exist */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await new Promise((r) => setTimeout(r, 500));
  });

  afterAll(async () => {
    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* may not exist */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await sql.end();
  });

  async function setupBudgetWithSessionLimit(
    maxBudgetMicrodollars: number,
    sessionLimitMicrodollars: number,
  ) {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    try { await invalidateBudget(orgId, "user", userId); } catch { /* may not exist */ }

    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy, session_limit_microdollars)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, ${maxBudgetMicrodollars}, 0, 'strict_block', ${sessionLimitMicrodollars})
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxBudgetMicrodollars},
                    spend_microdollars = 0,
                    session_limit_microdollars = ${sessionLimitMicrodollars},
                    updated_at = NOW()
    `;

    await new Promise((r) => setTimeout(r, 500));
    await syncBudget(orgId, "user", userId);
  }

  it("ST-4: concurrent same-session requests all get denied after first exceeds limit", async () => {
    await setupBudgetWithSessionLimit(100_000_000, 1); // 1 microdollar session limit

    const sessionId = `smoke-burst-${Date.now()}`;
    const concurrency = 5;

    // Fire 5 concurrent requests on the SAME session
    const requests = Array.from({ length: concurrency }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "x-nullspend-session": sessionId }),
        body: smallRequest({ messages: [{ role: "user", content: `Burst ${i}` }] }),
      }),
    );

    const results = await Promise.all(requests);
    const statuses: number[] = [];
    for (const r of results) {
      statuses.push(r.status);
      await r.text();
    }

    const denied = statuses.filter((s) => s === 429).length;
    const succeeded = statuses.filter((s) => s === 200).length;

    console.log(`[ST-4] Same-session burst: ${succeeded} succeeded, ${denied} denied`);

    // With 1 microdollar limit, most should be denied.
    // Allow at most 2 successes (DO reservation race window).
    expect(denied).toBeGreaterThan(0);
    expect(succeeded).toBeLessThanOrEqual(2);
    // No 500s
    expect(statuses.filter((s) => s !== 200 && s !== 429).length).toBe(0);
  }, 60_000);
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-trace — unique trace-ids for concurrent requests
// ────────────────────────────────────────────────────────────────────────

describe("Extracted from smoke-trace — unique trace-ids under concurrency", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  const TRACE_HEADER = "X-NullSpend-Trace-Id";

  it("generates unique trace-ids for concurrent requests", async () => {
    const requests = Array.from({ length: 5 }, () =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      }),
    );
    const responses = await Promise.all(requests);
    const traceIds = responses.map((r) => r.headers.get(TRACE_HEADER));

    for (const id of traceIds) {
      expect(id).toMatch(/^[0-9a-f]{32}$/);
    }
    // All should be unique
    expect(new Set(traceIds).size).toBe(traceIds.length);

    // Consume bodies to avoid connection leaks
    await Promise.all(responses.map((r) => r.text()));
  }, 60_000);
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-anthropic-known-issues — concurrent cost-event guards
// ────────────────────────────────────────────────────────────────────────
// The source file's `setupBudget` helper is NOT needed for the two extracted
// tests — they don't touch budgets, only assert that cost events land in
// Postgres. We keep the SQL handle + orgId lookup so we can query cost_events.

describe("Extracted from smoke-anthropic-known-issues — concurrent cost-event guards", () => {
  let sql: postgres.Sql;

  // Pace requests to stay under tier-1 Anthropic RPM during full-suite runs.
  beforeEach(anthropicRateLimitPace);

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(DATABASE_URL, { max: 5, idle_timeout: 10 });

    // Parity with source-file setup — verify the smoke key has an org.
    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

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
});

// ────────────────────────────────────────────────────────────────────────
// Extracted from smoke-advanced — latency / proxy overhead measurement
// ────────────────────────────────────────────────────────────────────────

describe("Extracted from smoke-advanced — proxy overhead / latency", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error("Proxy dev server is not running. Start it with `pnpm proxy:dev`.");
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required.");
    }
  });

  it("non-streaming request completes in under 10 seconds", async () => {
    const start = performance.now();
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say hi" }],
        max_tokens: 3,
      }),
    });
    const elapsed = performance.now() - start;

    expect(res.status).toBe(200);
    expect(elapsed).toBeLessThan(10_000);
    await res.json();
  }, 15_000);

  it("auth rejection is fast (under 500ms)", async () => {
    const start = performance.now();
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": "wrong-key",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    });
    const elapsed = performance.now() - start;

    expect(res.status).toBe(401);
    expect(elapsed).toBeLessThan(500);
    await res.text();
  });

  it("body validation rejection is fast (under 500ms)", async () => {
    const start = performance.now();
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: "not json",
    });
    const elapsed = performance.now() - start;

    expect(res.status).toBe(400);
    expect(elapsed).toBeLessThan(500);
    await res.text();
  });
});
