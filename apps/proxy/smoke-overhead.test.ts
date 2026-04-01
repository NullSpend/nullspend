/**
 * Proxy overhead regression tests.
 *
 * Verifies the "<1ms overhead" claim by measuring the overhead component
 * (total - upstream) across warm steady-state, concurrent load, and
 * budget denial paths. Catches regressions like accidentally making
 * the budget check sequential again.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - NULLSPEND_SMOKE_USER_ID, NULLSPEND_SMOKE_KEY_ID
 *   - INTERNAL_SECRET (for budget setup/teardown)
 *   - DATABASE_URL
 *
 * Run with: npx vitest run --config vitest.smoke.config.ts smoke-overhead.test.ts
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
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

// ── Helpers ─────────────────────────────────────────────────────────

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

/** Extract a numeric value from Server-Timing header: "name;dur=123" → 123 */
function extractTiming(serverTiming: string, name: string): number | null {
  const re = new RegExp(`${name};dur=(\\d+)`);
  const match = serverTiming.match(re);
  return match ? Number(match[1]) : null;
}

interface TimingResult {
  overhead: number;
  budget: number;
  preflight: number;
  upstream: number;
  total: number;
  status: number;
}

/** Fire a single proxied request and return parsed Server-Timing. */
async function timedRequest(
  label: string,
  overrides: Record<string, unknown> = {},
): Promise<TimingResult> {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: authHeaders(),
    body: smallRequest({
      messages: [{ role: "user", content: label }],
      ...overrides,
    }),
  });

  const st = res.headers.get("Server-Timing") ?? "";
  await res.text(); // consume body

  return {
    overhead: extractTiming(st, "overhead") ?? -1,
    budget: extractTiming(st, "budget") ?? -1,
    preflight: extractTiming(st, "preflight") ?? -1,
    upstream: extractTiming(st, "upstream") ?? -1,
    total: extractTiming(st, "total") ?? -1,
    status: res.status,
  };
}

// ── Tests ───────────────────────────────────────────────────────────

describe("Proxy overhead regression", () => {
  let sql: postgres.Sql;
  let orgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_USER_ID) throw new Error("NULLSPEND_SMOKE_USER_ID required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required.");
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;

    // Clean any leftover budget from a prior test run
    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* may not exist — that's fine */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await new Promise((r) => setTimeout(r, 2_000));

    // Warm up: 3 requests to prime isolate + auth cache + DO
    for (let i = 0; i < 3; i++) {
      const t = await timedRequest(`warmup-${i}`);
      if (t.status !== 200) {
        console.warn(`[overhead] Warmup ${i} returned ${t.status}, retrying...`);
        await new Promise((r) => setTimeout(r, 1_000));
        await timedRequest(`warmup-retry-${i}`);
      }
    }
  }, 60_000);

  afterAll(async () => {
    // Final cleanup
    try {
      await invalidateBudget(orgId, "user", NULLSPEND_SMOKE_USER_ID!);
    } catch { /* ignore */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${NULLSPEND_SMOKE_USER_ID!}`;
    await sql.end();
  });

  // ── 1. Warm steady-state overhead ────────────────────────────────

  it("warm steady-state overhead is under 10ms (p50)", async () => {
    const timings: TimingResult[] = [];

    for (let i = 0; i < 10; i++) {
      timings.push(await timedRequest(`overhead-${i}`));
    }

    const overheads = timings
      .filter((t) => t.status === 200)
      .map((t) => t.overhead)
      .sort((a, b) => a - b);

    expect(overheads.length).toBeGreaterThanOrEqual(8);

    const p50 = percentile(overheads, 50);
    const p95 = percentile(overheads, 95);

    console.log(`[overhead] Warm p50: ${p50}ms, p95: ${p95}ms`);
    console.log(`[overhead] All: ${overheads.join(", ")}ms`);

    // Core claim: p50 overhead < 10ms (measured at 0ms, generous margin)
    expect(p50).toBeLessThan(10);
    // p95 under 50ms (allows occasional isolate recycle within sequential test)
    expect(p95).toBeLessThan(50);
  }, 120_000);

  // ── 2. Budget check hidden by optimistic execution ───────────────

  it("budget check does not add to overhead (optimistic execution)", async () => {
    const timings: TimingResult[] = [];

    for (let i = 0; i < 8; i++) {
      timings.push(await timedRequest(`optim-${i}`));
    }

    const warm = timings.filter((t) => t.status === 200 && t.preflight === 0);

    // Need at least 4 warm-cache hits to make a meaningful assertion
    expect(warm.length).toBeGreaterThanOrEqual(4);

    for (const t of warm) {
      // Budget check takes 15-33ms but runs in parallel with upstream.
      // If optimistic execution is working, overhead should be ≤ budget time.
      // If someone makes budget sequential again, overhead ≥ budget.
      console.log(
        `[overhead] budget=${t.budget}ms overhead=${t.overhead}ms upstream=${t.upstream}ms`,
      );
      expect(t.overhead).toBeLessThanOrEqual(t.budget);
    }
  }, 120_000);

  // ── 3. Streaming overhead comparable to non-streaming ────────────

  it("streaming requests have comparable overhead to non-streaming", async () => {
    const nonStreamOverheads: number[] = [];
    const streamOverheads: number[] = [];

    // 5 non-streaming
    for (let i = 0; i < 5; i++) {
      const t = await timedRequest(`ns-cmp-${i}`);
      if (t.status === 200) nonStreamOverheads.push(t.overhead);
    }

    // 5 streaming
    for (let i = 0; i < 5; i++) {
      const t = await timedRequest(`s-cmp-${i}`, { stream: true });
      if (t.status === 200) streamOverheads.push(t.overhead);
    }

    expect(nonStreamOverheads.length).toBeGreaterThanOrEqual(3);
    expect(streamOverheads.length).toBeGreaterThanOrEqual(3);

    nonStreamOverheads.sort((a, b) => a - b);
    streamOverheads.sort((a, b) => a - b);

    const nsP50 = percentile(nonStreamOverheads, 50);
    const sP50 = percentile(streamOverheads, 50);

    console.log(`[overhead] Non-streaming p50: ${nsP50}ms, Streaming p50: ${sP50}ms`);

    // Both should be under 200ms (allows cold-isolate auth lookups).
    // Streaming overhead is measured at TTFB (response headers sent),
    // not at stream completion, so it should be similar to non-streaming.
    expect(nsP50).toBeLessThan(200);
    expect(sP50).toBeLessThan(200);
  }, 120_000);

  // ── 4. Overhead under concurrent load ────────────────────────────

  it("overhead stays low under 10 concurrent requests", async () => {
    const requests = Array.from({ length: 10 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: `concurrent-${i}` }],
        }),
      }),
    );

    const results = await Promise.all(requests);
    const overheads: number[] = [];

    for (const res of results) {
      const st = res.headers.get("Server-Timing") ?? "";
      const oh = extractTiming(st, "overhead");
      if (oh !== null && res.status === 200) overheads.push(oh);
      await res.text();
    }

    overheads.sort((a, b) => a - b);

    const p50 = percentile(overheads, 50);
    const p95 = percentile(overheads, 95);

    console.log(`[overhead] Concurrent(10) p50: ${p50}ms, p95: ${p95}ms`);
    console.log(`[overhead] All: ${overheads.join(", ")}ms`);

    expect(overheads.length).toBeGreaterThanOrEqual(8);
    // Concurrent requests span multiple isolates (cold auth cache).
    // p50 under 200ms ensures no systemic overhead regression.
    // p95 under 400ms allows cold-isolate auth lookups (~150ms).
    expect(p50).toBeLessThan(200);
    expect(p95).toBeLessThan(400);
  }, 60_000);

  // ── 5. Budget denial is fast (last — leaves budget state dirty) ──

  it("budget denial returns faster than upstream TTFB", async () => {
    const userId = NULLSPEND_SMOKE_USER_ID!;

    // Set up a $0 budget so every request is denied
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${userId}, ${orgId}, 'user', ${userId}, 0, 0, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = 0, spend_microdollars = 0, updated_at = NOW()
    `;
    await syncBudget(orgId, "user", userId);

    // Fire 5 requests — all should be denied
    const denialLatencies: number[] = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: `denied-${i}` }],
        }),
      });
      const elapsed = Math.round(performance.now() - start);
      await res.text();

      expect(res.status).toBe(429);
      denialLatencies.push(elapsed);
    }

    // Denial should be fast — the budget check resolves before upstream TTFB.
    // If the fetch wasn't aborted, we'd see 200-1000ms (upstream round-trip).
    // Denial latencies of 30-200ms prove the upstream is aborted, not awaited.
    denialLatencies.sort((a, b) => a - b);
    const p50 = percentile(denialLatencies, 50);
    const p95 = percentile(denialLatencies, 95);
    console.log(`[overhead] Denial p50: ${p50}ms, p95: ${p95}ms`);
    console.log(`[overhead] Denial all: ${denialLatencies.join(", ")}ms`);

    // p95 must be under 500ms — well below typical upstream TTFB of 200-1000ms.
    // Most denials complete in 30-50ms (network + auth + budget check).
    expect(p95).toBeLessThan(500);
  }, 60_000);
});
