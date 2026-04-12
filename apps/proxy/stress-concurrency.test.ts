/**
 * Concurrency stress tests.
 * Ramps up concurrent requests to find degradation thresholds,
 * cross-provider interference, and latency regression.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  authHeaders,
  anthropicAuthHeaders,
  smallRequest,
  smallAnthropicRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

const INTENSITY = (process.env.STRESS_INTENSITY ?? "medium") as "light" | "medium" | "heavy";

const CONCURRENCY = { light: 10, medium: 25, heavy: 50 } as const;
const BATCHES = { light: 3, medium: 5, heavy: 8 } as const;
const BATCH_SIZE = { light: 5, medium: 8, heavy: 12 } as const;

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

interface RequestResult {
  status: number;
  elapsed: number;
  provider: string;
  streaming: boolean;
}

describe(`Concurrency stress [${INTENSITY}]`, () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
  });

  // ── Baseline latency ──

  it("captures baseline latency (5 sequential requests)", async () => {
    const latencies: number[] = [];

    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `Baseline ${i}` }] }),
      });
      const elapsed = performance.now() - start;
      expect(res.status).toBe(200);
      await res.json();
      latencies.push(elapsed);
    }

    const sorted = latencies.sort((a, b) => a - b);
    const p50 = percentile(sorted, 50);
    console.log(`[stress] Baseline p50: ${p50.toFixed(0)}ms`);
  }, 60_000);

  // ── Concurrency ramp ──

  it(`ramps to ${CONCURRENCY[INTENSITY]} concurrent non-streaming requests`, async () => {
    const concurrency = CONCURRENCY[INTENSITY];
    const results: RequestResult[] = [];

    const requests = Array.from({ length: concurrency }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `Ramp ${i}` }] }),
      }).then(async (res) => {
        const elapsed = performance.now() - start;
        await res.text();
        results.push({ status: res.status, elapsed, provider: "openai", streaming: false });
      });
    });

    await Promise.all(requests);

    const successes = results.filter((r) => r.status === 200);
    const errors = results.filter((r) => r.status !== 200);
    const latencies = successes.map((r) => r.elapsed).sort((a, b) => a - b);

    console.log(`[stress] Ramp ${concurrency}: ${successes.length}/${results.length} succeeded`);
    if (latencies.length > 0) {
      console.log(
        `[stress] Latency p50: ${percentile(latencies, 50).toFixed(0)}ms, ` +
          `p95: ${percentile(latencies, 95).toFixed(0)}ms, ` +
          `p99: ${percentile(latencies, 99).toFixed(0)}ms`,
      );
    }
    if (errors.length > 0) {
      const statusCounts: Record<number, number> = {};
      for (const e of errors) statusCounts[e.status] = (statusCounts[e.status] ?? 0) + 1;
      console.log(`[stress] Errors:`, statusCounts);
    }

    // At least 80% should succeed (allow for rate limit / transient errors)
    expect(successes.length / results.length).toBeGreaterThanOrEqual(0.8);
  }, 300_000);

  // ── Concurrent streaming requests ──

  it(`ramps to ${CONCURRENCY[INTENSITY]} concurrent streaming requests`, async () => {
    const concurrency = CONCURRENCY[INTENSITY];
    const results: RequestResult[] = [];
    let doneCount = 0;

    const requests = Array.from({ length: concurrency }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: `StreamRamp ${i}` }],
        }),
      }).then(async (res) => {
        const text = await res.text();
        const elapsed = performance.now() - start;
        if (res.status === 200 && text.includes("[DONE]")) doneCount++;
        results.push({ status: res.status, elapsed, provider: "openai", streaming: true });
      });
    });

    await Promise.all(requests);

    const successes = results.filter((r) => r.status === 200);
    const latencies = successes.map((r) => r.elapsed).sort((a, b) => a - b);

    console.log(`[stress] Streaming ${concurrency}: ${successes.length}/${results.length} 200s, ${doneCount} [DONE]`);
    if (latencies.length > 0) {
      console.log(
        `[stress] Stream latency p50: ${percentile(latencies, 50).toFixed(0)}ms, ` +
          `p95: ${percentile(latencies, 95).toFixed(0)}ms`,
      );
    }

    expect(successes.length / results.length).toBeGreaterThanOrEqual(0.8);
    expect(doneCount).toBe(successes.length);
  }, 300_000);

  // ── Cross-provider interleave ──

  it("interleaves OpenAI and Anthropic requests concurrently", async () => {
    if (!ANTHROPIC_API_KEY) {
      console.log("[stress] Skipping cross-provider — no ANTHROPIC_API_KEY");
      return;
    }

    const count = Math.floor(CONCURRENCY[INTENSITY] / 2);
    const results: RequestResult[] = [];

    const openaiRequests = Array.from({ length: count }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `XP-OAI ${i}` }] }),
      }).then(async (res) => {
        const elapsed = performance.now() - start;
        await res.text();
        results.push({ status: res.status, elapsed, provider: "openai", streaming: false });
      });
    });

    const anthropicRequests = Array.from({ length: count }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: smallAnthropicRequest({ messages: [{ role: "user", content: `XP-ANT ${i}` }] }),
      }).then(async (res) => {
        const elapsed = performance.now() - start;
        await res.text();
        results.push({ status: res.status, elapsed, provider: "anthropic", streaming: false });
      });
    });

    await Promise.all([...openaiRequests, ...anthropicRequests]);

    const oaiResults = results.filter((r) => r.provider === "openai");
    const antResults = results.filter((r) => r.provider === "anthropic");
    const oaiSuccess = oaiResults.filter((r) => r.status === 200).length;
    const antSuccess = antResults.filter((r) => r.status === 200).length;

    console.log(`[stress] Cross-provider: OAI ${oaiSuccess}/${oaiResults.length}, ANT ${antSuccess}/${antResults.length}`);

    // Both providers should maintain >80% success under cross-load
    expect(oaiSuccess / oaiResults.length).toBeGreaterThanOrEqual(0.8);
    expect(antSuccess / antResults.length).toBeGreaterThanOrEqual(0.8);
  }, 300_000);

  // ── Sustained load ──

  it(`sustained: ${BATCHES[INTENSITY]} batches of ${BATCH_SIZE[INTENSITY]} concurrent`, async () => {
    const numBatches = BATCHES[INTENSITY];
    const batchSize = BATCH_SIZE[INTENSITY];
    const batchLatencies: number[] = [];
    let totalSuccess = 0;
    let totalFailed = 0;

    for (let batch = 0; batch < numBatches; batch++) {
      const batchStart = performance.now();

      const requests = Array.from({ length: batchSize }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: batch % 2 === 0,
            messages: [{ role: "user", content: `Sustained ${batch}-${i}` }],
          }),
        }),
      );

      const results = await Promise.all(requests);
      for (const res of results) {
        if (res.status === 200) totalSuccess++;
        else totalFailed++;
        await res.text();
      }

      batchLatencies.push(performance.now() - batchStart);
      await new Promise((r) => setTimeout(r, 300));
    }

    const sorted = batchLatencies.sort((a, b) => a - b);
    const p50 = percentile(sorted, 50);
    const p95 = percentile(sorted, 95);

    console.log(`[stress] Sustained: ${totalSuccess}/${totalSuccess + totalFailed} succeeded`);
    console.log(`[stress] Batch p50: ${p50.toFixed(0)}ms, p95: ${p95.toFixed(0)}ms`);

    // No batch degradation: p95 should stay under 20s
    expect(p95).toBeLessThan(20_000);
    expect(totalSuccess / (totalSuccess + totalFailed)).toBeGreaterThanOrEqual(0.9);
  }, 300_000);

  // ── Post-stress health check ──

  it("health endpoints respond normally after stress", async () => {
    const [health, ready] = await Promise.all([
      fetch(`${BASE}/health`),
      fetch(`${BASE}/health/ready`),
    ]);

    expect(health.status).toBe(200);
    expect(ready.status).toBe(200);

    const healthBody = await health.json();
    const readyBody = await ready.json();
    expect(healthBody.status).toBe("ok");
    expect(readyBody.status).toBe("ok");
  }, 10_000);
});
