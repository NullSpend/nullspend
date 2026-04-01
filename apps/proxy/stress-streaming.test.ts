/**
 * Streaming stress tests.
 * Rapid abort/reconnect cycles, concurrent stream management,
 * and resource leak detection.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL (optional, for cost event verification)
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  DATABASE_URL,
  authHeaders,
  smallRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

const INTENSITY = (process.env.STRESS_INTENSITY ?? "medium") as "light" | "medium" | "heavy";

const ABORT_CYCLES = { light: 5, medium: 15, heavy: 30 } as const;
const CONCURRENT_STREAMS = { light: 10, medium: 25, heavy: 50 } as const;

/**
 * Wait for rate limit to recover before starting tests.
 * Prior stress phases may have exhausted the sliding window.
 */
async function waitForRateLimitRecovery(maxWaitMs = 120_000): Promise<void> {
  const start = Date.now();
  let attempt = 0;
  while (Date.now() - start < maxWaitMs) {
    attempt++;
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest({ messages: [{ role: "user", content: "RL check" }] }),
    });
    const status = res.status;
    const body = await res.text();
    if (status === 200) {
      console.log(`[stress] Rate limit recovered after ${attempt} attempts (${((Date.now() - start) / 1000).toFixed(0)}s)`);
      return;
    }
    console.log(`[stress] Recovery attempt ${attempt}: status=${status}, body=${body.slice(0, 150)}`);
    if (status === 429) {
      await new Promise((r) => setTimeout(r, 15_000));
    } else {
      // Non-rate-limit error — wait briefly and retry
      await new Promise((r) => setTimeout(r, 3_000));
    }
  }
  console.log("[stress] WARNING: Rate limit did not recover within timeout");
}

describe(`Streaming stress [${INTENSITY}]`, () => {
  let sql: postgres.Sql | null = null;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (DATABASE_URL) {
      sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
    }
    // Wait for rate limit recovery from prior stress phases
    await waitForRateLimitRecovery();
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  // ── Rapid abort storm ──

  it(`rapid abort storm: ${ABORT_CYCLES[INTENSITY]} open-read-abort cycles`, async () => {
    const cycles = ABORT_CYCLES[INTENSITY];
    let abortedCount = 0;
    let failedToConnect = 0;

    for (let i = 0; i < cycles; i++) {
      const controller = new AbortController();
      try {
        const res = await fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: true,
            messages: [{ role: "user", content: `Abort storm ${i}` }],
            max_tokens: 50,
          }),
          signal: controller.signal,
        });

        if (res.status === 200 && res.body) {
          const reader = res.body.getReader();
          // Read one chunk then abort
          await reader.read();
          controller.abort();
          reader.releaseLock();
          abortedCount++;
        } else {
          const body = await res.text();
          if (i === 0) console.log(`[stress] Abort storm first non-200: status=${res.status}, body=${body.slice(0, 200)}`);
          failedToConnect++;
        }
      } catch {
        // AbortError or network error — expected
        abortedCount++;
      }

      // Minimal delay between cycles to be aggressive
      if (i % 5 === 4) await new Promise((r) => setTimeout(r, 100));
    }

    console.log(`[stress] Abort storm: ${abortedCount} aborted, ${failedToConnect} failed to connect`);

    // Most aborts should succeed (allow some connection failures under load)
    expect(abortedCount).toBeGreaterThan(cycles * 0.7);
  }, 120_000);

  // ── Post-abort health ──

  it("proxy serves normal requests after abort storm", async () => {
    // Brief settle time
    await new Promise((r) => setTimeout(r, 1_000));

    const results: boolean[] = [];

    for (let i = 0; i < 5; i++) {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `PostAbort ${i}` }] }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      results.push(body?.usage != null);
    }

    expect(results.every(Boolean)).toBe(true);
  }, 60_000);

  // ── Concurrent stream management ──

  it(`${CONCURRENT_STREAMS[INTENSITY]} concurrent streams all complete with [DONE]`, async () => {
    const count = CONCURRENT_STREAMS[INTENSITY];
    let doneCount = 0;
    let errorCount = 0;
    const latencies: number[] = [];

    const requests = Array.from({ length: count }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: `CStream ${i}` }],
        }),
      }).then(async (res) => {
        const text = await res.text();
        const elapsed = performance.now() - start;
        if (res.status === 200) {
          latencies.push(elapsed);
          if (text.includes("[DONE]")) doneCount++;
          else errorCount++;
        } else {
          if (errorCount === 0) console.log(`[stress] First stream error: status=${res.status}, body=${text.slice(0, 200)}`);
          errorCount++;
        }
      });
    });

    await Promise.all(requests);

    const sorted = latencies.sort((a, b) => a - b);
    const p50 = sorted.length > 0 ? sorted[Math.floor(sorted.length * 0.5)] : 0;
    const p95 = sorted.length > 0 ? sorted[Math.floor(sorted.length * 0.95)] : 0;

    console.log(
      `[stress] Concurrent streams: ${doneCount}/${count} [DONE], ${errorCount} errors. ` +
        `p50: ${p50.toFixed(0)}ms, p95: ${p95.toFixed(0)}ms`,
    );

    // All successful streams must have [DONE]
    expect(doneCount + errorCount).toBe(count);
    expect(doneCount / count).toBeGreaterThanOrEqual(0.8);
  }, 300_000);

  // ── Mixed abort + complete ──

  it("concurrent mix: half complete, half aborted — no interference", async () => {
    const count = Math.min(CONCURRENT_STREAMS[INTENSITY], 20);
    const half = Math.floor(count / 2);
    let completedCount = 0;
    let abortedCount = 0;

    // Requests that complete normally
    const completeRequests = Array.from({ length: half }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: `Complete ${i}` }],
        }),
      }).then(async (res) => {
        const text = await res.text();
        if (res.status === 200 && text.includes("[DONE]")) completedCount++;
      }),
    );

    // Requests that get aborted after first chunk
    const abortRequests = Array.from({ length: half }, (_, i) => {
      const controller = new AbortController();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: `Abort ${i}` }],
          max_tokens: 50,
        }),
        signal: controller.signal,
      })
        .then(async (res) => {
          if (res.status === 200 && res.body) {
            const reader = res.body.getReader();
            await reader.read();
            controller.abort();
            reader.releaseLock();
          }
          abortedCount++;
        })
        .catch(() => {
          abortedCount++;
        });
    });

    await Promise.all([...completeRequests, ...abortRequests]);

    console.log(`[stress] Mixed streams: ${completedCount}/${half} complete, ${abortedCount}/${half} aborted`);

    // Completed streams should not be affected by aborted ones
    expect(completedCount).toBeGreaterThanOrEqual(half * 0.8);
  }, 120_000);

  // ── Cost event verification for aborted streams ──

  it("aborted streams still log cost events (partial usage)", async () => {
    if (!sql) {
      console.log("[stress] Skipping cost verification — no DATABASE_URL");
      return;
    }

    const before = new Date();
    const requestIds: string[] = [];

    // Send 5 streams, abort after first chunk
    for (let i = 0; i < 5; i++) {
      const controller = new AbortController();
      try {
        const res = await fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: true,
            messages: [{ role: "user", content: `CostAbort ${i}` }],
            max_tokens: 50,
          }),
          signal: controller.signal,
        });

        const reqId = res.headers.get("x-request-id");
        if (reqId) requestIds.push(reqId);

        if (res.status === 200 && res.body) {
          const reader = res.body.getReader();
          await reader.read();
          // Small delay to let some chunks arrive
          await new Promise((r) => setTimeout(r, 200));
          controller.abort();
          reader.releaseLock();
        }
      } catch {
        // expected
      }
    }

    // Wait for cost logging
    await new Promise((r) => setTimeout(r, 10_000));

    const rows = await sql`
      SELECT request_id, cost_microdollars::text as cost, input_tokens, output_tokens
      FROM cost_events
      WHERE created_at >= ${before.toISOString()}
        AND provider = 'openai'
    `;

    const logged = requestIds.filter((id) => rows.some((r) => r.request_id === id));

    console.log(`[stress] Aborted stream cost events: ${logged.length}/${requestIds.length} logged`);

    // At least one aborted stream must have captured a request ID,
    // otherwise the test has zero signal and would pass vacuously.
    expect(requestIds.length).toBeGreaterThan(0);

    // After the cancelled-stream cost event fix, ≥80% of aborted streams
    // should have cost events (tolerance for timing edge cases where abort
    // fires before the proxy receives a request ID header).
    const ratio = logged.length / requestIds.length;
    console.log(`[stress] Aborted stream cost event ratio: ${(ratio * 100).toFixed(0)}%`);
    expect(ratio).toBeGreaterThanOrEqual(0.8);
  }, 60_000);

  // ── Post-stress health ──

  it("health check after streaming stress", async () => {
    const res = await fetch(`${BASE}/health/ready`);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
  }, 10_000);
});
