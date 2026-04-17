/**
 * Load tests for the live proxy.
 * Verifies the proxy handles concurrent and sustained traffic without degradation.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL for verifying all cost events were logged
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import { BASE, OPENAI_API_KEY, DATABASE_URL, authHeaders, smallRequest, isServerUp } from "./smoke-test-helpers.js";

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

describe("Load tests", () => {
  let sql: postgres.Sql | null = null;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (DATABASE_URL) {
      sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
    }
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("20 concurrent non-streaming requests all succeed", async () => {
    const requests = Array.from({ length: 20 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: `Load test NS ${i}` }],
        }),
      }),
    );

    const results = await Promise.all(requests);
    let successCount = 0;

    for (const res of results) {
      if (res.status === 200) {
        const body = await res.json();
        expect(body).toHaveProperty("usage");
        successCount++;
      } else {
        await res.text();
      }
    }

    expect(successCount).toBe(20);
  }, 60_000);

  it("20 concurrent streaming requests all complete with [DONE]", async () => {
    const requests = Array.from({ length: 20 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: `Load test S ${i}` }],
        }),
      }),
    );

    const results = await Promise.all(requests);
    let successCount = 0;

    for (const res of results) {
      if (res.status === 200) {
        const text = await res.text();
        expect(text).toContain("[DONE]");
        successCount++;
      } else {
        await res.text();
      }
    }

    expect(successCount).toBe(20);
  }, 60_000);

  it("sustained load: 10 sequential batches of 5 concurrent requests", async () => {
    const latencies: number[] = [];
    let totalSuccess = 0;
    let totalFailed = 0;

    for (let batch = 0; batch < 10; batch++) {
      const batchStart = performance.now();

      const requests = Array.from({ length: 5 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            messages: [{ role: "user", content: `Sustained ${batch}-${i}` }],
            stream: batch % 2 === 0,
          }),
        }),
      );

      const results = await Promise.all(requests);

      for (const res of results) {
        if (res.status === 200) {
          totalSuccess++;
        } else {
          totalFailed++;
        }
        await res.text();
      }

      const batchLatency = performance.now() - batchStart;
      latencies.push(batchLatency);

      // Brief pause between batches
      await new Promise((r) => setTimeout(r, 500));
    }

    expect(totalSuccess).toBe(50);
    expect(totalFailed).toBe(0);

    const sorted = latencies.sort((a, b) => a - b);
    const p50 = percentile(sorted, 50);
    const p95 = percentile(sorted, 95);

    // Log performance metrics
    console.log(`[load] Sustained: ${totalSuccess}/${totalSuccess + totalFailed} succeeded`);
    console.log(`[load] Batch latency p50: ${p50.toFixed(0)}ms, p95: ${p95.toFixed(0)}ms`);

    // p95 batch latency should be under 15s (5 concurrent OpenAI calls)
    expect(p95).toBeLessThan(15_000);
  }, 180_000);

  it("mixed load: streaming + non-streaming + error requests simultaneously", async () => {
    const requests = [
      // 5 non-streaming
      ...Array.from({ length: 5 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            messages: [{ role: "user", content: `Mixed NS ${i}` }],
          }),
        }),
      ),
      // 5 streaming
      ...Array.from({ length: 5 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: true,
            messages: [{ role: "user", content: `Mixed S ${i}` }],
          }),
        }),
      ),
      // 3 auth errors
      ...Array.from({ length: 3 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            "x-nullspend-key": "wrong-key",
          },
          body: smallRequest(),
        }),
      ),
      // 2 malformed body errors
      ...Array.from({ length: 2 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: "{invalid}",
        }),
      ),
    ];

    const results = await Promise.all(requests);

    // Non-streaming should succeed
    for (let i = 0; i < 5; i++) {
      expect(results[i].status).toBe(200);
      const body = await results[i].json();
      expect(body).toHaveProperty("usage");
    }

    // Streaming should complete
    for (let i = 5; i < 10; i++) {
      expect(results[i].status).toBe(200);
      const text = await results[i].text();
      expect(text).toContain("[DONE]");
    }

    // Auth errors
    for (let i = 10; i < 13; i++) {
      expect(results[i].status).toBe(401);
      await results[i].text();
    }

    // Malformed body
    for (let i = 13; i < 15; i++) {
      expect(results[i].status).toBe(400);
      await results[i].text();
    }
  }, 60_000);

  it("measures request latency distribution under load", async () => {
    const latencies: number[] = [];
    const count = 15;

    const requests = Array.from({ length: count }, (_, i) => {
      const start = performance.now();
      return fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: `Latency test ${i}` }],
        }),
      }).then(async (res) => {
        const elapsed = performance.now() - start;
        await res.text();
        return { status: res.status, elapsed };
      });
    });

    const results = await Promise.all(requests);

    for (const r of results) {
      if (r.status === 200) latencies.push(r.elapsed);
    }

    const sorted = latencies.sort((a, b) => a - b);
    const p50 = percentile(sorted, 50);
    const p95 = percentile(sorted, 95);
    const p99 = percentile(sorted, 99);

    console.log(`[load] Latency p50: ${p50.toFixed(0)}ms, p95: ${p95.toFixed(0)}ms, p99: ${p99.toFixed(0)}ms`);
    console.log(`[load] Success rate: ${latencies.length}/${count}`);

    expect(latencies.length).toBe(count);
    // p99 under 10 seconds for small requests
    expect(p99).toBeLessThan(10_000);
  }, 60_000);

  it("all cost events from load test are logged to database", async () => {
    if (!sql) {
      console.log("[load] Skipping DB verification — no DATABASE_URL");
      return;
    }

    const _before = new Date();
    const requestIds: string[] = [];

    // Send 10 concurrent requests
    const requests = Array.from({ length: 10 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: `Cost load ${i}` }],
        }),
      }),
    );

    const results = await Promise.all(requests);
    for (const res of results) {
      expect(res.status).toBe(200);
      const body = await res.json();
      const requestId = res.headers.get("x-request-id") ?? body.id;
      requestIds.push(requestId);
    }

    // Wait for all waitUntil cost logging to complete
    await new Promise((r) => setTimeout(r, 10_000));

    // Verify all 10 cost events exist in the database
    let found = 0;
    for (const id of requestIds) {
      const rows = await sql`
        SELECT id FROM cost_events
        WHERE request_id = ${id} AND provider = 'openai'
      `;
      if (rows.length > 0) found++;
    }

    console.log(`[load] Cost events logged: ${found}/${requestIds.length}`);
    expect(found).toBe(requestIds.length);
  }, 60_000);
});
