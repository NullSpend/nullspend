/**
 * Anthropic load smoke tests.
 * Validates concurrency, sustained load, latency distribution, and cost
 * logging reliability under pressure for the /v1/messages route.
 *
 * Uses smaller concurrency than OpenAI tests to conserve Anthropic credits.
 *
 * Requires: live proxy, ANTHROPIC_API_KEY, NULLSPEND_API_KEY, DATABASE_URL
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  ANTHROPIC_API_KEY,
  DATABASE_URL,
  anthropicAuthHeaders,
  isServerUp,
  waitForCostEvent,
} from "./smoke-test-helpers.js";

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

describe("Anthropic load tests", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");
    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("10 concurrent non-streaming Anthropic requests all succeed", async () => {
    const requests = Array.from({ length: 10 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Load ns ${i}` }],
        }),
      }),
    );

    const responses = await Promise.all(requests);
    const succeeded = responses.filter((r) => r.status === 200).length;
    expect(succeeded).toBe(10);

    for (const res of responses) {
      await res.text();
    }
  }, 60_000);

  it("10 concurrent streaming Anthropic requests all complete with message_stop", async () => {
    const requests = Array.from({ length: 10 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: `Load stream ${i}` }],
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

  it("sustained load: 5 sequential batches of 3 concurrent requests", async () => {
    let succeeded = 0;
    const total = 15;
    const latencies: number[] = [];

    for (let batch = 0; batch < 5; batch++) {
      const batchStart = performance.now();
      const requests = Array.from({ length: 3 }, (_, i) =>
        fetch(`${BASE}/v1/messages`, {
          method: "POST",
          headers: anthropicAuthHeaders(),
          body: JSON.stringify({
            model: "claude-3-haiku-20240307",
            max_tokens: 5,
            messages: [{ role: "user", content: `Sustained ${batch}-${i}` }],
          }),
        }),
      );

      const responses = await Promise.all(requests);
      const batchLatency = performance.now() - batchStart;
      latencies.push(batchLatency);

      for (const res of responses) {
        if (res.status === 200) succeeded++;
        await res.text();
      }

      await new Promise((r) => setTimeout(r, 500));
    }

    latencies.sort((a, b) => a - b);
    console.log(
      `[load] Sustained: ${succeeded}/${total} succeeded`,
    );
    console.log(
      `[load] Batch latency p50: ${Math.round(percentile(latencies, 50))}ms, p95: ${Math.round(percentile(latencies, 95))}ms`,
    );

    expect(succeeded).toBe(total);
  }, 120_000);

  it("mixed load: streaming + non-streaming + error requests simultaneously", async () => {
    const requests = [
      // Streaming
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed stream" }],
          stream: true,
        }),
      }),
      // Non-streaming
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: "Mixed non-stream" }],
        }),
      }),
      // Error (invalid model)
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "nonexistent-model-load",
          max_tokens: 5,
          messages: [{ role: "user", content: "Mixed error" }],
        }),
      }),
      // Streaming
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed stream 2" }],
          stream: true,
        }),
      }),
    ];

    const responses = await Promise.all(requests);
    expect(responses[0].status).toBe(200);
    expect(responses[1].status).toBe(200);
    expect(responses[2].status).toBe(400);
    expect(responses[3].status).toBe(200);

    for (const res of responses) {
      await res.text();
    }
  }, 60_000);

  it("measures Anthropic request latency distribution under load", async () => {
    const latencies: number[] = [];
    const count = 10;

    const requests = Array.from({ length: count }, async (_, i) => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Latency ${i}` }],
        }),
      });
      const elapsed = performance.now() - start;
      latencies.push(elapsed);
      await res.text();
      return res.status;
    });

    const statuses = await Promise.all(requests);
    const succeeded = statuses.filter((s) => s === 200).length;

    latencies.sort((a, b) => a - b);
    console.log(
      `[load] Latency p50: ${Math.round(percentile(latencies, 50))}ms, ` +
        `p95: ${Math.round(percentile(latencies, 95))}ms, ` +
        `p99: ${Math.round(percentile(latencies, 99))}ms`,
    );
    console.log(`[load] Success rate: ${succeeded}/${count}`);

    expect(succeeded).toBe(count);
  }, 60_000);

  it("all cost events from load test are logged to database", async () => {
    const requestIds: string[] = [];

    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Cost log check ${i} ${Date.now()}` }],
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

    let logged = 0;
    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 15_000, "anthropic");
      if (row) logged++;
    }

    console.log(`[load] Cost events logged: ${logged}/${requestIds.length}`);
    expect(logged).toBe(requestIds.length);
  }, 60_000);
});
