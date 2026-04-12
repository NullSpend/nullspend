/**
 * Known-issues smoke tests targeting specific bugs discovered through
 * research across Cloudflare Workers, Hyperdrive, node-postgres, and
 * the OpenAI API.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL for direct Supabase queries
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import { BASE, OPENAI_API_KEY, DATABASE_URL, authHeaders, smallRequest, isServerUp, waitForCostEvent } from "./smoke-test-helpers.js";

describe("Known issues: connection exhaustion & waitUntil reliability", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(DATABASE_URL, { max: 5, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("10 concurrent requests all produce cost events (connection concurrency guard)", async () => {
    const requestIds: string[] = [];

    const requests = Array.from({ length: 10 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: `Connection guard ${i}` }] }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      const requestId = res.headers.get("x-request-id") ?? body.id;
      requestIds.push(requestId);
    }

    // Wait for all waitUntil tasks to complete
    await new Promise((r) => setTimeout(r, 10_000));

    let foundCount = 0;
    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 15_000);
      if (row) foundCount++;
    }

    // All 10 must have cost events — no deadlocks, no dropped writes
    expect(foundCount).toBe(10);
  }, 120_000);

  it("20 requests in 4 batches all produce cost events (waitUntil reliability)", async () => {
    const allRequestIds: string[] = [];
    const BATCH_SIZE = 5;
    const BATCHES = 4;

    for (let batch = 0; batch < BATCHES; batch++) {
      const requests = Array.from({ length: BATCH_SIZE }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            messages: [{ role: "user", content: `waitUntil batch ${batch} req ${i}` }],
          }),
        }),
      );

      const responses = await Promise.all(requests);
      for (const res of responses) {
        expect(res.status).toBe(200);
        const body = await res.json();
        const requestId = res.headers.get("x-request-id") ?? body.id;
        allRequestIds.push(requestId);
      }

      // Brief pause between batches to let waitUntil complete
      await new Promise((r) => setTimeout(r, 3_000));
    }

    // Wait for final batch
    await new Promise((r) => setTimeout(r, 15_000));

    let foundCount = 0;
    for (const id of allRequestIds) {
      const row = await waitForCostEvent(sql, id, 15_000);
      if (row) foundCount++;
    }

    expect(foundCount).toBe(20);
  }, 180_000);
});
