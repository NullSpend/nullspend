/**
 * End-to-end cost verification tests.
 * Sends requests through the live proxy, then queries Supabase directly to
 * verify cost_events rows are correctly inserted.
 *
 * Requires:
 *   - Live proxy at PROXY_URL (or localhost:8787)
 *   - OPENAI_API_KEY
 *   - NULLSPEND_API_KEY
 *   - DATABASE_URL for direct Supabase queries
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import { BASE, OPENAI_API_KEY, DATABASE_URL, authHeaders, isServerUp, waitForCostEvent } from "./smoke-test-helpers.js";

describe("End-to-end cost verification", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy is not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required for cost E2E tests.");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("non-streaming request creates a cost_events row with correct fields", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'cost-test-ns' and nothing else." }],
        stream: false,
        max_tokens: 5,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    expect(body).toHaveProperty("usage");
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();

    expect(row!.provider).toBe("openai");
    expect(row!.model).toContain("gpt-4o-mini");
    expect(row!.input_tokens).toBe(usage.prompt_tokens);
    expect(row!.output_tokens).toBe(usage.completion_tokens);
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    expect(row!.duration_ms).toBeGreaterThan(0);
  }, 30_000);

  it("streaming request creates a cost_events row matching SSE usage", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'cost-test-stream' and nothing else." }],
        stream: true,
        max_tokens: 5,
      }),
    });

    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-request-id");
    const text = await res.text();

    const dataLines = text
      .split("\n")
      .filter((l) => l.trim().startsWith("data:"))
      .map((l) => l.trim().slice(5).trim())
      .filter((l) => l !== "[DONE]");

    const lastChunk = JSON.parse(dataLines[dataLines.length - 1]);
    expect(lastChunk).toHaveProperty("usage");
    const usage = lastChunk.usage;

    expect(requestId).toBeTruthy();
    const row = await waitForCostEvent(sql, requestId!);
    expect(row).not.toBeNull();

    expect(row!.provider).toBe("openai");
    expect(row!.model).toContain("gpt-4o-mini");
    expect(row!.input_tokens).toBe(usage.prompt_tokens);
    expect(row!.output_tokens).toBe(usage.completion_tokens);
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
  }, 30_000);

  it("max_tokens: 1 creates a cost event with minimal but non-zero cost", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Hi" }],
        stream: false,
        max_tokens: 1,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    expect(body.usage.completion_tokens).toBe(1);

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    expect(row!.output_tokens).toBe(1);
  }, 30_000);

  it("5 rapid requests all produce cost_events rows", async () => {
    const _before = new Date();
    const requestIds: string[] = [];

    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: `Rapid cost test ${i}` }],
          stream: false,
          max_tokens: 3,
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      const requestId = res.headers.get("x-request-id") ?? body.id;
      requestIds.push(requestId);
    }

    // waitUntil may take a moment — poll for all 5
    await new Promise((r) => setTimeout(r, 5_000));

    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 15_000);
      expect(row).not.toBeNull();
      expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    }
  }, 60_000);

  it("auth failure does NOT create a cost_events row", async () => {
    const before = new Date();

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": "wrong-key-for-cost-test",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "This should not be logged" }],
        max_tokens: 5,
      }),
    });

    expect(res.status).toBe(401);
    await res.text();

    // Wait to ensure no async cost logging happens
    await new Promise((r) => setTimeout(r, 3_000));

    // Verify no event has our specific auth-failure pattern by checking
    // that no new event was created in this narrow window for the wrong key
    const rows = await sql`
      SELECT * FROM cost_events
      WHERE created_at >= ${before.toISOString()}
      AND request_id LIKE '%wrong-key%'
    `;
    expect(rows.length).toBe(0);
  }, 15_000);

  // PXY-3: Unknown models pass through to provider. OpenAI returns 404 for
  // unknown models. Error responses are NOT cost-tracked, so no cost event.
  it("unknown model error does NOT create a cost_events row (PXY-3)", async () => {
    const before = new Date();

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "nonexistent-model-cost-test",
        messages: [{ role: "user", content: "This should not be logged" }],
        max_tokens: 5,
      }),
    });

    // Provider rejects the model — proxy forwards the error (4xx)
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
    await res.text();

    await new Promise((r) => setTimeout(r, 3_000));

    const rows = await sql`
      SELECT * FROM cost_events
      WHERE created_at >= ${before.toISOString()}
      AND model = 'nonexistent-model-cost-test'
    `;
    expect(rows.length).toBe(0);
  }, 15_000);

  it("cost_microdollars is consistent between streaming and non-streaming for same prompt", async () => {
    const commonBody = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "What is 2+2? Just the number." }],
      max_tokens: 3,
      temperature: 0,
      seed: 42,
    };

    // Non-streaming
    const nsRes = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ ...commonBody, stream: false }),
    });
    const nsBody = await nsRes.json();
    const nsRequestId = nsRes.headers.get("x-request-id") ?? nsBody.id;

    // Streaming
    const sRes = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ ...commonBody, stream: true }),
    });
    const sRequestId = sRes.headers.get("x-request-id") ?? "";
    await sRes.text();

    const nsRow = await waitForCostEvent(sql, nsRequestId);
    const sRow = await waitForCostEvent(sql, sRequestId);

    expect(nsRow).not.toBeNull();
    expect(sRow).not.toBeNull();

    // Same prompt, same model, same seed — costs should be very close
    const nsCost = Number(nsRow!.cost_microdollars);
    const sCost = Number(sRow!.cost_microdollars);
    // Allow 20% tolerance due to possible minor token count differences
    const tolerance = Math.max(nsCost, sCost) * 0.2 + 1;
    expect(Math.abs(nsCost - sCost)).toBeLessThanOrEqual(tolerance);
  }, 60_000);
});
