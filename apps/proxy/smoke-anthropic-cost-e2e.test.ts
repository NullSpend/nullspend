/**
 * End-to-end cost verification tests for Anthropic.
 * Sends requests through the live proxy, then queries Supabase directly to
 * verify cost_events rows are correctly inserted with provider: "anthropic".
 *
 * Requires:
 *   - Live proxy at PROXY_URL (or localhost:8787)
 *   - ANTHROPIC_API_KEY
 *   - NULLSPEND_API_KEY
 *   - DATABASE_URL for direct Supabase queries
 *
 * AUDIT NOTE: Anthropic uses usage.input_tokens / usage.output_tokens,
 * NOT usage.prompt_tokens / usage.completion_tokens like OpenAI.
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

describe("Anthropic end-to-end cost verification", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy is not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required for cost E2E tests.");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("non-streaming request creates a cost_events row with correct fields", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say 'cost-test-ns' and nothing else." }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    expect(usage).toBeDefined();

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();

    expect(row!.provider).toBe("anthropic");
    expect(row!.model).toContain("claude");
    expect(row!.input_tokens).toBe(usage.input_tokens);
    expect(row!.output_tokens).toBe(usage.output_tokens);
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    expect(row!.duration_ms).toBeGreaterThan(0);
  }, 30_000);

  it("streaming request creates a cost_events row with positive values", async () => {
    // AUDIT NOTE: Anthropic splits usage across message_start (input) and
    // message_delta (output) SSE events. Rather than reconstructing usage from
    // the raw stream, we verify the DB row has positive values.
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say 'cost-test-stream' and nothing else." }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-request-id");
    await res.text();

    expect(requestId).toBeTruthy();
    const row = await waitForCostEvent(sql, requestId!, 15_000, "anthropic");
    expect(row).not.toBeNull();

    expect(row!.provider).toBe("anthropic");
    expect(row!.model).toContain("claude");
    expect(row!.input_tokens).toBeGreaterThan(0);
    expect(row!.output_tokens).toBeGreaterThan(0);
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
  }, 30_000);

  it("model field contains dated version returned by Anthropic", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Hi" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    // Anthropic returns dated model like "claude-3-haiku-20240307"
    expect(body.model).toContain("claude");
    expect(body.model).toMatch(/\d{8}$/);

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(row!.model).toBe(body.model);
  }, 30_000);

  it("request_id in cost event matches x-request-id header", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Hi" }],
      }),
    });

    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-request-id");
    await res.text();

    expect(requestId).toBeTruthy();

    const row = await waitForCostEvent(sql, requestId!, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(row!.request_id).toBe(requestId);
  }, 30_000);

  it("auth failure does NOT create a cost_events row", async () => {
    const before = new Date();

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
        "x-nullspend-key": "wrong-key-for-cost-test",
      },
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "This should not be logged" }],
      }),
    });

    expect(res.status).toBe(401);
    await res.text();

    await new Promise((r) => setTimeout(r, 3_000));

    const rows = await sql`
      SELECT * FROM cost_events
      WHERE created_at >= ${before.toISOString()}
      AND provider = 'anthropic'
      AND model LIKE 'claude-3-haiku%'
    `;
    // No cost event should exist for a rejected request in this window
    // with this specific model (avoids false positives from concurrent tests)
    expect(rows.length).toBe(0);
  }, 15_000);

  // PXY-3: Unknown models pass through to provider. Anthropic returns 400.
  // Error responses are NOT cost-tracked, so no cost event.
  it("unknown model error does NOT create a cost_events row (PXY-3)", async () => {
    const before = new Date();

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "nonexistent-anthropic-cost-test",
        max_tokens: 5,
        messages: [{ role: "user", content: "This should not be logged" }],
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
      AND model = 'nonexistent-anthropic-cost-test'
    `;
    expect(rows.length).toBe(0);
  }, 15_000);
});
