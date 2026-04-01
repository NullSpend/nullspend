/**
 * Smoke tests for streaming body capture (Phase 2).
 * Verifies that streaming SSE responses are accumulated and stored in R2,
 * and retrievable via the internal /request-bodies endpoint.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY
 *   - ANTHROPIC_API_KEY
 *   - NULLSPEND_API_KEY (must have requestLoggingEnabled / pro subscription)
 *   - INTERNAL_SECRET (for /internal/request-bodies retrieval)
 *   - DATABASE_URL (for cost event polling)
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  DATABASE_URL,
  INTERNAL_SECRET,
  authHeaders,
  anthropicAuthHeaders,
  isServerUp,
  waitForCostEvent,
} from "./smoke-test-helpers.js";

describe("Streaming body capture E2E", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy is not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required.");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  async function fetchBodies(requestId: string): Promise<{ requestBody: unknown; responseBody: unknown } | null> {
    // Wait a bit for R2 write to complete (runs in waitUntil after cost processing)
    await new Promise((r) => setTimeout(r, 2000));

    const res = await fetch(
      `${BASE}/internal/request-bodies/${requestId}?ownerId=a6262022-9666-43af-b258-c870c8feb6be`,
      {
        headers: { Authorization: `Bearer ${INTERNAL_SECRET}` },
      },
    );
    if (!res.ok) return null;
    return res.json() as Promise<{ requestBody: unknown; responseBody: unknown }>;
  }

  it("captures OpenAI streaming response body as SSE text", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'body-capture-test' and nothing else." }],
        stream: true,
        max_tokens: 10,
      }),
    });

    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-request-id");
    expect(requestId).toBeTruthy();

    // Consume the stream fully
    const streamText = await res.text();
    expect(streamText).toContain("data:");

    // Wait for cost event (confirms stream processing completed)
    const costEvent = await waitForCostEvent(sql, requestId!);
    expect(costEvent).not.toBeNull();

    // Fetch stored bodies from R2 via internal endpoint
    const bodies = await fetchBodies(requestId!);
    expect(bodies).not.toBeNull();

    // Request body should be the original JSON
    expect(bodies!.requestBody).not.toBeNull();
    expect((bodies!.requestBody as Record<string, unknown>).model).toBe("gpt-4o-mini");

    // Response body should be SSE format wrapped
    expect(bodies!.responseBody).not.toBeNull();
    const responseBody = bodies!.responseBody as Record<string, unknown>;
    expect(responseBody._format).toBe("sse");
    expect(typeof responseBody.text).toBe("string");
    expect(responseBody.text as string).toContain("data:");
  }, 30_000);

  it("captures Anthropic streaming response body as SSE text", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        messages: [{ role: "user", content: "Say 'body-capture-test' and nothing else." }],
        stream: true,
        max_tokens: 10,
      }),
    });

    expect(res.status).toBe(200);
    // Anthropic's request-id is normalized to x-request-id by the proxy
    const requestId = res.headers.get("x-request-id");
    expect(requestId).toBeTruthy();

    // Consume the stream fully
    const streamText = await res.text();
    expect(streamText).toContain("event:");

    // Wait for cost event
    const costEvent = await waitForCostEvent(sql, requestId!, 15_000, "anthropic");
    expect(costEvent).not.toBeNull();

    // Fetch stored bodies
    const bodies = await fetchBodies(requestId!);
    expect(bodies).not.toBeNull();

    // Request body
    expect(bodies!.requestBody).not.toBeNull();
    expect((bodies!.requestBody as Record<string, unknown>).model).toBe("claude-3-haiku-20240307");

    // Response body — SSE format
    expect(bodies!.responseBody).not.toBeNull();
    const responseBody = bodies!.responseBody as Record<string, unknown>;
    expect(responseBody._format).toBe("sse");
    expect(typeof responseBody.text).toBe("string");
    expect(responseBody.text as string).toContain("event:");
    expect(responseBody.text as string).toContain("message_start");
  }, 30_000);

  // NOTE: The tests above require requestLoggingEnabled=true (pro/enterprise subscription).
  // The test below verifies the negative case — bodies are NOT stored when logging is disabled.
  // To run the positive tests, temporarily insert a pro subscription for the smoke test org.

  it("non-streaming request still stores JSON response body", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'json-body-test' and nothing else." }],
        stream: false,
        max_tokens: 10,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json() as Record<string, unknown>;
    const requestId = res.headers.get("x-request-id") ?? (body as { id?: string }).id;
    expect(requestId).toBeTruthy();

    // Wait for cost event
    const costEvent = await waitForCostEvent(sql, requestId!);
    expect(costEvent).not.toBeNull();

    // Fetch stored bodies
    const bodies = await fetchBodies(requestId!);
    expect(bodies).not.toBeNull();

    // Request body should be JSON
    expect(bodies!.requestBody).not.toBeNull();
    expect((bodies!.requestBody as Record<string, unknown>).model).toBe("gpt-4o-mini");

    // Response body should be parsed JSON (not SSE wrapper)
    expect(bodies!.responseBody).not.toBeNull();
    const responseBody = bodies!.responseBody as Record<string, unknown>;
    expect(responseBody._format).toBeUndefined(); // NOT SSE format
    expect(responseBody).toHaveProperty("choices");
  }, 30_000);
});

describe("Body capture disabled (no pro subscription)", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy is not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required.");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("does not store bodies when requestLoggingEnabled is false", async () => {
    // This test assumes the smoke test org does NOT have a pro/enterprise subscription.
    // If it does, this test will fail (bodies WILL be stored).
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'no-logging-test' and nothing else." }],
        stream: false,
        max_tokens: 5,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json() as Record<string, unknown>;
    const requestId = res.headers.get("x-request-id") ?? (body as { id?: string }).id;
    expect(requestId).toBeTruthy();

    // Wait for cost event to confirm request was processed
    const costEvent = await waitForCostEvent(sql, requestId!);
    expect(costEvent).not.toBeNull();

    // Wait a bit for any async R2 writes to complete
    await new Promise((r) => setTimeout(r, 2000));

    // Fetch bodies — should both be null since logging is disabled
    const bodiesRes = await fetch(
      `${BASE}/internal/request-bodies/${requestId}?ownerId=a6262022-9666-43af-b258-c870c8feb6be`,
      { headers: { Authorization: `Bearer ${INTERNAL_SECRET}` } },
    );
    expect(bodiesRes.ok).toBe(true);
    const bodies = await bodiesRes.json() as { requestBody: unknown; responseBody: unknown };
    expect(bodies.requestBody).toBeNull();
    expect(bodies.responseBody).toBeNull();
  }, 30_000);
});
