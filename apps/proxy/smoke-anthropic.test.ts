/**
 * Live smoke tests for the Anthropic proxy route.
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real Anthropic API key in ANTHROPIC_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run smoke-anthropic.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, ANTHROPIC_API_KEY, NULLSPEND_API_KEY, anthropicAuthHeaders, isServerUp } from "./smoke-test-helpers.js";

describe("Anthropic proxy smoke tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!ANTHROPIC_API_KEY) {
      throw new Error("ANTHROPIC_API_KEY env var is required for Anthropic smoke tests.");
    }
  });

  it("non-streaming request returns valid response with Anthropic fields", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say 'hello' and nothing else." }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("id");
    expect(body.type).toBe("message");
    expect(body).toHaveProperty("content");
    expect(body).toHaveProperty("usage");
    expect(body.usage).toHaveProperty("input_tokens");
    expect(body.usage).toHaveProperty("output_tokens");
    expect(body).toHaveProperty("stop_reason");
    expect(body).toHaveProperty("model");
  }, 30_000);

  it("streaming request returns valid SSE with event: and data: lines", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say 'hello' and nothing else." }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const text = await res.text();
    expect(text).toContain("event: message_start");
    expect(text).toContain("event: message_stop");
    expect(text).toContain("data:");
  }, 30_000);

  it("response has x-request-id header", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say ok" }],
      }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("x-request-id")).toBeTruthy();
    await res.text();
  }, 30_000);

  it("Authorization: Bearer auth path works end-to-end", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${ANTHROPIC_API_KEY}`,
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say ok" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.type).toBe("message");
    expect(body).toHaveProperty("usage");
  }, 30_000);

  // PXY-3: Unknown models now pass through to the provider (estimated cost, not rejected).
  // Anthropic returns 400 for unknown models. The proxy forwards the provider error.
  it("unknown Anthropic model passes through to provider (PXY-3)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "not-a-real-claude-model",
        max_tokens: 10,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Provider rejects the model — proxy forwards the error
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });

  it("invalid x-nullspend-key returns 401", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
        "x-nullspend-key": "wrong-key-here",
      },
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error.code).toBe("unauthorized");
  });

  it("missing x-nullspend-key returns 401", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
      },
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(401);
  });

  it("invalid JSON body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: "{this is not valid json!!!",
    });

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe("bad_request");
  });
});
