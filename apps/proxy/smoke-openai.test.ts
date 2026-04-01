/**
 * Live smoke tests for the OpenAI proxy route.
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run smoke-openai.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, OPENAI_API_KEY, NULLSPEND_API_KEY, isServerUp } from "./smoke-test-helpers.js";

describe("OpenAI proxy smoke tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  it("streaming request flows through and returns valid SSE", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'hello' and nothing else." }],
        stream: true,
        max_tokens: 10,
      }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const text = await res.text();
    expect(text).toContain("data:");
    expect(text).toContain("[DONE]");
  }, 30_000);

  it("non-streaming request returns valid JSON with usage", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Say 'hello' and nothing else." }],
        stream: false,
        max_tokens: 10,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("usage");
    expect(body.usage).toHaveProperty("prompt_tokens");
    expect(body.usage).toHaveProperty("completion_tokens");
    expect(body).toHaveProperty("model");
    expect(body).toHaveProperty("choices");
  }, 30_000);

  it("non-streaming error response (invalid model) forwards 4xx as-is", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({
        model: "not-a-real-model-xyz",
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
    const body = await res.json();
    expect(body).toHaveProperty("error");
  }, 30_000);

  it("invalid auth returns 401", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "x-nullspend-key": "wrong-key-here",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toHaveProperty("code", "unauthorized");
  });

  it("missing x-nullspend-key returns 401", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(401);
  });

  it("unsupported /v1/ path returns 404", async () => {
    const res = await fetch(`${BASE}/v1/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({ model: "text-embedding-3-small", input: "test" }),
    });

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error).toHaveProperty("code", "not_found");
  });

  it("invalid JSON body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: "{this is not valid json!!!",
    });

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toHaveProperty("code", "bad_request");
  });
});
