/**
 * Anthropic edge case smoke tests.
 * Validates request validation, streaming edge cases, and auth edge cases
 * for the /v1/messages route.
 *
 * Requires: live proxy, ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  BASE,
  ANTHROPIC_API_KEY,
  anthropicAuthHeaders,
  smallAnthropicRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

describe("Anthropic edge cases", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
  });

  // --- Request validation ---

  it("GET /v1/messages returns 404 (only POST allowed)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, { method: "GET" });
    expect(res.status).toBe(404);
    await res.text();
  });

  it("PUT /v1/messages returns 404", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "PUT",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(404);
    await res.text();
  });

  it("empty string body returns 400 bad_request", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "",
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe("bad_request");
  });

  it("JSON array body returns 400 (must be object)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify([{ role: "user", content: "hi" }]),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe("bad_request");
  });

  it("JSON null body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "null",
    });
    expect(res.status).toBe(400);
    await res.text();
  });

  it("truncated JSON body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: '{"model": "claude-3-haiku-20240307", "max_tokens":',
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe("bad_request");
  });

  it("HTML body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "<html><body>hello</body></html>",
    });
    expect(res.status).toBe(400);
    await res.text();
  });

  it("/v1/messages with trailing slash returns 404", async () => {
    const res = await fetch(`${BASE}/v1/messages/`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(404);
    await res.text();
  });

  it("JSON number body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "42",
    });
    expect(res.status).toBe(400);
    await res.text();
  });

  it("JSON boolean body returns 400", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "true",
    });
    expect(res.status).toBe(400);
    await res.text();
  });

  it("valid JSON body with extra unknown fields passes through", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "extra fields test" }],
        some_unknown_field: "should-be-forwarded",
        another_field: 42,
      }),
    });
    // Should either succeed or get an Anthropic error, not a proxy error
    expect([200, 400]).toContain(res.status);
    await res.text();
  }, 30_000);

  // --- Streaming edge cases ---

  it("streaming response has correct content-type header", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Content-type test" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    await res.text();
  }, 30_000);

  it("early client abort on streaming does not crash the proxy", async () => {
    const controller = new AbortController();
    try {
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 100,
          messages: [{ role: "user", content: "Abort test - write a story" }],
          stream: true,
        }),
        signal: controller.signal,
      });
      expect(res.status).toBe(200);
      setTimeout(() => controller.abort(), 200);
      await res.text();
    } catch {
      // AbortError expected
    }

    await new Promise((r) => setTimeout(r, 1_000));

    const health = await fetch(`${BASE}/health`);
    expect(health.ok).toBe(true);
  }, 30_000);

  it("handles 5 concurrent streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: `Concurrent stream ${i}` }],
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

  it("handles 5 concurrent non-streaming requests without errors", async () => {
    const requests = Array.from({ length: 5 }, (_, i) =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Concurrent ns ${i}` }],
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.type).toBe("message");
    }
  }, 60_000);

  it("handles mixed streaming and non-streaming requests concurrently", async () => {
    const requests = [
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed streaming" }],
          stream: true,
        }),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: "Mixed non-streaming" }],
        }),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 10,
          messages: [{ role: "user", content: "Mixed streaming 2" }],
          stream: true,
        }),
      }),
    ];

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      await res.text();
    }
  }, 60_000);

  it("x-request-id is unique across two sequential requests", async () => {
    const res1 = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "ID unique 1" }],
      }),
    });
    expect(res1.status).toBe(200);
    await res1.text();

    const res2 = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "ID unique 2" }],
      }),
    });
    expect(res2.status).toBe(200);
    await res2.text();

    const id1 = res1.headers.get("x-request-id");
    const id2 = res2.headers.get("x-request-id");
    expect(id1).toBeTruthy();
    expect(id2).toBeTruthy();
    expect(id1).not.toBe(id2);
  }, 30_000);

  it("non-streaming response includes x-request-id header", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Request ID check" }],
      }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("x-request-id")).toBeTruthy();
    await res.text();
  }, 30_000);

  // --- Auth edge cases ---

  it("empty string x-nullspend-key returns 401", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
        "x-nullspend-key": "",
      },
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(401);
    await res.text();
  });

  it("very long auth key (10KB) returns 401", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
        "x-nullspend-key": "x".repeat(10_000),
      },
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(401);
    await res.text();
  });

  it("auth key with special ASCII characters returns 401", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY!,
        "x-nullspend-key": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
      },
      body: smallAnthropicRequest(),
    });
    expect(res.status).toBe(401);
    await res.text();
  });
});
