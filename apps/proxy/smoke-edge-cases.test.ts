/**
 * Pressure-test smoke tests for the OpenAI proxy.
 * Tests edge cases, malformed inputs, concurrency, early aborts, and protocol violations.
 *
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run smoke-edge-cases.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, OPENAI_API_KEY, NULLSPEND_API_KEY, authHeaders, isServerUp } from "./smoke-test-helpers.js";

describe("Proxy pressure tests", () => {
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

  // ── HTTP method tests ──

  describe("HTTP method handling", () => {
    it("GET /v1/chat/completions returns 404 (only POST allowed)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "GET",
        headers: authHeaders(),
      });
      expect(res.status).toBe(404);
    });

    it("PUT /v1/chat/completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(404);
    });

    it("DELETE /v1/chat/completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      expect(res.status).toBe(404);
    });

    it("PATCH /v1/chat/completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "PATCH",
        headers: authHeaders(),
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(404);
    });
  });

  // ── Malformed body tests ──

  describe("Malformed request bodies", () => {
    it("empty string body returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "",
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("JSON array body returns 400 (must be object)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify([{ model: "gpt-4o-mini", messages: [] }]),
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("JSON string literal body returns 400 (must be object)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify("just a string"),
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("JSON number body returns 400 (must be object)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "42",
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("JSON null body returns 400 (must be object)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "null",
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("JSON boolean body returns 400 (must be object)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "true",
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("truncated JSON body returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: '{"model": "gpt-4o-mini", "messages": [{"role": "user", "con',
      });
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("HTML body returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "<html><body>not json</body></html>",
      });
      expect(res.status).toBe(400);
    });

    it("XML body returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: '<?xml version="1.0"?><request><model>gpt-4o</model></request>',
      });
      expect(res.status).toBe(400);
    });
  });

  // ── Auth edge cases ──

  describe("Authentication edge cases", () => {
    it("empty string x-nullspend-key returns 401", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(401);
    });

    it("very long auth key returns 401", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "x".repeat(10000),
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(401);
    });

    it("auth key with special ASCII characters returns 401", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "!@#$%^&*()_+-=[]{}|;':\",./<>?~`",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(401);
    });
  });

  // ── Routing edge cases ──

  describe("Route handling edge cases", () => {
    it("root path returns 404", async () => {
      const res = await fetch(`${BASE}/`);
      expect(res.status).toBe(404);
    });

    it("/v1/ without sub-path returns 404 with proper error", async () => {
      const res = await fetch(`${BASE}/v1/`, {
        method: "POST",
        headers: authHeaders(),
        body: "{}",
      });
      expect(res.status).toBe(404);
      const body = await res.json();
      expect(body.error.code).toBe("not_found");
    });

    it("/v1/models returns 404 (not yet supported)", async () => {
      const res = await fetch(`${BASE}/v1/models`, { headers: authHeaders() });
      expect(res.status).toBe(404);
    });

    it("/v1/chat/completions with trailing slash returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions/`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(404);
    });

    it("/v2/chat/completions returns 404 (wrong version)", async () => {
      const res = await fetch(`${BASE}/v2/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "{}",
      });
      expect(res.status).toBe(404);
    });

    it("/health returns 200 with service info", async () => {
      const res = await fetch(`${BASE}/health`);
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");
      expect(body.service).toBe("nullspend-proxy");
    });

    it("/health via POST still returns 200", async () => {
      const res = await fetch(`${BASE}/health`, { method: "POST" });
      expect(res.status).toBe(200);
    });
  });

  // ── OpenAI error forwarding ──

  describe("OpenAI error forwarding", () => {
    it("missing messages field forwards OpenAI 400 error", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model: "gpt-4o-mini" }),
      });
      expect(res.status).toBeGreaterThanOrEqual(400);
      expect(res.status).toBeLessThan(500);
      const body = await res.json();
      expect(body).toHaveProperty("error");
    }, 15_000);

    it("empty messages array forwards OpenAI 400 error", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
      });
      expect(res.status).toBeGreaterThanOrEqual(400);
      expect(res.status).toBeLessThan(500);
    }, 15_000);

    it("invalid OpenAI API key forwards 401 from upstream", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer sk-invalid-key-12345",
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      expect(res.status).toBe(401);
      const body = await res.json();
      expect(body).toHaveProperty("error");
    }, 15_000);

    it("forwards rate limit headers on error responses", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "not-a-real-model",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      // Even error responses may have rate-limit info; we just check it doesn't crash
      expect(res.status).toBeGreaterThanOrEqual(400);
    }, 15_000);
  });

  // ── Streaming edge cases ──

  describe("Streaming edge cases", () => {
    it("stream: true with already-set stream_options.include_usage: true works", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'ok'" }],
          stream: true,
          stream_options: { include_usage: true },
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("data:");
      expect(text).toContain("[DONE]");
    }, 30_000);

    it("stream: true with stream_options.include_usage: false gets overridden (usage in stream)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'ok'" }],
          stream: true,
          stream_options: { include_usage: false },
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");
      // The proxy force-overrides to include_usage: true, so usage should appear
      expect(text).toContain('"usage"');
    }, 30_000);

    it("streaming response has correct content-type header", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'hi'" }],
          stream: true,
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("text/event-stream");
      await res.text();
    }, 30_000);

    it("early client abort on streaming does not crash the proxy", async () => {
      const controller = new AbortController();

      const resPromise = fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "user",
              content: "Write a very long essay about the history of computing in extreme detail.",
            },
          ],
          stream: true,
          max_tokens: 500,
        }),
        signal: controller.signal,
      });

      const res = await resPromise;
      expect(res.status).toBe(200);

      const reader = res.body!.getReader();
      await reader.read();
      controller.abort();

      // Give the proxy a moment to handle the disconnect
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Verify proxy is still healthy after abort
      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    }, 30_000);
  });

  // ── Non-streaming edge cases ──

  describe("Non-streaming edge cases", () => {
    it("explicit stream: false returns valid JSON response", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'yes'" }],
          stream: false,
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
      expect(body).toHaveProperty("choices");
      expect(body.choices[0]).toHaveProperty("message");
    }, 30_000);

    it("omitted stream field defaults to non-streaming", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'yes'" }],
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("choices");
      expect(body).toHaveProperty("model");
    }, 30_000);

    it("non-streaming response includes x-request-id header", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'test'" }],
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get("x-request-id")).toBeTruthy();
      await res.json();
    }, 30_000);
  });

  // ── Concurrency tests ──

  describe("Concurrent request handling", () => {
    it("handles 5 concurrent streaming requests without errors", async () => {
      const requests = Array.from({ length: 5 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: `Say '${i}' and nothing else.` }],
            stream: true,
            max_tokens: 5,
          }),
        }),
      );

      const responses = await Promise.all(requests);

      for (const res of responses) {
        expect(res.status).toBe(200);
        const text = await res.text();
        expect(text).toContain("[DONE]");
      }
    }, 60_000);

    it("handles 5 concurrent non-streaming requests without errors", async () => {
      const requests = Array.from({ length: 5 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: `Say '${i}' and nothing else.` }],
            stream: false,
            max_tokens: 5,
          }),
        }),
      );

      const responses = await Promise.all(requests);

      for (const res of responses) {
        expect(res.status).toBe(200);
        const body = await res.json();
        expect(body).toHaveProperty("usage");
      }
    }, 60_000);

    it("handles mixed streaming and non-streaming requests concurrently", async () => {
      const streamReq = fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'stream'" }],
          stream: true,
          max_tokens: 5,
        }),
      });

      const nonStreamReq = fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'nostream'" }],
          stream: false,
          max_tokens: 5,
        }),
      });

      const errorReq = fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model: "not-real", messages: [{ role: "user", content: "hi" }] }),
      });

      const [streamRes, nonStreamRes, errorRes] = await Promise.all([
        streamReq,
        nonStreamReq,
        errorReq,
      ]);

      expect(streamRes.status).toBe(200);
      const streamText = await streamRes.text();
      expect(streamText).toContain("[DONE]");

      expect(nonStreamRes.status).toBe(200);
      const nonStreamBody = await nonStreamRes.json();
      expect(nonStreamBody).toHaveProperty("usage");

      expect(errorRes.status).toBeGreaterThanOrEqual(400);
    }, 60_000);
  });

  // ── Header transparency tests ──

  describe("Header transparency", () => {
    it("x-ratelimit headers are forwarded from successful response", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'header-test'" }],
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);

      // OpenAI typically returns these headers
      const rateLimitHeaders = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
      ];
      let foundAny = false;
      for (const h of rateLimitHeaders) {
        if (res.headers.get(h)) foundAny = true;
      }
      expect(foundAny).toBe(true);
      await res.json();
    }, 30_000);

    it("OpenAI-Organization header is forwarded upstream (no error)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "OpenAI-Organization": "org-test-fake" }),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'org'" }],
          max_tokens: 5,
        }),
      });
      // May return 401 from OpenAI if org is invalid, but proxy shouldn't crash
      expect([200, 401, 403]).toContain(res.status);
      await res.text();
    }, 30_000);
  });

  // ── Large payload tests ──

  describe("Large payload handling", () => {
    it("handles request with many messages (conversation context)", async () => {
      const messages = [];
      for (let i = 0; i < 50; i++) {
        messages.push(
          { role: "user", content: `Message ${i}: ${"padding ".repeat(20)}` },
          { role: "assistant", content: `Response ${i}: ${"padding ".repeat(20)}` },
        );
      }
      messages.push({ role: "user", content: "Summarize in one word." });

      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages,
          max_tokens: 10,
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
      expect(body.usage.prompt_tokens).toBeGreaterThan(100);
    }, 60_000);

    it("valid JSON body with extra unknown fields passes through to OpenAI", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'hi'" }],
          max_tokens: 5,
          unknown_field_abc: "should be ignored by OpenAI",
          another_random: 42,
        }),
      });
      // OpenAI ignores unknown fields (or returns an error, both are fine)
      expect(res.status).toBeLessThan(500);
      await res.text();
    }, 30_000);
  });

  // ── Content-Type edge cases ──

  describe("Content-Type edge cases", () => {
    it("missing Content-Type header still works (proxy sets it)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'ct'" }],
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);

    it("wrong Content-Type (text/plain) still processes if body is valid JSON", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "text/plain",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'ct2'" }],
          max_tokens: 5,
        }),
      });
      // Proxy always sets Content-Type: application/json upstream
      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);
  });

  // ── Model edge cases ──

  describe("Model handling edge cases", () => {
    it("works with model alias that OpenAI resolves", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say 'alias'" }],
          max_tokens: 5,
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.model).toBeTruthy();
    }, 30_000);

    it("missing model field forwards error from OpenAI", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      expect(res.status).toBeGreaterThanOrEqual(400);
      expect(res.status).toBeLessThan(500);
    }, 15_000);

    it("numeric model field forwards error from OpenAI", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: 12345,
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      expect(res.status).toBeGreaterThanOrEqual(400);
      expect(res.status).toBeLessThan(500);
    }, 15_000);
  });

  // ── Proxy resilience after errors ──

  describe("Proxy resilience", () => {
    it("proxy is healthy after a burst of error requests", async () => {
      const errorRequests = Array.from({ length: 10 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: "not json!!!",
        }),
      );

      const responses = await Promise.all(errorRequests);
      for (const res of responses) {
        expect(res.status).toBe(400);
        await res.text();
      }

      // Proxy should still be healthy
      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    });

    it("proxy is healthy after a burst of auth failures", async () => {
      const authFailRequests = Array.from({ length: 10 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            "x-nullspend-key": "wrong-key",
          },
          body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
        }),
      );

      const responses = await Promise.all(authFailRequests);
      for (const res of responses) {
        expect(res.status).toBe(401);
        await res.text();
      }

      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    });
  });
});
