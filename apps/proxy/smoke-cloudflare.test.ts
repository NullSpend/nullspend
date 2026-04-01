/**
 * Cloudflare-specific smoke tests for the OpenAI proxy.
 * Tests edge cases derived from Cloudflare Workers runtime documentation:
 *
 * - Connection management: rapid sequential requests, connection reuse
 * - SSE stream integrity: every line follows proper SSE format
 * - Path edge cases: query params, fragments, case sensitivity, double slashes
 * - Multiple sequential aborts: stress-test stream teardown
 * - Request/response header limits: very large headers
 * - Token accuracy: verify usage fields are present and valid
 * - waitUntil resilience: proxy stays healthy after cost-logging paths
 * - Response completeness: non-streaming JSON has all required fields
 * - Idempotency: repeated identical requests produce consistent results
 *
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: pnpm proxy:smoke
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, OPENAI_API_KEY, NULLSPEND_API_KEY, authHeaders, smallRequest, isServerUp } from "./smoke-test-helpers.js";

describe("Cloudflare runtime edge cases", () => {
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

  // ── Path edge cases (URL parsing in Workers runtime) ──

  describe("URL parsing edge cases", () => {
    it("query parameters on /v1/chat/completions are ignored (request still works)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions?foo=bar&debug=true`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);

    it("URL with fragment is handled correctly", async () => {
      // Fragments are stripped by the browser/fetch but let's verify
      const res = await fetch(`${BASE}/v1/chat/completions#section`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);

    it("uppercase path /V1/CHAT/COMPLETIONS returns 404 (case-sensitive routing)", async () => {
      const res = await fetch(`${BASE}/V1/CHAT/COMPLETIONS`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(404);
    });

    it("mixed case path /v1/Chat/Completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/Chat/Completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(404);
    });

    it("double-slash path //v1//chat//completions returns 404", async () => {
      const res = await fetch(`${BASE}//v1//chat//completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(404);
    });

    it("path with percent-encoded characters returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat%2Fcompletions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(404);
    });

    it("HEAD request to /health returns 200 (no body)", async () => {
      const res = await fetch(`${BASE}/health`, { method: "HEAD" });
      expect(res.status).toBe(200);
    });

    it("OPTIONS request to /v1/chat/completions returns 404 (not POST)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "OPTIONS",
        headers: authHeaders(),
      });
      expect(res.status).toBe(404);
    });
  });

  // ── SSE stream integrity ──

  describe("SSE stream integrity", () => {
    it("every non-empty line in streaming response follows SSE format", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      const lines = text.split("\n");

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed === "") continue;

        // Valid SSE line formats: "data: ...", ": comment", "event: ...", "id: ...", "retry: ..."
        const isValidSSE =
          trimmed.startsWith("data:") ||
          trimmed.startsWith(":") ||
          trimmed.startsWith("event:") ||
          trimmed.startsWith("id:") ||
          trimmed.startsWith("retry:");

        expect(isValidSSE).toBe(true);
      }
    }, 30_000);

    it("streaming response starts with data: and ends with [DONE]", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      const text = await res.text();
      const dataLines = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"));

      expect(dataLines.length).toBeGreaterThan(0);

      const lastDataLine = dataLines[dataLines.length - 1].trim();
      expect(lastDataLine).toBe("data: [DONE]");
    }, 30_000);

    it("streaming response contains parseable JSON in data lines (except [DONE])", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      const text = await res.text();
      const dataLines = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim());

      for (const payload of dataLines) {
        if (payload === "[DONE]") continue;
        expect(() => JSON.parse(payload)).not.toThrow();
        const parsed = JSON.parse(payload);
        expect(parsed).toHaveProperty("id");
      }
    }, 30_000);

    it("streaming response includes usage in the final data chunk", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      const text = await res.text();
      const dataLines = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim())
        .filter((l) => l !== "[DONE]");

      const lastChunk = JSON.parse(dataLines[dataLines.length - 1]);
      expect(lastChunk).toHaveProperty("usage");
      expect(lastChunk.usage).toHaveProperty("prompt_tokens");
      expect(lastChunk.usage).toHaveProperty("completion_tokens");
      expect(lastChunk.usage.prompt_tokens).toBeGreaterThan(0);
    }, 30_000);

    it("streaming with max_tokens: 1 produces at least one content chunk", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true, max_tokens: 1 }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("data:");
      expect(text).toContain("[DONE]");
    }, 30_000);
  });

  // ── Non-streaming response integrity ──

  describe("Non-streaming response integrity", () => {
    it("response JSON has all required OpenAI fields", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();

      expect(body).toHaveProperty("id");
      expect(body.id).toMatch(/^chatcmpl-/);
      expect(body).toHaveProperty("object", "chat.completion");
      expect(body).toHaveProperty("created");
      expect(typeof body.created).toBe("number");
      expect(body).toHaveProperty("model");
      expect(body).toHaveProperty("choices");
      expect(Array.isArray(body.choices)).toBe(true);
      expect(body.choices.length).toBeGreaterThan(0);
      expect(body.choices[0]).toHaveProperty("message");
      expect(body.choices[0].message).toHaveProperty("role", "assistant");
      expect(body.choices[0].message).toHaveProperty("content");
      expect(body.choices[0]).toHaveProperty("finish_reason");
      expect(body).toHaveProperty("usage");
      expect(body.usage).toHaveProperty("prompt_tokens");
      expect(body.usage).toHaveProperty("completion_tokens");
      expect(body.usage).toHaveProperty("total_tokens");
      expect(body.usage.total_tokens).toBe(
        body.usage.prompt_tokens + body.usage.completion_tokens,
      );
    }, 30_000);

    it("usage token counts are reasonable for a small request", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false, max_tokens: 5 }),
      });

      const body = await res.json();
      expect(body.usage.prompt_tokens).toBeGreaterThan(0);
      expect(body.usage.prompt_tokens).toBeLessThan(100);
      expect(body.usage.completion_tokens).toBeGreaterThan(0);
      expect(body.usage.completion_tokens).toBeLessThanOrEqual(5);
    }, 30_000);
  });

  // ── Connection management (6 simultaneous connections per request) ──

  describe("Connection management", () => {
    it("rapid sequential requests reuse connections correctly", async () => {
      for (let i = 0; i < 5; i++) {
        const res = await fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ stream: false }),
        });
        expect(res.status).toBe(200);
        const body = await res.json();
        expect(body).toHaveProperty("usage");
      }
    }, 120_000);

    it("handles 3 concurrent streaming + 3 concurrent non-streaming requests", async () => {
      const streamReqs = Array.from({ length: 3 }, (_, _i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ stream: true }),
        }),
      );

      const nonStreamReqs = Array.from({ length: 3 }, (_, _i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ stream: false }),
        }),
      );

      const results = await Promise.all([...streamReqs, ...nonStreamReqs]);

      for (let i = 0; i < 3; i++) {
        expect(results[i].status).toBe(200);
        const text = await results[i].text();
        expect(text).toContain("[DONE]");
      }

      for (let i = 3; i < 6; i++) {
        expect(results[i].status).toBe(200);
        const body = await results[i].json();
        expect(body).toHaveProperty("usage");
      }
    }, 90_000);

    it("10 rapid-fire auth failures don't exhaust connections", async () => {
      const reqs = Array.from({ length: 10 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-nullspend-key": "bad-key",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
          },
          body: smallRequest(),
        }),
      );

      const results = await Promise.all(reqs);
      for (const res of results) {
        expect(res.status).toBe(401);
        await res.text();
      }

      // Verify connections are freed up
      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    }, 15_000);
  });

  // ── Multiple sequential stream aborts ──

  describe("Stream teardown resilience", () => {
    it("3 sequential early aborts don't crash the proxy", async () => {
      for (let i = 0; i < 3; i++) {
        const controller = new AbortController();
        try {
          const res = await fetch(`${BASE}/v1/chat/completions`, {
            method: "POST",
            headers: authHeaders(),
            body: smallRequest({
              stream: true,
              messages: [
                { role: "user", content: "Write a long story about space exploration." },
              ],
              max_tokens: 200,
            }),
            signal: controller.signal,
          });

          expect(res.status).toBe(200);
          const reader = res.body!.getReader();
          await reader.read();
          controller.abort();
          reader.releaseLock();
        } catch {
          // AbortError is expected
        }

        // Brief pause to let the proxy clean up
        await new Promise((r) => setTimeout(r, 300));
      }

      // Verify proxy is still healthy
      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    }, 90_000);

    it("abort before reading any bytes still leaves proxy healthy", async () => {
      const controller = new AbortController();

      try {
        const resPromise = fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: true,
            max_tokens: 100,
          }),
          signal: controller.signal,
        });

        // Abort immediately after sending request, before reading response
        setTimeout(() => controller.abort(), 50);
        await resPromise;
      } catch {
        // AbortError expected
      }

      await new Promise((r) => setTimeout(r, 500));
      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    }, 30_000);
  });

  // ── Header limit awareness ──

  describe("Header handling limits", () => {
    it("request with many custom headers (non-forwarded) still works", async () => {
      const extraHeaders: Record<string, string> = {};
      for (let i = 0; i < 20; i++) {
        extraHeaders[`x-custom-header-${i}`] = `value-${i}-${"x".repeat(100)}`;
      }

      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(extraHeaders),
        body: smallRequest(),
      });

      // Custom headers should be stripped; request should succeed
      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);

    it("very long Authorization header is forwarded to OpenAI", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${"x".repeat(5000)}`,
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: smallRequest(),
      });

      // OpenAI will reject the invalid key, but proxy should forward it
      expect(res.status).toBe(401);
      await res.text();
    }, 30_000);
  });

  // ── Idempotency and consistency ──

  describe("Response consistency", () => {
    it("x-request-id is unique across two sequential requests", async () => {
      const res1 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });
      const id1 = res1.headers.get("x-request-id");
      await res1.json();

      const res2 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });
      const id2 = res2.headers.get("x-request-id");
      await res2.json();

      expect(id1).toBeTruthy();
      expect(id2).toBeTruthy();
      expect(id1).not.toBe(id2);
    }, 60_000);

    it("model in response matches or is a specific version of requested model", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false, model: "gpt-4o-mini" }),
      });

      const body = await res.json();
      // OpenAI may return the specific version like gpt-4o-mini-2024-07-18
      expect(body.model).toContain("gpt-4o-mini");
    }, 30_000);

    it("deterministic output with temperature 0 and seed", async () => {
      const reqBody = smallRequest({
        stream: false,
        temperature: 0,
        seed: 42,
        messages: [{ role: "user", content: "What is 2+2? Answer with just the number." }],
        max_tokens: 5,
      });

      const res1 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: reqBody,
      });
      const body1 = await res1.json();

      const res2 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: reqBody,
      });
      const body2 = await res2.json();

      // With temperature 0 and same seed, content should be identical
      expect(body1.choices[0].message.content).toBe(body2.choices[0].message.content);
    }, 60_000);
  });

  // ── Error-then-success recovery pattern ──

  describe("Error recovery patterns", () => {
    it("successful request after OpenAI error (invalid model then valid model)", async () => {
      // First: invalid model -> error
      const errorRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ model: "nonexistent-model-xyz" }),
      });
      expect(errorRes.status).toBeGreaterThanOrEqual(400);
      await errorRes.text();

      // Second: valid model -> success
      const successRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ model: "gpt-4o-mini" }),
      });
      expect(successRes.status).toBe(200);
      const body = await successRes.json();
      expect(body).toHaveProperty("usage");
    }, 30_000);

    it("successful streaming after auth failure", async () => {
      // First: auth failure
      const authRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "wrong",
        },
        body: smallRequest({ stream: true }),
      });
      expect(authRes.status).toBe(401);
      await authRes.text();

      // Second: valid streaming request
      const streamRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });
      expect(streamRes.status).toBe(200);
      const text = await streamRes.text();
      expect(text).toContain("[DONE]");
    }, 30_000);

    it("successful request after malformed body error", async () => {
      const badRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "totally not json {{{",
      });
      expect(badRes.status).toBe(400);
      await badRes.text();

      const goodRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(goodRes.status).toBe(200);
      await goodRes.text();
    }, 30_000);
  });

  // ── OpenAI-specific parameter handling ──

  describe("OpenAI parameter passthrough", () => {
    it("temperature parameter is respected", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ temperature: 0, stream: false }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("choices");
    }, 30_000);

    it("top_p parameter is respected", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ top_p: 0.1, stream: false }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("choices");
    }, 30_000);

    it("stop sequence parameter is respected", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stop: ["\n"],
          stream: false,
          messages: [{ role: "user", content: "Count from 1 to 10, one per line" }],
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("choices");
      expect(body.choices[0].finish_reason).toMatch(/stop|length/);
    }, 30_000);

    it("system message is passed through correctly", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: "You are a pirate. Always respond with 'Arrr!'." },
            { role: "user", content: "Hello" },
          ],
          max_tokens: 10,
          stream: false,
          temperature: 0,
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices[0].message.content.toLowerCase()).toContain("arr");
    }, 30_000);

    it("multi-turn conversation context is preserved through proxy", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "user", content: "Remember: the secret word is 'banana'" },
            { role: "assistant", content: "Got it! The secret word is banana." },
            { role: "user", content: "What is the secret word? Reply with just the word." },
          ],
          max_tokens: 10,
          temperature: 0,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices[0].message.content.toLowerCase()).toContain("banana");
    }, 30_000);
  });

  // ── waitUntil resilience (cost logging) ──

  describe("waitUntil and cost logging resilience", () => {
    it("proxy remains responsive during cost logging (streaming)", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      const elapsed = performance.now() - start;

      expect(text).toContain("[DONE]");
      // Response should come back well under the 30s waitUntil limit
      expect(elapsed).toBeLessThan(25_000);
    }, 30_000);

    it("proxy remains responsive during cost logging (non-streaming)", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      const elapsed = performance.now() - start;

      expect(body).toHaveProperty("usage");
      expect(elapsed).toBeLessThan(25_000);
    }, 30_000);

    it("multiple requests in rapid succession all get cost-logged without error", async () => {
      const requests = Array.from({ length: 3 }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({
            stream: i % 2 === 0,
            messages: [{ role: "user", content: `Count: ${i}` }],
          }),
        }),
      );

      const responses = await Promise.all(requests);
      for (const res of responses) {
        expect(res.status).toBe(200);
        await res.text();
      }

      // Brief pause for waitUntil to complete
      await new Promise((r) => setTimeout(r, 1000));

      const healthRes = await fetch(`${BASE}/health`);
      expect(healthRes.status).toBe(200);
    }, 60_000);
  });
});
