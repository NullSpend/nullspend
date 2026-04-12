/**
 * Resilience and chaos tests for the live proxy.
 * Tests behavior when dependencies fail or degrade.
 *
 * Approach: Since we can't inject faults into the live deployment's
 * dependencies, we test observable resilience patterns:
 * - Invalid OpenAI API key → upstream error forwarded transparently
 * - Invalid model → model validation blocks before upstream call
 * - Slow responses (high max_tokens) → timeout behavior
 * - Aborted streams → proxy stays healthy
 * - Health endpoints remain responsive under error conditions
 * - Rate limiter fail-open verification
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, OPENAI_API_KEY, NULLSPEND_API_KEY, authHeaders, smallRequest, isServerUp } from "./smoke-test-helpers.js";

describe("Resilience tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
  });

  // ── OpenAI error transparency ──

  describe("OpenAI error forwarding", () => {
    it("invalid OpenAI API key returns upstream 401 transparently", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer sk-invalid-key-for-resilience-test",
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(401);
      const body = await res.json();
      // OpenAI's error format should be forwarded as-is
      expect(body).toHaveProperty("error");
    }, 15_000);

    it("invalid model returns 400 from proxy (model validation)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ model: "nonexistent-model-resilience" }),
      });

      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("invalid_model");
    });

    it("proxy does NOT return 502 for invalid upstream key", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer sk-bad",
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: smallRequest(),
      });

      // Must be the upstream status (401), never 502
      expect(res.status).not.toBe(502);
      await res.text();
    }, 15_000);
  });

  // ── Health endpoint resilience ──

  describe("Health endpoint resilience", () => {
    it("/health stays responsive after a burst of errors", async () => {
      // Send 5 error-producing requests
      const errorRequests = Array.from({ length: 5 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer invalid",
            "x-nullspend-key": NULLSPEND_API_KEY!,
          },
          body: smallRequest(),
        }),
      );

      const results = await Promise.all(errorRequests);
      for (const r of results) {
        expect(r.status).toBe(401);
        await r.text();
      }

      // Health should still respond quickly
      const start = performance.now();
      const healthRes = await fetch(`${BASE}/health`);
      const elapsed = performance.now() - start;

      expect(healthRes.status).toBe(200);
      expect(elapsed).toBeLessThan(500);
    }, 30_000);

    it("/health/ready still works after auth failures", async () => {
      // Auth failure
      const authRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer bad",
          "x-nullspend-key": "bad",
        },
        body: smallRequest(),
      });
      expect(authRes.status).toBe(401);
      await authRes.text();

      // Ready endpoint should still be fine
      const readyRes = await fetch(`${BASE}/health/ready`);
      expect(readyRes.status).toBe(200);
      const body = await readyRes.json();
      expect(body.status).toBe("ok");
    });

    it("/health and /health/ready respond independently of each other", async () => {
      const [h, r] = await Promise.all([
        fetch(`${BASE}/health`),
        fetch(`${BASE}/health/ready`),
      ]);

      expect(h.status).toBe(200);
      expect(r.status).toBe(200);

      const healthBody = await h.json();
      const readyBody = await r.json();

      expect(healthBody.status).toBe("ok");
      expect(readyBody.status).toBe("ok");
    });
  });

  // ── Stream error recovery ──

  describe("Stream error recovery", () => {
    it("proxy recovers after client aborts 5 streaming requests", async () => {
      for (let i = 0; i < 5; i++) {
        const controller = new AbortController();
        try {
          const res = await fetch(`${BASE}/v1/chat/completions`, {
            method: "POST",
            headers: authHeaders(),
            body: smallRequest({
              stream: true,
              messages: [{ role: "user", content: `Abort test ${i}` }],
              max_tokens: 100,
            }),
            signal: controller.signal,
          });

          if (res.status === 200) {
            const reader = res.body!.getReader();
            await reader.read();
            controller.abort();
            reader.releaseLock();
          }
        } catch {
          // AbortError expected
        }
        await new Promise((r) => setTimeout(r, 200));
      }

      // Proxy should still be healthy and able to serve complete responses
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: false }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
    }, 60_000);

    it("streaming response with high max_tokens completes successfully", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          stream: true,
          messages: [{ role: "user", content: "Write 10 random words." }],
          max_tokens: 100,
        }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");
    }, 30_000);
  });

  // ── Error isolation ──

  describe("Error isolation", () => {
    it("auth error on one request doesn't affect the next", async () => {
      // Bad request
      const bad = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "wrong",
        },
        body: smallRequest(),
      });
      expect(bad.status).toBe(401);
      await bad.text();

      // Good request immediately after
      const good = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(good.status).toBe(200);
      const body = await good.json();
      expect(body).toHaveProperty("usage");
    }, 30_000);

    it("malformed body error doesn't affect subsequent valid requests", async () => {
      const bad = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "{{{invalid json",
      });
      expect(bad.status).toBe(400);
      await bad.text();

      const good = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(good.status).toBe(200);
      await good.json();
    }, 30_000);

    it("concurrent valid and invalid requests are isolated", async () => {
      const requests = [
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest(),
        }),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: "{bad json}",
        }),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ stream: true }),
        }),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer invalid",
            "x-nullspend-key": NULLSPEND_API_KEY!,
          },
          body: smallRequest(),
        }),
      ];

      const results = await Promise.all(requests);

      expect(results[0].status).toBe(200);
      await results[0].json();

      expect(results[1].status).toBe(400);
      await results[1].text();

      expect(results[2].status).toBe(200);
      const streamText = await results[2].text();
      expect(streamText).toContain("[DONE]");

      expect(results[3].status).toBe(401);
      await results[3].text();
    }, 30_000);
  });

  // ── Graceful degradation patterns ──

  describe("Graceful degradation", () => {
    it("proxy never returns 502 for normal operations", async () => {
      const operations = [
        fetch(`${BASE}/health`),
        fetch(`${BASE}/health/ready`),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest(),
        }),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ stream: true }),
        }),
        fetch(`${BASE}/not-a-route`),
      ];

      const results = await Promise.all(operations);
      for (const res of results) {
        expect(res.status).not.toBe(502);
        await res.text();
      }
    }, 30_000);

    it("request with extremely large max_tokens still works (proxy doesn't timeout)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({
          messages: [{ role: "user", content: "Say 'done'" }],
          max_tokens: 4096,
        }),
      });

      // Even with high max_tokens, the model should stop early for simple prompts
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
    }, 60_000);
  });
});
