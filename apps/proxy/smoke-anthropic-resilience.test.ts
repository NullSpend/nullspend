/**
 * Anthropic resilience smoke tests.
 * Validates error recovery, burst handling, and request isolation for
 * the /v1/messages route.
 *
 * Requires: live proxy, ANTHROPIC_API_KEY, NULLSPEND_API_KEY
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  BASE,
  ANTHROPIC_API_KEY,
  NULLSPEND_API_KEY,
  anthropicAuthHeaders,
  smallAnthropicRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

describe("Anthropic resilience", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
  });

  it("invalid Anthropic API key returns upstream error transparently (not 502)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "sk-ant-invalid-key-for-resilience-test",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: smallAnthropicRequest(),
    });

    // Anthropic returns 401 for invalid key; proxy should forward, not 502
    expect(res.status).toBe(401);
    expect(res.status).not.toBe(502);
    await res.text();
  }, 30_000);

  it("proxy does NOT return 502 for invalid upstream key", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "sk-ant-totally-bogus",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: smallAnthropicRequest(),
    });

    expect(res.status).not.toBe(502);
    await res.text();
  }, 30_000);

  it("/health stays responsive after a burst of Anthropic errors", async () => {
    const errorRequests = Array.from({ length: 5 }, () =>
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": "sk-ant-bad-key-burst",
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: smallAnthropicRequest(),
      }),
    );

    await Promise.all(errorRequests.map((r) => r.then((res) => res.text())));

    const start = performance.now();
    const health = await fetch(`${BASE}/health`);
    const elapsed = performance.now() - start;

    expect(health.ok).toBe(true);
    expect(elapsed).toBeLessThan(500);
  }, 30_000);

  it("/health/ready still works after Anthropic auth failures", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "sk-ant-bad-key-ready",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: smallAnthropicRequest(),
    });
    await res.text();

    const ready = await fetch(`${BASE}/health/ready`);
    expect(ready.ok).toBe(true);
    const body = await ready.json();
    expect(body.status).toBe("ok");
  }, 30_000);

  it("proxy recovers after client aborts 5 streaming Anthropic requests", async () => {
    for (let i = 0; i < 5; i++) {
      const controller = new AbortController();
      try {
        const res = await fetch(`${BASE}/v1/messages`, {
          method: "POST",
          headers: anthropicAuthHeaders(),
          body: JSON.stringify({
            model: "claude-3-haiku-20240307",
            max_tokens: 50,
            messages: [{ role: "user", content: `Abort recovery ${i}` }],
            stream: true,
          }),
          signal: controller.signal,
        });
        if (res.status === 200) {
          setTimeout(() => controller.abort(), 100);
          await res.text();
        }
      } catch {
        // AbortError expected
      }
      await new Promise((r) => setTimeout(r, 300));
    }

    // Proxy should still be healthy
    const valid = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Recovery check" }],
      }),
    });
    expect(valid.status).toBe(200);
    const body = await valid.json();
    expect(body.type).toBe("message");
  }, 60_000);

  it("auth error on one Anthropic request doesn't affect the next", async () => {
    const badRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "sk-ant-bad-isolation",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: smallAnthropicRequest(),
    });
    expect(badRes.status).not.toBe(200);
    await badRes.text();

    const goodRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });
    expect(goodRes.status).toBe(200);
    const body = await goodRes.json();
    expect(body.type).toBe("message");
  }, 30_000);

  it("malformed body error doesn't affect subsequent valid Anthropic requests", async () => {
    const badRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: "{invalid json!!!",
    });
    expect(badRes.status).toBe(400);
    await badRes.text();

    const goodRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: smallAnthropicRequest(),
    });
    expect(goodRes.status).toBe(200);
    const body = await goodRes.json();
    expect(body.type).toBe("message");
  }, 30_000);

  it("concurrent valid and invalid Anthropic requests are isolated", async () => {
    const requests = [
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: smallAnthropicRequest(),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": "sk-ant-bad-concurrent",
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: smallAnthropicRequest(),
      }),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: smallAnthropicRequest(),
      }),
    ];

    const [good1, bad, good2] = await Promise.all(requests);
    expect(good1.status).toBe(200);
    expect(bad.status).not.toBe(200);
    expect(good2.status).toBe(200);

    await good1.text();
    await bad.text();
    await good2.text();
  }, 30_000);

  it("proxy never returns 502 for normal Anthropic operations", async () => {
    const requests = [
      fetch(`${BASE}/health`),
      fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: smallAnthropicRequest(),
      }),
      fetch(`${BASE}/v1/nonexistent`, { method: "POST" }),
    ];

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).not.toBe(502);
      await res.text();
    }
  }, 30_000);

  it("request with max_tokens: 4096 still works (high output without timeout)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 4096,
        messages: [{ role: "user", content: "Say ok" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.type).toBe("message");
    expect(body.usage.output_tokens).toBeGreaterThan(0);
  }, 60_000);
});
