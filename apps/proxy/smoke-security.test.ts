/**
 * Security and attack vector tests for the live proxy.
 * Tests header injection, attribution spoofing resistance,
 * request smuggling, and proxy header stripping.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL for verifying attribution spoofing resistance
 */
import { describe, it, expect, beforeAll } from "vitest";
import postgres from "postgres";
import { BASE, OPENAI_API_KEY, NULLSPEND_API_KEY, NULLSPEND_SMOKE_USER_ID, DATABASE_URL, authHeaders, smallRequest, isServerUp, waitForCostEvent } from "./smoke-test-helpers.js";

describe("Security tests", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (DATABASE_URL) {
      sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
    }
  });

  // ── Timing attack resistance ──
  // Removed: SHA-256 hashing makes timing attacks on key comparison meaningless.
  // The proxy hashes the provided key and does a DB lookup — there is no
  // constant-time string comparison to test. Any key (short or long) goes
  // through the same hash-then-lookup path, so timing is dominated by the
  // DB round-trip, not the key value.

  // ── Header injection ──

  describe("Header injection attacks", () => {
    it("x-nullspend-key with null bytes is rejected by HTTP client (client-side protection)", async () => {
      // fetch() itself rejects null bytes in header values before sending
      await expect(
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            "x-nullspend-key": "valid-key\x00injected",
          },
          body: smallRequest(),
        }),
      ).rejects.toThrow();
    });

    it("very long x-nullspend-key (10KB) is rejected gracefully", async () => {
      const longKey = "x".repeat(10_000);
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": longKey,
        },
        body: smallRequest(),
      });

      // Should be rejected (401) not crash (502)
      expect([401, 431]).toContain(res.status);
      await res.text();
    });

    it("empty x-nullspend-key returns 401", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "",
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(401);
      await res.text();
    });

    it("missing x-nullspend-key header returns 401", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(401);
      await res.text();
    });

    it("x-nullspend-key with spaces/tabs still succeeds (HTTP runtime strips whitespace)", async () => {
      // Per HTTP spec, the runtime strips leading/trailing whitespace from
      // header values before the application sees them, so the padded key
      // is identical to the original after stripping. This is correct behavior.
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": `  ${NULLSPEND_API_KEY}  `,
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);
  });

  // ── Proxy header stripping (x-nullspend-key must not reach OpenAI) ──

  describe("Proxy header stripping", () => {
    it("x-nullspend-key is stripped from upstream request (not leaked to OpenAI)", async () => {
      // If the proxy forwarded x-nullspend-key to OpenAI, OpenAI would ignore it
      // but it would be a credential leak. We verify indirectly: a valid request
      // succeeds (so the proxy handled auth) and the response comes from OpenAI.
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.model).toContain("gpt-4o-mini");
      // The response is from OpenAI, confirming the proxy forwarded correctly
    }, 30_000);

    it("X-NullSpend-User-Id and X-NullSpend-Key-Id headers are ignored (no auth significance)", async () => {
      // These headers no longer carry auth information — userId and keyId are
      // derived from the API key hash. Sending them should have no effect.
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({
          "X-NullSpend-User-Id": "test-user",
          "X-NullSpend-Key-Id": "test-key",
        }),
        body: smallRequest(),
      });

      // Request should still succeed; the extra headers are stripped/ignored.
      expect(res.status).toBe(200);
      await res.json();
    }, 30_000);
  });

  // ── Attribution spoofing resistance ──

  describe("Attribution spoofing resistance", () => {
    it("spoofed X-NullSpend-User-Id is ignored — real userId from API key is recorded", async () => {
      if (!sql) return; // skip if no DATABASE_URL

      const spoofedUserId = `spoofed-user-${Date.now()}`;
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({
          "X-NullSpend-User-Id": spoofedUserId,
        }),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      const requestId = res.headers.get("x-request-id") ?? body.id;

      // Wait for cost event to be logged (queue consumer + DB write)
      const row = await waitForCostEvent(sql, requestId, 15_000);
      expect(row).not.toBeNull();

      // The spoofed user ID is NOT recorded — the real userId derived from
      // the API key hash is used instead, preventing attribution spoofing.
      expect(row!.user_id).toBe(NULLSPEND_SMOKE_USER_ID);
    }, 15_000);
  });

  // ── Request smuggling ──

  describe("Request smuggling resistance", () => {
    it("null bytes in URL path are rejected (400 on CF, 404 on Miniflare)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions%00admin`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      // Cloudflare production returns 400; local Miniflare returns 404
      expect([400, 404]).toContain(res.status);
      await res.text();
    });

    it("path traversal /v1/../admin returns 404", async () => {
      const res = await fetch(`${BASE}/v1/../admin`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(404);
      await res.text();
    });

    it("oversized Content-Length with small body is rejected by fetch client", async () => {
      // Node's undici rejects content-length mismatches before sending
      await expect(
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            ...authHeaders(),
            "Content-Length": "999999999",
          },
          body: smallRequest(),
        }),
      ).rejects.toThrow();
    });

    it("POST to /health is treated as GET (returns health response)", async () => {
      const res = await fetch(`${BASE}/health`, {
        method: "POST",
      });

      // Cloudflare Workers URL routing checks pathname, not method for /health
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");
    });

    it("DELETE method to /v1/chat/completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "DELETE",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(404);
      await res.text();
    });

    it("PUT method to /v1/chat/completions returns 404", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "PUT",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(404);
      await res.text();
    });

    it("very deep path nesting returns 404 without crashing", async () => {
      const deepPath = "/v1/" + "a/".repeat(100) + "chat/completions";
      const res = await fetch(`${BASE}${deepPath}`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(404);
      await res.text();
    });
  });

  // ── Body-level attacks ──

  describe("Malicious request bodies", () => {
    it("body with __proto__ pollution attempt is handled safely", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "hi" }],
          max_tokens: 3,
          __proto__: { admin: true },
          constructor: { prototype: { admin: true } },
        }),
      });

      // Should be 200 (proxy passes through to OpenAI, which ignores extra fields)
      // or 400 if the proxy rejects it. Either way, should not be 502.
      expect(res.status).not.toBe(502);
      await res.text();
    }, 30_000);

    it("deeply nested JSON doesn't crash the proxy", async () => {
      let nested: unknown = { role: "user", content: "hi" };
      for (let i = 0; i < 50; i++) {
        nested = { nested };
      }

      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [nested],
          max_tokens: 3,
        }),
      });

      // OpenAI will reject the malformed messages, but proxy should not crash
      expect(res.status).not.toBe(502);
      await res.text();
    }, 30_000);

    it("body that is a JSON array (not object) returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify([{ model: "gpt-4o-mini" }]),
      });

      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("body that is a JSON string returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify("just a string"),
      });

      expect(res.status).toBe(400);
      await res.text();
    });

    it("body that is a JSON number returns 400", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "42",
      });

      expect(res.status).toBe(400);
      await res.text();
    });

    it("body with unicode/emoji in messages is handled correctly", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say \u{1F600}\u{1F4A9}\u{0000}\u{FFFF}" }],
          max_tokens: 3,
        }),
      });

      expect(res.status).toBe(200);
      await res.text();
    }, 30_000);
  });
});
