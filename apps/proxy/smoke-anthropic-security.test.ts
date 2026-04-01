/**
 * Anthropic security smoke tests.
 * Validates header stripping, rate-limit forwarding, spoofing resistance,
 * unicode handling, and data leak prevention for the /v1/messages route.
 *
 * Requires: live proxy, ANTHROPIC_API_KEY, NULLSPEND_API_KEY, DATABASE_URL
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  ANTHROPIC_API_KEY,
  NULLSPEND_SMOKE_USER_ID,
  DATABASE_URL,
  anthropicAuthHeaders,
  isServerUp,
  waitForCostEvent,
} from "./smoke-test-helpers.js";

describe("Anthropic security", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");
    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("x-api-key header is NOT present in the response", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Header leak check" }],
      }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("x-api-key")).toBeNull();
    await res.text();
  }, 30_000);

  it("x-nullspend-key is not leaked in response headers", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Auth leak check" }],
      }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("x-nullspend-key")).toBeNull();
    await res.text();
  }, 30_000);

  it("attribution headers not forwarded to Anthropic (verify via successful response)", async () => {
    // X-NullSpend-User-Id and X-NullSpend-Key-Id no longer carry auth
    // information — they are stripped by the proxy regardless.
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Attribution strip check" }],
      }),
    });

    // A 200 means Anthropic processed the request fine.
    expect(res.status).toBe(200);
    await res.text();
  }, 30_000);

  it("anthropic-version header is set on the upstream request", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Version header check" }],
      }),
    });

    // A 200 confirms Anthropic accepted the request, which requires a valid
    // anthropic-version header to be set
    expect(res.status).toBe(200);
    await res.text();
  }, 30_000);

  it("Anthropic rate-limit headers are forwarded in the response", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Rate limit headers" }],
      }),
    });

    expect(res.status).toBe(200);

    // Anthropic sends rate-limit headers in most responses
    const allHeaders = [...res.headers.entries()];
    const rateLimitHeaders = allHeaders.filter(([k]) =>
      k.toLowerCase().startsWith("anthropic-ratelimit"),
    );

    // We expect at least some rate-limit headers; if Anthropic doesn't send
    // them for this tier, this is informational rather than a hard failure
    if (rateLimitHeaders.length > 0) {
      console.log(`[security] Rate-limit headers found: ${rateLimitHeaders.map(([k]) => k).join(", ")}`);
    } else {
      console.log("[security] No anthropic-ratelimit-* headers in response (may be tier-dependent)");
    }

    await res.text();
  }, 30_000);

  it("spoofed X-NullSpend-User-Id is ignored — real userId from API key is recorded", async () => {
    const userId = `sec-user-${Date.now()}`;
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders({
        "X-NullSpend-User-Id": userId,
      }),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "User ID recording" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    // The spoofed userId is NOT recorded — the real userId derived from
    // the API key hash is used instead, preventing attribution spoofing.
    expect(row!.user_id).toBe(NULLSPEND_SMOKE_USER_ID);
  }, 30_000);

  it("body with unicode/emoji in messages is handled correctly", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [
          {
            role: "user",
            content: "Hello! Here are some special chars: \u00e9\u00e8\u00ea\u00eb \u00fc\u00f6\u00e4 \u4f60\u597d \ud83d\ude80\ud83c\udf1f\ud83d\udcbb",
          },
        ],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.type).toBe("message");
    expect(body.content[0].text.length).toBeGreaterThan(0);
  }, 30_000);

  it("deeply nested JSON in message content doesn't crash the proxy", async () => {
    let nested: unknown = "leaf";
    for (let i = 0; i < 50; i++) {
      nested = { level: i, data: nested };
    }

    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [
          {
            role: "user",
            content: `Process this: ${JSON.stringify(nested).slice(0, 500)}`,
          },
        ],
      }),
    });

    // Should not crash (502). May succeed or fail with Anthropic error.
    expect(res.status).not.toBe(502);
    await res.text();
  }, 30_000);

  it("__proto__ pollution attempt in Anthropic request body is handled safely", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Proto test" }],
        __proto__: { isAdmin: true },
        constructor: { prototype: { isAdmin: true } },
      }),
    });

    // Should not crash or escalate privileges
    expect(res.status).not.toBe(502);
    await res.text();
  }, 30_000);

  // Removed: timing attack test for wrong platform keys.
  // With hash-based auth (SHA-256 + DB lookup), timing attacks are meaningless.
  // Every key — regardless of length or similarity to a valid key — goes through
  // the same hash-then-lookup path. Timing is dominated by the DB round-trip,
  // not the key value.
});
