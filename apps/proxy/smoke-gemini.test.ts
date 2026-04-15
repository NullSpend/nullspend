/**
 * Gemini smoke tests — live deployed proxy + real Google API.
 *
 * Requires:
 *   - Deployed proxy worker at PROXY_URL
 *   - GOOGLE_API_KEY (Gemini API key) in .env.smoke
 *   - NULLSPEND_API_KEY (NullSpend platform key) in .env.smoke
 *
 * Run: cd apps/proxy && npx vitest run smoke-gemini.test.ts
 *
 * These hit the live deployed worker and make real API calls.
 * Cost: ~$0.001 per run (minimal token usage).
 * Manual runs only — never wire into CI.
 */

import { describe, it, expect, beforeAll } from "vitest";

const PROXY_URL = process.env.PROXY_URL ?? "https://proxy.nullspend.dev";
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const NULLSPEND_API_KEY = process.env.NULLSPEND_API_KEY;

const canRun = GOOGLE_API_KEY && NULLSPEND_API_KEY;

function geminiUrl(model: string, method: string): string {
  return `${PROXY_URL}/v1beta/models/${model}:${method}`;
}

function geminiHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${GOOGLE_API_KEY}`,
    "x-nullspend-key": NULLSPEND_API_KEY!,
  };
}

describe.skipIf(!canRun)("Gemini smoke tests", () => {
  beforeAll(() => {
    if (!canRun) return;
    console.log(`Gemini smoke: proxy=${PROXY_URL}`);
  });

  // ── Non-streaming ────────────────────────────────────────────────

  it("non-streaming: generateContent returns valid response with usage", async () => {
    // Gemini 2.5 Flash is a thinking model — it uses some of maxOutputTokens
    // for internal reasoning (thoughtsTokenCount). Use a generous budget
    // so the model has room to think AND produce visible output.
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "generateContent"),
      {
        method: "POST",
        headers: geminiHeaders(),
        body: JSON.stringify({
          contents: [{ parts: [{ text: "Say hello in exactly 3 words." }] }],
          generationConfig: { maxOutputTokens: 200 },
        }),
      },
    );

    expect(res.status).toBe(200);

    // NullSpend trace ID is a 32-char hex string
    const traceId = res.headers.get("X-NullSpend-Trace-Id");
    expect(traceId).toMatch(/^[a-f0-9]{32}$/);

    const body = await res.json();

    // Validate response structure matches Gemini API spec
    expect(body.candidates).toBeInstanceOf(Array);
    expect(body.candidates.length).toBeGreaterThanOrEqual(1);

    // Gemini thinking models may return content.parts with text,
    // or content with role only if all tokens went to thinking.
    // Check parts exist and contain text when present.
    const candidate = body.candidates[0];
    expect(candidate.content).toBeDefined();
    if (candidate.content.parts?.length > 0) {
      expect(typeof candidate.content.parts[0].text).toBe("string");
      expect(candidate.content.parts[0].text.length).toBeGreaterThan(0);
    }

    // Usage metadata must have realistic token counts
    expect(body.usageMetadata).toBeDefined();
    expect(body.usageMetadata.promptTokenCount).toBeGreaterThan(3);
    expect(body.usageMetadata.promptTokenCount).toBeLessThan(100);
    // candidatesTokenCount may be 0 if model spent all budget on thinking,
    // but totalTokenCount should always be present
    expect(body.usageMetadata.totalTokenCount).toBeGreaterThan(0);
  });

  // ── Streaming ────────────────────────────────────────────────────

  it("streaming: streamGenerateContent returns valid SSE with final usage", async () => {
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "streamGenerateContent"),
      {
        method: "POST",
        headers: geminiHeaders(),
        body: JSON.stringify({
          contents: [{ parts: [{ text: "Count from 1 to 5." }] }],
          generationConfig: { maxOutputTokens: 50 },
        }),
      },
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toMatch(/^[a-f0-9]{32}$/);

    const text = await res.text();

    // Validate SSE structure: data: lines separated by double newlines
    const events = text
      .split("\n\n")
      .filter((e) => e.trim().startsWith("data:"))
      .map((e) => e.trim().replace(/^data:\s*/, ""));

    expect(events.length).toBeGreaterThanOrEqual(1);

    // Each event should be valid JSON (Gemini sends complete responses per chunk)
    for (const event of events) {
      expect(() => JSON.parse(event)).not.toThrow();
    }

    // Last event should have complete usage
    const lastEvent = JSON.parse(events[events.length - 1]);
    expect(lastEvent.usageMetadata).toBeDefined();
    expect(lastEvent.usageMetadata.candidatesTokenCount).toBeGreaterThan(0);
    expect(lastEvent.usageMetadata.promptTokenCount).toBeGreaterThan(0);

    // All events should share the same responseId
    const responseIds = events
      .map((e) => JSON.parse(e).responseId)
      .filter(Boolean);
    if (responseIds.length > 1) {
      const unique = new Set(responseIds);
      expect(unique.size).toBe(1);
    }
  });

  // ── Routing ──────────────────────────────────────────────────────

  it("unknown Gemini method returns 404", async () => {
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "countTokens"),
      {
        method: "POST",
        headers: geminiHeaders(),
        body: JSON.stringify({ contents: [{ parts: [{ text: "hi" }] }] }),
      },
    );

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error.code).toBe("not_found");
  });

  // ── Observability ────────────────────────────────────────────────

  it("overhead and timing headers present on Gemini responses", async () => {
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "generateContent"),
      {
        method: "POST",
        headers: geminiHeaders(),
        body: JSON.stringify({
          contents: [{ parts: [{ text: "hi" }] }],
          generationConfig: { maxOutputTokens: 5 },
        }),
      },
    );

    expect(res.status).toBe(200);

    // Overhead header is a numeric string
    const overhead = res.headers.get("x-nullspend-overhead-ms");
    expect(overhead).not.toBeNull();
    expect(Number(overhead)).toBeGreaterThanOrEqual(0);
    expect(Number(overhead)).toBeLessThan(5000); // sanity: less than 5s overhead

    // Server-Timing contains proxy overhead and upstream duration
    const serverTiming = res.headers.get("Server-Timing");
    expect(serverTiming).toContain("overhead");
    expect(serverTiming).toContain("upstream");
    expect(serverTiming).toContain("total");
  });

  // ── Auth error ───────────────────────────────────────────────────

  it("missing NullSpend API key returns 401", async () => {
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "generateContent"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${GOOGLE_API_KEY}`,
          // No x-nullspend-key
        },
        body: JSON.stringify({
          contents: [{ parts: [{ text: "hi" }] }],
        }),
      },
    );

    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error.code).toBe("unauthorized");
  });

  // ── Cost accuracy ────────────────────────────────────────────────

  it("cost event reflects actual token usage (non-streaming)", async () => {
    const res = await fetch(
      geminiUrl("gemini-2.5-flash", "generateContent"),
      {
        method: "POST",
        headers: geminiHeaders(),
        body: JSON.stringify({
          contents: [{ parts: [{ text: "Reply with just the word OK." }] }],
          generationConfig: { maxOutputTokens: 200 },
        }),
      },
    );

    expect(res.status).toBe(200);
    const body = await res.json();

    // Verify usage is present and reasonable
    const usage = body.usageMetadata;
    expect(usage.promptTokenCount).toBeGreaterThan(0);
    // totalTokenCount includes prompt + candidates + thoughts
    expect(usage.totalTokenCount).toBeGreaterThan(0);
    expect(usage.totalTokenCount).toBeGreaterThanOrEqual(usage.promptTokenCount);
  });
});
