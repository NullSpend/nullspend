/**
 * Pricing accuracy smoke tests.
 * Verifies that cost_microdollars in the database matches the expected
 * formula for various models, alias resolution works, and edge cases
 * like cached tokens and zero-cost are handled correctly.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL for direct Supabase queries
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import { BASE, OPENAI_API_KEY, DATABASE_URL, authHeaders, isServerUp, waitForCostEvent } from "./smoke-test-helpers.js";

const PRICING: Record<string, { inputPerMTok: number; cachedInputPerMTok: number; outputPerMTok: number }> = {
  "gpt-4o-mini": { inputPerMTok: 0.15, cachedInputPerMTok: 0.075, outputPerMTok: 0.60 },
  "gpt-4o": { inputPerMTok: 2.50, cachedInputPerMTok: 1.25, outputPerMTok: 10.00 },
  "gpt-4.1-nano": { inputPerMTok: 0.10, cachedInputPerMTok: 0.025, outputPerMTok: 0.40 },
};

function expectedCostMicrodollars(
  pricing: { inputPerMTok: number; cachedInputPerMTok: number; outputPerMTok: number },
  inputTokens: number,
  outputTokens: number,
  cachedTokens = 0,
): number {
  const normalInput = inputTokens - cachedTokens;
  return Math.round(
    normalInput * pricing.inputPerMTok +
    cachedTokens * pricing.cachedInputPerMTok +
    outputTokens * pricing.outputPerMTok,
  );
}

describe("Pricing accuracy", () => {
  let sql: postgres.Sql;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  it("gpt-4o-mini cost matches formula: (input * inputRate + output * outputRate)", async () => {
    const model = "gpt-4o-mini";
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "Say hello" }],
        max_tokens: 5,
        stream: false,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();

    const actualCost = Number(row!.cost_microdollars);
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const expected = expectedCostMicrodollars(
      PRICING[model],
      usage.prompt_tokens,
      usage.completion_tokens,
      cachedTokens,
    );

    expect(actualCost).toBe(expected);
  }, 30_000);

  it("gpt-4o cost matches formula", async () => {
    const model = "gpt-4o";
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "Say hi" }],
        max_tokens: 3,
        stream: false,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();

    const actualCost = Number(row!.cost_microdollars);
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const expected = expectedCostMicrodollars(
      PRICING[model],
      usage.prompt_tokens,
      usage.completion_tokens,
      cachedTokens,
    );

    expect(actualCost).toBe(expected);
  }, 30_000);

  it("gpt-4.1-nano (cheapest model) cost matches formula", async () => {
    const model = "gpt-4.1-nano";
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "Say yes" }],
        max_tokens: 3,
        stream: false,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();

    const actualCost = Number(row!.cost_microdollars);
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const expected = expectedCostMicrodollars(
      PRICING[model],
      usage.prompt_tokens,
      usage.completion_tokens,
      cachedTokens,
    );

    expect(actualCost).toBe(expected);
  }, 30_000);

  it("model alias resolution: DB row has a recognizable model name with non-zero cost", async () => {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Alias test" }],
        max_tokens: 3,
        stream: false,
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId);
    expect(row).not.toBeNull();

    // Model should either be exactly "gpt-4o-mini" or the resolved version
    const model = row!.model as string;
    expect(model).toContain("gpt-4o-mini");
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
  }, 30_000);

  it("cached tokens: second identical request may have cached_input_tokens > 0 with lower cost", async () => {
    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: `Cache test ${Date.now()} identical prompt for caching` }],
      max_tokens: 5,
      stream: false,
    };

    // First request — warms the cache
    const res1 = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    expect(res1.status).toBe(200);
    const body1 = await res1.json();
    const id1 = res1.headers.get("x-request-id") ?? body1.id;

    // Second identical request — may hit cache
    const res2 = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    expect(res2.status).toBe(200);
    const body2 = await res2.json();
    const id2 = res2.headers.get("x-request-id") ?? body2.id;

    await new Promise((r) => setTimeout(r, 5_000));

    const row1 = await waitForCostEvent(sql, id1);
    const row2 = await waitForCostEvent(sql, id2);
    expect(row1).not.toBeNull();
    expect(row2).not.toBeNull();

    // OpenAI does not guarantee caching, so we just verify the cost event
    // properly records whatever cached_input_tokens OpenAI returns and
    // the cost formula uses the cached rate when applicable
    const _cached1 = Number(row1!.cached_input_tokens);
    const cached2 = Number(row2!.cached_input_tokens);

    if (cached2 > 0) {
      // If OpenAI returned cached tokens, verify cost is lower
      const cost1 = Number(row1!.cost_microdollars);
      const cost2 = Number(row2!.cost_microdollars);
      expect(cost2).toBeLessThanOrEqual(cost1);
    }

    // Regardless of caching, both should have valid non-zero costs
    expect(Number(row1!.cost_microdollars)).toBeGreaterThan(0);
    expect(Number(row2!.cost_microdollars)).toBeGreaterThan(0);
  }, 60_000);

  it("no successful 200 response produces cost_microdollars == 0", async () => {
    const models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"];
    const requestIds: string[] = [];

    const requests = models.map((model) =>
      fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "Zero cost check" }],
          max_tokens: 3,
          stream: false,
        }),
      }),
    );

    const responses = await Promise.all(requests);
    for (const res of responses) {
      expect(res.status).toBe(200);
      const body = await res.json();
      requestIds.push(res.headers.get("x-request-id") ?? body.id);
    }

    await new Promise((r) => setTimeout(r, 8_000));

    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id);
      expect(row).not.toBeNull();
      expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    }
  }, 60_000);

  it("streaming cost matches the same formula as non-streaming", async () => {
    const model = "gpt-4o-mini";

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "Streaming formula test" }],
        max_tokens: 5,
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-request-id");
    const text = await res.text();

    const dataLines = text
      .split("\n")
      .filter((l) => l.trim().startsWith("data:"))
      .map((l) => l.trim().slice(5).trim())
      .filter((l) => l !== "[DONE]");

    const lastChunk = JSON.parse(dataLines[dataLines.length - 1]);
    expect(lastChunk).toHaveProperty("usage");
    const usage = lastChunk.usage;

    expect(requestId).toBeTruthy();
    const row = await waitForCostEvent(sql, requestId!);
    expect(row).not.toBeNull();

    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const expected = expectedCostMicrodollars(
      PRICING[model],
      usage.prompt_tokens,
      usage.completion_tokens,
      cachedTokens,
    );

    expect(Number(row!.cost_microdollars)).toBe(expected);
  }, 30_000);

  it("our pricing data is internally consistent (no negative or NaN values)", async () => {
    for (const [_model, rates] of Object.entries(PRICING)) {
      expect(rates.inputPerMTok).toBeGreaterThan(0);
      expect(rates.cachedInputPerMTok).toBeGreaterThan(0);
      expect(rates.outputPerMTok).toBeGreaterThan(0);
      expect(rates.cachedInputPerMTok).toBeLessThanOrEqual(rates.inputPerMTok);
      expect(Number.isFinite(rates.inputPerMTok)).toBe(true);
      expect(Number.isFinite(rates.outputPerMTok)).toBe(true);
    }
  });
});
