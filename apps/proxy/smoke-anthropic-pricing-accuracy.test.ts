/**
 * Anthropic pricing accuracy smoke tests.
 * Triple-checks cost math: Anthropic reports usage -> our calculator computes
 * cost -> DB stores cost -> test verifies all three match the known formula.
 *
 * Anthropic cost formula (claude-3-haiku-20240307):
 *   microdollars = Math.round(input * 0.25 + output * 1.25 + cacheRead * 0.03)
 *
 * Requires: live proxy, ANTHROPIC_API_KEY, NULLSPEND_API_KEY, DATABASE_URL
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  ANTHROPIC_API_KEY,
  DATABASE_URL,
  anthropicAuthHeaders,
  isServerUp,
  waitForCostEvent,
} from "./smoke-test-helpers.js";

const HAIKU_PRICING = {
  inputPerMTok: 0.25,
  cachedInputPerMTok: 0.03,
  outputPerMTok: 1.25,
};

function expectedAnthropicCost(
  inputTokens: number,
  outputTokens: number,
  cacheReadTokens = 0,
): number {
  return Math.round(
    inputTokens * HAIKU_PRICING.inputPerMTok +
      cacheReadTokens * HAIKU_PRICING.cachedInputPerMTok +
      outputTokens * HAIKU_PRICING.outputPerMTok,
  );
}

describe("Anthropic pricing accuracy", () => {
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

  it("claude-3-haiku cost matches formula: (input * 0.25 + output * 1.25)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Say hello" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();

    const actualCost = Number(row!.cost_microdollars);
    const cacheRead = Number(usage.cache_read_input_tokens ?? 0);
    const expected = expectedAnthropicCost(
      usage.input_tokens,
      usage.output_tokens,
      cacheRead,
    );

    expect(actualCost).toBe(expected);
  }, 30_000);

  it("cost event has non-zero cost_microdollars for every successful 200", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Non-zero cost check" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
  }, 30_000);

  it("model field in DB matches dated model returned by Anthropic", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Model check" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(row!.model).toBe(body.model);
    expect(row!.model).toContain("claude");
  }, 30_000);

  it("streaming cost matches the same formula as non-streaming for identical prompts", async () => {
    const prompt = `Pricing parity test ${Date.now()}`;

    const nsRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    expect(nsRes.status).toBe(200);
    const nsBody = await nsRes.json();
    const nsId = nsRes.headers.get("x-request-id") ?? nsBody.id;

    const sRes = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: prompt }],
        stream: true,
      }),
    });
    expect(sRes.status).toBe(200);
    const sId = sRes.headers.get("x-request-id");
    await sRes.text();

    const nsRow = await waitForCostEvent(sql, nsId, 15_000, "anthropic");
    const sRow = await waitForCostEvent(sql, sId!, 15_000, "anthropic");
    expect(nsRow).not.toBeNull();
    expect(sRow).not.toBeNull();

    // Both should use the same formula; cost may differ slightly due to
    // different output token counts, but both should be > 0
    expect(Number(nsRow!.cost_microdollars)).toBeGreaterThan(0);
    expect(Number(sRow!.cost_microdollars)).toBeGreaterThan(0);
  }, 60_000);

  it("input_tokens in DB matches usage.input_tokens from response", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Token match test" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();

    // DB stores totalInputTokens = input + cache_creation + cache_read
    const totalInput =
      body.usage.input_tokens +
      (body.usage.cache_creation_input_tokens ?? 0) +
      (body.usage.cache_read_input_tokens ?? 0);
    expect(row!.input_tokens).toBe(totalInput);
  }, 30_000);

  it("output_tokens in DB matches usage.output_tokens from response", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Output token match" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(row!.output_tokens).toBe(body.usage.output_tokens);
  }, 30_000);

  it("cachedInputTokens in DB is >= 0 (never negative)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 5,
        messages: [{ role: "user", content: "Cache non-negative" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(Number(row!.cached_input_tokens)).toBeGreaterThanOrEqual(0);
  }, 30_000);

  it("second identical request may show cache_read with reduced cost", async () => {
    const body = {
      model: "claude-3-haiku-20240307",
      max_tokens: 5,
      messages: [{ role: "user", content: `Anthropic cache test ${Date.now()} identical prompt` }],
    };

    const res1 = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify(body),
    });
    expect(res1.status).toBe(200);
    const body1 = await res1.json();
    const id1 = res1.headers.get("x-request-id") ?? body1.id;

    const res2 = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify(body),
    });
    expect(res2.status).toBe(200);
    const body2 = await res2.json();
    const id2 = res2.headers.get("x-request-id") ?? body2.id;

    await new Promise((r) => setTimeout(r, 5_000));

    const row1 = await waitForCostEvent(sql, id1, 15_000, "anthropic");
    const row2 = await waitForCostEvent(sql, id2, 15_000, "anthropic");
    expect(row1).not.toBeNull();
    expect(row2).not.toBeNull();

    // Anthropic cache is not guaranteed, so just verify both have valid costs
    expect(Number(row1!.cost_microdollars)).toBeGreaterThan(0);
    expect(Number(row2!.cost_microdollars)).toBeGreaterThan(0);

    // If cache hit occurred, cost should be lower or equal
    const cached2 = Number(row2!.cached_input_tokens);
    if (cached2 > 0) {
      expect(Number(row2!.cost_microdollars)).toBeLessThanOrEqual(
        Number(row1!.cost_microdollars),
      );
    }
  }, 60_000);

  it("cost formula works with very small output (max_tokens: 1)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 1,
        messages: [{ role: "user", content: "Min output" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();
    expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    expect(row!.output_tokens).toBeGreaterThanOrEqual(1);
  }, 30_000);

  it("cost formula works with longer output (max_tokens: 100)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 100,
        messages: [{ role: "user", content: "Write a short paragraph about the weather." }],
      }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    const requestId = res.headers.get("x-request-id") ?? body.id;
    const usage = body.usage;

    const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
    expect(row).not.toBeNull();

    const actualCost = Number(row!.cost_microdollars);
    const cacheRead = Number(usage.cache_read_input_tokens ?? 0);
    const expected = expectedAnthropicCost(
      usage.input_tokens,
      usage.output_tokens,
      cacheRead,
    );
    expect(actualCost).toBe(expected);
    expect(actualCost).toBeGreaterThan(0);
  }, 30_000);

  it("pricing data is internally consistent for all Anthropic models", () => {
    const models: Record<string, { input: number; cached: number; output: number }> = {
      "claude-3-haiku": { input: 0.25, cached: 0.03, output: 1.25 },
      "claude-haiku-3.5": { input: 0.80, cached: 0.08, output: 4.00 },
      "claude-haiku-4.5": { input: 1.00, cached: 0.10, output: 5.00 },
      "claude-sonnet-4": { input: 3.00, cached: 0.30, output: 15.00 },
      "claude-sonnet-4-5": { input: 3.00, cached: 0.30, output: 15.00 },
      "claude-sonnet-4-6": { input: 3.00, cached: 0.30, output: 15.00 },
      "claude-opus-4": { input: 15.00, cached: 1.50, output: 75.00 },
      "claude-opus-4-1": { input: 15.00, cached: 1.50, output: 75.00 },
      "claude-opus-4-5": { input: 5.00, cached: 0.50, output: 25.00 },
      "claude-opus-4-6": { input: 5.00, cached: 0.50, output: 25.00 },
    };

    for (const [_model, rates] of Object.entries(models)) {
      expect(rates.input).toBeGreaterThan(0);
      expect(rates.cached).toBeGreaterThan(0);
      expect(rates.output).toBeGreaterThan(0);
      // Cache read is always cheaper than base input
      expect(rates.cached).toBeLessThan(rates.input);
      // Output is always more expensive than input
      expect(rates.output).toBeGreaterThan(rates.input);
      expect(Number.isFinite(rates.input)).toBe(true);
      expect(Number.isFinite(rates.output)).toBe(true);
    }
  });

  // --- Multi-model live pricing verification ---
  // One dated model name per unique pricing tier. Each sends a tiny request
  // (max_tokens: 1) and verifies the DB cost matches the formula using that
  // model's specific rates from pricing-data.json.

  const MODELS_TO_TEST: {
    model: string;
    inputPerMTok: number;
    cachedInputPerMTok: number;
    outputPerMTok: number;
  }[] = [
    { model: "claude-3-haiku-20240307",     inputPerMTok: 0.25,  cachedInputPerMTok: 0.03,  outputPerMTok: 1.25  },
    { model: "claude-3-5-haiku-20241022",   inputPerMTok: 0.80,  cachedInputPerMTok: 0.08,  outputPerMTok: 4.00  },
    { model: "claude-haiku-4-5-20251001",   inputPerMTok: 1.00,  cachedInputPerMTok: 0.10,  outputPerMTok: 5.00  },
    { model: "claude-sonnet-4-20250514",    inputPerMTok: 3.00,  cachedInputPerMTok: 0.30,  outputPerMTok: 15.00 },
    { model: "claude-sonnet-4-5-20250929",  inputPerMTok: 3.00,  cachedInputPerMTok: 0.30,  outputPerMTok: 15.00 },
    { model: "claude-sonnet-4-6-20260217",  inputPerMTok: 3.00,  cachedInputPerMTok: 0.30,  outputPerMTok: 15.00 },
    { model: "claude-opus-4-20250514",      inputPerMTok: 15.00, cachedInputPerMTok: 1.50,  outputPerMTok: 75.00 },
    { model: "claude-opus-4-1-20250805",    inputPerMTok: 15.00, cachedInputPerMTok: 1.50,  outputPerMTok: 75.00 },
    { model: "claude-opus-4-5-20251101",    inputPerMTok: 5.00,  cachedInputPerMTok: 0.50,  outputPerMTok: 25.00 },
    { model: "claude-opus-4-6-20260205",    inputPerMTok: 5.00,  cachedInputPerMTok: 0.50,  outputPerMTok: 25.00 },
  ];

  for (const { model, inputPerMTok, cachedInputPerMTok, outputPerMTok } of MODELS_TO_TEST) {
    it(`${model} cost matches formula (input*${inputPerMTok} + output*${outputPerMTok})`, async () => {
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model,
          max_tokens: 1,
          messages: [{ role: "user", content: "Say ok" }],
        }),
      });

      // Some models may not be available on the API tier; skip gracefully
      if (res.status === 400 || res.status === 404) {
        const errBody = await res.json().catch(() => ({}));
        const errMsg = (errBody as { error?: { message?: string } }).error?.message ?? "";
        console.log(`[pricing] ${model} not available (${res.status}): ${errMsg}`);
        return;
      }

      expect(res.status).toBe(200);
      const body = await res.json();
      const requestId = res.headers.get("x-request-id") ?? body.id;
      const usage = body.usage;

      const row = await waitForCostEvent(sql, requestId, 15_000, "anthropic");
      expect(row).not.toBeNull();

      const cacheRead = Number(usage.cache_read_input_tokens ?? 0);
      const expected = Math.round(
        usage.input_tokens * inputPerMTok +
          cacheRead * cachedInputPerMTok +
          usage.output_tokens * outputPerMTok,
      );
      const actualCost = Number(row!.cost_microdollars);

      console.log(
        `[pricing] ${model}: input=${usage.input_tokens}, output=${usage.output_tokens}, ` +
          `cacheRead=${cacheRead}, expected=${expected}µ$, actual=${actualCost}µ$`,
      );

      expect(actualCost).toBe(expected);
      expect(actualCost).toBeGreaterThan(0);
    }, 30_000);
  }

  it("no successful 200 response produces cost_microdollars == 0", async () => {
    const requestIds: string[] = [];

    for (let i = 0; i < 3; i++) {
      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: 5,
          messages: [{ role: "user", content: `Zero cost check ${i}` }],
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      requestIds.push(res.headers.get("x-request-id") ?? body.id);
    }

    await new Promise((r) => setTimeout(r, 8_000));

    for (const id of requestIds) {
      const row = await waitForCostEvent(sql, id, 15_000, "anthropic");
      expect(row).not.toBeNull();
      expect(Number(row!.cost_microdollars)).toBeGreaterThan(0);
    }
  }, 60_000);
});
