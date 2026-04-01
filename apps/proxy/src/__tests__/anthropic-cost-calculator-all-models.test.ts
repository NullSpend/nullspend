import { describe, it, expect } from "vitest";
import { calculateAnthropicCost } from "../lib/anthropic-cost-calculator.js";

// ---------------------------------------------------------------------------
// Pricing rates per million tokens (microdollars) for all Anthropic models.
// The calculator uses costComponent(tokens, ratePerMTok) = tokens * rate
// when tokens > 0 and rate > 0, else 0.
// Final cost = Math.round(sum of components).
// ---------------------------------------------------------------------------

interface ModelRates {
  in: number;
  cached: number;
  w5m: number;
  w1h: number;
  out: number;
}

const SHORT_NAME_RATES: Record<string, ModelRates> = {
  "claude-sonnet-4-6":  { in: 3.00, cached: 0.30, w5m: 3.75, w1h: 6.00, out: 15.00 },
  "claude-haiku-3.5":   { in: 0.80, cached: 0.08, w5m: 1.00, w1h: 1.60, out: 4.00 },
  "claude-opus-4":      { in: 15.00, cached: 1.50, w5m: 18.75, w1h: 30.00, out: 75.00 },
  "claude-opus-4-6":    { in: 5.00, cached: 0.50, w5m: 6.25, w1h: 10.00, out: 25.00 },
  "claude-sonnet-4-5":  { in: 3.00, cached: 0.30, w5m: 3.75, w1h: 6.00, out: 15.00 },
  "claude-opus-4-5":    { in: 5.00, cached: 0.50, w5m: 6.25, w1h: 10.00, out: 25.00 },
  "claude-opus-4-1":    { in: 15.00, cached: 1.50, w5m: 18.75, w1h: 30.00, out: 75.00 },
  "claude-sonnet-4":    { in: 3.00, cached: 0.30, w5m: 3.75, w1h: 6.00, out: 15.00 },
  "claude-haiku-4-5":   { in: 1.00, cached: 0.10, w5m: 1.25, w1h: 2.00, out: 5.00 },
  "claude-haiku-3":     { in: 0.25, cached: 0.03, w5m: 0.30, w1h: 0.50, out: 1.25 },
};

// Dated/versioned model → alias it maps to (same rates)
const DATED_TO_ALIAS: Record<string, string> = {
  "claude-opus-4-6-20260205":    "claude-opus-4-6",
  "claude-sonnet-4-6-20260217":  "claude-sonnet-4-6",
  "claude-sonnet-4-5-20250929":  "claude-sonnet-4-5",
  "claude-opus-4-5-20251101":    "claude-opus-4-5",
  "claude-haiku-4-5-20251001":   "claude-haiku-4-5",
  "claude-opus-4-1-20250805":    "claude-opus-4-1",
  "claude-opus-4-20250514":      "claude-opus-4",
  "claude-sonnet-4-20250514":    "claude-sonnet-4",
  "claude-3-5-haiku-20241022":   "claude-haiku-3.5",
  "claude-3-haiku-20240307":     "claude-haiku-3",
  "claude-opus-4-0":             "claude-opus-4",
  "claude-sonnet-4-0":           "claude-sonnet-4",
};

// Build a combined map: all 22 models → rates
const ALL_MODEL_RATES: Record<string, ModelRates> = { ...SHORT_NAME_RATES };
for (const [dated, alias] of Object.entries(DATED_TO_ALIAS)) {
  ALL_MODEL_RATES[dated] = SHORT_NAME_RATES[alias];
}

// ---------------------------------------------------------------------------
// 1. Every Anthropic model: basic cost (no cache)
//    1000 input, 500 output, no cache tokens
//    Expected: Math.round(1000 * in + 500 * out)
// ---------------------------------------------------------------------------
describe("every Anthropic model: basic cost (no cache)", () => {
  for (const [model, rates] of Object.entries(ALL_MODEL_RATES)) {
    it(`${model}: 1000 in + 500 out`, () => {
      const result = calculateAnthropicCost(
        model,
        null,
        { input_tokens: 1000, output_tokens: 500 },
        null,
        `req-basic-${model}`,
        100,
      );

      // cost = Math.round(1000 * inputRate + 500 * outputRate)
      const expected = Math.round(1000 * rates.in + 500 * rates.out);
      expect(result.costMicrodollars).toBe(expected);
      expect(result.costMicrodollars).toBeGreaterThan(0);
      expect(result.inputTokens).toBe(1000);
      expect(result.outputTokens).toBe(500);
      expect(result.cachedInputTokens).toBe(0);
      expect(result.provider).toBe("anthropic");
    });
  }
});

// ---------------------------------------------------------------------------
// 2. Every Anthropic model: cached input cost
//    500 input, 200 output, 5000 cache_read
//    Expected: Math.round(500 * in + 5000 * cached + 200 * out)
// ---------------------------------------------------------------------------
describe("every Anthropic model: cached input cost", () => {
  for (const [model, rates] of Object.entries(SHORT_NAME_RATES)) {
    it(`${model}: 500 in + 200 out + 5000 cache_read`, () => {
      const result = calculateAnthropicCost(
        model,
        null,
        {
          input_tokens: 500,
          output_tokens: 200,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 5000,
        },
        null,
        `req-cached-${model}`,
        100,
      );

      // cost = Math.round(500 * in + 5000 * cached + 200 * out)
      const expected = Math.round(500 * rates.in + 5000 * rates.cached + 200 * rates.out);
      expect(result.costMicrodollars).toBe(expected);
      expect(result.inputTokens).toBe(500 + 5000); // totalInputTokens
      expect(result.cachedInputTokens).toBe(5000);
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Every Anthropic model: cache write cost (5m TTL)
//    100 input, 200 output, 3000 cache_creation, no cacheCreationDetail
//    Expected: Math.round(100 * in + 3000 * w5m + 200 * out)
// ---------------------------------------------------------------------------
describe("every Anthropic model: cache write cost (5m TTL)", () => {
  for (const [model, rates] of Object.entries(SHORT_NAME_RATES)) {
    it(`${model}: 100 in + 200 out + 3000 cache_write (5m)`, () => {
      const result = calculateAnthropicCost(
        model,
        null,
        {
          input_tokens: 100,
          output_tokens: 200,
          cache_creation_input_tokens: 3000,
          cache_read_input_tokens: 0,
        },
        null, // no detail → all cache_creation at 5m rate
        `req-write5m-${model}`,
        100,
      );

      // cost = Math.round(100 * in + 3000 * w5m + 200 * out)
      const expected = Math.round(100 * rates.in + 3000 * rates.w5m + 200 * rates.out);
      expect(result.costMicrodollars).toBe(expected);
      expect(result.inputTokens).toBe(100 + 3000); // totalInputTokens
    });
  }
});

// ---------------------------------------------------------------------------
// 4. Price tier groups verify identical rates
//    Each dated/versioned model must produce the exact same cost as its alias
//    for identical usage.
// ---------------------------------------------------------------------------
describe("price tier groups verify identical rates", () => {
  const testUsage = {
    input_tokens: 750,
    output_tokens: 400,
    cache_creation_input_tokens: 2000,
    cache_read_input_tokens: 8000,
  };

  for (const [dated, alias] of Object.entries(DATED_TO_ALIAS)) {
    it(`${dated} produces same cost as ${alias}`, () => {
      const datedResult = calculateAnthropicCost(
        dated,
        null,
        testUsage,
        null,
        `req-dated-${dated}`,
        100,
      );

      const aliasResult = calculateAnthropicCost(
        alias,
        null,
        testUsage,
        null,
        `req-alias-${alias}`,
        100,
      );

      // Both must produce the same non-zero cost
      expect(datedResult.costMicrodollars).toBe(aliasResult.costMicrodollars);
      expect(datedResult.costMicrodollars).toBeGreaterThan(0);

      // Verify all token fields match
      expect(datedResult.inputTokens).toBe(aliasResult.inputTokens);
      expect(datedResult.outputTokens).toBe(aliasResult.outputTokens);
      expect(datedResult.cachedInputTokens).toBe(aliasResult.cachedInputTokens);

      // Verify against manual calculation using the alias rates
      const rates = SHORT_NAME_RATES[alias];
      // cost = Math.round(750 * in + 2000 * w5m + 8000 * cached + 400 * out)
      const expected = Math.round(
        750 * rates.in + 2000 * rates.w5m + 8000 * rates.cached + 400 * rates.out,
      );
      expect(datedResult.costMicrodollars).toBe(expected);
    });
  }
});

// ---------------------------------------------------------------------------
// 5. Negative token edge cases
//    costComponent returns 0 when tokens <= 0, so negative values are safe.
// ---------------------------------------------------------------------------
describe("negative token edge cases", () => {
  it("negative input_tokens → costComponent returns 0 for the negative component", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: -500, output_tokens: 200 },
      null,
      "req-neg-input",
      100,
    );

    // Number(-500) || 0 → -500 (truthy), so inputTokens = -500
    // costComponent(-500, 3.00) → 0 (tokens <= 0)
    // cost = Math.round(0 + 200 * 15.00) = 3000
    expect(result.costMicrodollars).toBe(3000);
  });

  it("negative cache_creation_input_tokens → treated as 0 cost component", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 100,
        output_tokens: 200,
        cache_creation_input_tokens: -1000,
        cache_read_input_tokens: 0,
      },
      null,
      "req-neg-cache-create",
      100,
    );

    // Number(-1000) || 0 → -1000 (truthy), so cacheCreationTokens = -1000
    // costComponent(-1000, 3.75) → 0 (tokens <= 0)
    // cost = Math.round(100 * 3.00 + 0 + 0 + 200 * 15.00) = Math.round(300 + 3000) = 3300
    expect(result.costMicrodollars).toBe(3300);
  });

  it("negative cache_read_input_tokens → treated as 0 cost component", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 100,
        output_tokens: 200,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: -5000,
      },
      null,
      "req-neg-cache-read",
      100,
    );

    // Number(-5000) || 0 → -5000 (truthy), so cacheReadTokens = -5000
    // costComponent(-5000, 0.30) → 0 (tokens <= 0)
    // cost = Math.round(100 * 3.00 + 0 + 0 + 200 * 15.00) = Math.round(300 + 3000) = 3300
    expect(result.costMicrodollars).toBe(3300);
  });
});
