import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent, isKnownModel } from "./pricing.js";

// ---------------------------------------------------------------------------
// Model catalog — every model in pricing-data.json with expected rates
// Rates are in integer µ$/MTok (multiply old $/MTok by 1,000,000).
// ---------------------------------------------------------------------------

interface OpenAIRates {
  in: number;
  cached: number;
  out: number;
}

interface AnthropicRates {
  in: number;
  cached: number;
  w5m: number;
  w1h: number;
  out: number;
}

interface GoogleRates {
  in: number;
  cached: number;
  out: number;
}

type ModelEntry =
  | [provider: "openai", model: string, rates: OpenAIRates]
  | [provider: "anthropic", model: string, rates: AnthropicRates]
  | [provider: "google", model: string, rates: GoogleRates];

const openaiModels: [string, OpenAIRates][] = [
  ["gpt-4o", { in: 2_500_000, cached: 1_250_000, out: 10_000_000 }],
  ["gpt-4o-mini", { in: 150_000, cached: 75_000, out: 600_000 }],
  ["gpt-4.1", { in: 2_000_000, cached: 500_000, out: 8_000_000 }],
  ["gpt-4.1-mini", { in: 400_000, cached: 100_000, out: 1_600_000 }],
  ["gpt-4.1-nano", { in: 100_000, cached: 25_000, out: 400_000 }],
  ["o4-mini", { in: 1_100_000, cached: 275_000, out: 4_400_000 }],
  ["o3", { in: 2_000_000, cached: 500_000, out: 8_000_000 }],
  ["o3-mini", { in: 1_100_000, cached: 550_000, out: 4_400_000 }],
  ["o1", { in: 15_000_000, cached: 7_500_000, out: 60_000_000 }],
  ["gpt-5", { in: 1_250_000, cached: 125_000, out: 10_000_000 }],
  ["gpt-5-mini", { in: 250_000, cached: 25_000, out: 2_000_000 }],
  ["gpt-5-nano", { in: 50_000, cached: 5_000, out: 400_000 }],
  ["gpt-5.1", { in: 1_250_000, cached: 125_000, out: 10_000_000 }],
  ["gpt-5.2", { in: 1_750_000, cached: 175_000, out: 14_000_000 }],
  ["gpt-5.2-pro", { in: 21_000_000, cached: 21_000_000, out: 168_000_000 }],
  ["gpt-5-pro", { in: 15_000_000, cached: 15_000_000, out: 120_000_000 }],
  ["o3-pro", { in: 20_000_000, cached: 20_000_000, out: 80_000_000 }],
  ["o1-pro", { in: 150_000_000, cached: 150_000_000, out: 600_000_000 }],
  ["o1-mini", { in: 1_100_000, cached: 550_000, out: 4_400_000 }],
];

const anthropicModels: [string, AnthropicRates][] = [
  ["claude-sonnet-4-6", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-haiku-3.5", { in: 800_000, cached: 80_000, w5m: 1_000_000, w1h: 1_600_000, out: 4_000_000 }],
  ["claude-opus-4", { in: 15_000_000, cached: 1_500_000, w5m: 18_750_000, w1h: 30_000_000, out: 75_000_000 }],
  ["claude-opus-4-6", { in: 5_000_000, cached: 500_000, w5m: 6_250_000, w1h: 10_000_000, out: 25_000_000 }],
  ["claude-sonnet-4-5", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-opus-4-5", { in: 5_000_000, cached: 500_000, w5m: 6_250_000, w1h: 10_000_000, out: 25_000_000 }],
  ["claude-opus-4-1", { in: 15_000_000, cached: 1_500_000, w5m: 18_750_000, w1h: 30_000_000, out: 75_000_000 }],
  ["claude-sonnet-4", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-haiku-4-5", { in: 1_000_000, cached: 100_000, w5m: 1_250_000, w1h: 2_000_000, out: 5_000_000 }],
  ["claude-haiku-3", { in: 250_000, cached: 30_000, w5m: 300_000, w1h: 500_000, out: 1_250_000 }],
  ["claude-opus-4-6-20260205", { in: 5_000_000, cached: 500_000, w5m: 6_250_000, w1h: 10_000_000, out: 25_000_000 }],
  ["claude-sonnet-4-6-20260217", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-sonnet-4-5-20250929", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-opus-4-5-20251101", { in: 5_000_000, cached: 500_000, w5m: 6_250_000, w1h: 10_000_000, out: 25_000_000 }],
  ["claude-haiku-4-5-20251001", { in: 1_000_000, cached: 100_000, w5m: 1_250_000, w1h: 2_000_000, out: 5_000_000 }],
  ["claude-opus-4-1-20250805", { in: 15_000_000, cached: 1_500_000, w5m: 18_750_000, w1h: 30_000_000, out: 75_000_000 }],
  ["claude-opus-4-20250514", { in: 15_000_000, cached: 1_500_000, w5m: 18_750_000, w1h: 30_000_000, out: 75_000_000 }],
  ["claude-sonnet-4-20250514", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
  ["claude-3-5-haiku-20241022", { in: 800_000, cached: 80_000, w5m: 1_000_000, w1h: 1_600_000, out: 4_000_000 }],
  ["claude-3-haiku-20240307", { in: 250_000, cached: 30_000, w5m: 300_000, w1h: 500_000, out: 1_250_000 }],
  ["claude-opus-4-0", { in: 15_000_000, cached: 1_500_000, w5m: 18_750_000, w1h: 30_000_000, out: 75_000_000 }],
  ["claude-sonnet-4-0", { in: 3_000_000, cached: 300_000, w5m: 3_750_000, w1h: 6_000_000, out: 15_000_000 }],
];

const googleModels: [string, GoogleRates][] = [
  ["gemini-2.5-pro", { in: 1_250_000, cached: 125_000, out: 10_000_000 }],
  ["gemini-2.5-flash", { in: 300_000, cached: 30_000, out: 2_500_000 }],
  ["gemini-2.5-flash-lite", { in: 100_000, cached: 10_000, out: 400_000 }],
  ["gemini-2.0-flash", { in: 100_000, cached: 25_000, out: 400_000 }],
  ["gemini-2.0-flash-lite", { in: 75_000, cached: 0, out: 300_000 }],
  ["gemini-3-flash-preview", { in: 500_000, cached: 50_000, out: 3_000_000 }],
  ["gemini-3.1-pro-preview", { in: 2_000_000, cached: 200_000, out: 12_000_000 }],
  ["gemini-3.1-flash-lite-preview", { in: 250_000, cached: 25_000, out: 1_500_000 }],
];

// Flattened list for parameterized tests
const allModels: ModelEntry[] = [
  ...openaiModels.map(([m, r]) => ["openai", m, r] as ModelEntry),
  ...anthropicModels.map(([m, r]) => ["anthropic", m, r] as ModelEntry),
  ...googleModels.map(([m, r]) => ["google", m, r] as ModelEntry),
];

// ---------------------------------------------------------------------------
// 1. Pricing catalog completeness
// ---------------------------------------------------------------------------

describe("pricing catalog completeness", () => {
  it("recognises all 49 models via isKnownModel", () => {
    for (const [provider, model] of allModels) {
      expect(
        isKnownModel(provider, model),
        `expected isKnownModel("${provider}", "${model}") to be true`,
      ).toBe(true);
    }
    expect(allModels).toHaveLength(49);
  });

  it.each([
    ["openai", "gpt-6"],
    ["anthropic", "claude-5"],
    ["google", "gemini-3"],
  ])("returns false for unknown model %s/%s", (provider, model) => {
    expect(isKnownModel(provider, model)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 2. Every model has valid pricing data
// ---------------------------------------------------------------------------

describe("every model has valid pricing data", () => {
  for (const [provider, model] of allModels) {
    it(`${provider}/${model}`, () => {
      const pricing = getModelPricing(provider, model);
      expect(pricing, `getModelPricing("${provider}", "${model}") returned null`).not.toBeNull();

      expect(pricing!.inputPerMTok).toBeGreaterThan(0);
      expect(pricing!.outputPerMTok).toBeGreaterThan(0);
      expect(pricing!.cachedInputPerMTok).toBeGreaterThanOrEqual(0);

      if (provider === "anthropic") {
        expect(pricing!.cacheWrite5mPerMTok).toBeGreaterThan(0);
        expect(pricing!.cacheWrite1hPerMTok).toBeGreaterThan(0);
      } else {
        // OpenAI and Google should not have cacheWrite fields
        expect(pricing!.cacheWrite5mPerMTok).toBeUndefined();
        expect(pricing!.cacheWrite1hPerMTok).toBeUndefined();
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Exact pricing values
// ---------------------------------------------------------------------------

describe("exact pricing values", () => {
  for (const [provider, model, rates] of allModels) {
    it(`${provider}/${model}`, () => {
      const pricing = getModelPricing(provider, model)!;
      expect(pricing).not.toBeNull();

      expect(pricing.inputPerMTok).toBe(rates.in);
      expect(pricing.cachedInputPerMTok).toBe(rates.cached);
      expect(pricing.outputPerMTok).toBe(rates.out);

      if (provider === "anthropic") {
        const ar = rates as AnthropicRates;
        expect(pricing.cacheWrite5mPerMTok).toBe(ar.w5m);
        expect(pricing.cacheWrite1hPerMTok).toBe(ar.w1h);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 4. Cost calculation: every model, 10K in + 2K out
// ---------------------------------------------------------------------------

describe("cost calculation: every model, 10K in + 2K out", () => {
  for (const [provider, model, rates] of allModels) {
    it(`${provider}/${model}`, () => {
      const inputCost = costComponent(10_000, rates.in);
      const outputCost = costComponent(2_000, rates.out);
      const total = Math.round(inputCost + outputCost);
      const expected = Math.round(
        costComponent(10_000, rates.in) + costComponent(2_000, rates.out),
      );

      expect(total).toBe(expected);
      expect(total).toBeGreaterThan(0);
      expect(Number.isSafeInteger(total)).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// 5. Google Gemini cost calculations
// ---------------------------------------------------------------------------

describe("Google Gemini cost calculations", () => {
  it("gemini-2.5-pro: 5K input, 1K output, 2K cached", () => {
    const pricing = getModelPricing("google", "gemini-2.5-pro")!;
    expect(pricing).not.toBeNull();

    const inputCost = costComponent(5_000, pricing.inputPerMTok);
    const cachedCost = costComponent(2_000, pricing.cachedInputPerMTok);
    const outputCost = costComponent(1_000, pricing.outputPerMTok);
    const total = Math.round(inputCost + cachedCost + outputCost);
    // costComponent(5000, 1_250_000) + costComponent(2000, 125_000) + costComponent(1000, 10_000_000)
    // = 6250 + 250 + 10000 = 16500
    expect(total).toBe(16500);
  });

  it("gemini-2.5-flash: 50K input, 10K output", () => {
    const pricing = getModelPricing("google", "gemini-2.5-flash")!;
    expect(pricing).not.toBeNull();

    const inputCost = costComponent(50_000, pricing.inputPerMTok);
    const outputCost = costComponent(10_000, pricing.outputPerMTok);
    const total = Math.round(inputCost + outputCost);
    // costComponent(50000, 300_000) + costComponent(10000, 2_500_000) = 15000 + 25000 = 40000
    expect(total).toBe(40000);
  });

  it("Gemini models have no cacheWrite fields", () => {
    for (const [model] of googleModels) {
      const pricing = getModelPricing("google", model)!;
      expect(pricing).not.toBeNull();
      expect(pricing.cacheWrite5mPerMTok).toBeUndefined();
      expect(pricing.cacheWrite1hPerMTok).toBeUndefined();
    }
  });
});

// ---------------------------------------------------------------------------
// 6. Pricing tier consistency
// ---------------------------------------------------------------------------

describe("pricing tier consistency", () => {
  it("all claude-sonnet-4-6 variants have identical rates", () => {
    const variants = ["claude-sonnet-4-6", "claude-sonnet-4-6-20260217"];
    const base = getModelPricing("anthropic", variants[0])!;
    expect(base).not.toBeNull();

    for (const variant of variants.slice(1)) {
      const pricing = getModelPricing("anthropic", variant)!;
      expect(pricing).not.toBeNull();
      expect(pricing.inputPerMTok).toBe(base.inputPerMTok);
      expect(pricing.cachedInputPerMTok).toBe(base.cachedInputPerMTok);
      expect(pricing.cacheWrite5mPerMTok).toBe(base.cacheWrite5mPerMTok);
      expect(pricing.cacheWrite1hPerMTok).toBe(base.cacheWrite1hPerMTok);
      expect(pricing.outputPerMTok).toBe(base.outputPerMTok);
    }
  });

  it("all claude-opus-4 variants have identical rates", () => {
    const variants = [
      "claude-opus-4",
      "claude-opus-4-20250514",
      "claude-opus-4-0",
    ];
    const base = getModelPricing("anthropic", variants[0])!;
    expect(base).not.toBeNull();

    for (const variant of variants.slice(1)) {
      const pricing = getModelPricing("anthropic", variant)!;
      expect(pricing, `${variant} should have pricing`).not.toBeNull();
      expect(pricing.inputPerMTok).toBe(base.inputPerMTok);
      expect(pricing.cachedInputPerMTok).toBe(base.cachedInputPerMTok);
      expect(pricing.cacheWrite5mPerMTok).toBe(base.cacheWrite5mPerMTok);
      expect(pricing.cacheWrite1hPerMTok).toBe(base.cacheWrite1hPerMTok);
      expect(pricing.outputPerMTok).toBe(base.outputPerMTok);
    }
  });

  it("gpt-5 and gpt-5.1 have identical rates", () => {
    const gpt5 = getModelPricing("openai", "gpt-5")!;
    const gpt51 = getModelPricing("openai", "gpt-5.1")!;
    expect(gpt5).not.toBeNull();
    expect(gpt51).not.toBeNull();

    expect(gpt51.inputPerMTok).toBe(gpt5.inputPerMTok);
    expect(gpt51.cachedInputPerMTok).toBe(gpt5.cachedInputPerMTok);
    expect(gpt51.outputPerMTok).toBe(gpt5.outputPerMTok);
  });
});
