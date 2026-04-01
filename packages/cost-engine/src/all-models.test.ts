import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent, isKnownModel } from "./pricing.js";

// ---------------------------------------------------------------------------
// Model catalog — every model in pricing-data.json with expected rates
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
  ["gpt-4o", { in: 2.5, cached: 1.25, out: 10.0 }],
  ["gpt-4o-mini", { in: 0.15, cached: 0.075, out: 0.6 }],
  ["gpt-4.1", { in: 2.0, cached: 0.5, out: 8.0 }],
  ["gpt-4.1-mini", { in: 0.4, cached: 0.1, out: 1.6 }],
  ["gpt-4.1-nano", { in: 0.1, cached: 0.025, out: 0.4 }],
  ["o4-mini", { in: 1.1, cached: 0.275, out: 4.4 }],
  ["o3", { in: 2.0, cached: 0.5, out: 8.0 }],
  ["o3-mini", { in: 1.1, cached: 0.55, out: 4.4 }],
  ["o1", { in: 15.0, cached: 7.5, out: 60.0 }],
  ["gpt-5", { in: 1.25, cached: 0.125, out: 10.0 }],
  ["gpt-5-mini", { in: 0.25, cached: 0.025, out: 2.0 }],
  ["gpt-5-nano", { in: 0.05, cached: 0.005, out: 0.4 }],
  ["gpt-5.1", { in: 1.25, cached: 0.125, out: 10.0 }],
  ["gpt-5.2", { in: 1.75, cached: 0.175, out: 14.0 }],
];

const anthropicModels: [string, AnthropicRates][] = [
  ["claude-sonnet-4-6", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-haiku-3.5", { in: 0.8, cached: 0.08, w5m: 1.0, w1h: 1.6, out: 4.0 }],
  ["claude-opus-4", { in: 15.0, cached: 1.5, w5m: 18.75, w1h: 30.0, out: 75.0 }],
  ["claude-opus-4-6", { in: 5.0, cached: 0.5, w5m: 6.25, w1h: 10.0, out: 25.0 }],
  ["claude-sonnet-4-5", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-opus-4-5", { in: 5.0, cached: 0.5, w5m: 6.25, w1h: 10.0, out: 25.0 }],
  ["claude-opus-4-1", { in: 15.0, cached: 1.5, w5m: 18.75, w1h: 30.0, out: 75.0 }],
  ["claude-sonnet-4", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-haiku-4-5", { in: 1.0, cached: 0.1, w5m: 1.25, w1h: 2.0, out: 5.0 }],
  ["claude-haiku-3", { in: 0.25, cached: 0.03, w5m: 0.3, w1h: 0.5, out: 1.25 }],
  ["claude-opus-4-6-20260205", { in: 5.0, cached: 0.5, w5m: 6.25, w1h: 10.0, out: 25.0 }],
  ["claude-sonnet-4-6-20260217", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-sonnet-4-5-20250929", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-opus-4-5-20251101", { in: 5.0, cached: 0.5, w5m: 6.25, w1h: 10.0, out: 25.0 }],
  ["claude-haiku-4-5-20251001", { in: 1.0, cached: 0.1, w5m: 1.25, w1h: 2.0, out: 5.0 }],
  ["claude-opus-4-1-20250805", { in: 15.0, cached: 1.5, w5m: 18.75, w1h: 30.0, out: 75.0 }],
  ["claude-opus-4-20250514", { in: 15.0, cached: 1.5, w5m: 18.75, w1h: 30.0, out: 75.0 }],
  ["claude-sonnet-4-20250514", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
  ["claude-3-5-haiku-20241022", { in: 0.8, cached: 0.08, w5m: 1.0, w1h: 1.6, out: 4.0 }],
  ["claude-3-haiku-20240307", { in: 0.25, cached: 0.03, w5m: 0.3, w1h: 0.5, out: 1.25 }],
  ["claude-opus-4-0", { in: 15.0, cached: 1.5, w5m: 18.75, w1h: 30.0, out: 75.0 }],
  ["claude-sonnet-4-0", { in: 3.0, cached: 0.3, w5m: 3.75, w1h: 6.0, out: 15.0 }],
];

const googleModels: [string, GoogleRates][] = [
  ["gemini-2.5-pro", { in: 1.25, cached: 0.3125, out: 10.0 }],
  ["gemini-2.5-flash", { in: 0.15, cached: 0.0375, out: 0.6 }],
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
  it("recognises all 38 models via isKnownModel", () => {
    for (const [provider, model] of allModels) {
      expect(
        isKnownModel(provider, model),
        `expected isKnownModel("${provider}", "${model}") to be true`,
      ).toBe(true);
    }
    expect(allModels).toHaveLength(38);
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
      const expected = Math.round(10_000 * rates.in + 2_000 * rates.out);

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
    const expected = Math.round(5_000 * 1.25 + 2_000 * 0.3125 + 1_000 * 10.0);

    expect(total).toBe(expected);
  });

  it("gemini-2.5-flash: 50K input, 10K output", () => {
    const pricing = getModelPricing("google", "gemini-2.5-flash")!;
    expect(pricing).not.toBeNull();

    const inputCost = costComponent(50_000, pricing.inputPerMTok);
    const outputCost = costComponent(10_000, pricing.outputPerMTok);
    const total = Math.round(inputCost + outputCost);
    const expected = Math.round(50_000 * 0.15 + 10_000 * 0.6);

    expect(total).toBe(expected);
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
