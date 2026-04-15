import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent, RATE_SCALE } from "./pricing.js";

describe("getModelPricing", () => {
  it("returns pricing for known OpenAI model", () => {
    const pricing = getModelPricing("openai", "gpt-4o");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(2_500_000);
    expect(pricing!.cachedInputPerMTok).toBe(1_250_000);
    expect(pricing!.outputPerMTok).toBe(10_000_000);
    expect(pricing!.cacheWrite5mPerMTok).toBeUndefined();
    expect(pricing!.cacheWrite1hPerMTok).toBeUndefined();
  });

  it("returns pricing for known Anthropic model with cache write fields", () => {
    const pricing = getModelPricing("anthropic", "claude-sonnet-4-6");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(3_000_000);
    expect(pricing!.cachedInputPerMTok).toBe(300_000);
    expect(pricing!.cacheWrite5mPerMTok).toBe(3_750_000);
    expect(pricing!.cacheWrite1hPerMTok).toBe(6_000_000);
    expect(pricing!.outputPerMTok).toBe(15_000_000);
  });

  it("returns pricing for known Gemini model", () => {
    const pricing = getModelPricing("google", "gemini-2.5-flash");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(300_000);
    expect(pricing!.cachedInputPerMTok).toBe(30_000);
    expect(pricing!.outputPerMTok).toBe(2_500_000);
  });

  it("returns pricing for all 10 launch models", () => {
    const models = [
      ["openai", "gpt-4o"],
      ["openai", "gpt-4o-mini"],
      ["openai", "gpt-4.1"],
      ["openai", "gpt-4.1-mini"],
      ["openai", "o3-mini"],
      ["anthropic", "claude-sonnet-4-6"],
      ["anthropic", "claude-haiku-3.5"],
      ["anthropic", "claude-opus-4"],
      ["google", "gemini-2.5-pro"],
      ["google", "gemini-2.5-flash"],
    ];
    for (const [provider, model] of models) {
      const pricing = getModelPricing(provider, model);
      expect(pricing, `${provider}/${model} should exist`).not.toBeNull();
      expect(pricing!.inputPerMTok).toBeGreaterThan(0);
      expect(pricing!.outputPerMTok).toBeGreaterThan(0);
    }
  });

  it("returns null for unknown model", () => {
    expect(getModelPricing("openai", "gpt-99")).toBeNull();
    expect(getModelPricing("unknown", "model")).toBeNull();
  });
});

describe("costComponent", () => {
  it("returns correct unrounded microdollars", () => {
    // 1000 tokens × 2,500,000 µ$/MTok / 1,000,000 = 2500 µ$
    expect(costComponent(1000, 2_500_000)).toBe(2500.0);
  });

  it("returns 0 for zero tokens", () => {
    expect(costComponent(0, 10_000_000)).toBe(0);
  });

  it("returns 0 for zero rate", () => {
    expect(costComponent(5000, 0)).toBe(0);
  });

  it("handles small token counts", () => {
    // 1 token × 2,500,000 / 1,000,000 = 2.5 µ$
    expect(costComponent(1, 2_500_000)).toBe(2.5);
  });

  it("returns 0 for negative tokens (security guard)", () => {
    expect(costComponent(-1000, 2_500_000)).toBe(0);
  });

  it("returns 0 for negative rate (security guard)", () => {
    expect(costComponent(1000, -2_500_000)).toBe(0);
  });
});

describe("RATE_SCALE", () => {
  it("is 1,000,000", () => {
    expect(RATE_SCALE).toBe(1_000_000);
  });
});

describe("end-to-end cost calculation", () => {
  it("GPT-4o: 5000 input (1000 cached), 2000 output = 31250 microdollars", () => {
    const pricing = getModelPricing("openai", "gpt-4o")!;
    const uncachedInput = 4000;
    const cachedInput = 1000;
    const output = 2000;

    const cost = Math.round(
      costComponent(uncachedInput, pricing.inputPerMTok) +
        costComponent(cachedInput, pricing.cachedInputPerMTok) +
        costComponent(output, pricing.outputPerMTok),
    );

    // costComponent(4000, 2_500_000) + costComponent(1000, 1_250_000) + costComponent(2000, 10_000_000)
    // = 10000 + 1250 + 20000 = 31250
    expect(cost).toBe(31250);
    expect(cost / 1_000_000).toBeCloseTo(0.03125, 5);
  });

  it("Claude Sonnet: 2000 input, 500 cache write (5m), 300 cache read, 1000 output = 22965 microdollars", () => {
    const pricing = getModelPricing("anthropic", "claude-sonnet-4-6")!;

    const cost = Math.round(
      costComponent(2000, pricing.inputPerMTok) +
        costComponent(500, pricing.cacheWrite5mPerMTok!) +
        costComponent(300, pricing.cachedInputPerMTok) +
        costComponent(1000, pricing.outputPerMTok),
    );

    // costComponent(2000, 3_000_000) + costComponent(500, 3_750_000) + costComponent(300, 300_000) + costComponent(1000, 15_000_000)
    // = 6000 + 1875 + 90 + 15000 = 22965
    expect(cost).toBe(22965);
    expect(cost / 1_000_000).toBeCloseTo(0.022965, 6);
  });

  it("integer arithmetic eliminates IEEE 754 float imprecision in multiplication", () => {
    // With old float rates: 1 * 0.0375 = 0.037500000000000006 (IEEE 754 error)
    // With integer rates: (1 * 37500) / 1_000_000 — multiplication is exact
    const result = costComponent(1, 37_500);
    // The division by 1M may introduce ≤1 ULP error, but the multiplication is exact
    expect(result).toBeCloseTo(0.0375, 15);

    // Verify the multiplication step is exact for integer inputs
    const tokens = 7;
    const rate = 37_500;
    const product = tokens * rate; // 262500 — exact integer
    expect(Number.isInteger(product)).toBe(true);
    expect(product).toBe(262_500);
  });
});
