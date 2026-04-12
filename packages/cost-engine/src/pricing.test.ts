import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent } from "./pricing.js";

describe("getModelPricing", () => {
  it("returns pricing for known OpenAI model", () => {
    const pricing = getModelPricing("openai", "gpt-4o");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(2.5);
    expect(pricing!.cachedInputPerMTok).toBe(1.25);
    expect(pricing!.outputPerMTok).toBe(10.0);
    expect(pricing!.cacheWrite5mPerMTok).toBeUndefined();
    expect(pricing!.cacheWrite1hPerMTok).toBeUndefined();
  });

  it("returns pricing for known Anthropic model with cache write fields", () => {
    const pricing = getModelPricing("anthropic", "claude-sonnet-4-6");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(3.0);
    expect(pricing!.cachedInputPerMTok).toBe(0.3);
    expect(pricing!.cacheWrite5mPerMTok).toBe(3.75);
    expect(pricing!.cacheWrite1hPerMTok).toBe(6.0);
    expect(pricing!.outputPerMTok).toBe(15.0);
  });

  it("returns pricing for known Gemini model", () => {
    const pricing = getModelPricing("google", "gemini-2.5-flash");
    expect(pricing).not.toBeNull();
    expect(pricing!.inputPerMTok).toBe(0.15);
    expect(pricing!.cachedInputPerMTok).toBe(0.0375);
    expect(pricing!.outputPerMTok).toBe(0.6);
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
    expect(costComponent(1000, 2.5)).toBe(2500.0);
  });

  it("returns 0 for zero tokens", () => {
    expect(costComponent(0, 10.0)).toBe(0);
  });

  it("returns 0 for zero rate", () => {
    expect(costComponent(5000, 0)).toBe(0);
  });

  it("handles small token counts", () => {
    expect(costComponent(1, 2.5)).toBe(2.5);
  });

  it("returns 0 for negative tokens (security guard)", () => {
    expect(costComponent(-1000, 2.5)).toBe(0);
  });

  it("returns 0 for negative rate (security guard)", () => {
    expect(costComponent(1000, -2.5)).toBe(0);
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

    // 4000*2.50 + 1000*1.25 + 2000*10.00 = 10000 + 1250 + 20000 = 31250
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

    // 2000*3.00 + 500*3.75 + 300*0.30 + 1000*15.00 = 6000 + 1875 + 90 + 15000 = 22965
    expect(cost).toBe(22965);
    expect(cost / 1_000_000).toBeCloseTo(0.022965, 6);
  });

  it("IEEE 754 edge case: Math.round handles floating point safely", () => {
    // 0.1 + 0.2 !== 0.3 in IEEE 754, so test with values that actually produce fp error
    const raw = costComponent(1, 0.1) + costComponent(1, 0.2);
    // 1*0.1 + 1*0.2 = 0.30000000000000004 due to IEEE 754
    expect(raw).not.toBe(0.3);
    expect(Math.round(raw)).toBe(0);

    // Larger scale where rounding matters: verify it rounds correctly
    const big = costComponent(333, 0.3);
    expect(Math.round(big)).toBe(100);
  });
});
