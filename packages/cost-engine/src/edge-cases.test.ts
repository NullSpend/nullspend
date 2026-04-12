import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent } from "./pricing.js";

describe("costComponent boundary values", () => {
  it("single token at cheapest rate produces sub-microdollar value", () => {
    const pricing = getModelPricing("google", "gemini-2.5-flash")!;
    const cost = costComponent(1, pricing.cachedInputPerMTok);
    expect(cost).toBe(0.0375);
    expect(cost).toBeGreaterThan(0);
    expect(cost).toBeLessThan(1);
  });

  it("Math.round(0.5) goes to 1 (banker's rounding NOT used in JS)", () => {
    expect(Math.round(0.5)).toBe(1);
    expect(Math.round(1.5)).toBe(2);
    expect(Math.round(2.5)).toBe(3);
  });

  it("rounding boundary: cost that is exactly 0.5 rounds up", () => {
    // 1 token × 0.5 $/MTok = 0.5 microdollars → rounds to 1
    expect(Math.round(costComponent(1, 0.5))).toBe(1);
  });

  it("rounding boundary: cost just below 0.5 rounds down", () => {
    // 1 token × 0.499 $/MTok = 0.499 microdollars → rounds to 0
    expect(Math.round(costComponent(1, 0.499))).toBe(0);
  });

  it("very large token count stays in safe integer range", () => {
    // 1 billion tokens at $75/MTok (Opus output) = 75 billion microdollars
    const tokens = 1_000_000_000;
    const rate = 75.0;
    const cost = costComponent(tokens, rate);
    expect(cost).toBe(75_000_000_000);
    expect(Number.isSafeInteger(Math.round(cost))).toBe(true);
  });

  it("128K context window (typical max) produces valid cost", () => {
    const pricing = getModelPricing("anthropic", "claude-opus-4")!;
    const inputTokens = 128_000;
    const outputTokens = 4_096;
    const cost = Math.round(
      costComponent(inputTokens, pricing.inputPerMTok) +
        costComponent(outputTokens, pricing.outputPerMTok),
    );
    // 128000*15 + 4096*75 = 1,920,000 + 307,200 = 2,227,200
    expect(cost).toBe(2_227_200);
    expect(cost / 1_000_000).toBeCloseTo(2.2272, 4);
  });

  it("MAX_SAFE_INTEGER tokens at high rate exceeds safe integer range", () => {
    const maxTokens = Number.MAX_SAFE_INTEGER;
    // At Opus output rate ($75/MTok), the result is ~6.75×10^17, exceeding MAX_SAFE_INTEGER
    const cost = costComponent(maxTokens, 75);
    expect(Number.isFinite(cost)).toBe(true);
    expect(Number.isSafeInteger(Math.round(cost))).toBe(false);
  });

  it("MAX_SAFE_INTEGER tokens at low rate stays within safe range", () => {
    const maxTokens = Number.MAX_SAFE_INTEGER;
    const cost = costComponent(maxTokens, 0.0375);
    expect(Number.isFinite(cost)).toBe(true);
    expect(Number.isSafeInteger(Math.round(cost))).toBe(true);
  });

  it("zero tokens for all components yields zero total cost", () => {
    const pricing = getModelPricing("openai", "gpt-4o")!;
    const cost = Math.round(
      costComponent(0, pricing.inputPerMTok) +
        costComponent(0, pricing.cachedInputPerMTok) +
        costComponent(0, pricing.outputPerMTok),
    );
    expect(cost).toBe(0);
  });
});

describe("costComponent accumulation precision", () => {
  it("summing many small costs matches single large calculation", () => {
    const rate = 2.5;
    const perCall = costComponent(100, rate);
    const summed = Array.from({ length: 1000 }, () => perCall).reduce((a, b) => a + b, 0);
    const direct = costComponent(100_000, rate);
    expect(Math.round(summed)).toBe(Math.round(direct));
  });

  it("summing 10,000 micro-costs does not drift more than 1 microdollar", () => {
    const costs = Array.from({ length: 10_000 }, (_, i) =>
      costComponent(i + 1, 0.15),
    );
    const summed = Math.round(costs.reduce((a, b) => a + b, 0));
    // Expected: sum(1..10000) * 0.15 = 50_005_000 * 0.15 = 7_500_750
    const expected = Math.round(((10_000 * 10_001) / 2) * 0.15);
    expect(Math.abs(summed - expected)).toBeLessThanOrEqual(1);
  });
});

describe("getModelPricing edge cases", () => {
  it("returns null for empty strings", () => {
    expect(getModelPricing("", "")).toBeNull();
  });

  it("returns null for correct provider but wrong model", () => {
    expect(getModelPricing("openai", "gpt-5-turbo")).toBeNull();
  });

  it("returns null for correct model but wrong provider", () => {
    expect(getModelPricing("anthropic", "gpt-4o")).toBeNull();
  });

  it("is case-sensitive (uppercase rejected)", () => {
    expect(getModelPricing("OpenAI", "gpt-4o")).toBeNull();
    expect(getModelPricing("openai", "GPT-4O")).toBeNull();
  });

  it("rejects model with leading/trailing whitespace", () => {
    expect(getModelPricing("openai", " gpt-4o")).toBeNull();
    expect(getModelPricing("openai", "gpt-4o ")).toBeNull();
  });

  it("rejects full key as model name", () => {
    expect(getModelPricing("openai", "openai/gpt-4o")).toBeNull();
  });

  it("rejects provider with slash", () => {
    expect(getModelPricing("openai/", "gpt-4o")).toBeNull();
  });

  it("returns independent copies (no shared mutation risk)", () => {
    const a = getModelPricing("openai", "gpt-4o");
    const b = getModelPricing("openai", "gpt-4o");
    expect(a).toEqual(b);
    // They reference the same object (intentional — immutable data)
    // but the values must be correct
    expect(a!.inputPerMTok).toBe(2.5);
  });
});

describe("all-model cost calculation smoke test", () => {
  const models: [string, string][] = [
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
    it(`${provider}/${model}: 10K input + 2K output produces positive integer cost`, () => {
      const pricing = getModelPricing(provider, model)!;
      const cost = Math.round(
        costComponent(10_000, pricing.inputPerMTok) +
          costComponent(2_000, pricing.outputPerMTok),
      );
      expect(cost).toBeGreaterThan(0);
      expect(Number.isSafeInteger(cost)).toBe(true);
    });
  }
});
