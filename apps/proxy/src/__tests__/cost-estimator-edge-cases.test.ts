/**
 * Cost Estimator Edge Case Tests
 *
 * Additional edge case tests for estimateMaxCost. Uses real
 * @nullspend/cost-engine (no mocks) to validate actual behavior
 * with nullish coalescing, model caps, and safety margin.
 */
import { describe, it, expect } from "vitest";
import { estimateMaxCost } from "../lib/cost-estimator.js";

const baseBody = (model: string, overrides: Record<string, unknown> = {}) => ({
  model,
  messages: [{ role: "user", content: "hello" }],
  ...overrides,
});

describe("estimateMaxCost edge cases", () => {
  // --- max_tokens=0 edge case (nullish coalescing treats 0 as truthy) ---

  it("max_tokens: 0 is treated as truthy by ?? — output cap is 0, estimate is input-only", () => {
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: 0 }));
    const resultWithDefault = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini"));

    expect(result).toBeLessThan(resultWithDefault);
    expect(result).toBeGreaterThan(0); // input cost still contributes
  });

  it("max_completion_tokens: 0 behaves same as max_tokens: 0", () => {
    const withZero = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_completion_tokens: 0 }));
    const withDefault = estimateMaxCost("gpt-4o", baseBody("gpt-4o"));
    expect(withZero).toBeLessThan(withDefault);
  });

  it("max_tokens as negative number — costComponent returns 0 for negative tokens", () => {
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: -100 }));
    // Output cost is 0 (negative tokens), result is input-cost-only with margin
    expect(result).toBeGreaterThan(0);
    const withZero = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: 0 }));
    expect(result).toBe(withZero); // both yield 0 output cost
  });

  // --- Body size edge cases ---

  it("body with empty messages array produces valid estimate", () => {
    const result = estimateMaxCost("gpt-4o-mini", { model: "gpt-4o-mini", messages: [] });
    expect(Number.isInteger(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("very large body (100KB+) produces valid integer without overflow", () => {
    const largeContent = "a".repeat(100_000);
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", {
      messages: [{ role: "user", content: largeContent }],
      max_tokens: 100,
    }));
    expect(Number.isInteger(result)).toBe(true);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  // --- Model output caps ---

  it("all models in MODEL_OUTPUT_CAPS produce valid non-fallback estimates", () => {
    const modelsInCaps = [
      "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
      "o3", "o3-mini", "o4-mini", "o1",
      "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2",
    ];

    for (const model of modelsInCaps) {
      const result = estimateMaxCost(model, baseBody(model));
      expect(result).not.toBe(1_000_000); // not the unknown-model fallback
      expect(Number.isInteger(result)).toBe(true);
      expect(result).toBeGreaterThan(0);
    }
  });

  it("reasoning models (o3, o1) produce higher estimates than GPT models due to 100k cap", () => {
    const o3Estimate = estimateMaxCost("o3", baseBody("o3"));
    const gpt4oEstimate = estimateMaxCost("gpt-4o", baseBody("gpt-4o"));
    expect(o3Estimate).toBeGreaterThan(gpt4oEstimate);
  });

  // --- Safety margin verification ---

  it("applies exact 1.1x safety margin", () => {
    // Use max_tokens=0 to isolate input cost only (output cost = 0)
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: 0 }));

    // Compute expected: body stringified, / 4 chars per token, ceiled = input tokens
    const bodyStr = JSON.stringify(baseBody("gpt-4o-mini", { max_tokens: 0 }));
    const inputTokens = Math.ceil(bodyStr.length / 4);
    // gpt-4o-mini input rate: $0.15/MTok
    const inputCost = inputTokens * 0.15; // costComponent(tokens, rate) = tokens * rate
    const expected = Math.round(inputCost * 1.1);

    expect(result).toBe(expected);
  });

  // --- max_tokens vs max_completion_tokens precedence ---

  it("max_tokens used when max_completion_tokens is undefined", () => {
    const withMaxTokens = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_tokens: 500 }));
    const withDefault = estimateMaxCost("gpt-4o", baseBody("gpt-4o"));
    expect(withMaxTokens).toBeLessThan(withDefault);
  });

  it("max_completion_tokens takes precedence over max_tokens (nullish coalescing order)", () => {
    const withBoth = estimateMaxCost("o3", baseBody("o3", {
      max_completion_tokens: 200,
      max_tokens: 50_000,
    }));
    const withOnlyMaxTokens = estimateMaxCost("o3", baseBody("o3", {
      max_tokens: 50_000,
    }));

    expect(withBoth).toBeLessThan(withOnlyMaxTokens);
  });
});
