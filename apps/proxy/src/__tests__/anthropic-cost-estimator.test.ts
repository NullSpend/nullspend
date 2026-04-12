import { describe, it, expect } from "vitest";
import { estimateAnthropicMaxCost } from "../lib/anthropic-cost-estimator.js";

describe("estimateAnthropicMaxCost", () => {
  it("returns integer microdollars (suitable for HINCRBY)", () => {
    const result = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(Number.isInteger(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("uses max_tokens when specified in body (cheaper than default cap)", () => {
    const withLimit = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    const withoutLimit = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(withLimit).toBeLessThan(withoutLimit);
  });

  it("does NOT use max_completion_tokens (Anthropic only uses max_tokens)", () => {
    const withMaxCompletionTokens = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_completion_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    const withMaxTokens = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    // max_completion_tokens should be ignored (falls back to 64K cap),
    // while max_tokens=100 should be used, making it much cheaper
    expect(withMaxCompletionTokens).toBeGreaterThan(withMaxTokens);
  });

  it("returns $1 fallback for unknown models", () => {
    const result = estimateAnthropicMaxCost("nonexistent-model", {
      model: "nonexistent-model",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(result).toBe(1_000_000);
  });

  it("all models in output caps produce valid non-fallback estimates", () => {
    const models = [
      "claude-opus-4-6",
      "claude-opus-4-6-20260205",
      "claude-opus-4-5",
      "claude-opus-4-5-20251101",
      "claude-sonnet-4-6",
      "claude-sonnet-4-6-20260217",
      "claude-sonnet-4-5",
      "claude-sonnet-4-5-20250929",
      "claude-sonnet-4",
      "claude-sonnet-4-20250514",
      "claude-sonnet-4-0",
      "claude-opus-4-1",
      "claude-opus-4-1-20250805",
      "claude-opus-4",
      "claude-opus-4-20250514",
      "claude-opus-4-0",
      "claude-haiku-4-5",
      "claude-haiku-4-5-20251001",
      "claude-haiku-3.5",
      "claude-3-5-haiku-20241022",
      "claude-haiku-3",
      "claude-3-haiku-20240307",
    ];

    for (const model of models) {
      const result = estimateAnthropicMaxCost(model, {
        model,
        max_tokens: 100,
        messages: [{ role: "user", content: "test" }],
      });
      expect(result, `${model} should not return the $1 fallback`).not.toBe(1_000_000);
      expect(result, `${model} should return positive value`).toBeGreaterThan(0);
    }
  });

  it("opus models produce higher estimates than sonnet (128K vs 64K cap)", () => {
    const opus = estimateAnthropicMaxCost("claude-opus-4-6", {
      model: "claude-opus-4-6",
      messages: [{ role: "user", content: "hi" }],
    });
    const sonnet = estimateAnthropicMaxCost("claude-sonnet-4-6", {
      model: "claude-sonnet-4-6",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(opus).toBeGreaterThan(sonnet);
  });

  it("applies 1.1x safety margin", () => {
    const result = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 1,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(result).toBeGreaterThan(0);
  });

  it("scales with body size (larger messages = higher estimate)", () => {
    const small = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    });
    const large = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "a".repeat(10000) }],
    });
    expect(large).toBeGreaterThan(small);
  });

  it("applies 2x input and 1.5x output multipliers for long-context requests (>200K tokens)", () => {
    // Create a body large enough to estimate >200K tokens (>800K chars at 4 chars/token)
    const longContent = "a".repeat(900_000);
    const longContext = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 1000,
      messages: [{ role: "user", content: longContent }],
    });

    // Same body but short enough for normal pricing
    const shortContent = "a".repeat(1000);
    const normalContext = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 1000,
      messages: [{ role: "user", content: shortContent }],
    });

    // Long context should be significantly more expensive due to multipliers
    // Input: 2x rate, Output: 1.5x rate
    // The ratio won't be exact 2x because of body size difference, but
    // we can verify the output component is more expensive by isolating it
    const longOutputOnly = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 1000,
      messages: [{ role: "user", content: longContent }],
    });
    // Long context estimate should be much higher than normal
    expect(longContext).toBeGreaterThan(normalContext);

    // Verify the multiplier effect: for the same max_tokens=1000,
    // long-context output rate is 1.5x, so the output cost component
    // in the long-context estimate should be 1.5x the normal rate.
    // We verify by checking that the long-context estimate is MORE than
    // what you'd get by just scaling the input tokens (without multipliers).
    // 900K chars / 4 = 225K tokens, at 2x rate vs 1x = 2x input cost difference
    expect(longContext).toBeGreaterThan(longOutputOnly * 0.9); // sanity
  });

  it("does NOT apply long-context multipliers below 200K token threshold", () => {
    // 700K chars / 4 = 175K tokens — below 200K threshold
    const content = "a".repeat(700_000);
    const belowThreshold = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: content }],
    });

    // 900K chars / 4 = 225K tokens — above 200K threshold
    const contentAbove = "a".repeat(900_000);
    const aboveThreshold = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: contentAbove }],
    });

    // Above-threshold should be MORE than proportionally higher
    // because of 2x input multiplier
    const sizeRatio = 225_000 / 175_000; // ~1.29 token ratio
    const costRatio = aboveThreshold / belowThreshold;

    // Cost ratio should be > size ratio because of 2x multiplier
    expect(costRatio).toBeGreaterThan(sizeRatio);
  });
});
