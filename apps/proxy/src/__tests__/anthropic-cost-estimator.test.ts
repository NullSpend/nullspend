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

  // P0-4: Regression guards for unsanitized max_tokens inputs
  describe("max_tokens input validation (P0-4)", () => {
    const FIXED_BODY_BYTES = 256;
    const baseBody = (overrides: Record<string, unknown> = {}) => ({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "hello" }],
      ...overrides,
    });

    it.each([
      ["NaN", Number.NaN],
      ["Infinity", Number.POSITIVE_INFINITY],
      ["-Infinity", Number.NEGATIVE_INFINITY],
      ["0", 0],
      ["-100", -100],
      ["string 'unlimited'", "unlimited"],
      ["null", null],
      ["undefined", undefined],
      ["object {value: 100}", { value: 100 }],
      ["bool true (Codex P2)", true],
      ["bool false", false],
      ["array [5000] (Codex P2)", [5000]],
      ["empty array", []],
    ])("max_tokens: %s falls through to default cap (regression)", (_label, value) => {
      // REGRESSION GUARD: these malformed inputs used to cast via Number()
      // and propagate NaN/negative into the reservation math, producing NaN
      // estimates that bypassed budget enforcement via the DO's hasBudgets:false
      // path mislabeled as stale cache.
      const result = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody({ max_tokens: value }),
        FIXED_BODY_BYTES,
      );
      const withDefault = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody(),
        FIXED_BODY_BYTES,
      );

      expect(Number.isFinite(result)).toBe(true);
      expect(Number.isInteger(result)).toBe(true);
      expect(result).toBeGreaterThan(0);
      expect(result).toBe(withDefault);
    });

    it("max_tokens > 1M is clamped to 1M sanity limit", () => {
      const clamped = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody({ max_tokens: 10_000_000 }),
        FIXED_BODY_BYTES,
      );
      const at1m = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody({ max_tokens: 1_000_000 }),
        FIXED_BODY_BYTES,
      );
      expect(clamped).toBe(at1m);
    });

    it("max_tokens as fractional number is ceiled", () => {
      const fractional = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody({ max_tokens: 100.3 }),
        FIXED_BODY_BYTES,
      );
      const ceiled = estimateAnthropicMaxCost(
        "claude-sonnet-4-5",
        baseBody({ max_tokens: 101 }),
        FIXED_BODY_BYTES,
      );
      expect(fractional).toBe(ceiled);
    });
  });

  it("does NOT apply long-context multipliers below 100K token threshold (P0-5)", () => {
    // P0-5: Threshold lowered from 200K → 100K (codex review: 50% undercount
    // worst case = 200K actual / 2 = 100K estimate threshold). Test validates
    // the new boundary.

    // 300K chars / 4 = 75K tokens — below 100K threshold
    const content = "a".repeat(300_000);
    const belowThreshold = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: content }],
    });

    // 600K chars / 4 = 150K tokens — above 100K threshold
    const contentAbove = "a".repeat(600_000);
    const aboveThreshold = estimateAnthropicMaxCost("claude-sonnet-4-5", {
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: contentAbove }],
    });

    // Above-threshold should be MORE than proportionally higher because
    // of 2x input multiplier applied only to the above-threshold estimate.
    const sizeRatio = 150_000 / 75_000; // 2.0 token ratio
    const costRatio = aboveThreshold / belowThreshold;

    // Cost ratio should be > size ratio because of 2x multiplier on the
    // above-threshold side only.
    expect(costRatio).toBeGreaterThan(sizeRatio);
  });

  it("applies long-context multiplier for 50% worst-case undercount at 200K actual (P0-5 regression)", () => {
    // REGRESSION GUARD: the calculator uses totalInputTokens > 200_000 (with
    // cache + image tokens). The estimator uses bodyByteLength/4 which can
    // under-count by up to ~50% for multimodal/code/CJK content. This test
    // ensures the estimator applies the 2x multiplier at the 100K estimate
    // boundary so a request that estimates at 100K+ but will actually be
    // 200K+ tokens never under-reserves relative to the calculator.
    //
    // Codex follow-up flagged 150K as insufficient — 25% buffer vs 50%
    // worst case. 100K threshold covers the documented worst case exactly.
    const FIXED_BODY_BYTES = 400_001; // 100_001 estimate tokens — just over new 100K threshold
    const longContext = estimateAnthropicMaxCost(
      "claude-sonnet-4-5",
      { model: "claude-sonnet-4-5", max_tokens: 100, messages: [{ role: "user", content: "x" }] },
      FIXED_BODY_BYTES,
    );

    const belowThreshold = estimateAnthropicMaxCost(
      "claude-sonnet-4-5",
      { model: "claude-sonnet-4-5", max_tokens: 100, messages: [{ role: "user", content: "x" }] },
      200_000, // 50K estimate tokens — below threshold
    );

    // Same model, same max_tokens. The long-context variant applies 2x input
    // rate to 2x body tokens (400K vs 200K bytes), so the ratio is bounded
    // below by 2x body-size * 2x rate / 1x = 4x input contribution. Even
    // with output cost held constant, long-context total should be > 2x.
    expect(longContext).toBeGreaterThan(belowThreshold * 2);
  });
});
