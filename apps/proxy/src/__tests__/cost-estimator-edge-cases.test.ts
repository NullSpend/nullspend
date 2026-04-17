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
  // --- P0-4: invalid max_tokens inputs fall through to model default cap ---

  // Fixed bodyByteLength normalizes input-side cost across tests so output-side
  // behavior can be compared apples-to-apples. Without this, JSON serialization
  // length differs between baseBody and baseBody-with-extra-field and the
  // estimate shifts by a few microdollars.
  const FIXED_BODY_BYTES = 256;

  it("max_tokens: 0 falls through to model default (P0-4)", () => {
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: 0 }), FIXED_BODY_BYTES);
    const resultWithDefault = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini"), FIXED_BODY_BYTES);

    // Prior to P0-4, nullish-coalescing treated 0 as truthy and produced a
    // zero-output estimate. Now 0 is not positive → falls through to default.
    expect(result).toBe(resultWithDefault);
  });

  it("max_completion_tokens: 0 falls through to default (P0-4)", () => {
    const withZero = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_completion_tokens: 0 }), FIXED_BODY_BYTES);
    const withDefault = estimateMaxCost("gpt-4o", baseBody("gpt-4o"), FIXED_BODY_BYTES);
    expect(withZero).toBe(withDefault);
  });

  it("max_tokens as negative number falls through to default (P0-4)", () => {
    const result = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini", { max_tokens: -100 }), FIXED_BODY_BYTES);
    const withDefault = estimateMaxCost("gpt-4o-mini", baseBody("gpt-4o-mini"), FIXED_BODY_BYTES);
    // REGRESSION GUARD: negative max_tokens used to produce a degenerate
    // estimate. Now the gemini validation pattern rejects non-positive and
    // non-finite values, falling through to the model-specific default cap.
    expect(result).toBe(withDefault);
  });

  it.each([
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["-Infinity", Number.NEGATIVE_INFINITY],
    ["string 'unlimited'", "unlimited"],
    ["null", null],
    ["undefined", undefined],
    ["object {value: 100}", { value: 100 }],
    ["bool true (Codex P2)", true],
    ["bool false", false],
    ["array [5000] (Codex P2)", [5000]],
    ["empty array", []],
  ])("max_tokens: %s falls through to default (P0-4 regression)", (_label, value) => {
    // REGRESSION GUARD: these malformed inputs used to cast to Number()
    // and propagate NaN into the reservation math, producing NaN estimates
    // that bypassed budget enforcement via the NF-2 hasBudgets:false
    // mislabel path.
    const result = estimateMaxCost(
      "gpt-4o-mini",
      baseBody("gpt-4o-mini", { max_tokens: value as unknown as number }),
      FIXED_BODY_BYTES,
    );
    const withDefault = estimateMaxCost(
      "gpt-4o-mini",
      baseBody("gpt-4o-mini"),
      FIXED_BODY_BYTES,
    );

    expect(Number.isFinite(result)).toBe(true);
    expect(Number.isInteger(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
    expect(result).toBe(withDefault);
  });

  it("numeric string max_tokens IS coerced via Number() (matches gemini pattern)", () => {
    // Parity with gemini-cost-estimator.ts: Number() coercion is intentionally
    // lenient. "1000" → 1000. Documented so future refactors know this is not
    // a P0-4 bug.
    const withString = estimateMaxCost(
      "gpt-4o",
      baseBody("gpt-4o", { max_tokens: "1000" as unknown as number }),
      FIXED_BODY_BYTES,
    );
    const withNumber = estimateMaxCost(
      "gpt-4o",
      baseBody("gpt-4o", { max_tokens: 1000 }),
      FIXED_BODY_BYTES,
    );
    expect(withString).toBe(withNumber);
  });

  it("max_tokens > 1M is clamped to 1M sanity limit (P0-4)", () => {
    const clamped = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_tokens: 10_000_000 }));
    const at1m = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_tokens: 1_000_000 }));
    // 10M is clamped to 1M → identical estimate
    expect(clamped).toBe(at1m);
  });

  it("max_tokens as fractional number is ceiled (P0-4)", () => {
    const fractional = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_tokens: 100.3 }));
    const ceiled = estimateMaxCost("gpt-4o", baseBody("gpt-4o", { max_tokens: 101 }));
    expect(fractional).toBe(ceiled);
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
    // Use explicit small max_tokens to keep the expected computation simple.
    const body = baseBody("gpt-4o-mini", { max_tokens: 100 });
    const result = estimateMaxCost("gpt-4o-mini", body);

    // Compute expected: body stringified, / 4 chars per token, ceiled = input tokens
    const bodyStr = JSON.stringify(body);
    const inputTokens = Math.ceil(bodyStr.length / 4);
    // gpt-4o-mini rates: input $0.15/MTok, output $0.60/MTok
    const inputCost = inputTokens * 0.15;
    const outputCost = 100 * 0.60;
    const expected = Math.round((inputCost + outputCost) * 1.1);

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
