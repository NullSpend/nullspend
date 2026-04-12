import { describe, it, expect } from "vitest";
import { calculateOpenAICost } from "../lib/cost-calculator.js";

describe("calculateOpenAICost edge cases", () => {
  it("handles zero tokens for all fields", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 0, completion_tokens: 0 },
      "req-zero",
      10,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toEqual({ input: 0, cached: 0, output: 0 });
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
  });

  it("handles very large token counts without overflow", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 128000, completion_tokens: 16384 },
      "req-large",
      5000,
    );

    expect(result.inputTokens).toBe(128000);
    expect(result.outputTokens).toBe(16384);
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(Number.isFinite(result.costMicrodollars)).toBe(true);
    // input: 128000 * 2.50 = 320000, output: 16384 * 10.00 = 163840 → 483840
    expect(result.costMicrodollars).toBe(483840);
  });

  it("handles all-cached input (normalInput = 0)", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 100,
        prompt_tokens_details: { cached_tokens: 1000 },
      },
      "req-all-cached",
      200,
    );

    expect(result.cachedInputTokens).toBe(1000);
    // normalInput: 0 * 2.50 = 0
    // cached: 1000 * 1.25 = 1250
    // output: 100 * 10.00 = 1000
    // total: 2250
    expect(result.costMicrodollars).toBe(2250);
  });

  it("clamps negative cached_tokens to 0 (prevents cost inflation)", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 50,
        prompt_tokens_details: { cached_tokens: -10 },
      },
      "req-neg-cached",
      50,
    );

    // cached clamped to 0, so normalInput = 100 - 0 = 100
    // input: 100 * 2.50 = 250, output: 50 * 10.00 = 500
    // total: 750
    expect(result.cachedInputTokens).toBe(0);
    expect(result.costMicrodollars).toBe(750);
  });

  it("handles all-reasoning completion tokens", () => {
    const result = calculateOpenAICost(
      "o3-mini",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 5000,
        completion_tokens_details: { reasoning_tokens: 5000 },
      },
      "req-all-reason",
      1000,
    );

    expect(result.reasoningTokens).toBe(5000);
    // input: 100 * 1.10 = 110
    // output: 5000 * 4.40 = 22000
    // total: 22110
    expect(result.costMicrodollars).toBe(22110);
  });

  it("preserves durationMs and requestId accurately", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 10, completion_tokens: 5 },
      "req-id-preserves-exactly",
      99999,
    );

    expect(result.requestId).toBe("req-id-preserves-exactly");
    expect(result.durationMs).toBe(99999);
  });

  it("handles usage with prompt_tokens_details but no cached_tokens key", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 50,
        prompt_tokens_details: {} as any,
      },
      "req-empty-details",
      100,
    );

    expect(result.cachedInputTokens).toBe(0);
    // normalInput = 100, input cost: 100 * 2.50 = 250
    // output: 50 * 10.00 = 500
    expect(result.costMicrodollars).toBe(750);
  });

  it("handles completion_tokens_details with no reasoning_tokens key", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 50,
        completion_tokens_details: {} as any,
      },
      "req-empty-comp-details",
      100,
    );

    expect(result.reasoningTokens).toBe(0);
  });

  it("falls back from unknown request model to known response model", () => {
    const result = calculateOpenAICost(
      "ft:gpt-4o:my-org:custom-name:id",
      "gpt-4o",
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-finetune-fallback",
      100,
    );

    expect(result.model).toBe("gpt-4o");
    expect(result.costMicrodollars).toBeGreaterThan(0);
  });

  it("uses request model (not response model) when request model has pricing", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      "gpt-4o-2024-11-20",
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-prefer-request",
      100,
    );

    expect(result.model).toBe("gpt-4o");
  });

  it("returns zero cost when both request and response models are unknown", () => {
    const result = calculateOpenAICost(
      "some-custom-model",
      "another-unknown-model",
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-both-unknown",
      100,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeNull();
    expect(result.model).toBe("some-custom-model");
  });

  it("handles gpt-4o-mini pricing correctly", () => {
    const result = calculateOpenAICost(
      "gpt-4o-mini",
      null,
      { prompt_tokens: 1000000, completion_tokens: 1000000 },
      "req-mini-million",
      500,
    );

    // input: 1M * 0.15 = 150000
    // output: 1M * 0.60 = 600000
    // total: 750000
    expect(result.costMicrodollars).toBe(750000);
  });

  it("handles gpt-4.1-mini pricing correctly", () => {
    const result = calculateOpenAICost(
      "gpt-4.1-mini",
      null,
      { prompt_tokens: 1000, completion_tokens: 500 },
      "req-4.1-mini",
      100,
    );

    // input: 1000 * 0.40 = 400
    // output: 500 * 1.60 = 800
    // total: 1200
    expect(result.costMicrodollars).toBe(1200);
  });
});
