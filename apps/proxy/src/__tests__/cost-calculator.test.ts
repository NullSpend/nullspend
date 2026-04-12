import { describe, it, expect } from "vitest";
import { calculateOpenAICost } from "../lib/cost-calculator.js";

describe("calculateOpenAICost", () => {
  it("calculates cost for GPT-4o with cached tokens", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      "gpt-4o-2024-11-20",
      {
        prompt_tokens: 1000,
        completion_tokens: 500,
        prompt_tokens_details: { cached_tokens: 200 },
        completion_tokens_details: { reasoning_tokens: 0 },
      },
      "req-123",
      150,
    );

    expect(result.provider).toBe("openai");
    expect(result.model).toBe("gpt-4o");
    expect(result.inputTokens).toBe(1000);
    expect(result.outputTokens).toBe(500);
    expect(result.cachedInputTokens).toBe(200);
    expect(result.reasoningTokens).toBe(0);
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(result.requestId).toBe("req-123");
    expect(result.durationMs).toBe(150);
    expect(result.userId).toBeNull();
    expect(result.apiKeyId).toBeNull();

    // Cost breakdown:
    // normalInput: 800 tokens * 2.50 $/MTok = 2000
    // cached: 200 tokens * 1.25 $/MTok = 250
    // output: 500 tokens * 10.00 $/MTok = 5000
    // total: 7250 microdollars
    expect(result.costMicrodollars).toBe(7250);
    expect(result.costBreakdown).toEqual({ input: 2000, cached: 250, output: 5000 });
  });

  it("calculates cost for o3-mini with reasoning tokens", () => {
    const result = calculateOpenAICost(
      "o3-mini",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 2000,
        completion_tokens_details: { reasoning_tokens: 1500 },
      },
      "req-456",
      300,
    );

    expect(result.reasoningTokens).toBe(1500);
    // reasoning_tokens are included in completion_tokens — same output rate
    // input: 100 * 1.10 = 110
    // output: 2000 * 4.40 = 8800
    // total: 8910
    expect(result.costMicrodollars).toBe(8910);
    expect(result.costBreakdown).toEqual({
      input: 110, cached: 0, output: 8800,
      reasoning: 6600, // 1500 * 4.40
    });
  });

  it("calculates cost for GPT-4.1 standard call", () => {
    const result = calculateOpenAICost(
      "gpt-4.1",
      "gpt-4.1",
      {
        prompt_tokens: 500,
        completion_tokens: 200,
      },
      "req-789",
      100,
    );

    // input: 500 * 2.00 = 1000
    // output: 200 * 8.00 = 1600
    // total: 2600
    expect(result.costMicrodollars).toBe(2600);
    expect(result.costBreakdown).toEqual({ input: 1000, cached: 0, output: 1600 });
  });

  it("returns costMicrodollars: 0 for unknown model", () => {
    const result = calculateOpenAICost(
      "unknown-model-xyz",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 50,
      },
      "req-unknown",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeNull();
    expect(result.model).toBe("unknown-model-xyz");
  });

  it("handles missing usage fields gracefully (defaults to 0)", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 50,
      } as any,
      "req-defaults",
      75,
    );

    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
    // input: 100 * 2.50 = 250
    // output: 50 * 10.00 = 500
    // total: 750
    expect(result.costMicrodollars).toBe(750);
  });

  it("falls back to response model when request model has no pricing", () => {
    const result = calculateOpenAICost(
      "gpt-4o-2024-11-20",
      "gpt-4o",
      {
        prompt_tokens: 100,
        completion_tokens: 50,
      },
      "req-fallback",
      60,
    );

    // "gpt-4o-2024-11-20" is not in our pricing keys, fallback to "gpt-4o"
    expect(result.model).toBe("gpt-4o");
    expect(result.costMicrodollars).toBeGreaterThan(0);
  });

  it("clamps costMicrodollars to minimum of 0", () => {
    // With 0 tokens, cost should be exactly 0 (not negative)
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 0,
        completion_tokens: 0,
        prompt_tokens_details: { cached_tokens: 0 },
        completion_tokens_details: { reasoning_tokens: 0 },
      },
      "req-zero-tokens",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costMicrodollars).toBeGreaterThanOrEqual(0);
  });
});

describe("calculateOpenAICost actionId attribution", () => {
  it("includes actionId when provided in attribution", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-with-action",
      100,
      { userId: "user-1", apiKeyId: "key-1", actionId: "act-abc-123" },
    );

    expect(result.actionId).toBe("act-abc-123");
    expect(result.userId).toBe("user-1");
    expect(result.apiKeyId).toBe("key-1");
  });

  it("sets actionId to null when attribution has null actionId", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-null-action",
      100,
      { userId: "user-1", apiKeyId: "key-1", actionId: null },
    );

    expect(result.actionId).toBeNull();
  });

  it("sets actionId to null when no attribution provided", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-no-attr",
      100,
    );

    expect(result.actionId).toBeNull();
    expect(result.userId).toBeNull();
    expect(result.apiKeyId).toBeNull();
  });

  it("preserves actionId alongside correct cost calculation", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 500,
        prompt_tokens_details: { cached_tokens: 200 },
        completion_tokens_details: { reasoning_tokens: 0 },
      },
      "req-action-cost",
      250,
      { userId: "u-1", apiKeyId: "k-1", actionId: "act-xyz" },
    );

    expect(result.actionId).toBe("act-xyz");
    expect(result.costMicrodollars).toBe(7250);
    expect(result.inputTokens).toBe(1000);
    expect(result.cachedInputTokens).toBe(200);
  });
});

describe("calculateOpenAICost costBreakdown invariants", () => {
  it("costBreakdown is null when model is unknown", () => {
    const result = calculateOpenAICost(
      "nonexistent-model",
      null,
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-null-bd",
      50,
    );
    expect(result.costBreakdown).toBeNull();
    expect(result.costMicrodollars).toBe(0);
  });

  it("costBreakdown components sum exactly to costMicrodollars", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 500,
        prompt_tokens_details: { cached_tokens: 200 },
      },
      "req-sum-check",
      100,
    );

    expect(result.costBreakdown).not.toBeNull();
    const bd = result.costBreakdown!;
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });

  it("residual distribution preserves exact sum with fractional components", () => {
    // 3 input tokens * 2.50 = 7.5, 0 cached, 7 output tokens * 10.00 = 70
    // total = Math.round(7.5 + 0 + 70) = Math.round(77.5) = 78
    // rounded components: 8 + 0 + 70 = 78 → residual = 0 (no adjustment needed)
    const result1 = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 3, completion_tokens: 7 },
      "req-frac1",
      50,
    );
    expect(result1.costBreakdown).not.toBeNull();
    const bd1 = result1.costBreakdown!;
    expect(bd1.input! + bd1.output! + bd1.cached!).toBe(result1.costMicrodollars);

    // 1 input token * 2.50 = 2.5, 1 cached * 1.25 = 1.25, 1 output * 10.00 = 10
    // total = Math.round(2.5 + 1.25 + 10) = Math.round(13.75) = 14
    // rounded components: 3 + 1 + 10 = 14 → residual = 0
    const result2 = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 2,
        completion_tokens: 1,
        prompt_tokens_details: { cached_tokens: 1 },
      },
      "req-frac2",
      50,
    );
    expect(result2.costBreakdown).not.toBeNull();
    const bd2 = result2.costBreakdown!;
    expect(bd2.input! + bd2.output! + bd2.cached!).toBe(result2.costMicrodollars);

    // Use gpt-4.1-nano: input 0.10, cached 0.025, output 0.40
    // 3 input * 0.10 = 0.3, 7 cached * 0.025 = 0.175, 11 output * 0.40 = 4.4
    // total = Math.round(0.3 + 0.175 + 4.4) = Math.round(4.875) = 5
    // rounded components: 0 + 0 + 4 = 4 → residual = 1 → applied to output (largest)
    const result3 = calculateOpenAICost(
      "gpt-4.1-nano",
      null,
      {
        prompt_tokens: 10,
        completion_tokens: 11,
        prompt_tokens_details: { cached_tokens: 7 },
      },
      "req-frac3",
      50,
    );
    expect(result3.costBreakdown).not.toBeNull();
    const bd3 = result3.costBreakdown!;
    expect(bd3.input! + bd3.output! + bd3.cached!).toBe(result3.costMicrodollars);
  });

  it("costBreakdown keys are always in input/output/cached order", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-key-order",
      50,
    );
    expect(Object.keys(result.costBreakdown!)).toEqual(["input", "output", "cached"]);
  });

  it("reasoning is a subset of output cost, not additive", () => {
    const result = calculateOpenAICost(
      "o3-mini",
      null,
      {
        prompt_tokens: 100,
        completion_tokens: 2000,
        completion_tokens_details: { reasoning_tokens: 1500 },
      },
      "req-reason-subset",
      300,
    );

    const bd = result.costBreakdown!;
    expect(bd.reasoning).toBeDefined();
    expect(bd.reasoning!).toBeLessThanOrEqual(bd.output!);
    // Sum of input+cached+output still equals total (reasoning is informational)
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });
});
