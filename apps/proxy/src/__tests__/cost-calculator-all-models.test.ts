import { describe, it, expect } from "vitest";
import { calculateOpenAICost } from "../lib/cost-calculator.js";

/**
 * All OpenAI models from pricing-data.json with rates in integer µ$/MTok.
 * costComponent(tokens, rate) = (tokens * rate) / 1_000_000
 */
const ALL_MODELS = [
  { model: "gpt-4o", input: 2_500_000, cached: 1_250_000, output: 10_000_000 },
  { model: "gpt-4o-mini", input: 150_000, cached: 75_000, output: 600_000 },
  { model: "gpt-4.1", input: 2_000_000, cached: 500_000, output: 8_000_000 },
  { model: "gpt-4.1-mini", input: 400_000, cached: 100_000, output: 1_600_000 },
  { model: "gpt-4.1-nano", input: 100_000, cached: 25_000, output: 400_000 },
  { model: "o4-mini", input: 1_100_000, cached: 275_000, output: 4_400_000 },
  { model: "o3", input: 2_000_000, cached: 500_000, output: 8_000_000 },
  { model: "o3-mini", input: 1_100_000, cached: 550_000, output: 4_400_000 },
  { model: "o1", input: 15_000_000, cached: 7_500_000, output: 60_000_000 },
  { model: "gpt-5", input: 1_250_000, cached: 125_000, output: 10_000_000 },
  { model: "gpt-5-mini", input: 250_000, cached: 25_000, output: 2_000_000 },
  { model: "gpt-5-nano", input: 50_000, cached: 5_000, output: 400_000 },
  { model: "gpt-5.1", input: 1_250_000, cached: 125_000, output: 10_000_000 },
  { model: "gpt-5.2", input: 1_750_000, cached: 175_000, output: 14_000_000 },
] as const;

describe("every OpenAI model: basic cost calculation", () => {
  it.each(ALL_MODELS)(
    "$model — 1000 input, 500 output, 0 cached",
    ({ model, input, output }) => {
      const result = calculateOpenAICost(
        model,
        null,
        {
          prompt_tokens: 1000,
          completion_tokens: 500,
        },
        `req-basic-${model}`,
        100,
      );

      // expected = Math.round((1000 * inputRate + 500 * outputRate) / 1_000_000)
      const expected = Math.round((1000 * input + 500 * output) / 1_000_000);

      expect(result.costMicrodollars).toBe(expected);
      expect(result.model).toBe(model);
      expect(result.provider).toBe("openai");
    },
  );
});

describe("every OpenAI model: cached token cost", () => {
  it.each(ALL_MODELS)(
    "$model — 1000 prompt (200 cached), 500 output",
    ({ model, input, cached, output }) => {
      const result = calculateOpenAICost(
        model,
        null,
        {
          prompt_tokens: 1000,
          completion_tokens: 500,
          prompt_tokens_details: { cached_tokens: 200 },
        },
        `req-cached-${model}`,
        120,
      );

      // normalInput = 1000 - 200 = 800
      // expected = Math.round((800 * inputRate + 200 * cachedRate + 500 * outputRate) / 1_000_000)
      const expected = Math.round((800 * input + 200 * cached + 500 * output) / 1_000_000);

      expect(result.costMicrodollars).toBe(expected);
      expect(result.cachedInputTokens).toBe(200);
    },
  );
});

describe("gpt-5 family specific scenarios", () => {
  it("gpt-5: 10K input, 5K output, 2K cached", () => {
    const result = calculateOpenAICost(
      "gpt-5",
      null,
      {
        prompt_tokens: 10000,
        completion_tokens: 5000,
        prompt_tokens_details: { cached_tokens: 2000 },
      },
      "req-gpt5-large",
      500,
    );

    // normalInput = 10000 - 2000 = 8000
    // input cost:  costComponent(8000, 1_250_000) = 10000
    // cached cost: costComponent(2000, 125_000)   = 250
    // output cost: costComponent(5000, 10_000_000) = 50000
    // total: Math.round(10000 + 250 + 50000) = 60250
    expect(result.costMicrodollars).toBe(60250);
  });

  it("gpt-5-mini: 50K input, 10K output, no cached", () => {
    const result = calculateOpenAICost(
      "gpt-5-mini",
      null,
      {
        prompt_tokens: 50000,
        completion_tokens: 10000,
      },
      "req-gpt5mini-bulk",
      800,
    );

    // input cost:  costComponent(50000, 250_000)   = 12500
    // output cost: costComponent(10000, 2_000_000) = 20000
    // total: Math.round(12500 + 20000) = 32500
    expect(result.costMicrodollars).toBe(32500);
  });

  it("gpt-5-nano: 100K input, 50K output, no cached", () => {
    const result = calculateOpenAICost(
      "gpt-5-nano",
      null,
      {
        prompt_tokens: 100000,
        completion_tokens: 50000,
      },
      "req-gpt5nano-massive",
      1200,
    );

    // input cost:  costComponent(100000, 50_000)  = 5000
    // output cost: costComponent(50000, 400_000)  = 20000
    // total: Math.round(5000 + 20000) = 25000
    expect(result.costMicrodollars).toBe(25000);
  });

  it("gpt-5.1: identical pricing to gpt-5, same result", () => {
    const usage = {
      prompt_tokens: 10000,
      completion_tokens: 5000,
      prompt_tokens_details: { cached_tokens: 2000 },
    };

    const gpt5 = calculateOpenAICost("gpt-5", null, usage, "req-gpt5-cmp", 500);
    const gpt51 = calculateOpenAICost("gpt-5.1", null, usage, "req-gpt51-cmp", 500);

    // Both have input=1_250_000, cached=125_000, output=10_000_000
    // normalInput = costComponent(8000, 1_250_000) = 10000
    // cached = costComponent(2000, 125_000) = 250
    // output = costComponent(5000, 10_000_000) = 50000
    // total = 60250
    expect(gpt5.costMicrodollars).toBe(60250);
    expect(gpt51.costMicrodollars).toBe(60250);
    expect(gpt5.costMicrodollars).toBe(gpt51.costMicrodollars);
  });

  it("gpt-5.2: 1K input, 1K output — higher output rate", () => {
    const result = calculateOpenAICost(
      "gpt-5.2",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 1000,
      },
      "req-gpt52-output",
      200,
    );

    // input cost:  costComponent(1000, 1_750_000)  = 1750
    // output cost: costComponent(1000, 14_000_000) = 14000
    // total: Math.round(1750 + 14000) = 15750
    expect(result.costMicrodollars).toBe(15750);
  });
});

describe("reasoning model scenarios", () => {
  it("o1: 500 input, 10K output (8K reasoning) — reasoning included in completion_tokens", () => {
    const result = calculateOpenAICost(
      "o1",
      null,
      {
        prompt_tokens: 500,
        completion_tokens: 10000,
        completion_tokens_details: { reasoning_tokens: 8000 },
      },
      "req-o1-reasoning",
      2000,
    );

    // reasoning_tokens are part of completion_tokens, billed at the same output rate
    // input cost:  costComponent(500, 15_000_000)   = 7500
    // output cost: costComponent(10000, 60_000_000) = 600000
    // total: Math.round(7500 + 600000) = 607500
    expect(result.costMicrodollars).toBe(607500);
    expect(result.reasoningTokens).toBe(8000);
  });

  it("o3: 1K input, 5K output (4K reasoning)", () => {
    const result = calculateOpenAICost(
      "o3",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 5000,
        completion_tokens_details: { reasoning_tokens: 4000 },
      },
      "req-o3-reasoning",
      1500,
    );

    // input cost:  costComponent(1000, 2_000_000) = 2000
    // output cost: costComponent(5000, 8_000_000) = 40000
    // total: Math.round(2000 + 40000) = 42000
    expect(result.costMicrodollars).toBe(42000);
    expect(result.reasoningTokens).toBe(4000);
  });

  it("o4-mini: 2K input, 3K output (2K reasoning)", () => {
    const result = calculateOpenAICost(
      "o4-mini",
      null,
      {
        prompt_tokens: 2000,
        completion_tokens: 3000,
        completion_tokens_details: { reasoning_tokens: 2000 },
      },
      "req-o4mini-reasoning",
      800,
    );

    // input cost:  costComponent(2000, 1_100_000) = 2200
    // output cost: costComponent(3000, 4_400_000) = 13200
    // total: Math.round(2200 + 13200) = 15400
    expect(result.costMicrodollars).toBe(15400);
    expect(result.reasoningTokens).toBe(2000);
  });

  it("o3-mini: 1K input, 2K output (1K reasoning) — completeness check", () => {
    const result = calculateOpenAICost(
      "o3-mini",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 2000,
        completion_tokens_details: { reasoning_tokens: 1000 },
      },
      "req-o3mini-reasoning",
      600,
    );

    // input cost:  costComponent(1000, 1_100_000) = 1100
    // output cost: costComponent(2000, 4_400_000) = 8800
    // total: Math.round(1100 + 8800) = 9900
    expect(result.costMicrodollars).toBe(9900);
    expect(result.reasoningTokens).toBe(1000);
  });
});

describe("negative token edge cases", () => {
  it("negative prompt_tokens — clamped to 0", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: -100,
        completion_tokens: 500,
      },
      "req-neg-prompt",
      50,
    );

    // promptTokens clamped to 0, normalInput = 0 - 0 = 0
    // output = costComponent(500, 10_000_000) = 5000
    // total: Math.round(0 + 0 + 5000) = 5000
    expect(result.inputTokens).toBe(0);
    expect(result.costMicrodollars).toBe(5000);
  });

  it("negative cached_tokens — clamped to 0, prevents cost inflation", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 500,
        prompt_tokens_details: { cached_tokens: -200 },
      },
      "req-neg-cached",
      50,
    );

    // cachedTokens clamped to 0, normalInput = 1000 - 0 = 1000
    // input = costComponent(1000, 2_500_000) = 2500
    // cached = 0
    // output = costComponent(500, 10_000_000) = 5000
    // total: Math.round(2500 + 0 + 5000) = 7500
    expect(result.cachedInputTokens).toBe(0);
    expect(result.costMicrodollars).toBe(7500);
  });

  it("negative completion_tokens — clamped to 0", () => {
    const result = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: -500,
      },
      "req-neg-completion",
      50,
    );

    // completionTokens clamped to 0, output cost = 0
    // input = costComponent(1000, 2_500_000) = 2500
    // total: Math.round(2500 + 0 + 0) = 2500
    expect(result.outputTokens).toBe(0);
    expect(result.costMicrodollars).toBe(2500);
  });
});
