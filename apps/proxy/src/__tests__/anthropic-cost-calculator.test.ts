import { describe, it, expect } from "vitest";
import { calculateAnthropicCost } from "../lib/anthropic-cost-calculator.js";
import { getModelPricing } from "@nullspend/cost-engine";

describe("calculateAnthropicCost — bug avoidance (AC-1 through AC-7)", () => {
  it("AC-1: no double-counting via OTel normalization (Langfuse #12306)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 5,
        output_tokens: 503,
        cache_creation_input_tokens: 1253,
        cache_read_input_tokens: 128955,
      },
      null,
      "req-ac1",
      200,
    );

    expect(result.inputTokens).toBe(5 + 1253 + 128955);
    expect(result.costMicrodollars).toBe(50945);
    expect(result.costBreakdown).not.toBeNull();
    expect(result.costBreakdown!.input! + result.costBreakdown!.cached! + result.costBreakdown!.output!).toBe(50945);
  });

  it("AC-2: cache write uses independent rate, not base+premium (LiteLLM #6575)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 3,
        output_tokens: 550,
        cache_creation_input_tokens: 12304,
        cache_read_input_tokens: 0,
      },
      null,
      "req-ac2",
      150,
    );

    // 3*3.00 + 12304*3.75 + 0*0.30 + 550*15.00
    // = 9 + 46140 + 0 + 8250 = 54399
    expect(result.costMicrodollars).toBe(54399);
  });

  it("AC-3: cache costs NOT omitted (LiteLLM #5443)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 100,
        output_tokens: 200,
        cache_creation_input_tokens: 5000,
        cache_read_input_tokens: 50000,
      },
      null,
      "req-ac3",
      100,
    );

    // 100*3.00 + 5000*3.75 + 50000*0.30 + 200*15.00
    // = 300 + 18750 + 15000 + 3000 = 37050
    expect(result.costMicrodollars).toBe(37050);
    expect(result.costMicrodollars).toBeGreaterThan(
      Math.round(100 * 3.0 + 200 * 15.0),
    );
  });

  it("AC-4: streaming and non-streaming produce identical costs", () => {
    const usage = {
      input_tokens: 500,
      output_tokens: 300,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 10000,
    };

    const nonStreaming = calculateAnthropicCost(
      "claude-sonnet-4-6", null, usage, null, "req-ac4a", 100,
    );
    const streaming = calculateAnthropicCost(
      "claude-sonnet-4-6", null, usage, null, "req-ac4b", 200,
    );

    expect(nonStreaming.costMicrodollars).toBe(streaming.costMicrodollars);
  });

  it("AC-5a: 5-min and 1-hour TTL cache writes use different rates", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 10,
        output_tokens: 200,
        cache_creation_input_tokens: 556,
        cache_read_input_tokens: 0,
      },
      { ephemeral_5m_input_tokens: 456, ephemeral_1h_input_tokens: 100 },
      "req-ac5a",
      100,
    );

    // 10*3.00 + 456*3.75 + 100*6.00 + 0*0.30 + 200*15.00
    // = 30 + 1710 + 600 + 0 + 3000 = 5340
    expect(result.costMicrodollars).toBe(5340);
  });

  it("AC-5b: fallback when TTL breakdown absent (all assumed 5-min)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 10,
        output_tokens: 200,
        cache_creation_input_tokens: 556,
        cache_read_input_tokens: 0,
      },
      null,
      "req-ac5b",
      100,
    );

    // 10*3.00 + 556*3.75 + 0*0.30 + 200*15.00
    // = 30 + 2085 + 0 + 3000 = 5115
    expect(result.costMicrodollars).toBe(5115);
  });

  it("AC-6a: long context (>200K) applies 2x input, 1.5x output", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 5000,
        output_tokens: 1000,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 196000,
      },
      null,
      "req-ac6a",
      500,
    );

    // total input: 5000+0+196000 = 201000 > 200K → long context
    // 5000*6.00 + 0 + 196000*0.60 + 1000*22.50
    // = 30000 + 0 + 117600 + 22500 = 170100
    expect(result.costMicrodollars).toBe(170100);
  });

  it("AC-6b: exactly 200K total input uses base rates (threshold is >, not >=)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 5000,
        output_tokens: 1000,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 195000,
      },
      null,
      "req-ac6b",
      500,
    );

    // total input: 5000+0+195000 = 200000 — NOT > 200000, so base rates
    // 5000*3.00 + 0 + 195000*0.30 + 1000*15.00
    // = 15000 + 0 + 58500 + 15000 = 88500
    expect(result.costMicrodollars).toBe(88500);
  });

  it("AC-6c: just under 200K (199,999) uses base rates", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 5000,
        output_tokens: 1000,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 194999,
      },
      null,
      "req-ac6c",
      500,
    );

    // total input: 5000+0+194999 = 199999 — base rates
    // 5000*3.00 + 194999*0.30 + 1000*15.00
    // = 15000 + 58499.7 + 15000 = 88499.7 → 88500
    expect(result.costMicrodollars).toBe(88500);
  });

  it("AC-6d: total input includes cache tokens for threshold check", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 1000,
        output_tokens: 500,
        cache_creation_input_tokens: 100000,
        cache_read_input_tokens: 100000,
      },
      null,
      "req-ac6d",
      500,
    );

    // total: 1000+100000+100000 = 201000 > 200K → long context rates
    expect(result.inputTokens).toBe(201000);
    const baseCost = Math.round(
      1000 * 3.0 + 100000 * 3.75 + 100000 * 0.30 + 500 * 15.0,
    );
    expect(result.costMicrodollars).toBeGreaterThan(baseCost);
  });

  it("AC-7: extended thinking doesn't inflate output cost", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 5000 },
      null,
      "req-ac7",
      300,
    );

    // 100*3.00 + 5000*15.00 = 300 + 75000 = 75300
    expect(result.costMicrodollars).toBe(75300);
    expect(result.reasoningTokens).toBe(0);
  });
});

describe("calculateAnthropicCost — edge cases", () => {
  it("zero tokens produces cost 0", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 },
      null,
      "req-zero",
      10,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
  });

  it("missing cache fields (undefined) → cost based on input + output only", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-no-cache",
      50,
    );

    // 100*3.00 + 50*15.00 = 300 + 750 = 1050
    expect(result.costMicrodollars).toBe(1050);
    expect(result.cachedInputTokens).toBe(0);
  });

  it("very large token counts (128K context) produce finite result", () => {
    const result = calculateAnthropicCost(
      "claude-opus-4-6",
      null,
      { input_tokens: 128000, output_tokens: 64000 },
      null,
      "req-large",
      5000,
    );

    // 128000*5.00 + 64000*25.00 = 640000 + 1600000 = 2240000
    expect(result.costMicrodollars).toBe(2240000);
    expect(Number.isFinite(result.costMicrodollars)).toBe(true);
  });

  it("unknown model returns cost 0 with other fields populated", () => {
    const result = calculateAnthropicCost(
      "nonexistent-model",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-unknown",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeNull();
    expect(result.model).toBe("nonexistent-model");
    expect(result.inputTokens).toBe(100);
    expect(result.outputTokens).toBe(50);
    expect(result.provider).toBe("anthropic");
  });

  it("model alias fallback (requestModel fails, responseModel succeeds)", () => {
    const result = calculateAnthropicCost(
      "claude-3-5-haiku-latest",
      "claude-3-5-haiku-20241022",
      { input_tokens: 1000, output_tokens: 500 },
      null,
      "req-fallback",
      100,
    );

    expect(result.model).toBe("claude-3-5-haiku-20241022");
    expect(result.costMicrodollars).toBeGreaterThan(0);
    // 1000*0.80 + 500*4.00 = 800 + 2000 = 2800
    expect(result.costMicrodollars).toBe(2800);
  });

  it("all-cached input (input_tokens=0, only cache reads)", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 0, output_tokens: 100, cache_read_input_tokens: 50000 },
      null,
      "req-all-cached",
      100,
    );

    // 0 + 50000*0.30 + 100*15.00 = 0 + 15000 + 1500 = 16500
    expect(result.costMicrodollars).toBe(16500);
    expect(result.inputTokens).toBe(50000);
  });

  it("pure cache write (no reads, no output)", () => {
    const result = calculateAnthropicCost(
      "claude-haiku-3.5",
      null,
      { input_tokens: 100, output_tokens: 0, cache_creation_input_tokens: 50000 },
      null,
      "req-pure-write",
      100,
    );

    // 100*0.80 + 50000*1.00 + 0 + 0 = 80 + 50000 = 50080
    expect(result.costMicrodollars).toBe(50080);
  });

  it("attribution fields passed through correctly", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-attr",
      100,
      { userId: "user-123", apiKeyId: "key-456", actionId: "act-789" },
    );

    expect(result.userId).toBe("user-123");
    expect(result.apiKeyId).toBe("key-456");
    expect(result.actionId).toBe("act-789");
  });

  it("attribution fields default to null when omitted", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-no-attr",
      100,
    );

    expect(result.userId).toBeNull();
    expect(result.apiKeyId).toBeNull();
    expect(result.actionId).toBeNull();
  });

  it("Haiku 4.5 pricing verification", () => {
    const result = calculateAnthropicCost(
      "claude-haiku-4-5",
      null,
      { input_tokens: 1000, output_tokens: 500 },
      null,
      "req-haiku45",
      100,
    );

    // 1000*1.00 + 500*5.00 = 1000 + 2500 = 3500
    expect(result.costMicrodollars).toBe(3500);
    expect(result.costBreakdown).toEqual({ input: 1000, cached: 0, output: 2500 });
  });
});

describe("calculateAnthropicCost costBreakdown invariants", () => {
  it("costBreakdown is null when model is unknown", () => {
    const result = calculateAnthropicCost(
      "nonexistent-model",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-null-bd",
      50,
    );
    expect(result.costBreakdown).toBeNull();
    expect(result.costMicrodollars).toBe(0);
  });

  it("costBreakdown components sum exactly to costMicrodollars", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      {
        input_tokens: 500,
        output_tokens: 300,
        cache_creation_input_tokens: 2000,
        cache_read_input_tokens: 10000,
      },
      null,
      "req-sum-check",
      100,
    );

    expect(result.costBreakdown).not.toBeNull();
    const bd = result.costBreakdown!;
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });

  it("residual distribution preserves exact sum with fractional components", () => {
    // claude-haiku-3: in=0.25, cached=0.03, w5m=0.30, out=1.25
    // 3 input * 0.25 = 0.75, cache_write 7 * 0.30 = 2.1, cache_read 11 * 0.03 = 0.33, 5 output * 1.25 = 6.25
    // cached = 2.1 + 0.33 = 2.43
    // total = Math.round(0.75 + 2.43 + 6.25) = Math.round(9.43) = 9
    // rounded: 1 + 2 + 6 = 9 → residual = 0
    const result = calculateAnthropicCost(
      "claude-haiku-3",
      null,
      {
        input_tokens: 3,
        output_tokens: 5,
        cache_creation_input_tokens: 7,
        cache_read_input_tokens: 11,
      },
      null,
      "req-frac-anthropic",
      50,
    );
    expect(result.costBreakdown).not.toBeNull();
    const bd = result.costBreakdown!;
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });

  it("costBreakdown keys are always in input/output/cached order", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 50 },
      null,
      "req-key-order",
      50,
    );
    expect(Object.keys(result.costBreakdown!)).toEqual(["input", "output", "cached"]);
  });

  it("reasoning is never present for Anthropic", () => {
    const result = calculateAnthropicCost(
      "claude-sonnet-4-6",
      null,
      { input_tokens: 100, output_tokens: 5000 },
      null,
      "req-no-reasoning",
      300,
    );

    expect(result.costBreakdown).not.toBeNull();
    expect(result.costBreakdown!.reasoning).toBeUndefined();
  });
});

describe("calculateAnthropicCost — multi-model pricing verification", () => {
  const models: [string, string][] = [
    ["anthropic", "claude-sonnet-4-6"],
    ["anthropic", "claude-haiku-3.5"],
    ["anthropic", "claude-opus-4"],
    ["anthropic", "claude-opus-4-6"],
    ["anthropic", "claude-sonnet-4-5"],
    ["anthropic", "claude-opus-4-5"],
    ["anthropic", "claude-opus-4-1"],
    ["anthropic", "claude-sonnet-4"],
    ["anthropic", "claude-haiku-4-5"],
    ["anthropic", "claude-haiku-3"],
  ];

  for (const [provider, model] of models) {
    it(`${model}: 1K input + 500 output produces correct cost`, () => {
      const pricing = getModelPricing(provider, model)!;
      expect(pricing).not.toBeNull();

      const result = calculateAnthropicCost(
        model,
        null,
        { input_tokens: 1000, output_tokens: 500 },
        null,
        `req-${model}`,
        100,
      );

      const expectedCost = Math.round(
        1000 * pricing.inputPerMTok + 500 * pricing.outputPerMTok,
      );
      expect(result.costMicrodollars).toBe(expectedCost);
      expect(result.costMicrodollars).toBeGreaterThan(0);
    });
  }
});
