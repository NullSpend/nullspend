import { describe, it, expect } from "vitest";
import { calculateGeminiCost } from "../lib/gemini-cost-calculator.js";

describe("calculateGeminiCost — basic cost calculations", () => {
  it("gemini-2.5-flash basic cost (no cache)", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-flash-basic",
      100,
    );

    expect(result.provider).toBe("google");
    expect(result.model).toBe("gemini-2.5-flash");
    expect(result.inputTokens).toBe(1000);
    expect(result.outputTokens).toBe(500);
    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
    expect(result.requestId).toBe("req-flash-basic");
    expect(result.durationMs).toBe(100);
    expect(result.userId).toBeNull();
    expect(result.apiKeyId).toBeNull();
    expect(result.actionId).toBeNull();

    // Cost breakdown:
    // input: 1000 * 300000 / 1_000_000 = 150
    // output: 500 * 2500000 / 1_000_000 = 300
    // total: 450 microdollars
    expect(result.costMicrodollars).toBe(1550);
    expect(result.costBreakdown).toEqual({ input: 300, output: 1250, cached: 0 });
  });

  it("gemini-2.5-pro basic cost (no cache)", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-pro",
      null,
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-pro-basic",
      200,
    );

    expect(result.provider).toBe("google");
    expect(result.model).toBe("gemini-2.5-pro");
    expect(result.inputTokens).toBe(1000);
    expect(result.outputTokens).toBe(500);

    // Cost breakdown:
    // input: 1000 * 1250000 / 1_000_000 = 1250
    // output: 500 * 10000000 / 1_000_000 = 5000
    // total: 6250 microdollars
    expect(result.costMicrodollars).toBe(6250);
    expect(result.costBreakdown).toEqual({ input: 1250, output: 5000, cached: 0 });
  });

  it("cached input tokens use cachedInputPerMTok for cached portion", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 1000,
        candidatesTokenCount: 500,
        cachedContentTokenCount: 300,
      },
      "req-cached",
      150,
    );

    expect(result.inputTokens).toBe(1000);
    expect(result.cachedInputTokens).toBe(300);

    // Cost breakdown:
    // normalInput: 1000 - 300 = 700
    // input: 700 * 300000 / 1_000_000 = 105
    // cached: 300 * 30000 / 1_000_000 = 11.25 → round(11.25) = 11
    // output: 500 * 2500000 / 1_000_000 = 300
    // total: Math.round(105 + 11.25 + 300) = Math.round(416.25) = 416
    // rounded components: 105 + 11 + 300 = 416 → residual = 0
    expect(result.costMicrodollars).toBe(1469);
    expect(result.costBreakdown).toEqual({ input: 210, cached: 9, output: 1250 });
  });
});

describe("calculateGeminiCost — thinking tokens (informational)", () => {
  it("thoughtsTokenCount appears in costBreakdown.reasoning but does NOT increase total", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 1000,
        candidatesTokenCount: 500,
        thoughtsTokenCount: 200,
      },
      "req-thinking",
      120,
    );

    expect(result.reasoningTokens).toBe(200);

    // Thinking tokens are a SUBSET of candidatesTokenCount — same output rate, NOT additive
    // input: 1000 * 300000 / 1_000_000 = 150
    // output: 500 * 2500000 / 1_000_000 = 300
    // total: 450 (same as without thinking tokens)
    expect(result.costMicrodollars).toBe(1550);

    expect(result.costBreakdown).not.toBeNull();
    // reasoning: 200 * 2500000 / 1_000_000 = 120
    expect(result.costBreakdown!.reasoning).toBe(500);

    // Sum of input+cached+output still equals total (reasoning is informational)
    expect(result.costBreakdown!.input! + result.costBreakdown!.cached! + result.costBreakdown!.output!).toBe(
      result.costMicrodollars,
    );
  });

  it("zero thoughtsTokenCount omits reasoning from breakdown", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 100,
        candidatesTokenCount: 50,
        thoughtsTokenCount: 0,
      },
      "req-no-thinking",
      80,
    );

    expect(result.reasoningTokens).toBe(0);
    expect(result.costBreakdown).not.toBeNull();
    expect(result.costBreakdown!.reasoning).toBeUndefined();
  });
});

describe("calculateGeminiCost — unknown model and fallback", () => {
  it("unknown model returns costMicrodollars: 0, costBreakdown: null", () => {
    const result = calculateGeminiCost(
      "unknown-gemini-model",
      null,
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-unknown",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeNull();
    expect(result.model).toBe("unknown-gemini-model");
    expect(result.inputTokens).toBe(1000);
    expect(result.outputTokens).toBe(500);
    expect(result.provider).toBe("google");
  });

  it("dated alias resolves via prefix fallback — no responseModel needed", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash-preview-04-17",
      null,
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-alias",
      100,
    );

    // "gemini-2.5-flash-preview-04-17" resolves to "gemini-2.5-flash" via prefix
    expect(result.model).toBe("gemini-2.5-flash");
    expect(result.costMicrodollars).toBeGreaterThan(0);
    // input: 1000 * 300000 / 1_000_000 = 150
    // output: 500 * 2500000 / 1_000_000 = 300
    expect(result.costMicrodollars).toBe(1550);
  });

  it("response model fallback — requestModel fully unknown, responseModel known", () => {
    const result = calculateGeminiCost(
      "completely-unknown-model",
      "gemini-2.5-flash",
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-fallback",
      100,
    );

    // "completely-unknown-model" has no prefix match either, falls back to responseModel
    expect(result.model).toBe("gemini-2.5-flash");
    expect(result.costMicrodollars).toBe(1550);
  });

  it("both models unknown → cost 0", () => {
    const result = calculateGeminiCost(
      "unknown-model-a",
      "unknown-model-b",
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-both-unknown",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeNull();
    expect(result.model).toBe("unknown-model-a");
  });
});

describe("calculateGeminiCost — edge cases", () => {
  it("zero tokens → cost 0", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 0,
        candidatesTokenCount: 0,
        cachedContentTokenCount: 0,
        thoughtsTokenCount: 0,
      },
      "req-zero",
      10,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
  });

  it("null usage → all tokens 0, cost 0", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      null,
      "req-null-usage",
      50,
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
    // Known model with 0 tokens still produces a breakdown (all zeros)
    expect(result.costBreakdown).toEqual({ input: 0, output: 0, cached: 0 });
  });

  it("negative token values clamped to 0", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: -500,
        candidatesTokenCount: -200,
        cachedContentTokenCount: -100,
        thoughtsTokenCount: -50,
      },
      "req-negative",
      30,
    );

    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
    expect(result.costMicrodollars).toBe(0);
  });

  it("missing usage fields (undefined) → defaults to 0", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-pro",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 } as any,
      "req-partial",
      75,
    );

    expect(result.cachedInputTokens).toBe(0);
    expect(result.reasoningTokens).toBe(0);
    // input: 100 * 1250000 / 1_000_000 = 125
    // output: 50 * 10000000 / 1_000_000 = 500
    // total: 625
    expect(result.costMicrodollars).toBe(625);
  });
});

describe("calculateGeminiCost — adversarial inputs", () => {
  it("Infinity token counts produce finite cost (not NaN/Infinity)", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: Infinity,
        candidatesTokenCount: Infinity,
        cachedContentTokenCount: Infinity,
        thoughtsTokenCount: Infinity,
      },
      "req-infinity",
      100,
    );
    expect(Number.isFinite(result.costMicrodollars)).toBe(true);
    expect(result.inputTokens).toBe(0); // Infinity clamped to 0
    expect(result.outputTokens).toBe(0);
  });

  it("string token counts produce finite cost", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: "abc" as unknown as number,
        candidatesTokenCount: "def" as unknown as number,
      },
      "req-string",
      100,
    );
    expect(Number.isFinite(result.costMicrodollars)).toBe(true);
    expect(result.costMicrodollars).toBe(0);
  });

  it("cachedContentTokenCount clamped to promptTokenCount", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 100,
        candidatesTokenCount: 50,
        cachedContentTokenCount: 500, // exceeds prompt
      },
      "req-cached-exceeds",
      100,
    );
    // cached should be clamped to 100 (not 500)
    expect(result.cachedInputTokens).toBe(100);
    // normalInput = 100 - 100 = 0, so input cost is 0
    // total cost is just output: 50 * 600_000 / 1M = 30
    // plus cached: 100 * 37_500 / 1M = 3.75 → round = 4
    // total = 30 + 4 = 34 (approximately, depends on rounding)
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(Number.isFinite(result.costMicrodollars)).toBe(true);
  });
});

describe("calculateGeminiCost — tool definition tokens", () => {
  it("toolDefinitionTokens appears in costBreakdown.toolDefinition", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 1000, candidatesTokenCount: 500 },
      "req-tools",
      100,
      undefined,
      100,
    );

    expect(result.costBreakdown).not.toBeNull();
    // toolDefinition: 100 * 300000 / 1_000_000 = 15
    expect(result.costBreakdown!.toolDefinition).toBe(30);

    // Tool definition is a SUBSET of input cost, NOT additive
    // Total unchanged: 150 + 300 = 450
    expect(result.costMicrodollars).toBe(1550);
    expect(result.costBreakdown!.input! + result.costBreakdown!.cached! + result.costBreakdown!.output!).toBe(
      result.costMicrodollars,
    );
  });

  it("zero toolDefinitionTokens omits toolDefinition from breakdown", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-no-tools",
      50,
      undefined,
      0,
    );

    expect(result.costBreakdown).not.toBeNull();
    expect(result.costBreakdown!.toolDefinition).toBeUndefined();
  });
});

describe("calculateGeminiCost — attribution", () => {
  it("includes attribution fields when provided", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-attr",
      100,
      { userId: "user-123", apiKeyId: "key-456", actionId: "act-789" },
    );

    expect(result.userId).toBe("user-123");
    expect(result.apiKeyId).toBe("key-456");
    expect(result.actionId).toBe("act-789");
  });

  it("attribution fields default to null when omitted", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-no-attr",
      100,
    );

    expect(result.userId).toBeNull();
    expect(result.apiKeyId).toBeNull();
    expect(result.actionId).toBeNull();
  });

  it("attribution with null actionId passes through correctly", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-null-action",
      100,
      { userId: "user-1", apiKeyId: "key-1", actionId: null },
    );

    expect(result.userId).toBe("user-1");
    expect(result.apiKeyId).toBe("key-1");
    expect(result.actionId).toBeNull();
  });
});

describe("calculateGeminiCost — costBreakdown invariants", () => {
  it("costBreakdown is null when model is unknown", () => {
    const result = calculateGeminiCost(
      "nonexistent-model",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-null-bd",
      50,
    );
    expect(result.costBreakdown).toBeNull();
    expect(result.costMicrodollars).toBe(0);
  });

  it("costBreakdown components sum exactly to costMicrodollars", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 1000,
        candidatesTokenCount: 500,
        cachedContentTokenCount: 300,
      },
      "req-sum-check",
      100,
    );

    expect(result.costBreakdown).not.toBeNull();
    const bd = result.costBreakdown!;
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });

  it("costBreakdown keys are always in input/output/cached order", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-key-order",
      50,
    );
    expect(Object.keys(result.costBreakdown!)).toEqual(["input", "output", "cached"]);
  });

  it("reasoning is a subset of output cost, not additive", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-pro",
      null,
      {
        promptTokenCount: 100,
        candidatesTokenCount: 2000,
        thoughtsTokenCount: 1500,
      },
      "req-reason-subset",
      300,
    );

    const bd = result.costBreakdown!;
    expect(bd.reasoning).toBeDefined();
    expect(bd.reasoning!).toBeLessThanOrEqual(bd.output!);
    // reasoning: 1500 * 10000000 / 1_000_000 = 15000
    // output: 2000 * 10000000 / 1_000_000 = 20000
    expect(bd.reasoning).toBe(15000);
    expect(bd.output).toBe(20000);
    // Sum of input+cached+output still equals total (reasoning is informational)
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });

  it("residual distribution preserves exact sum with fractional components", () => {
    // gemini-2.5-flash: input 300000, cached 30000, output 2500000
    // 3 input * 300000 / 1_000_000 = 0.45
    // cached 7 * 30000 / 1_000_000 = 0.2625
    // output 11 * 2500000 / 1_000_000 = 6.6
    // total = Math.round(0.45 + 0.2625 + 6.6) = Math.round(7.3125) = 7
    // rounded: 0 + 0 + 7 = 7 → residual = 0
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      {
        promptTokenCount: 10,
        candidatesTokenCount: 11,
        cachedContentTokenCount: 7,
      },
      "req-frac",
      50,
    );
    expect(result.costBreakdown).not.toBeNull();
    const bd = result.costBreakdown!;
    expect(bd.input! + bd.output! + bd.cached!).toBe(result.costMicrodollars);
  });
});

describe("calculateGeminiCost — eventType", () => {
  it("eventType is always 'llm'", () => {
    const result = calculateGeminiCost(
      "gemini-2.5-flash",
      null,
      { promptTokenCount: 100, candidatesTokenCount: 50 },
      "req-event-type",
      50,
    );

    expect(result.eventType).toBe("llm");
  });
});
