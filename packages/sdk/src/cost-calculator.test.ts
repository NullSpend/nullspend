import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  calculateOpenAICostEvent,
  calculateAnthropicCostEvent,
} from "./cost-calculator.js";
import { getModelPricing, costComponent } from "@nullspend/cost-engine";

vi.mock("@nullspend/cost-engine", () => ({
  getModelPricing: vi.fn(),
  costComponent: vi.fn((tokens: number, rate: number) => {
    if (tokens <= 0 || rate <= 0) return 0;
    return tokens * rate;
  }),
}));

const mockedGetModelPricing = vi.mocked(getModelPricing);

// ---------------------------------------------------------------------------
// OpenAI cost calculation
// ---------------------------------------------------------------------------

describe("calculateOpenAICostEvent — OpenAI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calculates correct cost for a known model with pricing", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 2.5,
      outputPerMTok: 10,
      cachedInputPerMTok: 1.25,
    } as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      { prompt_tokens: 1000, completion_tokens: 500 },
      150,
      {},
    );

    expect(result.provider).toBe("openai");
    expect(result.model).toBe("gpt-4o");
    expect(result.inputTokens).toBe(1000);
    expect(result.outputTokens).toBe(500);
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(result.costBreakdown).toBeDefined();
    expect(result.costBreakdown!.input).toBeGreaterThan(0);
    expect(result.costBreakdown!.output).toBeGreaterThan(0);
    expect(result.durationMs).toBe(150);
    expect(result.eventType).toBe("llm");
  });

  it("returns 0 cost for an unknown model", () => {
    mockedGetModelPricing.mockReturnValue(null as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "unknown-model",
      { prompt_tokens: 100, completion_tokens: 50 },
      100,
      {},
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeUndefined();
    expect(result.inputTokens).toBe(100);
    expect(result.outputTokens).toBe(50);
  });

  it("reduces input cost with cached tokens", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 2.5,
      outputPerMTok: 10,
      cachedInputPerMTok: 1.25,
    } as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      {
        prompt_tokens: 1000,
        completion_tokens: 100,
        prompt_tokens_details: { cached_tokens: 800 },
      },
      100,
      {},
    );

    // normalInputTokens = 1000 - 800 = 200
    // costComponent called with (200, 2.5), (800, 1.25), (100, 10)
    expect(result.cachedInputTokens).toBe(800);
    expect(result.costBreakdown).toBeDefined();
    expect(result.costBreakdown!.cached).toBeGreaterThan(0);
  });

  it("tracks reasoning tokens as subset of output", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 15,
      outputPerMTok: 60,
      cachedInputPerMTok: 7.5,
    } as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "o1-preview",
      {
        prompt_tokens: 500,
        completion_tokens: 2000,
        completion_tokens_details: { reasoning_tokens: 1500 },
      },
      200,
      {},
    );

    expect(result.reasoningTokens).toBe(1500);
    expect(result.costBreakdown!.reasoning).toBeDefined();
    expect(result.costBreakdown!.reasoning).toBeGreaterThan(0);
  });

  it("handles zero tokens with zero cost", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 2.5,
      outputPerMTok: 10,
      cachedInputPerMTok: 1.25,
    } as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      { prompt_tokens: 0, completion_tokens: 0 },
      50,
      {},
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
  });

  it("passes through metadata (sessionId, traceId, tags)", () => {
    mockedGetModelPricing.mockReturnValue(null as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      { prompt_tokens: 10, completion_tokens: 5 },
      100,
      {
        sessionId: "sess-123",
        traceId: "trace-abc",
        tags: { env: "prod", team: "ml" },
      },
    );

    expect(result.sessionId).toBe("sess-123");
    expect(result.traceId).toBe("trace-abc");
    expect(result.tags).toEqual({ env: "prod", team: "ml" });
  });

  it("distributes rounding residual to largest component", () => {
    // Construct a scenario where rounding creates a residual
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 7,
      cachedInputPerMTok: 1,
    } as ReturnType<typeof getModelPricing>);

    // costComponent mock: input=300*3=900, cached=0, output=100*7=700
    // total = 1600, round(1600)=1600
    // round(900)=900, round(0)=0, round(700)=700 => sum=1600, residual=0
    // Let's use values that produce a residual by overriding costComponent
    vi.mocked(costComponent)
      .mockReturnValueOnce(100.4) // input
      .mockReturnValueOnce(0)     // cached
      .mockReturnValueOnce(200.3); // output
    // total = round(100.4 + 0 + 200.3) = round(300.7) = 301
    // roundedInput=100, roundedCached=0, roundedOutput=200 => sum=300
    // residual = 301 - 300 = 1 => added to output (largest)

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      { prompt_tokens: 100, completion_tokens: 50 },
      100,
      {},
    );

    expect(result.costMicrodollars).toBe(301);
    expect(result.costBreakdown!.output).toBe(201); // 200 + 1 residual
    expect(result.costBreakdown!.input).toBe(100);
    expect(result.costBreakdown!.cached).toBe(0);
  });

  it("does not include reasoning in costBreakdown when reasoningTokens is 0", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 2.5,
      outputPerMTok: 10,
      cachedInputPerMTok: 1.25,
    } as ReturnType<typeof getModelPricing>);

    const result = calculateOpenAICostEvent(
      "gpt-4o",
      { prompt_tokens: 100, completion_tokens: 50 },
      100,
      {},
    );

    expect(result.reasoningTokens).toBe(0);
    expect(result.costBreakdown!.reasoning).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Anthropic cost calculation
// ---------------------------------------------------------------------------

describe("calculateAnthropicCostEvent — Anthropic", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calculates correct cost for a known model", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 15,
      cachedInputPerMTok: 0.3,
      cacheWrite5mPerMTok: 3.75,
      cacheWrite1hPerMTok: undefined,
    } as unknown as ReturnType<typeof getModelPricing>);

    const result = calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      { input_tokens: 1000, output_tokens: 200 },
      null,
      150,
      {},
    );

    expect(result.provider).toBe("anthropic");
    expect(result.model).toBe("claude-sonnet-4-20250514");
    expect(result.inputTokens).toBe(1000); // totalInputTokens = 1000 + 0 + 0
    expect(result.outputTokens).toBe(200);
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(result.eventType).toBe("llm");
    expect(result.reasoningTokens).toBe(0);
  });

  it("calculates cache write cost with TTL detail (ephemeral breakdown)", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 15,
      cachedInputPerMTok: 0.3,
      cacheWrite5mPerMTok: 3.75,
      cacheWrite1hPerMTok: 30,
    } as unknown as ReturnType<typeof getModelPricing>);

    const usage = {
      input_tokens: 500,
      output_tokens: 100,
      cache_creation_input_tokens: 200,
      cache_read_input_tokens: 50,
    };
    const cacheDetail = {
      ephemeral_5m_input_tokens: 150,
      ephemeral_1h_input_tokens: 50,
    };

    const result = calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      usage,
      cacheDetail,
      100,
      {},
    );

    // totalInputTokens = 500 + 200 + 50 = 750, not > 200K
    expect(result.inputTokens).toBe(750);
    expect(result.cachedInputTokens).toBe(50); // cacheReadTokens
    expect(result.costMicrodollars).toBeGreaterThan(0);
    expect(result.costBreakdown).toBeDefined();
  });

  it("defaults to 5m rate when cache detail is absent", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 15,
      cachedInputPerMTok: 0.3,
      cacheWrite5mPerMTok: 3.75,
      cacheWrite1hPerMTok: 30,
    } as unknown as ReturnType<typeof getModelPricing>);

    const usage = {
      input_tokens: 500,
      output_tokens: 100,
      cache_creation_input_tokens: 200,
    };

    const result = calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      usage,
      null, // no detail
      100,
      {},
    );

    // Without detail, uses cacheWrite5mRate for all cache_creation_input_tokens
    expect(result.costMicrodollars).toBeGreaterThan(0);
    // costComponent should have been called with (200, 3.75) for cache write
    expect(costComponent).toHaveBeenCalledWith(200, 3.75);
  });

  it("doubles input rates and 1.5x output for long context (>200K tokens)", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 15,
      cachedInputPerMTok: 0.3,
      cacheWrite5mPerMTok: 3.75,
      cacheWrite1hPerMTok: 30,
    } as unknown as ReturnType<typeof getModelPricing>);

    const usage = {
      input_tokens: 210_000,
      output_tokens: 1000,
    };

    calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      usage,
      null,
      200,
      {},
    );

    // totalInputTokens = 210000, > 200K so long context
    // inputRate = 3 * 2 = 6, outputRate = 15 * 1.5 = 22.5
    expect(costComponent).toHaveBeenCalledWith(210_000, 6); // doubled input
    expect(costComponent).toHaveBeenCalledWith(1000, 22.5); // 1.5x output
  });

  it("handles zero tokens with zero cost", () => {
    mockedGetModelPricing.mockReturnValue({
      inputPerMTok: 3,
      outputPerMTok: 15,
      cachedInputPerMTok: 0.3,
    } as unknown as ReturnType<typeof getModelPricing>);

    const result = calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      { input_tokens: 0, output_tokens: 0 },
      null,
      50,
      {},
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
  });

  it("passes through metadata (sessionId, traceId, tags)", () => {
    mockedGetModelPricing.mockReturnValue(null as ReturnType<typeof getModelPricing>);

    const result = calculateAnthropicCostEvent(
      "claude-sonnet-4-20250514",
      { input_tokens: 10, output_tokens: 5 },
      null,
      100,
      {
        sessionId: "sess-456",
        traceId: "trace-def",
        tags: { environment: "staging" },
      },
    );

    expect(result.sessionId).toBe("sess-456");
    expect(result.traceId).toBe("trace-def");
    expect(result.tags).toEqual({ environment: "staging" });
  });

  it("returns 0 cost for unknown model", () => {
    mockedGetModelPricing.mockReturnValue(null as ReturnType<typeof getModelPricing>);

    const result = calculateAnthropicCostEvent(
      "unknown-model",
      { input_tokens: 100, output_tokens: 50 },
      null,
      100,
      {},
    );

    expect(result.costMicrodollars).toBe(0);
    expect(result.costBreakdown).toBeUndefined();
  });
});
