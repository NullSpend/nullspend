/**
 * Regression tests that simulate realistic OpenAI response patterns
 * end-to-end through the SSE parser and cost calculator.
 *
 * Tests cover:
 *   - Tool calling / function calling SSE chunks
 *   - Multiple choices (n > 1) SSE interleaving
 *   - Reasoning model usage (o3-mini reasoning_tokens)
 *   - Cached token cost adjustment
 *   - Model alias resolution
 *   - Zero-token edge cases
 *   - Extremely large token counts
 *   - Unicode content in SSE chunks
 *   - Malformed/partial upstream responses
 */
import { describe, it, expect } from "vitest";
import { createSSEParser } from "../lib/sse-parser.js";
import { calculateOpenAICost } from "../lib/cost-calculator.js";

function makeStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

async function drainStream(readable: ReadableStream<Uint8Array>): Promise<string> {
  const reader = readable.getReader();
  const decoder = new TextDecoder();
  let result = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
  }
  result += decoder.decode();
  return result;
}

describe("Tool calling SSE format", () => {
  it("extracts usage from tool call streaming response with function arguments", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"lo"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cation\\": \\"San Francisco\\"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
      'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":82,"completion_tokens":17,"total_tokens":99}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.model).toBe("gpt-4o-mini");
    expect(result.usage).not.toBeNull();
    expect(result.usage!.prompt_tokens).toBe(82);
    expect(result.usage!.completion_tokens).toBe(17);
    expect(result.toolCalls).toEqual([{ name: "get_weather", id: "call_xyz" }]);

    // Pass through cost calculator
    const cost = calculateOpenAICost("gpt-4o-mini", result.model, result.usage!, "req-tc", 150);
    expect(cost.inputTokens).toBe(82);
    expect(cost.outputTokens).toBe(17);
    // 82 * 0.15 + 17 * 0.60 = 12.3 + 10.2 = 22.5 → 23 microdollars
    expect(cost.costMicrodollars).toBe(23);
  });

  it("handles multiple parallel tool calls in a single response", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-multi","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-multi","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\": \\"NY\\"}"}},{"index":1,"function":{"arguments":"{\\"tz\\": \\"EST\\"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-multi","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
      'data: {"id":"chatcmpl-multi","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":150,"completion_tokens":40,"total_tokens":190}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage!.prompt_tokens).toBe(150);
    expect(result.usage!.completion_tokens).toBe(40);
    expect(result.toolCalls).toEqual([
      { name: "get_weather", id: "call_1" },
      { name: "get_time", id: "call_2" },
    ]);
  });
});

describe("Multi-choice (n > 1) SSE interleaving", () => {
  it("extracts usage from n=2 response with interleaved choice chunks", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":1,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Red"},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":1,"delta":{"content":"Blue"},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[{"index":1,"delta":{},"finish_reason":"stop"}]}\n\n',
      'data: {"id":"chatcmpl-n2","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage!.prompt_tokens).toBe(10);
    expect(result.usage!.completion_tokens).toBe(4);

    const cost = calculateOpenAICost("gpt-4o", "gpt-4o", result.usage!, "req-n2", 200);
    // 10 * 2.50 + 4 * 10.00 = 25 + 40 = 65
    expect(cost.costMicrodollars).toBe(65);
  });
});

describe("Reasoning model cost calculation", () => {
  it("o3-mini with heavy reasoning produces correct cost", () => {
    const cost = calculateOpenAICost(
      "o3-mini",
      "o3-mini",
      {
        prompt_tokens: 500,
        completion_tokens: 8000,
        completion_tokens_details: { reasoning_tokens: 7500 },
      },
      "req-reason",
      5000,
    );

    expect(cost.reasoningTokens).toBe(7500);
    expect(cost.outputTokens).toBe(8000);
    // input: 500 * 1.10 = 550
    // output: 8000 * 4.40 = 35200 (reasoning tokens billed at same rate)
    // total: 35750
    expect(cost.costMicrodollars).toBe(35750);
  });

  it("o1 with reasoning tokens and cached input", () => {
    const cost = calculateOpenAICost(
      "o1",
      null,
      {
        prompt_tokens: 2000,
        completion_tokens: 5000,
        prompt_tokens_details: { cached_tokens: 1500 },
        completion_tokens_details: { reasoning_tokens: 4000 },
      },
      "req-o1",
      10000,
    );

    expect(cost.cachedInputTokens).toBe(1500);
    expect(cost.reasoningTokens).toBe(4000);
    // normalInput: (2000-1500)=500 * 15.00 = 7500
    // cached: 1500 * 7.50 = 11250
    // output: 5000 * 60.00 = 300000
    // total: 318750
    expect(cost.costMicrodollars).toBe(318750);
  });
});

describe("Cached token cost adjustment", () => {
  it("100% cached input produces lower cost than uncached", () => {
    const uncached = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 1000, completion_tokens: 100 },
      "req-uc",
      100,
    );
    const cached = calculateOpenAICost(
      "gpt-4o",
      null,
      {
        prompt_tokens: 1000,
        completion_tokens: 100,
        prompt_tokens_details: { cached_tokens: 1000 },
      },
      "req-c",
      100,
    );

    expect(cached.costMicrodollars).toBeLessThan(uncached.costMicrodollars);
    // Uncached: 1000 * 2.50 + 100 * 10.00 = 2500 + 1000 = 3500
    expect(uncached.costMicrodollars).toBe(3500);
    // Cached: 1000 * 1.25 + 100 * 10.00 = 1250 + 1000 = 2250
    expect(cached.costMicrodollars).toBe(2250);

    expect(uncached.costMicrodollars - cached.costMicrodollars).toBe(1250);
  });

  it("partial cache produces cost between fully cached and uncached", () => {
    const partial = calculateOpenAICost(
      "gpt-4.1",
      null,
      {
        prompt_tokens: 10000,
        completion_tokens: 500,
        prompt_tokens_details: { cached_tokens: 6000 },
      },
      "req-partial",
      200,
    );

    // normalInput: 4000 * 2.00 = 8000
    // cached: 6000 * 0.50 = 3000
    // output: 500 * 8.00 = 4000
    // total: 15000
    expect(partial.costMicrodollars).toBe(15000);
  });
});

describe("Model alias resolution", () => {
  it("unknown request model falls back to known response model", () => {
    const cost = calculateOpenAICost(
      "gpt-4o-2024-11-20",
      "gpt-4o",
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-alias",
      50,
    );

    expect(cost.model).toBe("gpt-4o");
    expect(cost.costMicrodollars).toBeGreaterThan(0);
    // 100 * 2.50 + 50 * 10.00 = 250 + 500 = 750
    expect(cost.costMicrodollars).toBe(750);
  });

  it("both models unknown produces zero cost but preserves request model", () => {
    const cost = calculateOpenAICost(
      "ft:gpt-4o-mini:custom::abc123",
      "ft:gpt-4o-mini:custom::abc123",
      { prompt_tokens: 1000, completion_tokens: 500 },
      "req-ft",
      300,
    );

    expect(cost.model).toBe("ft:gpt-4o-mini:custom::abc123");
    expect(cost.costMicrodollars).toBe(0);
    expect(cost.inputTokens).toBe(1000);
    expect(cost.outputTokens).toBe(500);
  });

  it("request model takes priority over response model when both are known", () => {
    const cost = calculateOpenAICost(
      "gpt-4o-mini",
      "gpt-4o",
      { prompt_tokens: 100, completion_tokens: 50 },
      "req-priority",
      50,
    );

    expect(cost.model).toBe("gpt-4o-mini");
    // Uses gpt-4o-mini pricing: 100 * 0.15 + 50 * 0.60 = 15 + 30 = 45
    expect(cost.costMicrodollars).toBe(45);
  });
});

describe("Zero-token edge cases", () => {
  it("zero prompt and zero completion tokens produce zero cost", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 0, completion_tokens: 0 },
      "req-zero",
      10,
    );

    expect(cost.costMicrodollars).toBe(0);
    expect(cost.inputTokens).toBe(0);
    expect(cost.outputTokens).toBe(0);
  });

  it("zero completion tokens (prompt-only request for moderation)", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 500, completion_tokens: 0 },
      "req-promptonly",
      50,
    );

    // 500 * 2.50 = 1250, no output cost
    expect(cost.costMicrodollars).toBe(1250);
  });

  it("zero prompt tokens but nonzero completion (edge case)", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: 0, completion_tokens: 100 },
      "req-noPrompt",
      50,
    );

    // 0 + 100 * 10.00 = 1000
    expect(cost.costMicrodollars).toBe(1000);
  });
});

describe("Extremely large token counts", () => {
  it("200K context window call at Opus pricing", () => {
    const cost = calculateOpenAICost(
      "o1",
      null,
      {
        prompt_tokens: 200_000,
        completion_tokens: 16_000,
        prompt_tokens_details: { cached_tokens: 100_000 },
      },
      "req-huge",
      60_000,
    );

    // normalInput: 100000 * 15.00 = 1,500,000
    // cached: 100000 * 7.50 = 750,000
    // output: 16000 * 60.00 = 960,000
    // total: 3,210,000
    expect(cost.costMicrodollars).toBe(3_210_000);
    expect(Number.isSafeInteger(cost.costMicrodollars)).toBe(true);
    // $3.21
    expect(cost.costMicrodollars / 1_000_000).toBeCloseTo(3.21, 2);
  });

  it("maximal nano model call stays in safe integer range", () => {
    const cost = calculateOpenAICost(
      "gpt-4.1-nano",
      null,
      { prompt_tokens: 1_000_000, completion_tokens: 500_000 },
      "req-nano-max",
      30_000,
    );

    // 1M * 0.10 + 500K * 0.40 = 100,000 + 200,000 = 300,000
    expect(cost.costMicrodollars).toBe(300_000);
    expect(Number.isSafeInteger(cost.costMicrodollars)).toBe(true);
  });
});

describe("Unicode in SSE chunks", () => {
  it("emoji content in streaming response is passed through and parsed", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello! 🌍🎉"}}]}\n\n',
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"日本語テスト"}}]}\n\n',
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(output).toContain("🌍🎉");
    expect(output).toContain("日本語テスト");
    expect(result.usage!.prompt_tokens).toBe(10);
    expect(result.usage!.completion_tokens).toBe(8);
  });

  it("multi-byte UTF-8 split across chunk boundaries", async () => {
    // "€" is 3 bytes in UTF-8: 0xE2 0x82 0xAC
    const encoder = new TextEncoder();
    const fullPayload =
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Price: €100"}}]}\n\n' +
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n' +
      "data: [DONE]\n\n";

    const bytes = encoder.encode(fullPayload);
    // Split right in the middle of the € character (after first byte of 3-byte sequence)
    const euroStart = fullPayload.indexOf("€");
    const byteOffset = encoder.encode(fullPayload.slice(0, euroStart)).length;

    const chunk1 = bytes.slice(0, byteOffset + 1); // first byte of €
    const chunk2 = bytes.slice(byteOffset + 1);     // rest including remaining 2 bytes of €

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(chunk1);
        controller.enqueue(chunk2);
        controller.close();
      },
    });

    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(output).toContain("€100");
    expect(result.usage!.prompt_tokens).toBe(5);
  });
});

describe("Malformed upstream response handling", () => {
  it("stream ending abruptly without [DONE] still resolves usage from last valid chunk", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
      // no [DONE] — stream just ends
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).not.toBeNull();
    expect(result.usage!.prompt_tokens).toBe(5);
  });

  it("stream with only [DONE] and no content/usage chunks resolves null", async () => {
    const stream = makeStream(["data: [DONE]\n\n"]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("usage chunk with missing fields defaults safely in cost calculator", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: undefined, completion_tokens: undefined } as any,
      "req-missing",
      50,
    );

    expect(cost.inputTokens).toBe(0);
    expect(cost.outputTokens).toBe(0);
    expect(cost.costMicrodollars).toBe(0);
  });

  it("usage chunk with NaN tokens produces zero cost (not NaN)", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: NaN, completion_tokens: NaN } as any,
      "req-nan",
      50,
    );

    expect(cost.costMicrodollars).toBe(0);
    expect(cost.inputTokens).toBe(0);
    expect(cost.outputTokens).toBe(0);
  });

  it("usage chunk with negative tokens clamps to zero cost via costComponent", () => {
    const cost = calculateOpenAICost(
      "gpt-4o",
      null,
      { prompt_tokens: -100, completion_tokens: -50 } as any,
      "req-neg",
      50,
    );

    // costComponent returns 0 for tokens <= 0
    expect(cost.costMicrodollars).toBe(0);
  });
});

describe("Cost calculation for every OpenAI model in catalog", () => {
  const openaiModels = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "o4-mini", "o3", "o3-mini", "o1",
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2",
  ];

  for (const model of openaiModels) {
    it(`${model}: cost with 1K input, 500 output, 200 cached`, () => {
      const cost = calculateOpenAICost(
        model,
        null,
        {
          prompt_tokens: 1000,
          completion_tokens: 500,
          prompt_tokens_details: { cached_tokens: 200 },
        },
        `req-${model}`,
        100,
      );

      expect(cost.model).toBe(model);
      expect(cost.costMicrodollars).toBeGreaterThan(0);
      expect(Number.isSafeInteger(cost.costMicrodollars)).toBe(true);
      expect(cost.inputTokens).toBe(1000);
      expect(cost.outputTokens).toBe(500);
      expect(cost.cachedInputTokens).toBe(200);
    });
  }
});

describe("End-to-end: SSE parser → cost calculator pipeline", () => {
  it("realistic GPT-4o streaming response with cached tokens produces correct cost", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-e2e","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-e2e","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-e2e","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"content":" is 42."},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-e2e","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
      'data: {"id":"chatcmpl-e2e","model":"gpt-4o-2024-11-20","choices":[],"usage":{"prompt_tokens":2000,"completion_tokens":50,"total_tokens":2050,"prompt_tokens_details":{"cached_tokens":1500},"completion_tokens_details":{"reasoning_tokens":0}}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    // SSE parser captured the response model alias
    expect(result.model).toBe("gpt-4o-2024-11-20");
    expect(result.usage).not.toBeNull();

    // Cost calculator: request model "gpt-4o" should be used
    const cost = calculateOpenAICost("gpt-4o", result.model, result.usage!, "req-e2e", 1500);

    expect(cost.model).toBe("gpt-4o");
    expect(cost.inputTokens).toBe(2000);
    expect(cost.cachedInputTokens).toBe(1500);

    // normalInput: 500 * 2.50 = 1250
    // cached: 1500 * 1.25 = 1875
    // output: 50 * 10.00 = 500
    // total: 3625
    expect(cost.costMicrodollars).toBe(3625);
  });

  it("non-streaming GPT-4.1 response with all zeros", () => {
    const cost = calculateOpenAICost(
      "gpt-4.1",
      "gpt-4.1",
      {
        prompt_tokens: 0,
        completion_tokens: 0,
        prompt_tokens_details: { cached_tokens: 0 },
        completion_tokens_details: { reasoning_tokens: 0 },
      },
      "req-allzero",
      5,
    );

    expect(cost.costMicrodollars).toBe(0);
    expect(cost.model).toBe("gpt-4.1");
  });
});
