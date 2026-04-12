import { describe, it, expect } from "vitest";
import { createSSEParser } from "../lib/sse-parser.js";

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

function makeByteStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
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

describe("SSE parser", () => {
  it("extracts usage from standard streaming response with final chunk", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":0}}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result).not.toBeNull();
    expect(result!.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      prompt_tokens_details: { cached_tokens: 2 },
      completion_tokens_details: { reasoning_tokens: 0 },
    });
    expect(result!.model).toBe("gpt-4o");
  });

  it("resolves null when stream ends without usage (interrupted)", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toBeNull();
    expect(result!.model).toBe("gpt-4o");
  });

  it("handles SSE data split across multiple chunks (line boundary edge case)", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"inde',
      'x":0,"delta":{"content":"Hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toEqual({ prompt_tokens: 5, completion_tokens: 1 });
  });

  it("handles multi-byte UTF-8 character (emoji) split across chunk boundary", async () => {
    const encoder = new TextEncoder();
    const full = encoder.encode(
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello 🌍"}}]}\n\n' +
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n' +
      "data: [DONE]\n\n",
    );

    // Split in the middle of the globe emoji (4 bytes: F0 9F 8C 8D)
    const emojiStart = Array.from(full).findIndex((_, i) => full[i] === 0xf0 && full[i + 1] === 0x9f);
    const chunk1 = full.slice(0, emojiStart + 2);
    const chunk2 = full.slice(emojiStart + 2);

    const stream = makeByteStream([chunk1, chunk2]);

    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(output).toContain("🌍");
    expect(result!.usage).toEqual({ prompt_tokens: 3, completion_tokens: 2 });
  });

  it("handles malformed SSE chunks without crashing (resolves null usage)", async () => {
    const stream = makeStream([
      "data: {not valid json}\n\n",
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toBeNull();
  });

  it("handles data: prefix without trailing space (per SSE spec)", async () => {
    const stream = makeStream([
      'data:{"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      "data:[DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1 });
  });

  it("ignores SSE comment lines (: prefix)", async () => {
    const stream = makeStream([
      ": this is a keep-alive comment\n",
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toEqual({ prompt_tokens: 7, completion_tokens: 3 });
  });

  it("passes all bytes through unmodified to the client", async () => {
    const rawSSE =
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n' +
      "data: [DONE]\n\n";

    const stream = makeStream([rawSSE]);
    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    await resultPromise;

    expect(output).toBe(rawSSE);
  });

  it("handles \\r\\n line endings (SSE spec allows both)", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\r\n\r\n',
      "data: [DONE]\r\n\r\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.usage).toEqual({ prompt_tokens: 4, completion_tokens: 2 });
    expect(result!.model).toBe("gpt-4o");
  });

  it("extracts single tool call from streaming response", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-tc","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-tc","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"SF\\"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-tc","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.toolCalls).toEqual([{ name: "get_weather", id: "call_xyz" }]);
  });

  it("extracts multiple parallel tool calls", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-mt","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-mt","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.toolCalls).toEqual([
      { name: "get_weather", id: "call_1" },
      { name: "get_time", id: "call_2" },
    ]);
  });

  it("returns null toolCalls when no tool calls in stream", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.toolCalls).toBeNull();
  });

  it("ignores argument-only chunks without id/name", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-tc2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-tc2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"test\\"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-tc2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-tc2","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.toolCalls).toEqual([{ name: "search", id: "call_a" }]);
  });

  it("captures model from first chunk", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.model).toBe("gpt-4o-2024-11-20");
  });

  it("extracts finish_reason from final streaming chunk", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.finishReason).toBe("stop");
  });

  it("extracts finish_reason=tool_calls", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"name":"get_weather"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.finishReason).toBe("tool_calls");
  });

  it("finishReason is null when stream has no finish_reason", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.finishReason).toBeNull();
  });

  it("finishReason is null on cancelled stream", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    // Cancel the stream instead of draining it
    await readable.cancel();
    const result = await resultPromise;

    expect(result!.finishReason).toBeNull();
    expect(result!.cancelled).toBe(true);
  });

  it("firstChunkMs is set on the first chunk and not overwritten", async () => {
    const stream = makeStream([
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.firstChunkMs).toBeTypeOf("number");
    expect(result!.firstChunkMs).toBeGreaterThan(0);
  });

  it("firstChunkMs is null when stream has no chunks (empty body)", async () => {
    const stream = makeStream([]);

    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result!.firstChunkMs).toBeNull();
  });

  it("firstChunkMs is null when cancelled before any chunks arrive", async () => {
    // Use a stream that never pushes chunks
    const stream = new ReadableStream<Uint8Array>({
      start() { /* never enqueue — simulates cancel before data */ },
      cancel() { /* no-op */ },
    });

    const { readable, resultPromise } = createSSEParser(stream);
    await readable.cancel();
    const result = await resultPromise;

    expect(result!.firstChunkMs).toBeNull();
    expect(result!.cancelled).toBe(true);
  });
});
