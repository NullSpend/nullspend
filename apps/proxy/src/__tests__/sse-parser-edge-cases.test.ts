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

describe("SSE parser edge cases", () => {
  it("handles completely empty stream (zero bytes)", async () => {
    const stream = makeStream([]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("handles stream with only whitespace/empty lines", async () => {
    const stream = makeStream(["\n\n\n\n"]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("handles stream with only SSE comments (keep-alive pings)", async () => {
    const stream = makeStream([
      ": keep-alive\n\n",
      ": another ping\n\n",
      ": third comment\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("handles stream with only [DONE] and no content chunks", async () => {
    const stream = makeStream(["data: [DONE]\n\n"]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("uses last usage object when multiple usage chunks appear", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":15}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 20, completion_tokens: 15 });
  });

  it("handles 1-byte-at-a-time chunking (worst-case fragmentation)", async () => {
    const encoder = new TextEncoder();
    const fullPayload =
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n' +
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n' +
      "data: [DONE]\n\n";

    const bytes = encoder.encode(fullPayload);
    const singleByteChunks: Uint8Array[] = [];
    for (let i = 0; i < bytes.length; i++) {
      singleByteChunks.push(bytes.slice(i, i + 1));
    }

    const stream = makeByteStream(singleByteChunks);
    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(output).toBe(fullPayload);
    expect(result.usage).toEqual({ prompt_tokens: 3, completion_tokens: 1 });
    expect(result.model).toBe("gpt-4o");
  });

  it("handles 'data:' prefix split across chunk boundary", async () => {
    const stream = makeStream([
      "dat",
      'a: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1 });
  });

  it("handles mixed \\r\\n and \\n line endings in same stream", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}\r\n\r\n',
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
      "data: [DONE]\r\n\r\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 5, completion_tokens: 2 });
  });

  it("handles very long content field (10KB+ JSON payload)", async () => {
    const longContent = "x".repeat(15000);
    const stream = makeStream([
      `data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"${longContent}"}}]}\n\n`,
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":4000}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(output).toContain(longContent);
    expect(result.usage).toEqual({ prompt_tokens: 100, completion_tokens: 4000 });
  });

  it("handles usage with all optional detail fields present", async () => {
    const stream = makeStream([
      'data: {"model":"o3","choices":[],"usage":{"prompt_tokens":500,"completion_tokens":2000,"prompt_tokens_details":{"cached_tokens":300},"completion_tokens_details":{"reasoning_tokens":1500}}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage!.prompt_tokens).toBe(500);
    expect(result.usage!.completion_tokens).toBe(2000);
    expect(result.usage!.prompt_tokens_details!.cached_tokens).toBe(300);
    expect(result.usage!.completion_tokens_details!.reasoning_tokens).toBe(1500);
  });

  it("handles usage chunk with extra unknown fields (forward compatibility)", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"some_future_field":42}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage!.prompt_tokens).toBe(10);
    expect(result.usage!.completion_tokens).toBe(5);
    expect((result.usage as any).total_tokens).toBe(15);
  });

  it("handles multiple consecutive malformed JSON lines without crashing", async () => {
    const stream = makeStream([
      "data: {broken\n\n",
      "data: also{broken\n\n",
      "data: [[[not json]]]\n\n",
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1 });
  });

  it("resolves resultPromise on stream cancel() with partial data", async () => {
    let _controllerRef: ReadableStreamDefaultController<Uint8Array>;
    const encoder = new TextEncoder();

    const source = new ReadableStream<Uint8Array>({
      start(controller) {
        _controllerRef = controller;
        controller.enqueue(encoder.encode(
          'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
        ));
      },
      cancel() {
        // upstream cancel
      },
    });

    const { readable, resultPromise } = createSSEParser(source);
    const reader = readable.getReader();

    await reader.read();
    await reader.cancel();

    const result = await resultPromise;
    expect(result.usage).toBeNull();
    expect(result.model).toBe("gpt-4o");
  });

  it("handles data event with no space after colon", async () => {
    const stream = makeStream([
      'data:{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2}}\n\n',
      "data:[DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 2, completion_tokens: 2 });
  });

  it("handles data event with multiple spaces after colon", async () => {
    const stream = makeStream([
      'data:   {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":6,"completion_tokens":3}}\n\n',
      "data:   [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 6, completion_tokens: 3 });
  });

  it("handles stream with many rapid small content chunks (simulates fast model)", async () => {
    const chunks: string[] = [];
    for (let i = 0; i < 100; i++) {
      chunks.push(`data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"w${i}"}}]}\n\n`);
    }
    chunks.push('data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":100}}\n\n');
    chunks.push("data: [DONE]\n\n");

    const stream = makeStream(chunks);
    const { readable, resultPromise } = createSSEParser(stream);
    const output = await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 50, completion_tokens: 100 });
    expect(output).toContain("w99");
  });

  it("handles newline inside JSON string content (escaped \\n)", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"line1\\nline2\\nline3"}}]}\n\n',
      'data: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 8, completion_tokens: 4 });
  });

  it("handles chunk containing only partial 'data:' prefix at end of buffer", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"A"}}]}\n\nda',
      'ta: {"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const { readable, resultPromise } = createSSEParser(stream);
    await drainStream(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1 });
  });
});
