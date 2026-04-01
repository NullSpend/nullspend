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

describe("SSE parser cancelled flag", () => {
  it("sets cancelled: false on normal stream completion", async () => {
    const stream = makeStream([
      'data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hi"}}]}\n\n',
      'data: {"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
      "data: [DONE]\n\n",
    ]);

    const { readable, resultPromise } = createSSEParser(stream);

    // Consume the readable stream so flush() fires and resolves resultPromise
    const reader = readable.getReader();
    while (!(await reader.read()).done) {
      // drain
    }

    const result = await resultPromise;

    expect(result.cancelled).toBe(false);
    expect(result.usage).not.toBeNull();
  });

  it("sets cancelled: true when stream is cancelled", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode('data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hi"}}]}\n\n'),
        );
      },
    });

    const { readable, resultPromise } = createSSEParser(stream);

    // Read one chunk, then cancel
    const reader = readable.getReader();
    await reader.read();
    await reader.cancel();

    const result = await resultPromise;
    expect(result.cancelled).toBe(true);
    expect(result.model).toBe("gpt-4o");
  });

  it("preserves partial usage on cancel", async () => {
    const encoder = new TextEncoder();

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        // Send usage before cancel
        controller.enqueue(
          encoder.encode('data: {"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'),
        );
      },
    });

    const { readable, resultPromise } = createSSEParser(stream);

    const reader = readable.getReader();
    await reader.read();
    await reader.cancel();

    const result = await resultPromise;
    expect(result.cancelled).toBe(true);
    // Usage was captured before cancel
    expect(result.usage).not.toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
  });
});
