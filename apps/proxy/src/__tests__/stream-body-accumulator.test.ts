import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn((p: Promise<unknown>) => { p.catch(() => {}); }),
}));

const mockEmitMetric = vi.fn();
vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { createStreamBodyAccumulator } from "../lib/body-storage.js";

const encoder = new TextEncoder();

async function pipeChunks(
  accumulator: ReturnType<typeof createStreamBodyAccumulator>,
  chunks: string[],
): Promise<string> {
  const readable = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });

  const output = readable.pipeThrough(accumulator.transform);
  const reader = output.getReader();
  const outputChunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    outputChunks.push(value);
  }

  const decoder = new TextDecoder();
  return outputChunks.map((c) => decoder.decode(c, { stream: true })).join("") +
    decoder.decode(new Uint8Array(), { stream: false });
}

describe("StreamBodyAccumulator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("passes all chunks through unmodified", async () => {
    const accumulator = createStreamBodyAccumulator();
    const output = await pipeChunks(accumulator, [
      "data: {\"id\":\"1\"}\n\n",
      "data: {\"id\":\"2\"}\n\n",
      "data: [DONE]\n\n",
    ]);

    expect(output).toBe(
      "data: {\"id\":\"1\"}\n\ndata: {\"id\":\"2\"}\n\ndata: [DONE]\n\n",
    );
  });

  it("accumulates body text from all chunks", async () => {
    const accumulator = createStreamBodyAccumulator();
    await pipeChunks(accumulator, [
      "data: {\"id\":\"1\"}\n\n",
      "data: {\"id\":\"2\"}\n\n",
    ]);

    expect(accumulator.getBody()).toBe(
      "data: {\"id\":\"1\"}\n\ndata: {\"id\":\"2\"}\n\n",
    );
  });

  it("truncates at 1MB and sets overflow flag", async () => {
    const accumulator = createStreamBodyAccumulator();
    // Create a chunk that exceeds 1MB
    const bigChunk = "x".repeat(1_048_576 + 100);
    await pipeChunks(accumulator, [bigChunk]);

    expect(accumulator.getBody().length).toBe(1_048_576);
    expect(accumulator.overflow).toBe(true);
    expect(mockEmitMetric).toHaveBeenCalledWith("body_storage_overflow", { type: "response_sse" });
  });

  it("truncates across multiple chunks", async () => {
    const accumulator = createStreamBodyAccumulator();
    const halfMB = "a".repeat(524_288);
    const overHalfMB = "b".repeat(524_289);
    await pipeChunks(accumulator, [halfMB, overHalfMB]);

    expect(accumulator.getBody().length).toBe(1_048_576);
    expect(accumulator.overflow).toBe(true);
    // Body should be half 'a' and the rest 'b' up to 1MB
    expect(accumulator.getBody().startsWith(halfMB)).toBe(true);
  });

  it("handles empty stream", async () => {
    const accumulator = createStreamBodyAccumulator();
    await pipeChunks(accumulator, []);

    expect(accumulator.getBody()).toBe("");
    expect(accumulator.overflow).toBe(false);
  });

  it("handles multi-byte UTF-8 correctly", async () => {
    const accumulator = createStreamBodyAccumulator();
    // Emoji is 4 bytes in UTF-8
    await pipeChunks(accumulator, ["data: 🎉\n\n", "data: 你好\n\n"]);

    expect(accumulator.getBody()).toBe("data: 🎉\n\ndata: 你好\n\n");
  });

  it("handles stream cancellation — preserves partial buffer", async () => {
    const accumulator = createStreamBodyAccumulator();

    const readable = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("data: {\"partial\":true}\n\n"));
        // Don't close — simulate cancellation
      },
    });

    const output = readable.pipeThrough(accumulator.transform);
    const reader = output.getReader();

    // Read the one chunk
    const { value } = await reader.read();
    expect(value).toBeTruthy();

    // Cancel the stream
    await reader.cancel();

    // Buffer should contain the partial data
    expect(accumulator.getBody()).toBe("data: {\"partial\":true}\n\n");
  });

  it("skips further accumulation after overflow", async () => {
    const accumulator = createStreamBodyAccumulator();
    const almostMB = "x".repeat(1_048_575);
    await pipeChunks(accumulator, [almostMB, "ab", "more_data_that_should_not_be_added"]);

    // Should have exactly 1MB (1_048_575 + 1 byte from "ab")
    expect(accumulator.getBody().length).toBe(1_048_576);
    expect(accumulator.overflow).toBe(true);
  });

  it("handles large single chunk", async () => {
    const accumulator = createStreamBodyAccumulator();
    const chunk = "data: " + "z".repeat(500_000) + "\n\n";
    await pipeChunks(accumulator, [chunk]);

    expect(accumulator.getBody()).toBe(chunk);
    expect(accumulator.overflow).toBe(false);
  });

  it("multiple getBody() calls return consistent result", async () => {
    const accumulator = createStreamBodyAccumulator();
    await pipeChunks(accumulator, ["data: hello\n\n"]);

    expect(accumulator.getBody()).toBe("data: hello\n\n");
    expect(accumulator.getBody()).toBe("data: hello\n\n");
  });
});
