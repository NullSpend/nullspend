import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn((p: Promise<unknown>) => {
    p.catch(() => {});
  }),
}));

import { createGeminiSSEParser } from "../lib/gemini-sse-parser.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a ReadableStream from an array of SSE event strings. */
function sseStream(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(event));
      }
      controller.close();
    },
  });
}

/** Shortcut: wrap a GenerateContentResponse object into an SSE `data:` frame. */
function geminiEvent(payload: Record<string, unknown>): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/** Drain a ReadableStream to a string (passthrough verification). */
async function drainStream(
  readable: ReadableStream<Uint8Array>,
): Promise<string> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Gemini SSE parser", () => {
  describe("Usage extraction", () => {
    it("extracts usage, model, and finishReason from a basic text response", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hello world" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-abc123",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      });
      expect(result.model).toBe("gemini-2.5-pro-preview-05-06");
      expect(result.finishReason).toBe("STOP");
      expect(result.cancelled).toBe(false);
      expect(result.toolCalls).toBeNull();
    });

    it("captures LAST usageMetadata from multi-chunk response", async () => {
      const events = [
        geminiEvent({
          candidates: [
            { content: { role: "model", parts: [{ text: "Hello" }] } },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 2,
            totalTokenCount: 12,
          },
          modelVersion: "gemini-2.5-flash-preview-04-17",
          responseId: "resp-multi-1",
        }),
        geminiEvent({
          candidates: [
            { content: { role: "model", parts: [{ text: " world" }] } },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 5,
            totalTokenCount: 15,
          },
          responseId: "resp-multi-1",
        }),
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "!" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 8,
            totalTokenCount: 18,
          },
          responseId: "resp-multi-1",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      // Final chunk's usage is the complete one
      expect(result.usage).toEqual({
        promptTokenCount: 10,
        candidatesTokenCount: 8,
        totalTokenCount: 18,
      });
      expect(result.model).toBe("gemini-2.5-flash-preview-04-17");
      expect(result.finishReason).toBe("STOP");
    });

    it("returns null usage when no chunks contain usageMetadata", async () => {
      const events = [
        geminiEvent({
          candidates: [
            { content: { role: "model", parts: [{ text: "Hello" }] } },
          ],
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-no-usage",
        }),
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: " world" }] },
              finishReason: "STOP",
            },
          ],
          responseId: "resp-no-usage",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toBeNull();
    });

    it("preserves thoughtsTokenCount in usageMetadata", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Thought result" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 50,
            candidatesTokenCount: 20,
            thoughtsTokenCount: 150,
            totalTokenCount: 220,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-thoughts",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        promptTokenCount: 50,
        candidatesTokenCount: 20,
        thoughtsTokenCount: 150,
        totalTokenCount: 220,
      });
      expect(result.usage!.thoughtsTokenCount).toBe(150);
    });
  });

  describe("Tool call extraction", () => {
    it("extracts function call with explicit id", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  {
                    functionCall: {
                      name: "get_weather",
                      args: { city: "San Francisco" },
                      id: "call_abc123",
                    },
                  },
                ],
              },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 30,
            candidatesTokenCount: 10,
            totalTokenCount: 40,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-fc1",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toEqual([
        { name: "get_weather", id: "call_abc123" },
      ]);
      expect(result.finishReason).toBe("STOP");
    });

    it("generates UUID for function call without id", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  {
                    functionCall: {
                      name: "search_docs",
                      args: { query: "test" },
                      // no id field
                    },
                  },
                ],
              },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 20,
            candidatesTokenCount: 8,
            totalTokenCount: 28,
          },
          modelVersion: "gemini-2.5-flash-preview-04-17",
          responseId: "resp-fc-noid",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls![0].name).toBe("search_docs");
      // Generated UUID should be a valid UUID v4 format
      expect(result.toolCalls![0].id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      );
    });

    it("extracts multiple function calls from one chunk", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  {
                    functionCall: {
                      name: "get_weather",
                      args: { city: "NYC" },
                      id: "call_1",
                    },
                  },
                  {
                    functionCall: {
                      name: "get_time",
                      args: { timezone: "EST" },
                      id: "call_2",
                    },
                  },
                ],
              },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 40,
            candidatesTokenCount: 15,
            totalTokenCount: 55,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-fc-multi",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toEqual([
        { name: "get_weather", id: "call_1" },
        { name: "get_time", id: "call_2" },
      ]);
    });
  });

  describe("Stream lifecycle", () => {
    it("captures partial usage on client cancellation", async () => {
      // Use a stream that never closes — the cancel path fires
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const encoder = new TextEncoder();
          controller.enqueue(
            encoder.encode(
              geminiEvent({
                candidates: [
                  { content: { role: "model", parts: [{ text: "Hello" }] } },
                ],
                usageMetadata: {
                  promptTokenCount: 20,
                  candidatesTokenCount: 3,
                  totalTokenCount: 23,
                },
                modelVersion: "gemini-2.5-pro-preview-05-06",
                responseId: "resp-cancel",
              }),
            ),
          );
          // Do NOT close — simulates an ongoing stream
        },
      });

      const { readable, resultPromise } = createGeminiSSEParser(stream);
      const reader = readable.getReader();

      // Read first chunk
      await reader.read();
      // Cancel mid-stream
      await reader.cancel();

      const result = await resultPromise;

      expect(result.cancelled).toBe(true);
      expect(result.usage).toEqual({
        promptTokenCount: 20,
        candidatesTokenCount: 3,
        totalTokenCount: 23,
      });
      expect(result.model).toBe("gemini-2.5-pro-preview-05-06");
    });

    it("captures firstChunkMs on first chunk", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Hi" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 5,
            candidatesTokenCount: 2,
            totalTokenCount: 7,
          },
          modelVersion: "gemini-2.5-flash-preview-04-17",
          responseId: "resp-ttfb",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.firstChunkMs).toBeTypeOf("number");
      expect(result.firstChunkMs).toBeGreaterThan(0);
    });

    it("passes all bytes through unmodified", async () => {
      const raw = geminiEvent({
        candidates: [
          {
            content: { role: "model", parts: [{ text: "pass-through" }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 5,
          candidatesTokenCount: 3,
          totalTokenCount: 8,
        },
        modelVersion: "gemini-2.5-pro-preview-05-06",
        responseId: "resp-pt",
      });

      const { readable, resultPromise } = createGeminiSSEParser(sseStream([raw]));
      const output = await drainStream(readable);
      await resultPromise;

      expect(output).toBe(raw);
    });
  });

  describe("responseId extraction", () => {
    it("captures responseId from chunks", async () => {
      const events = [
        geminiEvent({
          candidates: [
            { content: { role: "model", parts: [{ text: "Part 1" }] } },
          ],
          responseId: "resp-id-test-42",
          modelVersion: "gemini-2.5-pro-preview-05-06",
        }),
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "Part 2" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 6,
            totalTokenCount: 16,
          },
          responseId: "resp-id-test-42",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.responseId).toBe("resp-id-test-42");
    });

    it("returns null responseId when absent from all chunks", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "No ID" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 5,
            candidatesTokenCount: 2,
            totalTokenCount: 7,
          },
          modelVersion: "gemini-2.5-flash-preview-04-17",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.responseId).toBeNull();
    });
  });

  describe("Edge cases", () => {
    it("skips malformed JSON silently without crashing", async () => {
      const events = [
        "data: {invalid json here}\n\n",
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "After bad" }] },
              finishReason: "STOP",
            },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 3,
            totalTokenCount: 13,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-malformed",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      // Parser should recover and capture data from the valid chunk
      expect(result.usage).toEqual({
        promptTokenCount: 10,
        candidatesTokenCount: 3,
        totalTokenCount: 13,
      });
      expect(result.model).toBe("gemini-2.5-pro-preview-05-06");
      expect(result.finishReason).toBe("STOP");
    });

    it("handles empty candidates array without crashing", async () => {
      const events = [
        geminiEvent({
          candidates: [],
          usageMetadata: {
            promptTokenCount: 15,
            candidatesTokenCount: 0,
            totalTokenCount: 15,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-empty-cand",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        promptTokenCount: 15,
        candidatesTokenCount: 0,
        totalTokenCount: 15,
      });
      expect(result.finishReason).toBeNull();
      expect(result.toolCalls).toBeNull();
    });

    it("captures finishReason SAFETY correctly", async () => {
      const events = [
        geminiEvent({
          candidates: [
            {
              content: { role: "model", parts: [] },
              finishReason: "SAFETY",
            },
          ],
          usageMetadata: {
            promptTokenCount: 20,
            candidatesTokenCount: 0,
            totalTokenCount: 20,
          },
          modelVersion: "gemini-2.5-pro-preview-05-06",
          responseId: "resp-safety",
        }),
      ];

      const { readable, resultPromise } = createGeminiSSEParser(sseStream(events));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.finishReason).toBe("SAFETY");
      expect(result.usage!.candidatesTokenCount).toBe(0);
    });

    it("drops oversized line buffer and logs warning", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      // Create a single data line that exceeds 1MB WITHOUT a newline
      // so it sits in the lineBuffer and triggers the safety valve.
      // (Safety valve is 1MB to allow large Gemini SSE events)
      const hugePayload = "data: " + "x".repeat(1_100_000);
      const validEvent = geminiEvent({
        candidates: [
          {
            content: { role: "model", parts: [{ text: "OK" }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 5,
          candidatesTokenCount: 2,
          totalTokenCount: 7,
        },
        modelVersion: "gemini-2.5-pro-preview-05-06",
        responseId: "resp-oversized",
      });

      // First chunk: oversized without newline (triggers buffer drop)
      // Second chunk: valid event (parser recovers)
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(hugePayload));
          controller.enqueue(encoder.encode(validEvent));
          controller.close();
        },
      });

      const { readable, resultPromise } = createGeminiSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("[gemini-sse-parser] Dropping oversized line buffer:"),
        expect.any(Number),
        "bytes",
      );
      // After buffer drop, the valid event should still be parsed
      expect(result.model).toBe("gemini-2.5-pro-preview-05-06");
      expect(result.finishReason).toBe("STOP");

      warnSpy.mockRestore();
    });

    it("handles empty stream gracefully", async () => {
      const { readable, resultPromise } = createGeminiSSEParser(sseStream([]));
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toBeNull();
      expect(result.model).toBeNull();
      expect(result.finishReason).toBeNull();
      expect(result.toolCalls).toBeNull();
      expect(result.cancelled).toBe(false);
      expect(result.responseId).toBeNull();
      expect(result.firstChunkMs).toBeNull();
    });

    it("reassembles SSE data split across chunk boundaries", async () => {
      const fullSSE = geminiEvent({
        candidates: [
          {
            content: { role: "model", parts: [{ text: "Split test" }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 12,
          candidatesTokenCount: 4,
          totalTokenCount: 16,
        },
        modelVersion: "gemini-2.5-pro-preview-05-06",
        responseId: "resp-split",
      });

      const encoder = new TextEncoder();
      const bytes = encoder.encode(fullSSE);
      const mid = Math.floor(bytes.length / 2);
      const chunk1 = bytes.slice(0, mid);
      const chunk2 = bytes.slice(mid);

      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(chunk1);
          controller.enqueue(chunk2);
          controller.close();
        },
      });

      const { readable, resultPromise } = createGeminiSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        promptTokenCount: 12,
        candidatesTokenCount: 4,
        totalTokenCount: 16,
      });
      expect(result.model).toBe("gemini-2.5-pro-preview-05-06");
      expect(result.finishReason).toBe("STOP");
      expect(result.responseId).toBe("resp-split");
    });
  });
});
