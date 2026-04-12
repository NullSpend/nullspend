import { describe, it, expect } from "vitest";
import { createAnthropicSSEParser } from "../lib/anthropic-sse-parser.js";

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

function buildAnthropicSSEStream(
  events: Array<{ event: string; data: Record<string, unknown> }>,
): ReadableStream<Uint8Array> {
  const lines = events
    .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`)
    .join("");
  return makeStream([lines]);
}

describe("Anthropic SSE parser", () => {
  describe("Usage extraction", () => {
    it("extracts basic usage from message_start + message_delta", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_1",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 25, output_tokens: 1 },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 15 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        input_tokens: 25,
        output_tokens: 15,
        cache_creation_input_tokens: undefined,
        cache_read_input_tokens: undefined,
      });
      expect(result.model).toBe("claude-sonnet-4-5-20250929");
      expect(result.stopReason).toBe("end_turn");
    });

    it("extracts cached request usage with all cache fields", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_2",
              model: "claude-sonnet-4-5-20250929",
              usage: {
                input_tokens: 5,
                output_tokens: 1,
                cache_creation_input_tokens: 1253,
                cache_read_input_tokens: 128955,
              },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 503 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toEqual({
        input_tokens: 5,
        output_tokens: 503,
        cache_creation_input_tokens: 1253,
        cache_read_input_tokens: 128955,
      });
    });

    it("does not double-count cumulative cache tokens from message_delta", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_3",
              model: "claude-sonnet-4-5-20250929",
              usage: {
                input_tokens: 5,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 128955,
              },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: {
              output_tokens: 200,
              cache_read_input_tokens: 128955,
            },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.cache_read_input_tokens).toBe(128955);
    });

    it("uses message_delta input_tokens as override (server tool case)", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_4",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 25, output_tokens: 1 },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { input_tokens: 50, output_tokens: 15 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.input_tokens).toBe(50);
      expect(result.usage!.output_tokens).toBe(15);
    });

    it("extracts cacheCreationDetail from message_start", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_5",
              model: "claude-sonnet-4-5-20250929",
              usage: {
                input_tokens: 50,
                output_tokens: 1,
                cache_creation_input_tokens: 556,
                cache_read_input_tokens: 0,
                cache_creation: {
                  ephemeral_5m_input_tokens: 456,
                  ephemeral_1h_input_tokens: 100,
                },
              },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 30 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.cacheCreationDetail).toEqual({
        ephemeral_5m_input_tokens: 456,
        ephemeral_1h_input_tokens: 100,
      });
    });

    it("returns null cacheCreationDetail when cache_creation sub-object is absent", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_6",
              model: "claude-sonnet-4-5-20250929",
              usage: {
                input_tokens: 25,
                output_tokens: 1,
                cache_creation_input_tokens: 100,
                cache_read_input_tokens: 0,
              },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 10 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.cacheCreationDetail).toBeNull();
    });
  });

  describe("Stream lifecycle", () => {
    it("handles extended thinking sequence with multiple content blocks", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_7",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 100, output_tokens: 1 },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "thinking", thinking: "" },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "thinking_delta", thinking: "Let me think..." },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "signature_delta", signature: "sig_abc" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 0 },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 1,
            content_block: { type: "text", text: "" },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 1,
            delta: { type: "text_delta", text: "Hello!" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 1 },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 350 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.output_tokens).toBe(350);
      expect(result.model).toBe("claude-sonnet-4-5-20250929");
      expect(result.stopReason).toBe("end_turn");
    });

    it("ignores ping events interspersed in the stream", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_8",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 10, output_tokens: 1 },
            },
          },
        },
        { event: "ping", data: { type: "ping" } },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "" },
          },
        },
        { event: "ping", data: { type: "ping" } },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: "Hi" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 0 },
        },
        { event: "ping", data: { type: "ping" } },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 5 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.input_tokens).toBe(10);
      expect(result.usage!.output_tokens).toBe(5);
    });

    it("captures partial usage on error event mid-stream", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_9",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 40, output_tokens: 1 },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "" },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: "Partial" },
          },
        },
        {
          event: "error",
          data: {
            type: "error",
            error: {
              type: "overloaded_error",
              message: "Overloaded",
            },
          },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.input_tokens).toBe(40);
      expect(result.usage!.output_tokens).toBe(1);
      expect(result.stopReason).toBeNull();
    });

    it("returns partial usage on stream cancellation before message_delta", async () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const encoder = new TextEncoder();
          const sseData =
            `event: message_start\n` +
            `data: ${JSON.stringify({
              type: "message_start",
              message: {
                id: "msg_10",
                model: "claude-sonnet-4-5-20250929",
                usage: { input_tokens: 25, output_tokens: 1 },
              },
            })}\n\n` +
            `event: content_block_delta\n` +
            `data: ${JSON.stringify({
              type: "content_block_delta",
              index: 0,
              delta: { type: "text_delta", text: "Hello" },
            })}\n\n`;
          controller.enqueue(encoder.encode(sseData));
        },
      });

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      const reader = readable.getReader();

      await reader.read();
      await reader.cancel();

      const result = await resultPromise;

      expect(result.usage).not.toBeNull();
      expect(result.usage!.input_tokens).toBe(25);
      expect(result.model).toBe("claude-sonnet-4-5-20250929");
    });
  });

  describe("Tool call extraction", () => {
    it("extracts tool use from content_block_start event", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tc1",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 50, output_tokens: 1 },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "tool_use", id: "toolu_01A", name: "get_weather" },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "input_json_delta", partial_json: '{"city":"SF"}' },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 0 },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "tool_use" },
            usage: { output_tokens: 20 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toEqual([{ name: "get_weather", id: "toolu_01A" }]);
    });

    it("extracts multiple tool uses", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tc2",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 50, output_tokens: 1 },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "tool_use", id: "toolu_01A", name: "get_weather" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 0 },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 1,
            content_block: { type: "tool_use", id: "toolu_01B", name: "get_time" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 1 },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "tool_use" },
            usage: { output_tokens: 30 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toEqual([
        { name: "get_weather", id: "toolu_01A" },
        { name: "get_time", id: "toolu_01B" },
      ]);
    });

    it("returns null toolCalls when no tool use blocks", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tc3",
              model: "claude-sonnet-4-5-20250929",
              usage: { input_tokens: 10, output_tokens: 1 },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "" },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: "Hello!" },
          },
        },
        {
          event: "content_block_stop",
          data: { type: "content_block_stop", index: 0 },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 5 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.toolCalls).toBeNull();
    });
  });

  describe("Edge cases", () => {
    it("reassembles SSE data split across chunk boundaries", async () => {
      const encoder = new TextEncoder();
      const fullSSE =
        `event: message_start\n` +
        `data: ${JSON.stringify({
          type: "message_start",
          message: {
            id: "msg_11",
            model: "claude-sonnet-4-5-20250929",
            usage: { input_tokens: 30, output_tokens: 1 },
          },
        })}\n\n` +
        `event: message_delta\n` +
        `data: ${JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 20 },
        })}\n\n` +
        `event: message_stop\n` +
        `data: ${JSON.stringify({ type: "message_stop" })}\n\n`;

      const bytes = encoder.encode(fullSSE);
      const mid = Math.floor(bytes.length / 2);
      const chunk1 = bytes.slice(0, mid);
      const chunk2 = bytes.slice(mid);

      const stream = makeByteStream([chunk1, chunk2]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.input_tokens).toBe(30);
      expect(result.usage!.output_tokens).toBe(20);
    });

    it("returns null usage and model for empty stream (ping + message_stop only)", async () => {
      const stream = buildAnthropicSSEStream([
        { event: "ping", data: { type: "ping" } },
        { event: "message_stop", data: { type: "message_stop" } },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage).toBeNull();
      expect(result.model).toBeNull();
    });

    it("captures model from message_start payload", async () => {
      const stream = buildAnthropicSSEStream([
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_12",
              model: "claude-opus-4-20250918",
              usage: { input_tokens: 5, output_tokens: 1 },
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 10 },
          },
        },
        {
          event: "message_stop",
          data: { type: "message_stop" },
        },
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.model).toBe("claude-opus-4-20250918");
    });

    it("passes all bytes through unmodified", async () => {
      const rawSSE =
        `event: message_start\n` +
        `data: ${JSON.stringify({
          type: "message_start",
          message: {
            id: "msg_13",
            model: "claude-sonnet-4-5-20250929",
            usage: { input_tokens: 5, output_tokens: 1 },
          },
        })}\n\n` +
        `event: content_block_delta\n` +
        `data: ${JSON.stringify({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello!" },
        })}\n\n` +
        `event: message_stop\n` +
        `data: ${JSON.stringify({ type: "message_stop" })}\n\n`;

      const stream = makeStream([rawSSE]);
      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      const output = await drainStream(readable);
      await resultPromise;

      expect(output).toBe(rawSSE);
    });

    it("handles \\r\\n line endings per SSE spec", async () => {
      const sseWithCRLF =
        `event: message_start\r\n` +
        `data: ${JSON.stringify({
          type: "message_start",
          message: {
            id: "msg_14",
            model: "claude-sonnet-4-5-20250929",
            usage: { input_tokens: 12, output_tokens: 1 },
          },
        })}\r\n\r\n` +
        `event: message_delta\r\n` +
        `data: ${JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 8 },
        })}\r\n\r\n` +
        `event: message_stop\r\n` +
        `data: ${JSON.stringify({ type: "message_stop" })}\r\n\r\n`;

      const stream = makeStream([sseWithCRLF]);
      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.usage!.input_tokens).toBe(12);
      expect(result.usage!.output_tokens).toBe(8);
      expect(result.model).toBe("claude-sonnet-4-5-20250929");
    });
  });

  describe("firstChunkMs", () => {
    it("captures firstChunkMs on the first chunk", async () => {
      const stream = makeStream([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
      ]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.firstChunkMs).toBeTypeOf("number");
      expect(result.firstChunkMs).toBeGreaterThan(0);
    });

    it("firstChunkMs is null when stream has no chunks", async () => {
      const stream = makeStream([]);

      const { readable, resultPromise } = createAnthropicSSEParser(stream);
      await drainStream(readable);
      const result = await resultPromise;

      expect(result.firstChunkMs).toBeNull();
    });
  });
});
