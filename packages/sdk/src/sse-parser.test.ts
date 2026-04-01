import { describe, it, expect } from "vitest";
import {
  createOpenAISSEParser,
  createAnthropicSSEParser,
} from "./sse-parser.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function consume(readable: ReadableStream<Uint8Array>): Promise<string> {
  const reader = readable.getReader();
  const decoder = new TextDecoder();
  let result = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
  }
  return result;
}

// ---------------------------------------------------------------------------
// OpenAI SSE parser
// ---------------------------------------------------------------------------

describe("createOpenAISSEParser", () => {
  it("extracts usage from a complete stream", async () => {
    const chunks = [
      `data: ${JSON.stringify({ id: "chatcmpl-1", model: "gpt-4o", choices: [{ delta: { content: "Hi" } }] })}\n\n`,
      `data: ${JSON.stringify({ id: "chatcmpl-1", model: "gpt-4o", choices: [{ delta: {} }], usage: { prompt_tokens: 10, completion_tokens: 20 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 10, completion_tokens: 20 });
    expect(result.model).toBe("gpt-4o");
  });

  it("extracts the model from the first chunk", async () => {
    const chunks = [
      `data: ${JSON.stringify({ model: "gpt-4o-mini", choices: [] })}\n\n`,
      `data: ${JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 5, completion_tokens: 3 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;
    expect(result.model).toBe("gpt-4o-mini");
  });

  it("returns null usage when stream has no usage chunk", async () => {
    const chunks = [
      `data: ${JSON.stringify({ model: "gpt-4o", choices: [{ delta: { content: "Hello" } }] })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBe("gpt-4o");
  });

  it("handles malformed JSON gracefully", async () => {
    const chunks = [
      "data: {not valid json\n\n",
      `data: ${JSON.stringify({ model: "gpt-4o", usage: { prompt_tokens: 1, completion_tokens: 2 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 1, completion_tokens: 2 });
    expect(result.model).toBe("gpt-4o");
  });

  it("handles [DONE] sentinel without error", async () => {
    const chunks = ["data: [DONE]\n\n"];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("handles partial chunks split across multiple enqueues", async () => {
    // Split a single SSE message across two chunks
    const full = `data: ${JSON.stringify({ model: "gpt-4o", usage: { prompt_tokens: 42, completion_tokens: 7 } })}\n\ndata: [DONE]\n\n`;
    const mid = Math.floor(full.length / 2);
    const chunks = [full.slice(0, mid), full.slice(mid)];

    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ prompt_tokens: 42, completion_tokens: 7 });
  });

  it("handles multi-byte UTF-8 without corruption", async () => {
    const chunks = [
      `data: ${JSON.stringify({ model: "gpt-4o", choices: [{ delta: { content: "\u00e9\u00e0\u00fc\ud83d\ude00" } }] })}\n\n`,
      `data: ${JSON.stringify({ model: "gpt-4o", usage: { prompt_tokens: 10, completion_tokens: 5 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    const output = await consume(readable);
    const result = await resultPromise;

    expect(output).toContain("\u00e9\u00e0\u00fc\ud83d\ude00");
    expect(result.usage).toEqual({ prompt_tokens: 10, completion_tokens: 5 });
  });

  it("passes through bytes unmodified (passthrough integrity)", async () => {
    const raw = [
      `data: ${JSON.stringify({ model: "gpt-4o", choices: [] })}\n\n`,
      `data: ${JSON.stringify({ usage: { prompt_tokens: 1, completion_tokens: 2 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const expected = raw.join("");
    const { readable } = createOpenAISSEParser(toStream(raw));
    const output = await consume(readable);
    expect(output).toBe(expected);
  });

  it("handles an empty stream", async () => {
    const { readable, resultPromise } = createOpenAISSEParser(toStream([]));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("resolves with null usage on cancelled stream", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            `data: ${JSON.stringify({ model: "gpt-4o", choices: [{ delta: { content: "Hello" } }] })}\n\n`,
          ),
        );
        // Do not close — simulate cancellation
      },
    });

    const { readable, resultPromise } = createOpenAISSEParser(body);
    const reader = readable.getReader();
    // Read one chunk then cancel
    await reader.read();
    await reader.cancel();

    const result = await resultPromise;
    // Usage was never emitted before cancel
    expect(result.usage).toBeNull();
    expect(result.model).toBe("gpt-4o");
  });

  it("captures cached_tokens in prompt_tokens_details", async () => {
    const usage = {
      prompt_tokens: 100,
      completion_tokens: 50,
      prompt_tokens_details: { cached_tokens: 80 },
    };
    const chunks = [
      `data: ${JSON.stringify({ model: "gpt-4o", usage })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage!.prompt_tokens_details!.cached_tokens).toBe(80);
  });

  it("captures reasoning_tokens in completion_tokens_details", async () => {
    const usage = {
      prompt_tokens: 200,
      completion_tokens: 150,
      completion_tokens_details: { reasoning_tokens: 120 },
    };
    const chunks = [
      `data: ${JSON.stringify({ model: "o1-preview", usage })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage!.completion_tokens_details!.reasoning_tokens).toBe(120);
    expect(result.model).toBe("o1-preview");
  });

  it("ignores SSE comment lines (starting with colon)", async () => {
    const chunks = [
      ": this is a comment\n\n",
      `data: ${JSON.stringify({ model: "gpt-4o", usage: { prompt_tokens: 5, completion_tokens: 3 } })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const { readable, resultPromise } = createOpenAISSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;
    expect(result.usage).toEqual({ prompt_tokens: 5, completion_tokens: 3 });
  });
});

// ---------------------------------------------------------------------------
// Anthropic SSE parser
// ---------------------------------------------------------------------------

describe("createAnthropicSSEParser", () => {
  it("extracts usage from message_start event", async () => {
    const chunks = [
      `event: message_start\ndata: ${JSON.stringify({
        type: "message_start",
        message: {
          model: "claude-sonnet-4-20250514",
          usage: { input_tokens: 50, output_tokens: 0 },
        },
      })}\n\n`,
      `event: content_block_delta\ndata: ${JSON.stringify({
        type: "content_block_delta",
        delta: { text: "Hello" },
      })}\n\n`,
      `event: message_delta\ndata: ${JSON.stringify({
        type: "message_delta",
        usage: { output_tokens: 20 },
      })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
    ];
    const { readable, resultPromise } = createAnthropicSSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toEqual({ input_tokens: 50, output_tokens: 20 });
    expect(result.model).toBe("claude-sonnet-4-20250514");
  });

  it("extracts output_tokens from message_delta", async () => {
    const chunks = [
      `event: message_start\ndata: ${JSON.stringify({
        type: "message_start",
        message: {
          model: "claude-sonnet-4-20250514",
          usage: { input_tokens: 100, output_tokens: 0 },
        },
      })}\n\n`,
      `event: message_delta\ndata: ${JSON.stringify({
        type: "message_delta",
        usage: { output_tokens: 42 },
      })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
    ];
    const { readable, resultPromise } = createAnthropicSSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage!.output_tokens).toBe(42);
  });

  it("extracts the model from message_start", async () => {
    const chunks = [
      `event: message_start\ndata: ${JSON.stringify({
        type: "message_start",
        message: { model: "claude-opus-4-20250514", usage: { input_tokens: 10, output_tokens: 0 } },
      })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
    ];
    const { readable, resultPromise } = createAnthropicSSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;
    expect(result.model).toBe("claude-opus-4-20250514");
  });

  it("extracts cache creation detail with ephemeral TTLs", async () => {
    const chunks = [
      `event: message_start\ndata: ${JSON.stringify({
        type: "message_start",
        message: {
          model: "claude-sonnet-4-20250514",
          usage: {
            input_tokens: 200,
            output_tokens: 0,
            cache_creation_input_tokens: 150,
            cache_read_input_tokens: 50,
            cache_creation: {
              ephemeral_5m_input_tokens: 100,
              ephemeral_1h_input_tokens: 50,
            },
          },
        },
      })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
    ];
    const { readable, resultPromise } = createAnthropicSSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.cacheCreationDetail).toEqual({
      ephemeral_5m_input_tokens: 100,
      ephemeral_1h_input_tokens: 50,
    });
    expect(result.usage!.cache_creation_input_tokens).toBe(150);
    expect(result.usage!.cache_read_input_tokens).toBe(50);
  });

  it("resolves early on message_stop", async () => {
    let controllerRef: ReadableStreamDefaultController<Uint8Array>;
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controllerRef = controller;
        controller.enqueue(
          encoder.encode(
            `event: message_start\ndata: ${JSON.stringify({
              type: "message_start",
              message: { model: "claude-sonnet-4-20250514", usage: { input_tokens: 10, output_tokens: 0 } },
            })}\n\n`,
          ),
        );
        controller.enqueue(
          encoder.encode(
            `event: message_delta\ndata: ${JSON.stringify({
              type: "message_delta",
              usage: { output_tokens: 5 },
            })}\n\n`,
          ),
        );
        controller.enqueue(
          encoder.encode(
            `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
          ),
        );
        // Stream is NOT closed yet
      },
    });

    const { readable, resultPromise } = createAnthropicSSEParser(body);

    // Start consuming in background
    const consumePromise = consume(readable).catch(() => "");

    // resultPromise should resolve once message_stop is processed, even before stream close
    const result = await resultPromise;
    expect(result.usage!.output_tokens).toBe(5);
    expect(result.model).toBe("claude-sonnet-4-20250514");

    // Clean up: close the stream so consume finishes
    controllerRef!.close();
    await consumePromise;
  });

  it("resolves with null usage on cancelled stream", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", delta: { text: "Hi" } })}\n\n`,
          ),
        );
        // No message_start emitted, no close
      },
    });

    const { readable, resultPromise } = createAnthropicSSEParser(body);
    const reader = readable.getReader();
    await reader.read();
    await reader.cancel();

    const result = await resultPromise;
    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
  });

  it("returns null usage when stream has no usage events", async () => {
    const chunks = [
      `event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`,
    ];
    const { readable, resultPromise } = createAnthropicSSEParser(toStream(chunks));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.cacheCreationDetail).toBeNull();
  });

  it("passes through bytes unmodified (passthrough integrity)", async () => {
    const raw = [
      `event: message_start\ndata: ${JSON.stringify({
        type: "message_start",
        message: { model: "claude-sonnet-4-20250514", usage: { input_tokens: 10, output_tokens: 0 } },
      })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
    ];
    const expected = raw.join("");
    const { readable } = createAnthropicSSEParser(toStream(raw));
    const output = await consume(readable);
    expect(output).toBe(expected);
  });

  it("handles empty stream", async () => {
    const { readable, resultPromise } = createAnthropicSSEParser(toStream([]));
    await consume(readable);
    const result = await resultPromise;

    expect(result.usage).toBeNull();
    expect(result.model).toBeNull();
    expect(result.cacheCreationDetail).toBeNull();
  });
});
