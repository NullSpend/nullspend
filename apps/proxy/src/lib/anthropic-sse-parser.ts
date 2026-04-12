import type {
  AnthropicRawUsage,
  AnthropicCacheCreationDetail,
} from "./anthropic-types.js";

export interface AnthropicSSEResult {
  usage: AnthropicRawUsage | null;
  cacheCreationDetail: AnthropicCacheCreationDetail | null;
  model: string | null;
  stopReason: string | null;
  toolCalls: { name: string; id: string }[] | null;
  cancelled: boolean;
  firstChunkMs: number | null;
}

/**
 * Create a TransformStream that passes Anthropic SSE bytes through unmodified
 * while extracting usage, model, cache detail, and stop reason from the stream.
 *
 * Anthropic uses named events (`event: message_start`, `event: message_delta`, etc.)
 * unlike OpenAI's single `data:` line format. Two extraction points:
 *   - `message_start` → input tokens, cache tokens, model, cache creation detail
 *   - `message_delta`  → output tokens (cumulative), stop reason, optional input overrides
 *
 * Uses a two-line state machine tracking `currentEventType` from `event:` lines.
 * Falls back to `parsed.type` when `currentEventType` is null (chunk boundary split).
 */
export function createAnthropicSSEParser(
  upstreamBody: ReadableStream<Uint8Array>,
): {
  readable: ReadableStream<Uint8Array>;
  resultPromise: Promise<AnthropicSSEResult>;
} {
  let resolveResult: (value: AnthropicSSEResult) => void;
  const resultPromise = new Promise<AnthropicSSEResult>((resolve) => {
    resolveResult = resolve;
  });

  let capturedUsage: AnthropicRawUsage | null = null;
  let capturedCacheDetail: AnthropicCacheCreationDetail | null = null;
  let capturedModel: string | null = null;
  let capturedStopReason: string | null = null;
  let capturedToolCalls: { name: string; id: string }[] | null = null;
  let firstChunkMs: number | null = null;
  let resolved = false;

  let currentEventType: string | null = null;
  let lineBuffer = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const MAX_LINE_LENGTH = 65_536; // 64KB — safety valve for malformed streams

  function resolve(result: AnthropicSSEResult): void {
    if (resolved) return;
    resolved = true;
    resolveResult(result);
  }

  let wasCancelled = false;

  function buildResult(): AnthropicSSEResult {
    return {
      usage: capturedUsage,
      cacheCreationDetail: capturedCacheDetail,
      model: capturedModel,
      stopReason: capturedStopReason,
      toolCalls: capturedToolCalls,
      cancelled: wasCancelled,
      firstChunkMs,
    };
  }

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      if (firstChunkMs === null) firstChunkMs = performance.now();
      controller.enqueue(chunk);

      const text = decoder.decode(chunk, { stream: true });
      lineBuffer += text;

      // Safety valve: drop oversized incomplete lines to prevent memory exhaustion
      if (!lineBuffer.includes("\n") && lineBuffer.length > MAX_LINE_LENGTH) {
        console.warn("[anthropic-sse-parser] Dropping oversized line buffer:", lineBuffer.length, "bytes");
        lineBuffer = "";
        return;
      }

      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop()!;

      for (const line of lines) {
        processLine(line);
      }
    },

    flush() {
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      lineBuffer += remaining;

      if (lineBuffer.trim()) {
        processLine(lineBuffer);
      }

      resolve(buildResult());
    },

    cancel() {
      wasCancelled = true;
      resolve(buildResult());
    },
  });

  function processLine(line: string): void {
    const trimmed = line.trim();

    if (!trimmed) {
      currentEventType = null;
      return;
    }

    if (trimmed.startsWith(":")) return;

    if (trimmed.startsWith("event:")) {
      let eventValue = trimmed.slice(6);
      if (eventValue.startsWith(" ")) eventValue = eventValue.slice(1);
      currentEventType = eventValue.trim();
      return;
    }

    if (!trimmed.startsWith("data:")) return;

    let payload = trimmed.slice(5);
    if (payload.startsWith(" ")) payload = payload.slice(1);
    payload = payload.trim();

    try {
      const parsed = JSON.parse(payload);
      const eventType = currentEventType ?? parsed.type ?? null;

      if (eventType === "message_start") {
        const msg = parsed.message;
        if (msg) {
          if (msg.model) capturedModel = msg.model;

          if (msg.usage && typeof msg.usage === "object") {
            capturedUsage = {
              input_tokens: Number(msg.usage.input_tokens) || 0,
              output_tokens: Number(msg.usage.output_tokens) || 0,
              cache_creation_input_tokens:
                msg.usage.cache_creation_input_tokens != null
                  ? Number(msg.usage.cache_creation_input_tokens)
                  : undefined,
              cache_read_input_tokens:
                msg.usage.cache_read_input_tokens != null
                  ? Number(msg.usage.cache_read_input_tokens)
                  : undefined,
            };

            if (
              msg.usage.cache_creation &&
              typeof msg.usage.cache_creation === "object"
            ) {
              capturedCacheDetail = {
                ephemeral_5m_input_tokens:
                  msg.usage.cache_creation.ephemeral_5m_input_tokens,
                ephemeral_1h_input_tokens:
                  msg.usage.cache_creation.ephemeral_1h_input_tokens,
              };
            }
          }
        }
      } else if (eventType === "message_delta") {
        if (parsed.usage && typeof parsed.usage === "object") {
          if (!capturedUsage) {
            capturedUsage = {
              input_tokens: 0,
              output_tokens: 0,
            };
          }

          if (parsed.usage.output_tokens !== undefined) {
            capturedUsage.output_tokens = Number(parsed.usage.output_tokens);
          }
          if (parsed.usage.input_tokens !== undefined) {
            capturedUsage.input_tokens = Number(parsed.usage.input_tokens);
          }
          if (parsed.usage.cache_creation_input_tokens !== undefined) {
            capturedUsage.cache_creation_input_tokens = Number(
              parsed.usage.cache_creation_input_tokens,
            );
          }
          if (parsed.usage.cache_read_input_tokens !== undefined) {
            capturedUsage.cache_read_input_tokens = Number(
              parsed.usage.cache_read_input_tokens,
            );
          }
        }

        if (parsed.delta?.stop_reason) {
          capturedStopReason = parsed.delta.stop_reason;
        }
      } else if (eventType === "content_block_start") {
        const block = parsed.content_block;
        if (block?.type === "tool_use" && block.name && block.id) {
          if (!capturedToolCalls) capturedToolCalls = [];
          capturedToolCalls.push({ name: block.name, id: block.id });
        }
      } else if (eventType === "message_stop") {
        resolve(buildResult());
      }

      currentEventType = null;
    } catch {
      currentEventType = null;
    }
  }

  const readable = upstreamBody.pipeThrough(transform);

  return { readable, resultPromise };
}
