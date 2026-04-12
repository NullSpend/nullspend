// ---------------------------------------------------------------------------
// SSE types
// ---------------------------------------------------------------------------

export interface OpenAISSEUsage {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

export interface OpenAISSEResult {
  usage: OpenAISSEUsage | null;
  model: string | null;
}

export interface AnthropicSSEUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

export interface AnthropicCacheCreationDetail {
  ephemeral_5m_input_tokens?: number;
  ephemeral_1h_input_tokens?: number;
}

export interface AnthropicSSEResult {
  usage: AnthropicSSEUsage | null;
  cacheCreationDetail: AnthropicCacheCreationDetail | null;
  model: string | null;
}

// ---------------------------------------------------------------------------
// Shared line-buffered TransformStream infrastructure
// ---------------------------------------------------------------------------

const MAX_LINE_LENGTH = 65_536; // 64KB safety valve

// ---------------------------------------------------------------------------
// OpenAI SSE parser
// ---------------------------------------------------------------------------

/**
 * Create a TransformStream that passes SSE bytes through unmodified
 * while extracting the usage object and response model from the stream.
 *
 * Simplified version of the proxy's parser — only extracts usage and model.
 */
export function createOpenAISSEParser(body: ReadableStream<Uint8Array>): {
  readable: ReadableStream<Uint8Array>;
  resultPromise: Promise<OpenAISSEResult>;
} {
  let resolveResult: (value: OpenAISSEResult) => void;
  const resultPromise = new Promise<OpenAISSEResult>((resolve) => {
    resolveResult = resolve;
  });

  let capturedUsage: OpenAISSEUsage | null = null;
  let capturedModel: string | null = null;
  let lineBuffer = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });

  function processLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(":")) return;
    if (!trimmed.startsWith("data:")) return;

    let payload = trimmed.slice(5);
    if (payload.startsWith(" ")) payload = payload.slice(1);
    payload = payload.trim();
    if (payload === "[DONE]") return;

    try {
      const parsed = JSON.parse(payload);
      if (!capturedModel && parsed.model) capturedModel = parsed.model;
      if (parsed.usage && typeof parsed.usage === "object") {
        capturedUsage = parsed.usage;
      }
    } catch {
      // Malformed JSON — skip
    }
  }

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      controller.enqueue(chunk);
      const text = decoder.decode(chunk, { stream: true });
      lineBuffer += text;

      if (!lineBuffer.includes("\n") && lineBuffer.length > MAX_LINE_LENGTH) {
        lineBuffer = "";
        return;
      }

      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop()!;
      for (const line of lines) processLine(line);
    },

    flush() {
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      lineBuffer += remaining;
      if (lineBuffer.trim()) processLine(lineBuffer);
      resolveResult({ usage: capturedUsage, model: capturedModel });
    },

    cancel() {
      resolveResult({ usage: capturedUsage, model: capturedModel });
    },
  } as Transformer<Uint8Array, Uint8Array> & { cancel(): void });

  const readable = body.pipeThrough(transform);
  return { readable, resultPromise };
}

// ---------------------------------------------------------------------------
// Anthropic SSE parser
// ---------------------------------------------------------------------------

/**
 * Create a TransformStream that passes Anthropic SSE bytes through unmodified
 * while extracting usage, model, and cache detail from the stream.
 *
 * Handles named events (message_start, message_delta, message_stop).
 */
export function createAnthropicSSEParser(body: ReadableStream<Uint8Array>): {
  readable: ReadableStream<Uint8Array>;
  resultPromise: Promise<AnthropicSSEResult>;
} {
  let resolveResult: (value: AnthropicSSEResult) => void;
  const resultPromise = new Promise<AnthropicSSEResult>((resolve) => {
    resolveResult = resolve;
  });

  let capturedUsage: AnthropicSSEUsage | null = null;
  let capturedCacheDetail: AnthropicCacheCreationDetail | null = null;
  let capturedModel: string | null = null;
  let resolved = false;
  let currentEventType: string | null = null;
  let lineBuffer = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });

  function resolve(result: AnthropicSSEResult): void {
    if (resolved) return;
    resolved = true;
    resolveResult(result);
  }

  function buildResult(): AnthropicSSEResult {
    return {
      usage: capturedUsage,
      cacheCreationDetail: capturedCacheDetail,
      model: capturedModel,
    };
  }

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
            capturedUsage = { input_tokens: 0, output_tokens: 0 };
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
      } else if (eventType === "message_stop") {
        resolve(buildResult());
      }

      currentEventType = null;
    } catch {
      currentEventType = null;
    }
  }

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      controller.enqueue(chunk);
      const text = decoder.decode(chunk, { stream: true });
      lineBuffer += text;

      if (!lineBuffer.includes("\n") && lineBuffer.length > MAX_LINE_LENGTH) {
        lineBuffer = "";
        return;
      }

      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop()!;
      for (const line of lines) processLine(line);
    },

    flush() {
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      lineBuffer += remaining;
      if (lineBuffer.trim()) processLine(lineBuffer);
      resolve(buildResult());
    },

    cancel() {
      resolve(buildResult());
    },
  } as Transformer<Uint8Array, Uint8Array> & { cancel(): void });

  const readable = body.pipeThrough(transform);
  return { readable, resultPromise };
}
