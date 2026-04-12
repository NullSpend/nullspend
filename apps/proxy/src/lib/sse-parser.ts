export interface SSEResult {
  usage: OpenAIUsage | null;
  model: string | null;
  finishReason: string | null;
  toolCalls: { name: string; id: string }[] | null;
  cancelled: boolean;
  firstChunkMs: number | null;
}

export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

/**
 * Create a TransformStream that passes SSE bytes through unmodified
 * while extracting the usage object and response model from the stream.
 *
 * Uses a single TextDecoder with { stream: true } for multi-byte UTF-8 safety.
 * Line-buffered: only processes complete \n-terminated lines.
 */
export function createSSEParser(upstreamBody: ReadableStream<Uint8Array>): {
  readable: ReadableStream<Uint8Array>;
  resultPromise: Promise<SSEResult>;
} {
  let resolveResult: (value: SSEResult) => void;
  const resultPromise = new Promise<SSEResult>((resolve) => {
    resolveResult = resolve;
  });

  let capturedUsage: OpenAIUsage | null = null;
  let capturedModel: string | null = null;
  let capturedFinishReason: string | null = null;
  let capturedToolCalls: { name: string; id: string }[] | null = null;
  let firstChunkMs: number | null = null;
  let lineBuffer = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const MAX_LINE_LENGTH = 65_536; // 64KB — safety valve for malformed streams

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      if (firstChunkMs === null) firstChunkMs = performance.now();
      controller.enqueue(chunk);

      const text = decoder.decode(chunk, { stream: true });
      lineBuffer += text;

      // Safety valve: drop oversized incomplete lines to prevent memory exhaustion
      if (!lineBuffer.includes("\n") && lineBuffer.length > MAX_LINE_LENGTH) {
        console.warn("[sse-parser] Dropping oversized line buffer:", lineBuffer.length, "bytes");
        lineBuffer = "";
        return;
      }

      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop()!;

      for (const line of lines) {
        processLine(line);
      }
    },

    flush(_controller) {
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      lineBuffer += remaining;

      if (lineBuffer.trim()) {
        processLine(lineBuffer);
      }

      resolveResult({
        usage: capturedUsage,
        model: capturedModel,
        finishReason: capturedFinishReason,
        toolCalls: capturedToolCalls,
        cancelled: false,
        firstChunkMs,
      });
    },

    cancel() {
      resolveResult({
        usage: capturedUsage,
        model: capturedModel,
        finishReason: capturedFinishReason,
        toolCalls: capturedToolCalls,
        cancelled: true,
        firstChunkMs,
      });
    },
  });

  function processLine(line: string): void {
    const trimmed = line.trim();

    if (!trimmed) return;
    if (trimmed.startsWith(":")) return;

    if (!trimmed.startsWith("data:")) return;

    let payload = trimmed.slice(5);
    if (payload.startsWith(" ")) {
      payload = payload.slice(1);
    }
    payload = payload.trim();

    if (payload === "[DONE]") return;

    try {
      const parsed = JSON.parse(payload);

      if (!capturedModel && parsed.model) {
        capturedModel = parsed.model;
      }

      if (parsed.usage && typeof parsed.usage === "object") {
        capturedUsage = parsed.usage;
      }

      const finishReason = parsed.choices?.[0]?.finish_reason ?? parsed.choices?.[0]?.delta?.finish_reason;
      if (finishReason && typeof finishReason === "string") {
        capturedFinishReason = finishReason;
      }

      const deltaToolCalls = parsed.choices?.[0]?.delta?.tool_calls;
      if (Array.isArray(deltaToolCalls)) {
        for (const tc of deltaToolCalls) {
          if (tc.id && tc.function?.name) {
            if (!capturedToolCalls) capturedToolCalls = [];
            capturedToolCalls.push({ name: tc.function.name, id: tc.id });
          }
        }
      }
    } catch {
      // Malformed JSON — skip silently
    }
  }

  const readable = upstreamBody.pipeThrough(transform);

  return { readable, resultPromise };
}
