/**
 * Gemini SSE streaming response parser.
 *
 * Gemini's streaming format (streamGenerateContent?alt=sse) is simpler than
 * both OpenAI and Anthropic:
 * - Only `data:` lines (no `event:` lines like Anthropic)
 * - Each `data:` line is a COMPLETE GenerateContentResponse (not a delta)
 * - No `[DONE]` sentinel like OpenAI — stream closes when done
 * - Usage comes in each chunk's `usageMetadata` — final chunk has complete counts
 * - `responseId` shared across all chunks of the same response
 *
 * Parser strategy: accumulate the LAST usageMetadata and finishReason seen,
 * since the final chunk has the complete token counts.
 *
 * Ref: https://ai.google.dev/gemini-api/docs/api-overview
 */

import type { GeminiUsageMetadata, GeminiResponse } from "./gemini-types.js";

export interface GeminiSSEResult {
  usage: GeminiUsageMetadata | null;
  model: string | null;
  finishReason: string | null;
  toolCalls: { name: string; id: string }[] | null;
  cancelled: boolean;
  firstChunkMs: number | null;
  responseId: string | null;
}

/**
 * Create a TransformStream that passes Gemini SSE bytes through unmodified
 * while extracting usage, model, finish reason, and tool calls from the stream.
 */
export function createGeminiSSEParser(
  upstreamBody: ReadableStream<Uint8Array>,
): {
  readable: ReadableStream<Uint8Array>;
  resultPromise: Promise<GeminiSSEResult>;
} {
  let resolveResult: (value: GeminiSSEResult) => void;
  const resultPromise = new Promise<GeminiSSEResult>((resolve) => {
    resolveResult = resolve;
  });

  let capturedUsage: GeminiUsageMetadata | null = null;
  let capturedModel: string | null = null;
  let capturedFinishReason: string | null = null;
  let capturedToolCalls: { name: string; id: string }[] | null = null;
  let capturedResponseId: string | null = null;
  let firstChunkMs: number | null = null;
  let resolved = false;

  let lineBuffer = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });
  // Gemini sends each SSE event as a complete GenerateContentResponse on a single
  // data: line. Large text chunks or tool-call arguments can exceed 64KB per event.
  // 1MB matches the proxy's MAX_BODY_SIZE and prevents legitimate events from being dropped.
  const MAX_LINE_LENGTH = 1_048_576; // 1MB safety valve

  function resolve(result: GeminiSSEResult): void {
    if (resolved) return;
    resolved = true;
    resolveResult(result);
  }

  let wasCancelled = false;

  function buildResult(): GeminiSSEResult {
    return {
      usage: capturedUsage,
      model: capturedModel,
      finishReason: capturedFinishReason,
      toolCalls: capturedToolCalls,
      cancelled: wasCancelled,
      firstChunkMs,
      responseId: capturedResponseId,
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
        console.warn("[gemini-sse-parser] Dropping oversized line buffer:", lineBuffer.length, "bytes");
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

    // Skip empty lines, comments, and non-data lines
    if (!trimmed || trimmed.startsWith(":") || !trimmed.startsWith("data:")) return;

    let payload = trimmed.slice(5); // Remove "data:"
    if (payload.startsWith(" ")) payload = payload.slice(1); // Remove optional space
    payload = payload.trim();

    try {
      const parsed: GeminiResponse = JSON.parse(payload);

      // Capture responseId (shared across all chunks)
      if (parsed.responseId) {
        capturedResponseId = parsed.responseId;
      }

      // Capture model version
      if (parsed.modelVersion) {
        capturedModel = parsed.modelVersion;
      }

      // Accumulate usage — always take the latest, final chunk has complete counts
      if (parsed.usageMetadata) {
        capturedUsage = parsed.usageMetadata;
      }

      // Extract from first candidate
      const candidate = parsed.candidates?.[0];
      if (candidate) {
        // Capture finish reason (only present in final chunk)
        if (candidate.finishReason) {
          capturedFinishReason = candidate.finishReason;
        }

        // Extract function calls from parts
        if (candidate.content?.parts) {
          for (const part of candidate.content.parts) {
            if (part.functionCall) {
              if (!capturedToolCalls) capturedToolCalls = [];
              capturedToolCalls.push({
                name: part.functionCall.name,
                id: part.functionCall.id ?? crypto.randomUUID(),
              });
            }
          }
        }
      }
    } catch {
      // Malformed JSON chunk — skip silently, same pattern as Anthropic parser
    }
  }

  const readable = upstreamBody.pipeThrough(transform);

  return { readable, resultPromise };
}
