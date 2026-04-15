/**
 * Gemini chat adapter.
 *
 * Implements ProviderAdapter for Google's native Gemini REST API:
 *   POST /v1beta/models/{model}:generateContent     (non-streaming)
 *   POST /v1beta/models/{model}:streamGenerateContent (streaming)
 *
 * Key differences from OpenAI/Anthropic:
 *   - Model is in the URL path, not the request body
 *   - Streaming is determined by endpoint suffix, not a body field
 *   - Auth uses x-goog-api-key, not Authorization: Bearer or x-api-key
 *   - Usage is in usageMetadata (not usage), with different field names
 *   - No request-id response header — uses responseId from response body
 *   - SSE format: each data: line is a complete response (not a delta)
 *
 * Ref: https://ai.google.dev/gemini-api/docs/api-overview
 */

import type { ProviderAdapter, StreamResult, ParsedResponse, CostEventInsert } from "../lib/provider-types.js";
import type { Attribution } from "../routes/shared.js";
import type { GeminiUsageMetadata, GeminiResponse } from "../lib/gemini-types.js";
import { buildGeminiUpstreamHeaders, buildGeminiClientHeaders } from "../lib/gemini-headers.js";
import { createGeminiSSEParser, type GeminiSSEResult } from "../lib/gemini-sse-parser.js";
import { calculateGeminiCost } from "../lib/gemini-cost-calculator.js";
import { estimateGeminiMaxCost } from "../lib/gemini-cost-estimator.js";
import { GEMINI_BASE_URL } from "../lib/constants.js";
import { isAllowedUpstream } from "../lib/upstream-allowlist.js";
import { errorResponse } from "../lib/errors.js";
import type { RequestContext } from "../lib/context.js";

/**
 * Parse model name from a Gemini URL path: /v1beta/models/{model}:{method}
 *
 * Validates model name contains only safe characters (alphanumeric, hyphens,
 * dots, underscores). Rejects percent-encoded traversal sequences, slashes,
 * and other path-injection characters to prevent SSRF on the upstream host.
 */
const SAFE_MODEL_NAME = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;

function extractModelFromPath(pathname: string): string {
  const match = pathname.match(/^\/v1beta\/models\/([^:]+):/);
  const model = match?.[1] ?? "unknown";
  // Reject model names with path traversal or unsafe characters
  if (!SAFE_MODEL_NAME.test(model)) return "unknown";
  return model;
}

export const geminiChatAdapter: ProviderAdapter = {
  provider: "google",
  eventType: "llm",
  supportsStreaming: true,

  extractModel(request: Request): string {
    return extractModelFromPath(new URL(request.url).pathname);
  },

  isStreaming(request: Request): boolean {
    return new URL(request.url).pathname.endsWith(":streamGenerateContent");
  },

  estimateMaxCost(model: string, body: Record<string, unknown>, bodyByteLength: number): number {
    return estimateGeminiMaxCost(model, body, bodyByteLength);
  },

  buildUpstreamHeaders(request: Request): HeadersInit {
    return buildGeminiUpstreamHeaders(request);
  },

  getUpstreamUrl(env: Env, request: Request, ctx: { traceId: string }): string | Response {
    const url = new URL(request.url);
    const model = extractModelFromPath(url.pathname);
    const isStream = url.pathname.endsWith(":streamGenerateContent");
    const method = isStream ? "streamGenerateContent" : "generateContent";

    // Support x-nullspend-upstream override (same pattern as OpenAI adapter)
    const upstreamHeader = request.headers.get("x-nullspend-upstream");
    if (upstreamHeader) {
      if (!isAllowedUpstream(upstreamHeader)) {
        const resp = errorResponse(
          "invalid_upstream",
          "The specified upstream URL is not supported",
          400,
        );
        resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
        return resp;
      }
      const base = upstreamHeader.replace(/\/+$/, "");
      const upstream = `${base}/v1beta/models/${model}:${method}`;
      return isStream ? `${upstream}?alt=sse` : upstream;
    }

    const upstream = `${GEMINI_BASE_URL}/models/${model}:${method}`;
    // Auto-enforce ?alt=sse for streaming to prevent user error
    return isStream ? `${upstream}?alt=sse` : upstream;
  },

  getRequestBody(ctx: RequestContext, _isStreaming: boolean): string {
    // Gemini body passes through unmodified — native format
    return ctx.bodyText;
  },

  buildClientHeaders(upstreamResponse: Response, apiVersion: string): Headers {
    return buildGeminiClientHeaders(upstreamResponse, apiVersion);
  },

  extractRequestId(_upstreamResponse: Response): string {
    // Gemini has no request-id response header.
    // Google's responseId is in the body, not headers — captured separately
    // in parseNonStreamingResponse and SSE parser, exposed as _ns_google_response_id tag.
    return crypto.randomUUID();
  },

  extractResponseTags(_upstreamResponse: Response, body: Record<string, unknown>, isUnpricedModel: boolean): Record<string, string> {
    const tags: Record<string, string> = {};

    // Request metadata from generationConfig
    const config = body.generationConfig as { maxOutputTokens?: number; temperature?: number } | undefined;
    if (typeof config?.maxOutputTokens === "number") tags._ns_max_tokens = String(config.maxOutputTokens);
    if (typeof config?.temperature === "number") tags._ns_temperature = String(config.temperature);

    // Tool count from tools array
    const tools = body.tools as Array<{ functionDeclarations?: unknown[] }> | undefined;
    if (Array.isArray(tools)) {
      const totalDecls = tools.reduce((sum, t) => sum + (Array.isArray(t.functionDeclarations) ? t.functionDeclarations.length : 0), 0);
      if (totalDecls > 0) tags._ns_tool_count = String(totalDecls);
    }

    if (isUnpricedModel) tags._ns_unpriced = "true";

    return tags;
  },

  createSSEParser(upstreamBody: ReadableStream<Uint8Array>): {
    readable: ReadableStream<Uint8Array>;
    resultPromise: Promise<StreamResult>;
  } {
    const { readable, resultPromise } = createGeminiSSEParser(upstreamBody);
    return {
      readable,
      resultPromise: resultPromise.then((r: GeminiSSEResult): StreamResult => ({
        hasUsage: r.usage != null && (r.usage.promptTokenCount != null || r.usage.candidatesTokenCount != null),
        usage: r.usage,
        model: r.model,
        stopReason: r.finishReason,
        toolCalls: r.toolCalls,
        cancelled: r.cancelled,
        firstChunkMs: r.firstChunkMs,
        providerExtra: {
          responseId: r.responseId,
        },
      })),
    };
  },

  parseNonStreamingResponse(parsed: Record<string, unknown>): ParsedResponse {
    const resp = parsed as GeminiResponse;
    const usage = resp.usageMetadata ?? null;
    const model = resp.modelVersion ?? null;
    const candidate = resp.candidates?.[0];
    const finishReason: string | null = candidate?.finishReason ?? null;

    let toolCalls: { name: string; id: string }[] | null = null;
    try {
      if (candidate?.content?.parts) {
        const funcParts = candidate.content.parts.filter((p) => p.functionCall);
        if (funcParts.length > 0) {
          toolCalls = funcParts.map((p) => ({
            name: p.functionCall!.name,
            id: p.functionCall!.id ?? crypto.randomUUID(),
          }));
        }
      }
    } catch { /* malformed parts — proceed without */ }

    return {
      hasUsage: usage != null && (usage.promptTokenCount != null || usage.candidatesTokenCount != null),
      usage,
      model,
      stopReason: finishReason,
      toolCalls,
      providerExtra: {
        responseId: resp.responseId ?? null,
      },
    };
  },

  calculateCost(
    requestModel: string,
    responseModel: string | null,
    parsed: ParsedResponse | StreamResult,
    requestId: string,
    durationMs: number,
    attribution: Attribution,
    toolDefinitionTokens: number,
  ): CostEventInsert {
    return calculateGeminiCost(
      requestModel,
      responseModel,
      parsed.usage as GeminiUsageMetadata,
      requestId,
      durationMs,
      attribution,
      toolDefinitionTokens,
    );
  },

  extractPostResponseTags(parsed: ParsedResponse | StreamResult): Record<string, string> {
    const tags: Record<string, string> = {};
    const usage = parsed.usage as GeminiUsageMetadata | null;

    // Thinking tokens tag (like OpenAI reasoning, Anthropic cache split)
    if (usage?.thoughtsTokenCount != null && usage.thoughtsTokenCount > 0) {
      tags._ns_thinking_tokens = String(usage.thoughtsTokenCount);
    }

    // Google response ID for cross-referencing with Google's systems
    const responseId = parsed.providerExtra?.responseId as string | null;
    if (responseId) {
      tags._ns_google_response_id = responseId;
    }

    return tags;
  },
};
