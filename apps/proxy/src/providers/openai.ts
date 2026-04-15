/**
 * OpenAI chat completions adapter.
 *
 * Implements ProviderAdapter for /v1/chat/completions.
 * Wraps existing OpenAI-specific modules (headers, SSE parser, cost calculator).
 */

import type { ProviderAdapter, StreamResult, ParsedResponse, CostEventInsert } from "../lib/provider-types.js";
import type { Attribution } from "../routes/shared.js";
import { buildUpstreamHeaders, buildClientHeaders } from "../lib/headers.js";
import { ensureStreamOptions } from "../lib/request-utils.js";
import { createSSEParser, type SSEResult } from "../lib/sse-parser.js";
import { calculateOpenAICost } from "../lib/cost-calculator.js";
import { estimateMaxCost } from "../lib/cost-estimator.js";
import { OPENAI_BASE_URL } from "../lib/constants.js";
import { isAllowedUpstream } from "../lib/upstream-allowlist.js";
import { errorResponse } from "../lib/errors.js";
import type { RequestContext } from "../lib/context.js";

export const openaiChatAdapter: ProviderAdapter = {
  provider: "openai",
  eventType: "llm",
  supportsStreaming: true,

  prepareBody(body: Record<string, unknown>, isStreaming: boolean): void {
    if (isStreaming) {
      ensureStreamOptions(body);
    }
  },

  estimateMaxCost(model: string, body: Record<string, unknown>, bodyByteLength: number): number {
    return estimateMaxCost(model, body, bodyByteLength);
  },

  buildUpstreamHeaders(request: Request): HeadersInit {
    return buildUpstreamHeaders(request);
  },

  getUpstreamUrl(env: Env, request: Request, ctx: { traceId: string }): string | Response {
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
      return `${upstreamHeader.replace(/\/+$/, "")}/v1/chat/completions`;
    }
    return `${OPENAI_BASE_URL}/v1/chat/completions`;
  },

  getRequestBody(ctx: RequestContext, isStreaming: boolean): string {
    // When streaming, body may have been mutated by prepareBody (ensureStreamOptions),
    // so we re-serialize. Non-streaming uses the original text to avoid double-parse.
    return isStreaming ? JSON.stringify(ctx.body) : ctx.bodyText;
  },

  buildClientHeaders(upstreamResponse: Response, apiVersion: string): Headers {
    return buildClientHeaders(upstreamResponse, apiVersion);
  },

  extractRequestId(upstreamResponse: Response): string {
    return upstreamResponse.headers.get("x-request-id") ?? crypto.randomUUID();
  },

  extractResponseTags(upstreamResponse: Response, body: Record<string, unknown>, isUnpricedModel: boolean): Record<string, string> {
    const tags: Record<string, string> = {};

    // Rate limit proximity
    const rlRemReqs = upstreamResponse.headers.get("x-ratelimit-remaining-requests");
    const rlRemToks = upstreamResponse.headers.get("x-ratelimit-remaining-tokens");
    if (rlRemReqs) tags._ns_ratelimit_remaining_requests = rlRemReqs;
    if (rlRemToks) tags._ns_ratelimit_remaining_tokens = rlRemToks;

    // Request metadata
    const maxTokens = (body.max_completion_tokens ?? body.max_tokens) as number | undefined;
    if (typeof maxTokens === "number") tags._ns_max_tokens = String(maxTokens);
    if (typeof body.temperature === "number") tags._ns_temperature = String(body.temperature);
    if (Array.isArray(body.tools) && body.tools.length > 0) tags._ns_tool_count = String(body.tools.length);
    if (isUnpricedModel) tags._ns_unpriced = "true";

    return tags;
  },

  createSSEParser(upstreamBody: ReadableStream<Uint8Array>): {
    readable: ReadableStream<Uint8Array>;
    resultPromise: Promise<StreamResult>;
  } {
    const { readable, resultPromise } = createSSEParser(upstreamBody);
    return {
      readable,
      resultPromise: resultPromise.then((r: SSEResult): StreamResult => ({
        hasUsage: r.usage != null,
        usage: r.usage,
        model: r.model,
        stopReason: r.finishReason,
        toolCalls: r.toolCalls,
        cancelled: r.cancelled,
        firstChunkMs: r.firstChunkMs,
      })),
    };
  },

  parseNonStreamingResponse(parsed: Record<string, unknown>): ParsedResponse {
    const usage = (parsed as { usage?: unknown }).usage;
    const model = (parsed as { model?: string }).model ?? null;
    const choices = (parsed as { choices?: Array<{ finish_reason?: string; message?: { tool_calls?: Array<{ id: string; function: { name: string } }> } }> }).choices;

    const finishReason: string | null = choices?.[0]?.finish_reason ?? null;

    let toolCalls: { name: string; id: string }[] | null = null;
    try {
      toolCalls = choices?.[0]?.message?.tool_calls?.map(
        (tc) => ({ name: tc.function.name, id: tc.id }),
      ) ?? null;
    } catch { /* malformed tool_calls — proceed without */ }

    return {
      hasUsage: usage != null,
      usage,
      model,
      stopReason: finishReason,
      toolCalls,
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
    return calculateOpenAICost(
      requestModel,
      responseModel,
      parsed.usage as import("../lib/sse-parser.js").OpenAIUsage,
      requestId,
      durationMs,
      attribution,
      toolDefinitionTokens,
    );
  },

  // OpenAI has no provider-specific post-response tags
};
