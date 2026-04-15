/**
 * Anthropic messages adapter.
 *
 * Implements ProviderAdapter for /v1/messages.
 * Wraps existing Anthropic-specific modules (headers, SSE parser, cost calculator).
 * Handles Anthropic-unique features: cache creation detail, long-context detection,
 * beta header tracking.
 */

import type { ProviderAdapter, StreamResult, ParsedResponse, CostEventInsert } from "../lib/provider-types.js";
import type { Attribution } from "../routes/shared.js";
import { buildAnthropicUpstreamHeaders, buildAnthropicClientHeaders } from "../lib/anthropic-headers.js";
import { createAnthropicSSEParser, type AnthropicSSEResult } from "../lib/anthropic-sse-parser.js";
import { calculateAnthropicCost } from "../lib/anthropic-cost-calculator.js";
import type { AnthropicCacheCreationDetail, AnthropicRawUsage } from "../lib/anthropic-types.js";
import { estimateAnthropicMaxCost } from "../lib/anthropic-cost-estimator.js";
import { ANTHROPIC_BASE_URL } from "../lib/constants.js";
import { emitMetric } from "../lib/metrics.js";
import type { RequestContext } from "../lib/context.js";

export const anthropicChatAdapter: ProviderAdapter = {
  provider: "anthropic",
  eventType: "llm",
  supportsStreaming: true,

  // Anthropic doesn't need prepareBody (no ensureStreamOptions)

  estimateMaxCost(model: string, body: Record<string, unknown>, bodyByteLength: number): number {
    return estimateAnthropicMaxCost(model, body, bodyByteLength);
  },

  buildUpstreamHeaders(request: Request): HeadersInit {
    return buildAnthropicUpstreamHeaders(request);
  },

  getUpstreamUrl(_env: Env, _request: Request, _ctx: { traceId: string }): string {
    return `${ANTHROPIC_BASE_URL}/v1/messages`;
  },

  getRequestBody(ctx: RequestContext, _isStreaming: boolean): string {
    // Anthropic always uses the original body text
    return ctx.bodyText;
  },

  emitPreFetchMetrics(request: Request, model: string): void {
    // PXY-14: Track anthropic-beta usage for audit
    const betaHeader = request.headers.get("anthropic-beta");
    if (betaHeader) {
      emitMetric("anthropic_beta_used", { beta: betaHeader, model });
    }
  },

  buildClientHeaders(upstreamResponse: Response, apiVersion: string): Headers {
    return buildAnthropicClientHeaders(upstreamResponse, apiVersion);
  },

  extractRequestId(upstreamResponse: Response): string {
    return upstreamResponse.headers.get("request-id") ?? crypto.randomUUID();
  },

  extractResponseTags(upstreamResponse: Response, body: Record<string, unknown>, isUnpricedModel: boolean): Record<string, string> {
    const tags: Record<string, string> = {};

    // Rate limit proximity (Anthropic-specific header names)
    const rlRemReqs = upstreamResponse.headers.get("anthropic-ratelimit-requests-remaining");
    const rlRemToks = upstreamResponse.headers.get("anthropic-ratelimit-tokens-remaining");
    if (rlRemReqs) tags._ns_ratelimit_remaining_requests = rlRemReqs;
    if (rlRemToks) tags._ns_ratelimit_remaining_tokens = rlRemToks;

    // Request metadata
    if (typeof body.max_tokens === "number") tags._ns_max_tokens = String(body.max_tokens);
    if (typeof body.temperature === "number") tags._ns_temperature = String(body.temperature);
    if (Array.isArray(body.tools) && body.tools.length > 0) tags._ns_tool_count = String(body.tools.length);
    if (isUnpricedModel) tags._ns_unpriced = "true";

    return tags;
  },

  createSSEParser(upstreamBody: ReadableStream<Uint8Array>): {
    readable: ReadableStream<Uint8Array>;
    resultPromise: Promise<StreamResult>;
  } {
    const { readable, resultPromise } = createAnthropicSSEParser(upstreamBody);
    return {
      readable,
      resultPromise: resultPromise.then((r: AnthropicSSEResult): StreamResult => ({
        hasUsage: r.usage != null,
        usage: r.usage,
        model: r.model,
        stopReason: r.stopReason,
        toolCalls: r.toolCalls,
        cancelled: r.cancelled,
        firstChunkMs: r.firstChunkMs,
        providerExtra: {
          cacheCreationDetail: r.cacheCreationDetail,
        },
      })),
    };
  },

  parseNonStreamingResponse(parsed: Record<string, unknown>): ParsedResponse {
    const usage = (parsed as { usage?: AnthropicRawUsage }).usage;
    const model = (parsed as { model?: string }).model ?? null;
    const stopReason: string | null = (parsed as { stop_reason?: string }).stop_reason ?? null;
    const content = (parsed as { content?: Array<{ type: string; id?: string; name?: string }> }).content;

    let toolCalls: { name: string; id: string }[] | null = null;
    try {
      if (Array.isArray(content)) {
        const toolUseBlocks = content.filter((b) => b.type === "tool_use");
        if (toolUseBlocks.length > 0) {
          toolCalls = toolUseBlocks.map((b) => ({ name: b.name!, id: b.id! }));
        }
      }
    } catch { /* malformed content blocks — proceed without */ }

    // Extract cacheCreationDetail from usage for non-streaming
    let cacheCreationDetail: AnthropicCacheCreationDetail | null = null;
    if (usage) {
      const cacheCreation = (usage as unknown as { cache_creation?: unknown }).cache_creation;
      if (cacheCreation && typeof cacheCreation === "object") {
        cacheCreationDetail = cacheCreation as AnthropicCacheCreationDetail;
      }
    }

    return {
      hasUsage: usage != null,
      usage,
      model,
      stopReason,
      toolCalls,
      providerExtra: {
        cacheCreationDetail,
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
    const cacheCreationDetail = (parsed.providerExtra?.cacheCreationDetail as AnthropicCacheCreationDetail) ?? null;
    return calculateAnthropicCost(
      requestModel,
      responseModel,
      parsed.usage as AnthropicRawUsage,
      cacheCreationDetail,
      requestId,
      durationMs,
      attribution,
      toolDefinitionTokens,
    );
  },

  extractPostResponseTags(parsed: ParsedResponse | StreamResult): Record<string, string> {
    const tags: Record<string, string> = {};
    const usage = parsed.usage as AnthropicRawUsage | null;
    if (!usage) return tags;

    // Cache read/write split
    if (usage.cache_creation_input_tokens != null) {
      tags._ns_cache_write_tokens = String(usage.cache_creation_input_tokens);
    }
    if (usage.cache_read_input_tokens != null) {
      tags._ns_cache_read_tokens = String(usage.cache_read_input_tokens);
    }

    // Long-context detection (>200K total input)
    const totalInput = (usage.input_tokens ?? 0)
      + (usage.cache_creation_input_tokens ?? 0)
      + (usage.cache_read_input_tokens ?? 0);
    if (totalInput > 200_000) {
      tags._ns_long_context = "true";
    }

    return tags;
  },
};
