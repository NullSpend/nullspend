/**
 * Provider adapter interface for the generic route handler.
 *
 * Each provider (OpenAI, Anthropic, Gemini...) implements this interface
 * as a plain object with function properties. The generic handler in
 * provider-handler.ts calls these methods at the appropriate lifecycle
 * points. Provider-specific logic is isolated here; shared budget,
 * webhook, and body storage logic stays in the handler.
 *
 * Architecture: apps/proxy/CLAUDE.md → "Entry & Routing"
 * Design doc: docs/plans/p2-1-provider-adapter-dedup.md
 */

import type { RequestContext } from "./context.js";
import type { Attribution, Provider } from "../routes/shared.js";

// ── Cost event insert type (re-export for adapter convenience) ──────
import type { NewCostEventRow } from "@nullspend/db";
export type CostEventInsert = Omit<NewCostEventRow, "id" | "createdAt">;

// ── Adapter interface ───────────────────────────────────────────────

export interface ProviderAdapter {
  /** Provider identifier — used in metrics, logs, webhooks. */
  readonly provider: Provider;

  /** Cost event type — "llm" for chat, "embedding" for embeddings. */
  readonly eventType: string;

  /** Whether this endpoint type supports streaming responses. */
  readonly supportsStreaming: boolean;

  // ── Model & streaming detection ────────────────────────────────

  /**
   * Extract the model identifier from the request.
   * Default (when absent): reads `body.model` via extractModelFromBody.
   * Gemini overrides this to parse the model from the URL path
   * (`/v1beta/models/{model}:generateContent`).
   */
  extractModel?(request: Request, body: Record<string, unknown>): string;

  /**
   * Determine whether this request is streaming.
   * Default (when absent): `body.stream === true`.
   * Gemini overrides this because streaming is determined by the endpoint
   * suffix (`:streamGenerateContent` vs `:generateContent`), not a body field.
   */
  isStreaming?(request: Request, body: Record<string, unknown>): boolean;

  // ── Pre-request ─────────────────────────────────────────────────

  /** Optional body mutation before upstream fetch (e.g., ensureStreamOptions for OpenAI). */
  prepareBody?(body: Record<string, unknown>, isStreaming: boolean): void;

  /** Pre-request cost estimate for budget enforcement. */
  estimateMaxCost(model: string, body: Record<string, unknown>, bodyByteLength: number): number;

  /** Build provider-specific headers for the upstream request. */
  buildUpstreamHeaders(request: Request): HeadersInit;

  /**
   * Resolve the upstream URL. May validate headers (e.g., OpenAI x-nullspend-upstream).
   * Returns the URL string on success, or a Response to return early on validation failure.
   * Accepts Request so provider can read provider-specific routing headers.
   */
  getUpstreamUrl(env: Env, request: Request, ctx: { traceId: string }): string | Response;

  /**
   * Build the request body string for the upstream fetch.
   * OpenAI: `JSON.stringify(body)` when streaming (to include stream_options),
   * otherwise `bodyText`. Anthropic: always `bodyText`.
   */
  getRequestBody(ctx: RequestContext, isStreaming: boolean): string;

  /** Optional per-provider pre-fetch metric emission (e.g., Anthropic beta header tracking). */
  emitPreFetchMetrics?(request: Request, model: string): void;

  // ── Response context ────────────────────────────────────────────

  /** Build provider-specific client response headers. */
  buildClientHeaders(upstreamResponse: Response, apiVersion: string): Headers;

  /** Extract the provider-specific request ID from the upstream response. */
  extractRequestId(upstreamResponse: Response): string;

  /**
   * Extract provider-specific response tags (rate limits, request metadata).
   * Merged into enrichment.tags alongside ctx.tags.
   */
  extractResponseTags(upstreamResponse: Response, body: Record<string, unknown>, isUnpricedModel: boolean): Record<string, string>;

  // ── Streaming ───────────────────────────────────────────────────

  /**
   * Create the SSE parser TransformStream for streaming responses.
   * Returns a readable stream that passes bytes through to the client
   * and a promise that resolves with the parsed result when the stream
   * completes.
   *
   * Only called when supportsStreaming is true and the request is streaming.
   */
  createSSEParser(upstreamBody: ReadableStream<Uint8Array>): {
    readable: ReadableStream<Uint8Array>;
    resultPromise: Promise<StreamResult>;
  };

  // ── Non-streaming response parsing ──────────────────────────────

  /**
   * Parse a non-streaming JSON response body into a normalized result.
   * Extracts usage, model, stopReason, toolCalls, and provider-specific data.
   */
  parseNonStreamingResponse(parsed: Record<string, unknown>): ParsedResponse;

  // ── Cost calculation ────────────────────────────────────────────

  /**
   * Calculate cost from a streaming or non-streaming result.
   * The adapter wraps the underlying provider-specific calculator and
   * handles type-specific fields (e.g., Anthropic cacheCreationDetail).
   * Returns a CostEventInsert ready for logCostEventQueued.
   */
  calculateCost(
    requestModel: string,
    responseModel: string | null,
    parsed: ParsedResponse | StreamResult,
    requestId: string,
    durationMs: number,
    attribution: Attribution,
    toolDefinitionTokens: number,
  ): CostEventInsert;

  /**
   * Optional post-response metadata tags (e.g., Anthropic cache tags, long-context flag).
   * Merged into the cost event tags.
   */
  extractPostResponseTags?(parsed: ParsedResponse | StreamResult): Record<string, string>;
}

// ── Shared result types ─────────────────────────────────────────────

/**
 * Normalized streaming result. Both SSE parsers (OpenAI, Anthropic)
 * resolve to this shape. Provider-specific data lives in providerExtra.
 */
export interface StreamResult {
  /** Whether the upstream provided usage data. */
  hasUsage: boolean;
  /** Raw usage object (provider-specific shape, passed to calculateCost). */
  usage: unknown;
  model: string | null;
  stopReason: string | null;
  toolCalls: { name: string; id: string }[] | null;
  cancelled: boolean;
  firstChunkMs: number | null;
  /** Provider-specific data needed by calculateCost and extractPostResponseTags. */
  providerExtra?: Record<string, unknown>;
}

/**
 * Normalized non-streaming parsed response.
 */
export interface ParsedResponse {
  /** Whether the upstream provided usage data. */
  hasUsage: boolean;
  /** Raw usage object (provider-specific shape, passed to calculateCost). */
  usage: unknown;
  model: string | null;
  stopReason: string | null;
  toolCalls: { name: string; id: string }[] | null;
  /** Provider-specific data needed by calculateCost and extractPostResponseTags. */
  providerExtra?: Record<string, unknown>;
}
