import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import { errorResponse } from "../lib/errors.js";
import {
  buildAnthropicUpstreamHeaders,
  buildAnthropicClientHeaders,
} from "../lib/anthropic-headers.js";
import { appendTimingHeaders } from "../lib/headers.js";
import { extractModelFromBody } from "../lib/request-utils.js";
import { createAnthropicSSEParser } from "../lib/anthropic-sse-parser.js";
import { calculateAnthropicCost } from "../lib/anthropic-cost-calculator.js";
import type { AnthropicCacheCreationDetail } from "../lib/anthropic-types.js";
import { isKnownModel } from "@nullspend/cost-engine";
import { logCostEventQueued, getCostEventQueue } from "../lib/cost-event-queue.js";
import { ANTHROPIC_BASE_URL } from "../lib/constants.js";
import type { BudgetEntity } from "../lib/budget-do-lookup.js";
import { estimateAnthropicMaxCost } from "../lib/anthropic-cost-estimator.js";
import { checkBudget, reconcileBudgetQueued, getReconcileQueue } from "../lib/budget-orchestrator.js";
import { sanitizeUpstreamError } from "../lib/sanitize-upstream-error.js";
import { stripNsPrefix } from "../lib/validation.js";
import { emitMetric } from "../lib/metrics.js";
import { writeLatencyDataPoint } from "../lib/write-metric.js";
import { handleBudgetDenials, dispatchVelocityRecoveryWebhooks, dispatchCostEventWebhooks, type Attribution, type EnrichmentFields } from "./shared.js";
import { storeRequestBody, storeResponseBody, storeStreamingResponseBody, createStreamBodyAccumulator } from "../lib/body-storage.js";

const UPSTREAM_TIMEOUT_MS = 120_000;


export async function handleAnthropicMessages(
  request: Request,
  env: Env,
  ctx: RequestContext,
): Promise<Response> {
  const requestModel = extractModelFromBody(ctx.body);
  const safeModel = requestModel.slice(0, 200);

  // --- Provider restriction ---
  if (ctx.auth.allowedProviders && !ctx.auth.allowedProviders.includes("anthropic")) {
    emitMetric("mandate_denied", { reason: "provider_not_allowed", provider: "anthropic", model: safeModel });
    const resp = errorResponse("mandate_violation", "Provider anthropic is not allowed by this key's policy", 403, {
      mandate: "allowed_providers",
      requested: "anthropic",
      allowed: ctx.auth.allowedProviders,
    });
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return resp;
  }

  // --- Model restriction ---
  if (ctx.auth.allowedModels && !ctx.auth.allowedModels.includes(requestModel)) {
    emitMetric("mandate_denied", { reason: "model_not_allowed", provider: "anthropic", model: safeModel });
    const resp = errorResponse("mandate_violation", `Model ${safeModel} is not allowed by this key's policy. Allowed: ${ctx.auth.allowedModels.join(", ")}`, 403, {
      mandate: "allowed_models",
      requested: safeModel,
      allowed: ctx.auth.allowedModels,
    });
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return resp;
  }

  const attribution: Attribution = {
    userId: ctx.auth.userId,
    apiKeyId: ctx.auth.keyId,
    actionId: stripNsPrefix("ns_act_", request.headers.get("x-nullspend-action-id")),
  };

  const isUnpricedModel = !isKnownModel("anthropic", requestModel);
  if (isUnpricedModel) {
    emitMetric("unknown_model", { provider: "anthropic", model: requestModel });
  }

  const toolDefinitionTokens = Array.isArray(ctx.body.tools) && ctx.body.tools.length > 0
    ? Math.ceil(JSON.stringify(ctx.body.tools).length / 4)
    : 0;

  const isStreaming = ctx.body.stream === true;

  // --- Budget enforcement + optimistic upstream fetch ---
  const estimate = estimateAnthropicMaxCost(requestModel, ctx.body, ctx.bodyByteLength);

  let reservationId: string | null = null;
  let budgetEntities: BudgetEntity[] = [];
  let budgetStatus: "skipped" | "approved" | "denied" = "skipped";

  // Start upstream fetch optimistically — runs in parallel with budget check.
  // Budget DO RPC takes 15-33ms; upstream TTFB is >200ms, so the budget
  // result is always known before the provider sends any data back.
  const upstreamHeaders = buildAnthropicUpstreamHeaders(request);
  const budgetAbort = new AbortController();
  const startTime = performance.now();
  const upstreamFetchPromise = fetch(
    `${ANTHROPIC_BASE_URL}/v1/messages`,
    {
      method: "POST",
      headers: upstreamHeaders,
      body: ctx.bodyText,
      signal: AbortSignal.any([budgetAbort.signal, AbortSignal.timeout(UPSTREAM_TIMEOUT_MS)]),
    },
  );

  try {
    const budgetStartMs = performance.now();
    const outcome = await checkBudget(env, ctx, estimate);
    if (ctx.stepTiming) ctx.stepTiming.budgetCheckMs = Math.round(performance.now() - budgetStartMs);

    budgetStatus = outcome.status;
    reservationId = outcome.reservationId;
    budgetEntities = outcome.budgetEntities;

    const denialResponse = handleBudgetDenials(outcome, ctx, env, "anthropic", requestModel, estimate, budgetEntities);
    if (denialResponse) {
      budgetAbort.abort();
      upstreamFetchPromise.catch((e) => { if (e?.name !== "AbortError") console.warn("[anthropic-route] Upstream fetch error after budget denial:", e); });
      return denialResponse;
    }

    dispatchVelocityRecoveryWebhooks(outcome, ctx, env, "anthropic");
  } catch {
    budgetAbort.abort();
    upstreamFetchPromise.catch((e) => { if (e?.name !== "AbortError") console.warn("[anthropic-route] Upstream fetch error after budget error:", e); });
    const budgetUnavailResp = errorResponse("budget_unavailable", "Budget service unavailable", 503);
    budgetUnavailResp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return budgetUnavailResp;
  }

  // Budget approved — await the already-in-flight upstream fetch
  try {
    const upstreamResponse = await upstreamFetchPromise;
    const upstreamDurationMs = Math.round(performance.now() - startTime);

    // Capture provider rate limit proximity in tags for analytics
    const rateLimitTags: Record<string, string> = {};
    const rlRemReqs = upstreamResponse.headers.get("anthropic-ratelimit-requests-remaining");
    const rlRemToks = upstreamResponse.headers.get("anthropic-ratelimit-tokens-remaining");
    if (rlRemReqs) rateLimitTags._ns_ratelimit_remaining_requests = rlRemReqs;
    if (rlRemToks) rateLimitTags._ns_ratelimit_remaining_tokens = rlRemToks;

    // Capture request metadata in tags for analytics
    const metadataTags: Record<string, string> = {};
    if (typeof ctx.body.max_tokens === "number") metadataTags._ns_max_tokens = String(ctx.body.max_tokens);
    if (typeof ctx.body.temperature === "number") metadataTags._ns_temperature = String(ctx.body.temperature);
    if (Array.isArray(ctx.body.tools) && ctx.body.tools.length > 0) metadataTags._ns_tool_count = String(ctx.body.tools.length);
    if (isUnpricedModel) metadataTags._ns_unpriced = "true";

    const enrichment: EnrichmentFields = {
      upstreamDurationMs,
      sessionId: ctx.sessionId,
      traceId: ctx.traceId,
      toolDefinitionTokens,
      tags: { ...ctx.tags, ...rateLimitTags, ...metadataTags },
      budgetStatus,
      estimatedCostMicrodollars: estimate,
      orgId: ctx.auth.orgId,
    };

    const requestId =
      upstreamResponse.headers.get("request-id") ?? crypto.randomUUID();

    if (!upstreamResponse.ok) {
      if (reservationId) {
        waitUntil(
          reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, ctx.connectionString),
        );
      }
      const clientHeaders = buildAnthropicClientHeaders(upstreamResponse, ctx.resolvedApiVersion);
      clientHeaders.set("X-NullSpend-Trace-Id", ctx.traceId);
      if (ctx.sessionId) clientHeaders.set("X-NullSpend-Session", ctx.sessionId);
      const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, upstreamDurationMs, ctx.stepTiming);
      emitMetric("proxy_latency", { provider: "anthropic", model: requestModel, overheadMs, upstreamMs: upstreamDurationMs, totalMs, streaming: false });
      writeLatencyDataPoint(env, "anthropic", requestModel, false, upstreamResponse.status, overheadMs, upstreamDurationMs, totalMs, undefined, ctx.auth.userId);
      const sanitizedBody = await sanitizeUpstreamError(upstreamResponse, "anthropic");
      // Log error request/response bodies — valuable for debugging upstream failures
      const bodyBucket = (env as Record<string, unknown>).BODY_STORAGE as R2Bucket | undefined;
      if (ctx.requestLoggingEnabled && bodyBucket) {
        waitUntil(storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText));
        waitUntil(storeResponseBody(bodyBucket, ctx.ownerId, requestId, sanitizedBody));
      }
      clientHeaders.set("content-type", "application/json");
      return new Response(sanitizedBody, {
        status: upstreamResponse.status,
        headers: clientHeaders,
      });
    }

    const clientHeaders = buildAnthropicClientHeaders(upstreamResponse, ctx.resolvedApiVersion);
    clientHeaders.set("X-NullSpend-Trace-Id", ctx.traceId);
    if (ctx.sessionId) clientHeaders.set("X-NullSpend-Session", ctx.sessionId);

    if (isStreaming) {
      if (ctx.stepTiming) ctx.stepTiming.ttfbMs = upstreamDurationMs;
      const bodyBucket = (env as Record<string, unknown>).BODY_STORAGE as R2Bucket | undefined;
      return handleStreaming(
        upstreamResponse,
        clientHeaders,
        requestModel,
        requestId,
        startTime,
        env,
        attribution,
        reservationId,
        budgetEntities,
        ctx.connectionString,
        enrichment,
        ctx,
        estimate,
        ctx.requestLoggingEnabled && bodyBucket ? bodyBucket : null,
      );
    }

    return await handleNonStreaming(
      upstreamResponse,
      clientHeaders,
      requestModel,
      requestId,
      startTime,
      env,
      attribution,
      reservationId,
      budgetEntities,
      ctx.connectionString,
      enrichment,
      ctx,
    );
  } catch (err) {
    // Record failed request in latency metrics so timeout spikes are visible
    const failTotalMs = Math.round(performance.now() - ctx.requestStartMs);
    const failUpstreamMs = Math.round(performance.now() - startTime);
    const failOverheadMs = Math.max(0, failTotalMs - failUpstreamMs);
    writeLatencyDataPoint(env, "anthropic", requestModel, isStreaming, 502, failOverheadMs, failUpstreamMs, failTotalMs, undefined, ctx.auth.userId);

    if (reservationId) {
      waitUntil(
        reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, ctx.connectionString),
      );
    }
    throw err;
  }
}

function handleStreaming(
  upstreamResponse: Response,
  clientHeaders: Headers,
  requestModel: string,
  requestId: string,
  startTime: number,
  env: Env,
  attribution: Attribution,
  reservationId: string | null,
  budgetEntities: BudgetEntity[],
  connectionString: string,
  enrichment: EnrichmentFields,
  ctx: RequestContext,
  estimate: number,
  bodyBucket: R2Bucket | null,
): Response {
  const upstreamBody = upstreamResponse.body;
  if (!upstreamBody) {
    if (reservationId) {
      waitUntil(
        reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, connectionString),
      );
    }
    const noBodyResp = errorResponse("upstream_error", "No response body from upstream", 502);
    noBodyResp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return noBodyResp;
  }

  // Insert body accumulator between upstream and SSE parser when logging is enabled
  const accumulator = bodyBucket ? createStreamBodyAccumulator() : null;
  const parserInput = accumulator
    ? upstreamBody.pipeThrough(accumulator.transform)
    : upstreamBody;

  const { readable, resultPromise } = createAnthropicSSEParser(parserInput);

  waitUntil(
    resultPromise.then(async (result) => {
      try {
        const durationMs = Math.round(performance.now() - startTime);

        if (!result?.usage) {
          // Stream cancelled by client — reconcile with pre-request estimate
          // to prevent cost evasion.
          const reconcileCost = result?.cancelled ? estimate : 0;
          if (result?.cancelled) {
            emitMetric("stream_cancelled", { model: requestModel, estimate });

            // Best-effort cost event — failure must NOT prevent budget reconciliation
            try {
              await logCostEventQueued(getCostEventQueue(env), connectionString, {
                requestId,
                provider: "anthropic",
                model: result.model ?? requestModel,
                inputTokens: 0,
                outputTokens: 0,
                cachedInputTokens: 0,
                reasoningTokens: 0,
                costMicrodollars: reconcileCost,
                costBreakdown: null,
                durationMs,
                ...attribution,
                ...enrichment,
                tags: { ...enrichment.tags, _ns_estimated: "true", _ns_cancelled: "true" },
                toolCallsRequested: result.toolCalls,
                stopReason: null,
                source: "proxy" as const,
                eventType: "llm" as const,
              });
              emitMetric("cost_event_estimated", { model: requestModel, estimate: reconcileCost });
            } catch (costErr) {
              console.error("[anthropic-route] Failed to log cancelled stream cost event:", { requestId, error: costErr });
            }
          }
          console.warn(
            "[anthropic-route] Streaming response completed without usage data.",
            { requestId, model: requestModel, durationMs, cancelled: result?.cancelled ?? false },
          );
          if (reservationId) {
            await reconcileBudgetQueued(
              getReconcileQueue(env), env, ctx.ownerId, reservationId, reconcileCost, budgetEntities, connectionString,
            );
          }
          // Store streaming body for all no-usage exits (cancelled or malformed — both are valuable for debugging)
          try {
            if (accumulator && bodyBucket) {
              const sseBody = accumulator.getBody();
              if (sseBody.length > 0) {
                await storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText);
                await storeStreamingResponseBody(bodyBucket, ctx.ownerId, requestId, sseBody);
              }
            }
          } catch (bodyErr) {
            console.error("[anthropic-route] Failed to store streaming body:", bodyErr);
          }
          return;
        }

        const costEvent = calculateAnthropicCost(
          requestModel,
          result.model,
          result.usage,
          result.cacheCreationDetail,
          requestId,
          durationMs,
          attribution,
          enrichment.toolDefinitionTokens,
        );

        // Write AE data point at stream completion with full duration
        const streamTotalMs = Math.round(performance.now() - ctx.requestStartMs);
        const streamOverheadMs = Math.max(0, streamTotalMs - durationMs);
        const ttfbMs = result.firstChunkMs != null ? Math.round(result.firstChunkMs - startTime) : undefined;
        writeLatencyDataPoint(env, "anthropic", requestModel, true, 200, streamOverheadMs, durationMs, streamTotalMs, ttfbMs, ctx.auth.userId);

        // Capture cache read/write split and long-context detection in tags for analytics
        const cacheTags: Record<string, string> = {};
        if (result.usage?.cache_creation_input_tokens != null) {
          cacheTags._ns_cache_write_tokens = String(result.usage.cache_creation_input_tokens);
        }
        if (result.usage?.cache_read_input_tokens != null) {
          cacheTags._ns_cache_read_tokens = String(result.usage.cache_read_input_tokens);
        }
        const totalInput = (result.usage?.input_tokens ?? 0)
          + (result.usage?.cache_creation_input_tokens ?? 0)
          + (result.usage?.cache_read_input_tokens ?? 0);
        if (totalInput > 200_000) {
          cacheTags._ns_long_context = "true";
        }

        await logCostEventQueued(getCostEventQueue(env), connectionString, {
          ...costEvent,
          ...enrichment,
          tags: { ...enrichment.tags, ...cacheTags },
          toolCallsRequested: result.toolCalls,
          stopReason: result.stopReason,
          source: "proxy" as const,
        });

        if (reservationId) {
          await reconcileBudgetQueued(
            getReconcileQueue(env), env, ctx.ownerId, reservationId, costEvent.costMicrodollars, budgetEntities, connectionString,
          );
        }

        try {
          await dispatchCostEventWebhooks(ctx, env, "anthropic", costEvent, enrichment, budgetEntities, result.toolCalls);
        } catch (err) {
          console.error("[anthropic-route] Webhook dispatch failed:", err);
        }

        // Store streaming body after all cost processing is complete
        try {
          if (accumulator && bodyBucket) {
            const sseBody = accumulator.getBody();
            if (sseBody.length > 0) {
              await storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText);
              await storeStreamingResponseBody(bodyBucket, ctx.ownerId, requestId, sseBody);
            }
          }
        } catch (bodyErr) {
          console.error("[anthropic-route] Failed to store streaming body:", bodyErr);
        }
      } catch (err) {
        console.error(
          "[anthropic-route] Failed to process streaming cost event:",
          err,
        );
        if (reservationId) {
          try {
            await reconcileBudgetQueued(
              getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, connectionString,
            );
          } catch { /* already logged inside reconcileBudgetQueued */ }
        }
      }
    }),
  );

  clientHeaders.set("cache-control", "no-cache, no-transform");
  clientHeaders.set("x-accel-buffering", "no");
  clientHeaders.set("connection", "keep-alive");
  const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, enrichment.upstreamDurationMs, ctx.stepTiming);
  emitMetric("proxy_latency", { provider: "anthropic", model: requestModel, overheadMs, upstreamMs: enrichment.upstreamDurationMs, totalMs, streaming: true });
  // AE data point is written at stream completion inside the waitUntil callback
  // (see resultPromise.then) — NOT here, because overhead/total at this point
  // only reflects TTFB, not the full stream duration.

  return new Response(readable, {
    status: upstreamResponse.status,
    headers: clientHeaders,
  });
}

async function handleNonStreaming(
  upstreamResponse: Response,
  clientHeaders: Headers,
  requestModel: string,
  requestId: string,
  startTime: number,
  env: Env,
  attribution: Attribution,
  reservationId: string | null,
  budgetEntities: BudgetEntity[],
  connectionString: string,
  enrichment: EnrichmentFields,
  ctx: RequestContext,
): Promise<Response> {
  const responseText = await upstreamResponse.text();
  // Capture upstream duration after .text() completes so response body
  // transfer time is attributed to the upstream, not to proxy overhead.
  const upstreamDurationWithBody = Math.round(performance.now() - startTime);
  enrichment = { ...enrichment, upstreamDurationMs: upstreamDurationWithBody };

  try {
    const parsed = JSON.parse(responseText);
    const responseModel: string | null = parsed.model ?? null;
    const usage = parsed.usage;

    const stopReason: string | null = parsed.stop_reason ?? null;

    let toolCallsRequested: { name: string; id: string }[] | null = null;
    try {
      if (Array.isArray(parsed.content)) {
        const toolUseBlocks = parsed.content
          .filter((b: { type: string }) => b.type === "tool_use");
        if (toolUseBlocks.length > 0) {
          toolCallsRequested = toolUseBlocks.map(
            (b: { id: string; name: string }) => ({ name: b.name, id: b.id }),
          );
        }
      }
    } catch { /* malformed content blocks — proceed without */ }

    if (usage) {
      const cacheCreationDetail: AnthropicCacheCreationDetail | null =
        usage.cache_creation &&
        typeof usage.cache_creation === "object"
          ? usage.cache_creation
          : null;

      const costEvent = calculateAnthropicCost(
        requestModel,
        responseModel,
        usage,
        cacheCreationDetail,
        requestId,
        upstreamDurationWithBody,
        attribution,
        enrichment.toolDefinitionTokens,
      );

      // Capture cache read/write split and long-context detection in tags for analytics
      const cacheTags: Record<string, string> = {};
      if (usage.cache_creation_input_tokens != null) {
        cacheTags._ns_cache_write_tokens = String(usage.cache_creation_input_tokens);
      }
      if (usage.cache_read_input_tokens != null) {
        cacheTags._ns_cache_read_tokens = String(usage.cache_read_input_tokens);
      }
      const totalInput = (usage.input_tokens ?? 0)
        + (usage.cache_creation_input_tokens ?? 0)
        + (usage.cache_read_input_tokens ?? 0);
      if (totalInput > 200_000) {
        cacheTags._ns_long_context = "true";
      }

      waitUntil(logCostEventQueued(getCostEventQueue(env), connectionString, {
        ...costEvent,
        ...enrichment,
        tags: { ...enrichment.tags, ...cacheTags },
        toolCallsRequested,
        stopReason,
        source: "proxy" as const,
      }));

      if (reservationId) {
        waitUntil(
          reconcileBudgetQueued(
            getReconcileQueue(env), env, ctx.ownerId, reservationId, costEvent.costMicrodollars, budgetEntities, connectionString,
          ),
        );
      }

      // Webhook dispatch (separate waitUntil — independent of log/reconcile)
      waitUntil(dispatchCostEventWebhooks(ctx, env, "anthropic", costEvent, enrichment, budgetEntities, toolCallsRequested));
    } else if (reservationId) {
      waitUntil(
        reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, connectionString),
      );
    }
  } catch {
    console.error(
      "[anthropic-route] Failed to parse non-streaming response for cost tracking",
    );
    if (reservationId) {
      waitUntil(
        reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, reservationId, 0, budgetEntities, connectionString),
      );
    }
  }

  const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, enrichment.upstreamDurationMs, ctx.stepTiming);
  emitMetric("proxy_latency", { provider: "anthropic", model: requestModel, overheadMs, upstreamMs: enrichment.upstreamDurationMs, totalMs, streaming: false });
  writeLatencyDataPoint(env, "anthropic", requestModel, false, upstreamResponse.status, overheadMs, enrichment.upstreamDurationMs, totalMs, undefined, ctx.auth.userId);

  // Body logging — fire-and-forget in waitUntil (pro/enterprise only, non-streaming only)
  const bodyBucket = (env as Record<string, unknown>).BODY_STORAGE as R2Bucket | undefined;
  if (ctx.requestLoggingEnabled && bodyBucket) {
    waitUntil(storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText));
    waitUntil(storeResponseBody(bodyBucket, ctx.ownerId, requestId, responseText));
  }

  return new Response(responseText, {
    status: upstreamResponse.status,
    headers: clientHeaders,
  });
}
