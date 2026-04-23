/**
 * Generic provider route handler.
 *
 * Implements the full proxy lifecycle (auth → budget → upstream → cost → reconcile)
 * parameterized by a ProviderAdapter. Each provider implements ~10 adapter methods;
 * all shared logic lives here.
 *
 * Critical ordering constraint (from PXY-3 and PXY-4 fixes):
 *   1. Cost event write (queue or direct)
 *   2. Budget reconciliation (DO RPC)
 *   3. Webhook dispatch (threshold crossings from reconcile result)
 *   4. Body storage (R2 write)
 *
 * See docs/plans/p2-1-provider-adapter-dedup.md for design decisions.
 */

import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import { errorResponse } from "../lib/errors.js";
import { appendTimingHeaders } from "../lib/headers.js";
import { extractModelFromBody } from "../lib/request-utils.js";
import { isKnownModel } from "@nullspend/cost-engine";
import { logCostEventQueued, getCostEventQueue } from "../lib/cost-event-queue.js";
import { optionalBinding } from "../lib/env.js";
import { checkBudget, reconcileBudget, type ReconcileOutcome } from "../lib/budget-orchestrator.js";
import { resolvePeriodBounds } from "../lib/period-math.js";
import { sanitizeUpstreamError } from "../lib/sanitize-upstream-error.js";
import { stripNsPrefix } from "../lib/validation.js";
import { emitMetric } from "../lib/metrics.js";
import { writeLatencyDataPoint } from "../lib/write-metric.js";
import { handleBudgetDenials, dispatchVelocityRecoveryWebhooks, dispatchCostEventWebhooks, buildBudgetHeaders, type Attribution, type EnrichmentFields } from "./shared.js";
import { storeRequestBody, storeResponseBody, storeStreamingResponseBody, createStreamBodyAccumulator } from "../lib/body-storage.js";
import { resolveUpstreamTimeoutMs } from "../lib/constants.js";
import type { BudgetEntity } from "../lib/budget-do-lookup.js";
import type { ProviderAdapter, StreamResult } from "../lib/provider-types.js";

/**
 * Handle a provider route request using the provided adapter.
 *
 * This is the single code path for all LLM proxy routes (OpenAI, Anthropic, etc.).
 * Provider-specific behavior is isolated in the adapter; everything else is shared.
 */
export async function handleProviderRequest(
  request: Request,
  env: Env,
  ctx: RequestContext,
  adapter: ProviderAdapter,
): Promise<Response> {
  const provider = adapter.provider;
  const requestModel = adapter.extractModel?.(request, ctx.body) ?? extractModelFromBody(ctx.body);
  const safeModel = requestModel.slice(0, 200);

  // --- Provider restriction ---
  if (ctx.auth.allowedProviders && !ctx.auth.allowedProviders.includes(provider)) {
    emitMetric("mandate_denied", { reason: "provider_not_allowed", provider, model: safeModel });
    const resp = errorResponse("mandate_violation", `Provider ${provider} is not allowed by this key's policy`, 403, {
      mandate: "allowed_providers",
      requested: provider,
      allowed: ctx.auth.allowedProviders,
    });
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return resp;
  }

  // --- Model restriction ---
  if (ctx.auth.allowedModels && !ctx.auth.allowedModels.includes(requestModel)) {
    emitMetric("mandate_denied", { reason: "model_not_allowed", provider, model: safeModel });
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

  const isUnpricedModel = !isKnownModel(provider, requestModel);
  if (isUnpricedModel) {
    emitMetric("unknown_model", { provider, model: requestModel });
  }

  const toolDefinitionTokens = Array.isArray(ctx.body.tools) && ctx.body.tools.length > 0
    ? Math.ceil(JSON.stringify(ctx.body.tools).length / 4)
    : 0;

  const isStreaming = adapter.supportsStreaming
    && (adapter.isStreaming?.(request, ctx.body) ?? ctx.body.stream === true);

  // Provider-specific adapter hooks — wrapped in try/catch so a buggy adapter
  // returns a structured 502, not an opaque Cloudflare error.
  //
  // P-10: each hook is timed individually so a slow hook is observable. The
  // emitMetric call uses { provider, hook, ms } so dashboards can alert when
  // any single hook crosses a latency threshold (e.g. P95 > 50ms). Without
  // per-hook timing, a rogue change in one adapter could add 100ms silently.
  let estimate: number;
  try {
    if (adapter.prepareBody) {
      const t0 = performance.now();
      adapter.prepareBody(ctx.body, isStreaming);
      emitMetric("adapter_hook_ms", {
        provider, hook: "prepareBody", ms: Math.round(performance.now() - t0),
      });
    }
    if (adapter.emitPreFetchMetrics) {
      const t0 = performance.now();
      adapter.emitPreFetchMetrics(request, requestModel);
      emitMetric("adapter_hook_ms", {
        provider, hook: "emitPreFetchMetrics", ms: Math.round(performance.now() - t0),
      });
    }
    const t0 = performance.now();
    estimate = adapter.estimateMaxCost(requestModel, ctx.body, ctx.bodyByteLength);
    emitMetric("adapter_hook_ms", {
      provider, hook: "estimateMaxCost", ms: Math.round(performance.now() - t0),
    });
  } catch (adapterErr) {
    console.error(`[${provider}-route] Adapter hook failed:`, adapterErr);
    emitMetric("adapter_error", { provider, hook: "pre_request" });
    const resp = errorResponse("internal_error", "Provider adapter error", 502);
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return resp;
  }

  // --- Loop detection context ---
  // Compute content hash from the raw body text for loop detection.
  // SHA-256 of first 8KB, truncated to 8 hex chars.
  let loopContext: { provider: string; model: string; contentHash: string } | null = null;
  if (ctx.bodyText) {
    const bodySlice = ctx.bodyText.slice(0, 8192);
    const hashBuffer = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bodySlice));
    const hashArray = new Uint8Array(hashBuffer);
    const contentHash = Array.from(hashArray.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    loopContext = { provider, model: safeModel, contentHash };
  }

  // --- Budget enforcement (CX-5: check BEFORE upstream fetch) ---

  let reservationId: string | null = null;
  let budgetEntities: BudgetEntity[] = [];
  let budgetStatus: "skipped" | "approved" | "denied" = "skipped";
  let loopWarningHeader: string | null = null;

  try {
    const budgetStartMs = performance.now();
    const outcome = await checkBudget(env, ctx, estimate, ctx.finalize, loopContext);
    if (ctx.stepTiming) ctx.stepTiming.budgetCheckMs = Math.round(performance.now() - budgetStartMs);

    budgetStatus = outcome.status;
    reservationId = outcome.reservationId;
    budgetEntities = outcome.budgetEntities;

    // Loop detection warning header (approaching threshold)
    if (outcome.loopCount != null && outcome.loopMaxCalls != null) {
      loopWarningHeader = `${outcome.loopCount}/${outcome.loopMaxCalls}`;
    }

    if (ctx.finalize) {
      emitMetric("finalization_request", { provider, entityType: outcome.deniedEntityType ?? "unknown", approved: outcome.status !== "denied" });
    }

    const denialResponse = await handleBudgetDenials(outcome, ctx, env, provider, requestModel, estimate, budgetEntities);
    if (denialResponse) {
      return denialResponse;
    }

    dispatchVelocityRecoveryWebhooks(outcome, ctx, env, provider);
  } catch (err) {
    console.error(`[${provider}-route] Budget check or denial handling failed:`, err);
    const budgetUnavailResp = errorResponse("budget_unavailable", "Budget service unavailable", 503);
    budgetUnavailResp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return budgetUnavailResp;
  }

  // Budget approved — resolve upstream URL (may return early Response on validation failure)
  const upstreamUrlOrResponse = adapter.getUpstreamUrl(env, request, ctx);
  if (upstreamUrlOrResponse instanceof Response) {
    // Upstream validation failed (e.g., OpenAI invalid_upstream)
    // Budget was reserved but upstream is invalid — reconcile with 0
    if (reservationId) {
      waitUntil(
        reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, 0, budgetEntities, ctx.connectionString),
      );
    }
    return upstreamUrlOrResponse;
  }
  const upstreamUrl = upstreamUrlOrResponse;

  const upstreamHeaders = adapter.buildUpstreamHeaders(request);
  const timeoutMs = resolveUpstreamTimeoutMs(env as Record<string, unknown>);
  const startTime = performance.now();

  try {
    // getRequestBody inside try so JSON.stringify failures are caught
    // and budget reservation is reconciled (Finding 2 from Codex review)
    const requestBody = adapter.getRequestBody(ctx, isStreaming);

    // P1-6: Propagate client abort to upstream. Without `request.signal`,
    // a client that disconnects mid-stream leaves the proxy paying the
    // upstream provider for the full generation even though nothing reads
    // the response. Combining signals means either timeout OR client
    // abort cancels the upstream fetch.
    const upstreamResponse = await fetch(upstreamUrl, {
      method: "POST",
      headers: upstreamHeaders,
      body: requestBody,
      signal: AbortSignal.any([AbortSignal.timeout(timeoutMs), request.signal]),
    });
    const upstreamDurationMs = Math.round(performance.now() - startTime);

    // Provider-specific response tags (rate limits, request metadata)
    const responseTags = adapter.extractResponseTags(upstreamResponse, ctx.body, isUnpricedModel);

    // PR-2d: resolve billing period bounds for this request. Stamped on every
    // cost event (streaming/non-streaming, usage/no-usage paths) via the
    // enrichment spread so reconciliation lands on the ORIGINAL period row.
    const periods = resolvePeriodBounds(ctx.auth, Date.now());
    const enrichment: EnrichmentFields = {
      upstreamDurationMs,
      sessionId: ctx.sessionId,
      traceId: ctx.traceId,
      toolDefinitionTokens,
      tags: { ...ctx.tags, ...responseTags },
      customerId: ctx.customerId,
      budgetStatus,
      estimatedCostMicrodollars: estimate,
      orgId: ctx.auth.orgId,
      periodStart: new Date(periods.periodStart),
      periodEnd: new Date(periods.periodEnd),
    };

    const requestId = adapter.extractRequestId(upstreamResponse);

    // --- Upstream error ---
    if (!upstreamResponse.ok) {
      if (reservationId) {
        waitUntil(
          reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, 0, budgetEntities, ctx.connectionString),
        );
      }
      const clientHeaders = adapter.buildClientHeaders(upstreamResponse, ctx.resolvedApiVersion);
      clientHeaders.set("X-NullSpend-Trace-Id", ctx.traceId);
      clientHeaders.set("x-request-id", requestId);
      if (ctx.sessionId) clientHeaders.set("X-NullSpend-Session", ctx.sessionId);
      const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, upstreamDurationMs, ctx.stepTiming);
      emitMetric("proxy_latency", { provider, model: requestModel, overheadMs, upstreamMs: upstreamDurationMs, totalMs, streaming: false });
      writeLatencyDataPoint(env, provider, requestModel, false, upstreamResponse.status, overheadMs, upstreamDurationMs, totalMs, undefined, ctx.auth.userId);
      const sanitizedBody = await sanitizeUpstreamError(upstreamResponse, provider);
      const bodyBucket = optionalBinding(env, "BODY_STORAGE");
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

    // --- Success response ---
    const clientHeaders = adapter.buildClientHeaders(upstreamResponse, ctx.resolvedApiVersion);
    clientHeaders.set("X-NullSpend-Trace-Id", ctx.traceId);
    clientHeaders.set("x-request-id", requestId);
    if (ctx.sessionId) clientHeaders.set("X-NullSpend-Session", ctx.sessionId);
    if (loopWarningHeader) clientHeaders.set("X-NullSpend-Loop-Count", loopWarningHeader);

    for (const [k, v] of Object.entries(buildBudgetHeaders(budgetEntities, estimate))) {
      clientHeaders.set(k, v);
    }

    if (isStreaming) {
      if (ctx.stepTiming) ctx.stepTiming.ttfbMs = upstreamDurationMs;
      const bodyBucket = optionalBinding(env, "BODY_STORAGE");
      return handleStreaming(
        upstreamResponse, clientHeaders, requestModel, requestId, startTime,
        env, adapter, attribution, reservationId, budgetEntities,
        ctx.connectionString, enrichment, ctx, estimate,
        ctx.requestLoggingEnabled && bodyBucket ? bodyBucket : null,
      );
    }

    return await handleNonStreaming(
      upstreamResponse, clientHeaders, requestModel, requestId, startTime,
      env, adapter, attribution, reservationId, budgetEntities,
      ctx.connectionString, enrichment, ctx,
    );
  } catch (err) {
    const failTotalMs = Math.round(performance.now() - ctx.requestStartMs);
    const failUpstreamMs = Math.round(performance.now() - startTime);
    const failOverheadMs = Math.max(0, failTotalMs - failUpstreamMs);
    writeLatencyDataPoint(env, provider, requestModel, isStreaming, 502, failOverheadMs, failUpstreamMs, failTotalMs, undefined, ctx.auth.userId);

    if (reservationId) {
      waitUntil(
        reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, 0, budgetEntities, ctx.connectionString),
      );
    }
    throw err;
  }
}

// ── Streaming handler ─────────────────────────────────────────────────

function handleStreaming(
  upstreamResponse: Response,
  clientHeaders: Headers,
  requestModel: string,
  requestId: string,
  startTime: number,
  env: Env,
  adapter: ProviderAdapter,
  attribution: Attribution,
  reservationId: string | null,
  budgetEntities: BudgetEntity[],
  connectionString: string,
  enrichment: EnrichmentFields,
  ctx: RequestContext,
  estimate: number,
  bodyBucket: R2Bucket | null,
): Response {
  const provider = adapter.provider;
  const upstreamBody = upstreamResponse.body;

  if (!upstreamBody) {
    if (reservationId) {
      waitUntil(
        reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, 0, budgetEntities, connectionString),
      );
    }
    const noBodyResp = errorResponse("upstream_error", "No response body from upstream", 502);
    noBodyResp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return noBodyResp;
  }

  // Insert body accumulator between upstream and SSE parser when logging is enabled.
  // P1-5 fix (2026-04-16): wrap upstreamBody in an explicit reader so we catch
  // upstream errors mid-stream. Per WHATWG spec, writable-abort SHOULD fire
  // transformer.cancel() — but workerd behavior on that path is not empirically
  // verified, and codex review confirmed the hang is real. The wrapper's
  // start() drains upstream via .getReader(), so any abort / network error
  // surfaces as a rejected promise we can resolve against resultPromise.
  let resolveUpstreamError: (err: Error) => void;
  const upstreamErrorPromise = new Promise<Error>((resolve) => {
    resolveUpstreamError = resolve;
  });
  const trackedUpstream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = upstreamBody.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            break;
          }
          controller.enqueue(value);
        }
      } catch (err) {
        // Upstream errored mid-stream. Signal the error before propagating
        // downstream so the resultPromise race below can fire cleanly.
        resolveUpstreamError(err instanceof Error ? err : new Error(String(err)));
        try {
          controller.error(err);
        } catch {
          // already closed
        }
      }
    },
    cancel(reason) {
      void upstreamBody.cancel(reason).catch(() => {
        // best-effort
      });
    },
  });

  const accumulator = bodyBucket ? createStreamBodyAccumulator() : null;
  const parserInput = accumulator
    ? trackedUpstream.pipeThrough(accumulator.transform)
    : trackedUpstream;

  const { readable, resultPromise } = adapter.createSSEParser(parserInput);

  // P1-5 escape hatch: race resultPromise against upstreamErrorPromise so
  // waitUntil() always receives a settled value. Without this race, a
  // mid-stream upstream error could cause the SSE parser's flush/cancel
  // hooks to never fire (workerd writable-abort semantics are not
  // empirically verified), leaving the reservation hung until the DO alarm
  // reclaim cycle. That's the bug codex confirmed as real in the Wave 2
  // audit. The synthetic result routes through the existing
  // `!result.hasUsage` branch below, which reconciles cost via the
  // estimate and tags the cost event with `_ns_cancelled: "true"`.
  const effectiveResult: Promise<StreamResult> = Promise.race([
    resultPromise,
    upstreamErrorPromise.then((err) => {
      console.error(
        `[${provider}-route] Upstream stream errored mid-flight:`,
        { requestId, error: err.message },
      );
      emitMetric("stream_upstream_error", {
        model: requestModel,
        provider,
      });
      return {
        usage: null,
        model: null,
        finishReason: null,
        toolCalls: null,
        cancelled: true,
        firstChunkMs: null,
      } satisfies StreamResult;
    }),
  ]);

  waitUntil(
    effectiveResult.then(async (result: StreamResult) => {
      try {
        const durationMs = Math.round(performance.now() - startTime);

        if (!result.hasUsage) {
          // PXY-4: Always use estimate when usage is missing
          const reconcileCost = estimate;
          const isCancelled = result.cancelled;

          if (isCancelled) {
            emitMetric("stream_cancelled", { model: requestModel, estimate });
          } else {
            emitMetric("stream_no_usage", { model: requestModel, estimate });
          }

          // Best-effort cost event
          try {
            await logCostEventQueued(getCostEventQueue(env), connectionString, {
              requestId,
              provider,
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
              tags: {
                ...enrichment.tags,
                _ns_estimated: "true",
                ...(isCancelled ? { _ns_cancelled: "true" } : { _ns_no_usage: "true" }),
              },
              toolCallsRequested: result.toolCalls ?? null,
              stopReason: null,
              source: "proxy" as const,
              eventType: adapter.eventType as "llm",
            });
            emitMetric("cost_event_estimated", { model: requestModel, estimate: reconcileCost });
          } catch (costErr) {
            console.error(`[${provider}-route] Failed to log stream cost event:`, { requestId, error: costErr });
          }

          console.warn(
            `[${provider}-route] Streaming response completed without usage data.`,
            { requestId, model: requestModel, durationMs, cancelled: isCancelled },
          );
          if (reservationId) {
            await reconcileBudget(
              env, ctx.ownerId, ctx.auth.orgId, reservationId, reconcileCost, budgetEntities, connectionString,
            );
          }
          try {
            if (accumulator && bodyBucket) {
              const sseBody = accumulator.getBody();
              if (sseBody.length > 0) {
                await storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText);
                await storeStreamingResponseBody(bodyBucket, ctx.ownerId, requestId, sseBody);
              }
            }
          } catch (bodyErr) {
            console.error(`[${provider}-route] Failed to store streaming body:`, bodyErr);
          }
          return;
        }

        // Usage present — calculate real cost
        const costEvent = adapter.calculateCost(
          requestModel, result.model, result, requestId, durationMs, attribution, enrichment.toolDefinitionTokens,
        );

        // PXY-3: Unknown model cost fallback
        let loggedCost = costEvent.costMicrodollars;
        let unpricedTags: Record<string, string> = {};
        if (costEvent.costMicrodollars === 0 && (costEvent.inputTokens + costEvent.outputTokens) > 0) {
          loggedCost = estimate;
          unpricedTags = { _ns_unpriced: "true", _ns_estimated: "true" };
          emitMetric("cost_event_unpriced", { model: requestModel, estimate });
        }

        // Write AE data point at stream completion with full duration
        const streamTotalMs = Math.round(performance.now() - ctx.requestStartMs);
        const streamOverheadMs = Math.max(0, streamTotalMs - durationMs);
        const ttfbMs = result.firstChunkMs != null ? Math.round(result.firstChunkMs - startTime) : undefined;
        writeLatencyDataPoint(env, provider, requestModel, true, 200, streamOverheadMs, durationMs, streamTotalMs, ttfbMs, ctx.auth.userId);

        // Provider-specific post-response tags (e.g., Anthropic cache tags)
        const postTags = adapter.extractPostResponseTags?.(result) ?? {};

        await logCostEventQueued(getCostEventQueue(env), connectionString, {
          ...costEvent,
          costMicrodollars: loggedCost,
          ...enrichment,
          tags: { ...enrichment.tags, ...postTags, ...unpricedTags },
          toolCallsRequested: result.toolCalls,
          stopReason: result.stopReason,
          source: "proxy" as const,
        });

        let reconcileResult: ReconcileOutcome | undefined;
        if (reservationId) {
          reconcileResult = await reconcileBudget(
            env, ctx.ownerId, ctx.auth.orgId, reservationId, loggedCost, budgetEntities, connectionString,
          );
        }

        try {
          await dispatchCostEventWebhooks(ctx, env, provider, costEvent, enrichment, budgetEntities, result.toolCalls, reconcileResult?.thresholdCrossings);
        } catch (err) {
          console.error(`[${provider}-route] Webhook dispatch failed:`, err);
        }

        try {
          if (accumulator && bodyBucket) {
            const sseBody = accumulator.getBody();
            if (sseBody.length > 0) {
              await storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText);
              await storeStreamingResponseBody(bodyBucket, ctx.ownerId, requestId, sseBody);
            }
          }
        } catch (bodyErr) {
          console.error(`[${provider}-route] Failed to store streaming body:`, bodyErr);
        }
      } catch (err) {
        console.error(`[${provider}-route] Failed to process streaming cost event:`, err);
        // P0 fix: reconcile with the estimate, not 0. The upstream LLM call
        // already succeeded and the client received the stream. Reconciling
        // with 0 would silently erase the spend from budget enforcement.
        // Using `estimate` ensures the reservation is converted to real spend
        // even when cost logging infrastructure is down.
        if (reservationId) {
          try {
            await reconcileBudget(
              env, ctx.ownerId, ctx.auth.orgId, reservationId, estimate, budgetEntities, connectionString,
            );
          } catch { /* already logged inside reconcileBudget */ }
        }
        emitMetric("stream_cost_logging_failure", { model: requestModel, estimate, provider });
      }
    }),
  );

  clientHeaders.set("cache-control", "no-cache, no-transform");
  clientHeaders.set("x-accel-buffering", "no");
  clientHeaders.set("connection", "keep-alive");
  const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, enrichment.upstreamDurationMs, ctx.stepTiming);
  emitMetric("proxy_latency", { provider, model: requestModel, overheadMs, upstreamMs: enrichment.upstreamDurationMs, totalMs, streaming: true });

  return new Response(readable, {
    status: upstreamResponse.status,
    headers: clientHeaders,
  });
}

// ── Non-streaming handler ─────────────────────────────────────────────

async function handleNonStreaming(
  upstreamResponse: Response,
  clientHeaders: Headers,
  requestModel: string,
  requestId: string,
  startTime: number,
  env: Env,
  adapter: ProviderAdapter,
  attribution: Attribution,
  reservationId: string | null,
  budgetEntities: BudgetEntity[],
  connectionString: string,
  enrichment: EnrichmentFields,
  ctx: RequestContext,
): Promise<Response> {
  const provider = adapter.provider;
  const responseText = await upstreamResponse.text();
  const upstreamDurationWithBody = Math.round(performance.now() - startTime);
  enrichment = { ...enrichment, upstreamDurationMs: upstreamDurationWithBody };

  try {
    const parsed = JSON.parse(responseText);
    const result = adapter.parseNonStreamingResponse(parsed);

    if (result.hasUsage) {
      const costEvent = adapter.calculateCost(
        requestModel, result.model, result, requestId, upstreamDurationWithBody, attribution, enrichment.toolDefinitionTokens,
      );

      // PXY-3: Unknown model cost fallback
      const estimate = enrichment.estimatedCostMicrodollars ?? 0;
      let loggedCostNS = costEvent.costMicrodollars;
      let unpricedTagsNS: Record<string, string> = {};
      if (costEvent.costMicrodollars === 0 && (costEvent.inputTokens + costEvent.outputTokens) > 0) {
        loggedCostNS = estimate;
        unpricedTagsNS = { _ns_unpriced: "true", _ns_estimated: "true" };
        emitMetric("cost_event_unpriced", { model: requestModel, estimate });
      }

      // Provider-specific post-response tags
      const postTags = adapter.extractPostResponseTags?.(result) ?? {};

      waitUntil(logCostEventQueued(getCostEventQueue(env), connectionString, {
        ...costEvent,
        costMicrodollars: loggedCostNS,
        ...enrichment,
        tags: { ...enrichment.tags, ...postTags, ...unpricedTagsNS },
        toolCallsRequested: result.toolCalls,
        stopReason: result.stopReason,
        source: "proxy" as const,
      }));

      waitUntil((async () => {
        let crossings: import("../durable-objects/user-budget.js").ThresholdCrossing[] | undefined;
        if (reservationId) {
          const reconcileResult = await reconcileBudget(
            env, ctx.ownerId, ctx.auth.orgId, reservationId, loggedCostNS, budgetEntities, connectionString,
          );
          crossings = reconcileResult?.thresholdCrossings;
        }
        await dispatchCostEventWebhooks(ctx, env, provider, costEvent, enrichment, budgetEntities, result.toolCalls, crossings);
      })());
    } else if (reservationId) {
      waitUntil(
        reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, 0, budgetEntities, connectionString),
      );
    }
  } catch {
    // P0 fix: reconcile with estimate, not 0. The upstream returned 200
    // (non-streaming), so the provider charged us. Using 0 would silently
    // erase the spend from budget enforcement during cost parsing failures.
    const fallbackCost = enrichment.estimatedCostMicrodollars ?? 0;
    console.error(`[${provider}-route] Failed to parse non-streaming response for cost tracking, reconciling with estimate=${fallbackCost}`);
    if (reservationId) {
      waitUntil(
        reconcileBudget(env, ctx.ownerId, ctx.auth.orgId, reservationId, fallbackCost, budgetEntities, connectionString),
      );
    }
    emitMetric("nonstream_cost_parsing_failure", { model: requestModel, estimate: fallbackCost, provider });
  }

  const { totalMs, overheadMs } = appendTimingHeaders(clientHeaders, ctx.requestStartMs, enrichment.upstreamDurationMs, ctx.stepTiming);
  emitMetric("proxy_latency", { provider, model: requestModel, overheadMs, upstreamMs: enrichment.upstreamDurationMs, totalMs, streaming: false });
  writeLatencyDataPoint(env, provider, requestModel, false, upstreamResponse.status, overheadMs, enrichment.upstreamDurationMs, totalMs, undefined, ctx.auth.userId);

  const bodyBucket = optionalBinding(env, "BODY_STORAGE");
  if (ctx.requestLoggingEnabled && bodyBucket) {
    waitUntil(storeRequestBody(bodyBucket, ctx.ownerId, requestId, ctx.bodyText));
    waitUntil(storeResponseBody(bodyBucket, ctx.ownerId, requestId, responseText));
  }

  return new Response(responseText, {
    status: upstreamResponse.status,
    headers: clientHeaders,
  });
}
