import { handleMcpBudgetCheck, handleMcpEvents } from "./routes/mcp.js";
import { matchProviderRoute } from "./providers/registry.js";
import { handleProviderRequest } from "./routes/provider-handler.js";
import { handleBudgetInvalidation, handleVelocityState, handleRequestBodies, handlePlanCounterIncrement } from "./routes/internal.js";
import { handleMetrics } from "./routes/metrics.js";
import { handleFeatureFlags } from "./routes/health.js";
import { handleReconcilePlanCounterCron } from "./routes/reconcile-plan-counter-cron.js";
import { handlePolicy } from "./routes/policy.js";
import { authenticateRequest } from "./lib/auth.js";
import { resolveApiVersion } from "./lib/api-version.js";
import { errorResponse } from "./lib/errors.js";
import { createWebhookDispatcher } from "./lib/webhook-dispatch.js";
import { mergeTags } from "./lib/tags.js";
import { parseCustomerHeader, resolveCustomerId } from "./lib/customer.js";
import { resolveTraceId } from "./lib/trace-context.js";
import { emitMetric } from "./lib/metrics.js";
import { optionalBinding } from "./lib/env.js";
import type { IngressMetadata, RequestContext, RouteHandler } from "./lib/context.js";
import { resolveNullspendRequestId } from "./lib/context.js";
import { stampNullspendHeaders } from "./lib/headers.js";
import { handleCostEventQueue, COST_EVENT_QUEUE_NAME } from "./cost-event-queue-handler.js";
import { handleCostEventDlq, COST_EVENT_DLQ_NAME } from "./cost-event-dlq-handler.js";
import { handleWebhookQueue, WEBHOOK_QUEUE_NAME } from "./webhook-queue-handler.js";
import { handleWebhookDlq, WEBHOOK_DLQ_NAME } from "./webhook-dlq-handler.js";
import type { CostEventMessage } from "./lib/cost-event-queue.js";
import type { WebhookQueueMessage } from "./lib/webhook-queue.js";

export { UserBudgetDO } from "./durable-objects/user-budget.js";

const MAX_BODY_SIZE = 1_048_576; // 1MB

// MCP routes have a different lifecycle (no upstream fetch) — registered directly.
// Provider routes (OpenAI, Anthropic, etc.) use the adapter registry.
const mcpRoutes = new Map<string, RouteHandler>();
mcpRoutes.set("/v1/mcp/budget/check", handleMcpBudgetCheck);
mcpRoutes.set("/v1/mcp/events", handleMcpEvents);

/**
 * Rate limiting via Cloudflare native bindings.
 * Counters are on the same machine as the Worker — ~0ms overhead.
 * Limits are configured in wrangler.jsonc (per-IP: 120/min, per-key: 600/min).
 */
async function applyRateLimit(
  request: Request,
  env: Env,
): Promise<Response | null> {
  const clientIp = request.headers.get("cf-connecting-ip") ?? "unknown";

  try {
    // Per-IP rate limit (abuse/DDoS protection)
    const { success: ipOk } = await env.IP_RATE_LIMITER.limit({ key: clientIp });
    if (!ipOk) {
      emitMetric("request_error", { status: 429, reason: "ip_rate_limited" });
      return Response.json(
        { error: { code: "rate_limited", message: "Too many requests", details: null } },
        { status: 429, headers: { "Retry-After": "60" } },
      );
    }

    // Per-key rate limit (runaway agent protection)
    const rateLimitKey = request.headers.get("x-nullspend-key");
    if (rateLimitKey && rateLimitKey.length <= 128) {
      const { success: keyOk } = await env.KEY_RATE_LIMITER.limit({ key: rateLimitKey });
      if (!keyOk) {
        emitMetric("request_error", { status: 429, reason: "key_rate_limited" });
        return Response.json(
          { error: { code: "rate_limited", message: "Too many requests", details: null } },
          { status: 429, headers: { "Retry-After": "60" } },
        );
      }
    }
  } catch (err) {
    // Fail-open: if rate limiter binding is unavailable, allow the request
    console.error("[proxy] Rate limiter error:", err);
  }

  return null;
}

async function parseRequestBody(
  request: Request,
): Promise<{ body: Record<string, unknown>; bodyText: string; bodyByteLength: number; error?: undefined } | { body?: undefined; bodyText?: undefined; bodyByteLength?: undefined; error: Response }> {
  const contentLength = request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > MAX_BODY_SIZE) {
    return {
      error: errorResponse("payload_too_large", `Body exceeds ${MAX_BODY_SIZE} bytes`, 413),
    };
  }

  // P0-3 (codex follow-up): stream the body and enforce MAX_BODY_SIZE on
  // the running total. The prior Wave 2 implementation used
  // `await request.text()` which buffers the FULL body before our check
  // could fire — so a client lying about or omitting Content-Length could
  // still force the Worker to allocate up to the runtime limit (~100MB)
  // before we returned 413. That closed the accounting contract but left
  // the DoS window open. Streaming + early-cancel closes both.
  if (!request.body) {
    return {
      error: errorResponse("bad_request", "Missing request body", 400),
    };
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > MAX_BODY_SIZE) {
        // Cancel the upstream reader so the Worker doesn't continue
        // buffering bytes from a malicious or buggy client. Fire-and-forget
        // — failures here are irrelevant because we're returning 413 anyway.
        reader.cancel("body exceeds MAX_BODY_SIZE").catch(() => {});
        emitMetric("body_size_post_read_exceeded", {
          bodyByteLength: totalBytes, // lower bound: at least this many bytes
          declaredContentLength: contentLength ?? "absent",
        });
        return {
          error: errorResponse("payload_too_large", `Body exceeds ${MAX_BODY_SIZE} bytes`, 413),
        };
      }
      chunks.push(value);
    }
  } catch {
    return {
      error: errorResponse("bad_request", "Could not read request body", 400),
    };
  }

  // Combine accumulated chunks and decode UTF-8 on the full buffer
  // (decoding chunk-by-chunk could split multi-byte sequences at
  // boundaries — a single decode call at the end is safe).
  const combined = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const bodyText = new TextDecoder("utf-8").decode(combined);
  const bodyByteLength = totalBytes;

  try {
    const parsed = JSON.parse(bodyText);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        error: errorResponse("bad_request", "Request body must be a JSON object", 400),
      };
    }
    return { body: parsed, bodyText, bodyByteLength };
  } catch {
    return {
      error: errorResponse("bad_request", "Invalid JSON body", 400),
    };
  }
}

const MAX_SESSION_ID_LENGTH = 256;

export default {
  /**
   * CF Worker cron dispatcher (PR-2d / Decision #35).
   *
   * Wire new crons via the `switch` below — do NOT invoke work directly in
   * this function. `ctx.waitUntil(...)` lets the tick ack immediately so the
   * CF scheduler marks the trigger as succeeded even if the underlying job
   * runs long (up to the 30s CPU cap). Dispatching by `controller.cron`
   * string keeps multiple cron entries (when we add them) self-documenting.
   */
  async scheduled(
    controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    emitMetric("scheduled_cron_tick", { cron: controller.cron });
    switch (controller.cron) {
      case "*/15 * * * *":
        ctx.waitUntil(handleReconcilePlanCounterCron(env));
        break;
      default:
        emitMetric("scheduled_unknown_cron", { cron: controller.cron });
        console.warn("[proxy] scheduled: unknown cron expression", controller.cron);
    }
  },

  async queue(
    batch: MessageBatch<CostEventMessage | WebhookQueueMessage>,
    env: Env,
  ): Promise<void> {
    if (batch.queue === WEBHOOK_DLQ_NAME) {
      await handleWebhookDlq(batch as MessageBatch<WebhookQueueMessage>);
    } else if (batch.queue === WEBHOOK_QUEUE_NAME) {
      await handleWebhookQueue(batch as MessageBatch<WebhookQueueMessage>, env);
    } else if (batch.queue === COST_EVENT_DLQ_NAME) {
      await handleCostEventDlq(batch as MessageBatch<CostEventMessage>, env);
    } else if (batch.queue === COST_EVENT_QUEUE_NAME) {
      await handleCostEventQueue(batch as MessageBatch<CostEventMessage>, env);
    } else {
      throw new Error(`Unknown queue: ${batch.queue}`);
    }
  },

  async fetch(
    request: Request,
    env: Env,
    _ctx: ExecutionContext,
  ): Promise<Response> {
    const requestStartMs = performance.now();
    const forceDbPersist = optionalBinding(env, "FORCE_DB_PERSIST") === "true";
    const skipDbPersist = optionalBinding(env, "SKIP_DB_PERSIST") === "true";
    const skipDbWrites = !forceDbPersist && skipDbPersist;

    // PR-2c codex-round-3 H3: resolve trace ID + nullspendRequestId + build
    // IngressMetadata at the TOP of fetch so pre-auth early-returns (401,
    // rate-limit 429, body-parse 400) can stamp response headers uniformly.
    const traceId = resolveTraceId(request);
    const nullspendRequestId = resolveNullspendRequestId(request, (reason) =>
      emitMetric("nullspend_request_id_invalid", { reason }),
    );
    // sessionId stays null until after the length-validation check runs below;
    // pre-auth 401/429/400 responses don't carry it. ctx.sessionId will pick
    // up the validated value when RequestContext is constructed later.
    const meta: IngressMetadata = { traceId, nullspendRequestId, sessionId: null };
    const stamp = (resp: Response): Response => stampNullspendHeaders(resp, meta);

    try {
      const url = new URL(request.url);

      // Health routes stay outside the pipeline (no auth needed)
      if (url.pathname === "/health") {
        return stamp(Response.json({ status: "ok", service: "nullspend-proxy" }));
      }

      if (url.pathname === "/health/ready") {
        return stamp(Response.json({ status: "ok", service: "nullspend-proxy" }));
      }

      // codex-final review: stamp internal / metrics routes too so the
      // "X-NullSpend-Request-Id on every response" contract holds uniformly.
      // These endpoints don't feed client-retry dedup (internal auth is a
      // shared secret, metrics is observability), but stamping keeps log
      // correlation consistent across all proxy responses.
      if (url.pathname === "/health/metrics") {
        return stamp(await handleMetrics(request, env));
      }
      if (url.pathname === "/health/feature-flags" && request.method === "GET") {
        return stamp(handleFeatureFlags(env));
      }

      // Internal endpoints — separate auth pipeline (shared secret, not API key)
      if (url.pathname === "/internal/budget/invalidate" && request.method === "POST") {
        return stamp(await handleBudgetInvalidation(request, env));
      }
      if (url.pathname === "/internal/budget/velocity-state" && request.method === "GET") {
        return stamp(await handleVelocityState(request, env));
      }
      if (url.pathname.startsWith("/internal/request-bodies/") && request.method === "GET") {
        return stamp(await handleRequestBodies(request, env));
      }
      if (url.pathname === "/internal/plan-counter/increment" && request.method === "POST") {
        return stamp(await handlePlanCounterIncrement(request, env));
      }

      // Policy endpoint — GET with API key auth, no body parsing
      if (url.pathname === "/v1/policy" && request.method === "GET") {
        const connectionString = env.HYPERDRIVE.connectionString;
        const rateLimitResult = await applyRateLimit(request, env);
        if (rateLimitResult) {
          return stamp(rateLimitResult);
        }

        let auth;
        try {
          auth = await authenticateRequest(request, env, connectionString);
        } catch (err) {
          console.error("[proxy] Auth DB error:", err instanceof Error ? err.message : "Unknown error");
          emitMetric("request_error", { status: 503, reason: "auth_db_error" });
          const resp = errorResponse("service_unavailable", "Authentication service temporarily unavailable", 503);
          return stamp(resp);
        }
        if (!auth) {
          emitMetric("request_error", { status: 401, reason: "unauthorized" });
          const resp = errorResponse("unauthorized", "Invalid or missing authentication header", 401);
          return stamp(resp);
        }
        if (!auth.orgId) {
          const resp = errorResponse("forbidden", "API key must be associated with an organization", 403);
          return stamp(resp);
        }
        return stamp(await handlePolicy(request, env, auth, traceId));
      }

      // Route lookup: provider registry first, then MCP routes
      const providerAdapter = request.method === "POST" ? matchProviderRoute(url.pathname) : undefined;
      const mcpHandler = !providerAdapter && request.method === "POST" ? mcpRoutes.get(url.pathname) : undefined;
      if (!providerAdapter && !mcpHandler) {
        emitMetric("request_error", { status: 404, reason: "not_found" });
        if (url.pathname.startsWith("/v1/")) {
          const resp = errorResponse("not_found", "This endpoint is not yet supported", 404);
          return stamp(resp);
        }
        const resp = errorResponse("not_found", "Not found", 404);
        return stamp(resp);
      }

      // Rate limit first, then auth separately so DB errors produce 503, not 401
      const connectionString = env.HYPERDRIVE.connectionString;
      const preFlightStartMs = performance.now();
      const rateLimitResult = await applyRateLimit(request, env);
      if (rateLimitResult) {
        return stamp(rateLimitResult);
      }

      let auth;
      try {
        auth = await authenticateRequest(request, env, connectionString);
      } catch (err) {
        console.error("[proxy] Auth DB error:", err instanceof Error ? err.message : "Unknown error");
        emitMetric("request_error", { status: 503, reason: "auth_db_error" });
        const resp = errorResponse("service_unavailable", "Authentication service temporarily unavailable", 503);
        return stamp(resp);
      }
      const preFlightMs = Math.round(performance.now() - preFlightStartMs);

      if (!auth) {
        emitMetric("request_error", { status: 401, reason: "unauthorized" });
        const resp = errorResponse("unauthorized", "Invalid or missing authentication header", 401);
        return stamp(resp);
      }

      // S-1: Header-only validation (customer mandate, allowlist, session-id
      // length) runs BEFORE body parse so requests that can't possibly succeed
      // fail fast without consuming bandwidth/CPU on body read. Customer/tags
      // resolution depends only on request headers + auth.defaultTags — no
      // body dependency.
      const tags = mergeTags(auth.defaultTags, request.headers.get("x-nullspend-tags"));

      const customerHeaderResult = parseCustomerHeader(request.headers.get("x-nullspend-customer"));
      const customerId = resolveCustomerId(customerHeaderResult, tags);

      // CX-4: Enforce requireCustomerId — 400 when header is missing/malformed
      if (auth.requireCustomerId && !customerId) {
        emitMetric("customer_required_missing", { keyId: auth.keyId });
        const resp = errorResponse("customer_required",
          "This API key requires X-NullSpend-Customer header",
          400);
        return stamp(resp);
      }

      // CX-2: Validate customer ID against API key's allowed customers
      if (customerId && auth.allowedCustomers && !auth.allowedCustomers.includes(customerId)) {
        emitMetric("customer_not_allowed", { customerId, keyId: auth.keyId });
        const resp = errorResponse("customer_not_allowed",
          "Customer ID not authorized for this API key",
          403);
        return stamp(resp);
      }

      const rawSessionId = request.headers.get("x-nullspend-session");
      if (rawSessionId && rawSessionId.length > MAX_SESSION_ID_LENGTH) {
        emitMetric("request_error", { status: 400, reason: "session_id_too_long" });
        const resp = errorResponse("bad_request", `x-nullspend-session exceeds ${MAX_SESSION_ID_LENGTH} characters`, 400);
        return stamp(resp);
      }
      // PR-2c codex-round-3 H3: sync validated sessionId onto meta so any
      // stamp() call after this point carries the correct X-NullSpend-Session.
      meta.sessionId = rawSessionId?.trim() || null;

      // Body parse (sequential — budget check needs the parsed body)
      const bodyStartMs = performance.now();
      const result = await parseRequestBody(request);
      const bodyParseMs = Math.round(performance.now() - bodyStartMs);
      if (result.error) {
        emitMetric("request_error", { status: result.error.status, reason: "bad_request" });
        return stamp(result.error);
      }

      // Build context
      const webhookDispatcher = auth.hasWebhooks
        ? createWebhookDispatcher(optionalBinding(env, "WEBHOOK_QUEUE"), auth.orgId ?? auth.userId)
        : null;

      if (auth.hasWebhooks && !webhookDispatcher) {
        console.warn("[proxy] User has webhooks but WEBHOOK_QUEUE binding is not configured");
      }

      const resolvedApiVersion = resolveApiVersion(
        request.headers.get("nullspend-version"),
        auth.apiVersion,
      );

      const finalize = request.headers.get("x-nullspend-finalize") === "1";

      const ctx: RequestContext = {
        body: result.body,
        bodyText: result.bodyText,
        bodyByteLength: result.bodyByteLength,
        auth,
        ownerId: auth.orgId ?? auth.userId,
        connectionString,
        skipDbWrites,
        // PR-2c codex-round-3 H3: RequestContext extends IngressMetadata — these
        // three fields come from `meta`, set at ingress before any early-return.
        sessionId: meta.sessionId,
        traceId: meta.traceId,
        nullspendRequestId: meta.nullspendRequestId,
        tags,
        customerId,
        customerWarning: customerHeaderResult.warning,
        webhookDispatcher,
        resolvedApiVersion,
        requestStartMs,
        finalize,
        stepTiming: { preFlightMs, bodyParseMs },
        requestLoggingEnabled: auth.requestLoggingEnabled,
      };

      // Observability: track customer attribution rate
      emitMetric("customer_attribution", { present: ctx.customerId ? "true" : "false" });

      const response = providerAdapter
        ? await handleProviderRequest(request, env, ctx, providerAdapter)
        : await mcpHandler!(request, env, ctx);
      if (Object.keys(ctx.tags).length > 0) {
        response.headers.set("X-NullSpend-Effective-Tags", JSON.stringify(ctx.tags));
      }
      if (ctx.customerWarning) {
        response.headers.set("X-NullSpend-Warning", ctx.customerWarning);
      }
      // PR-2c codex-round-3 H3: stamp final response too — covers the happy
      // path 200, upstream 502, and any other response that bubbles up from
      // route handlers without going through an early-return stamp.
      return stamp(response);
    } catch (err) {
      console.error("[proxy] Unhandled error:", { traceId, err });
      emitMetric("request_error", { status: 500, reason: "internal_error" });
      const resp = errorResponse("internal_error", "Internal server error", 500);
      return stamp(resp);
    }
  },
};
