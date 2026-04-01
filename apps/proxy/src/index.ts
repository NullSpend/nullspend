import { handleChatCompletions } from "./routes/openai.js";
import { handleAnthropicMessages } from "./routes/anthropic.js";
import { handleMcpBudgetCheck, handleMcpEvents } from "./routes/mcp.js";
import { handleBudgetInvalidation, handleVelocityState, handleRequestBodies } from "./routes/internal.js";
import { handleMetrics } from "./routes/metrics.js";
import { handlePolicy } from "./routes/policy.js";
import { authenticateRequest } from "./lib/auth.js";
import { resolveApiVersion } from "./lib/api-version.js";
import { errorResponse } from "./lib/errors.js";
import { createWebhookDispatcher } from "./lib/webhook-dispatch.js";
import { mergeTags } from "./lib/tags.js";
import { resolveTraceId } from "./lib/trace-context.js";
import { emitMetric } from "./lib/metrics.js";
import type { RequestContext, RouteHandler } from "./lib/context.js";
import { handleReconciliationQueue } from "./queue-handler.js";
import { handleDlqQueue, DLQ_QUEUE_NAME } from "./dlq-handler.js";
import { handleCostEventQueue, COST_EVENT_QUEUE_NAME } from "./cost-event-queue-handler.js";
import { handleCostEventDlq, COST_EVENT_DLQ_NAME } from "./cost-event-dlq-handler.js";
import { handleWebhookQueue, WEBHOOK_QUEUE_NAME } from "./webhook-queue-handler.js";
import { handleWebhookDlq, WEBHOOK_DLQ_NAME } from "./webhook-dlq-handler.js";
import type { ReconciliationMessage } from "./lib/reconciliation-queue.js";
import type { CostEventMessage } from "./lib/cost-event-queue.js";
import type { WebhookQueueMessage } from "./lib/webhook-queue.js";

export { UserBudgetDO } from "./durable-objects/user-budget.js";

const MAX_BODY_SIZE = 1_048_576; // 1MB

const routes = new Map<string, RouteHandler>();
routes.set("/v1/chat/completions", handleChatCompletions);
routes.set("/v1/messages", handleAnthropicMessages);
routes.set("/v1/mcp/budget/check", handleMcpBudgetCheck);
routes.set("/v1/mcp/events", handleMcpEvents);

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

  let bodyText: string;
  try {
    bodyText = await request.text();
  } catch {
    return {
      error: errorResponse("bad_request", "Could not read request body", 400),
    };
  }

  // Content-Length pre-check above catches well-behaved clients.
  // Workers runtime enforces its own body size limits on request.text().
  // No need for a redundant TextEncoder().encode() copy just to check byte length.

  try {
    const parsed = JSON.parse(bodyText);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        error: errorResponse("bad_request", "Request body must be a JSON object", 400),
      };
    }
    return { body: parsed, bodyText, bodyByteLength: bodyText.length };
  } catch {
    return {
      error: errorResponse("bad_request", "Invalid JSON body", 400),
    };
  }
}

const MAX_SESSION_ID_LENGTH = 256;

export default {
  async queue(
    batch: MessageBatch<ReconciliationMessage | CostEventMessage | WebhookQueueMessage>,
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
    } else if (batch.queue === DLQ_QUEUE_NAME) {
      await handleDlqQueue(batch as MessageBatch<ReconciliationMessage>, env);
    } else {
      await handleReconciliationQueue(batch as MessageBatch<ReconciliationMessage>, env);
    }
  },

  async fetch(
    request: Request,
    env: Env,
    _ctx: ExecutionContext,
  ): Promise<Response> {
    const requestStartMs = performance.now();
    const forceDbPersist = (env as Record<string, unknown>).FORCE_DB_PERSIST === "true";
    const skipDbPersist = (env as Record<string, unknown>).SKIP_DB_PERSIST === "true";
    const skipDbWrites = !forceDbPersist && skipDbPersist;

    // Resolve trace ID early so it's available in the catch block for 500 responses
    const traceId = resolveTraceId(request);

    try {
      const url = new URL(request.url);

      // Health routes stay outside the pipeline (no auth needed)
      if (url.pathname === "/health") {
        return Response.json({ status: "ok", service: "nullspend-proxy" });
      }

      if (url.pathname === "/health/ready") {
        return Response.json({ status: "ok", service: "nullspend-proxy" });
      }

      if (url.pathname === "/health/metrics") {
        return handleMetrics(request, env);
      }

      // Internal endpoints — separate auth pipeline (shared secret, not API key)
      if (url.pathname === "/internal/budget/invalidate" && request.method === "POST") {
        return handleBudgetInvalidation(request, env);
      }
      if (url.pathname === "/internal/budget/velocity-state" && request.method === "GET") {
        return handleVelocityState(request, env);
      }
      if (url.pathname.startsWith("/internal/request-bodies/") && request.method === "GET") {
        return handleRequestBodies(request, env);
      }

      // Policy endpoint — GET with API key auth, no body parsing
      if (url.pathname === "/v1/policy" && request.method === "GET") {
        const connectionString = env.HYPERDRIVE.connectionString;
        const [rateLimitResult, auth] = await Promise.all([
          applyRateLimit(request, env),
          authenticateRequest(request, connectionString),
        ]);
        if (rateLimitResult) {
          rateLimitResult.headers.set("X-NullSpend-Trace-Id", traceId);
          return rateLimitResult;
        }
        if (!auth) {
          emitMetric("request_error", { status: 401, reason: "unauthorized" });
          const resp = errorResponse("unauthorized", "Invalid or missing authentication header", 401);
          resp.headers.set("X-NullSpend-Trace-Id", traceId);
          return resp;
        }
        if (!auth.orgId) {
          const resp = errorResponse("forbidden", "API key must be associated with an organization", 403);
          resp.headers.set("X-NullSpend-Trace-Id", traceId);
          return resp;
        }
        return handlePolicy(request, env, auth, traceId);
      }

      // Route lookup
      const handler = request.method === "POST" ? routes.get(url.pathname) : undefined;
      if (!handler) {
        emitMetric("request_error", { status: 404, reason: "not_found" });
        if (url.pathname.startsWith("/v1/")) {
          const resp = errorResponse("not_found", "This endpoint is not yet supported", 404);
          resp.headers.set("X-NullSpend-Trace-Id", traceId);
          return resp;
        }
        const resp = errorResponse("not_found", "Not found", 404);
        resp.headers.set("X-NullSpend-Trace-Id", traceId);
        return resp;
      }

      // Rate limit + auth in parallel (neither reads request body)
      const connectionString = env.HYPERDRIVE.connectionString;
      const preFlightStartMs = performance.now();
      const [rateLimitResult, auth] = await Promise.all([
        applyRateLimit(request, env),
        authenticateRequest(request, connectionString),
      ]);
      const preFlightMs = Math.round(performance.now() - preFlightStartMs);

      if (rateLimitResult) {
        rateLimitResult.headers.set("X-NullSpend-Trace-Id", traceId);
        return rateLimitResult;
      }
      if (!auth) {
        emitMetric("request_error", { status: 401, reason: "unauthorized" });
        const resp = errorResponse("unauthorized", "Invalid or missing authentication header", 401);
        resp.headers.set("X-NullSpend-Trace-Id", traceId);
        return resp;
      }

      // Body parse (sequential — budget check needs the parsed body)
      const bodyStartMs = performance.now();
      const result = await parseRequestBody(request);
      const bodyParseMs = Math.round(performance.now() - bodyStartMs);
      if (result.error) {
        emitMetric("request_error", { status: result.error.status, reason: "bad_request" });
        result.error.headers.set("X-NullSpend-Trace-Id", traceId);
        return result.error;
      }

      // Build context
      const webhookDispatcher = auth.hasWebhooks
        ? createWebhookDispatcher((env as Record<string, unknown>).WEBHOOK_QUEUE as Queue | undefined, auth.orgId ?? auth.userId)
        : null;

      if (auth.hasWebhooks && !webhookDispatcher) {
        console.warn("[proxy] User has webhooks but WEBHOOK_QUEUE binding is not configured");
      }

      const resolvedApiVersion = resolveApiVersion(
        request.headers.get("nullspend-version"),
        auth.apiVersion,
      );

      const tags = mergeTags(auth.defaultTags, request.headers.get("x-nullspend-tags"));

      const rawSessionId = request.headers.get("x-nullspend-session");
      if (rawSessionId && rawSessionId.length > MAX_SESSION_ID_LENGTH) {
        emitMetric("request_error", { status: 400, reason: "session_id_too_long" });
        return errorResponse("bad_request", `x-nullspend-session exceeds ${MAX_SESSION_ID_LENGTH} characters`, 400);
      }

      const ctx: RequestContext = {
        body: result.body,
        bodyText: result.bodyText,
        bodyByteLength: result.bodyByteLength,
        auth,
        ownerId: auth.orgId ?? auth.userId,
        connectionString,
        skipDbWrites,
        sessionId: rawSessionId?.trim() || null,
        traceId,
        tags,
        webhookDispatcher,
        resolvedApiVersion,
        requestStartMs,
        stepTiming: { preFlightMs, bodyParseMs },
        requestLoggingEnabled: auth.requestLoggingEnabled,
      };

      const response = await handler(request, env, ctx);
      if (Object.keys(ctx.tags).length > 0) {
        response.headers.set("X-NullSpend-Effective-Tags", JSON.stringify(ctx.tags));
      }
      return response;
    } catch (err) {
      console.error("[proxy] Unhandled error:", { traceId, err });
      emitMetric("request_error", { status: 500, reason: "internal_error" });
      const resp = errorResponse("internal_error", "Internal server error", 500);
      resp.headers.set("X-NullSpend-Trace-Id", traceId);
      return resp;
    }
  },
};

