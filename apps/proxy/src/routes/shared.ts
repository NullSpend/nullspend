import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import { errorResponse } from "../lib/errors.js";
import { getWebhookEndpoints, getWebhookEndpointsWithSecrets } from "../lib/webhook-cache.js";
import type { BudgetEntity } from "../lib/budget-do-lookup.js";
import {
  buildVelocityExceededPayload,
  buildVelocityRecoveredPayload,
  buildSessionLimitExceededPayload,
  buildTagBudgetExceededPayload,
  buildBudgetExceededPayload,
  buildCostEventPayload,
  buildThinCostEventPayload,
  CURRENT_API_VERSION,
} from "../lib/webhook-events.js";
import { dispatchToEndpoints } from "../lib/webhook-dispatch.js";
import { detectThresholdCrossings } from "../lib/webhook-thresholds.js";
import { expireRotatedSecrets } from "../lib/webhook-expiry.js";
import { emitMetric } from "../lib/metrics.js";

export type Provider = "openai" | "anthropic" | "mcp";

export type Attribution = { userId: string | null; apiKeyId: string | null; actionId: string | null };

export interface EnrichmentFields {
  upstreamDurationMs: number;
  sessionId: string | null;
  traceId: string;
  toolDefinitionTokens: number;
  tags: Record<string, string>;
  budgetStatus: "skipped" | "approved" | "denied";
  estimatedCostMicrodollars: number;
  orgId: string | null;
}

interface BudgetCheckOutcome {
  status: "approved" | "denied" | "skipped";
  reservationId: string | null;
  budgetEntities: BudgetEntity[];
  velocityDenied?: boolean;
  sessionLimitDenied?: boolean;
  tagBudgetDenied?: boolean;
  deniedEntityType?: string;
  deniedEntityId?: string;
  velocityDetails?: { limitMicrodollars: number; windowSeconds: number; currentMicrodollars: number };
  retryAfterSeconds?: number;
  sessionId?: string;
  sessionSpend?: number;
  sessionLimit?: number;
  tagKey?: string;
  tagValue?: string;
  maxBudget?: number;
  spend?: number;
  reserved?: number;
  velocityRecovered?: Array<{
    entityType: string;
    entityId: string;
    velocityLimitMicrodollars: number;
    velocityWindowSeconds: number;
    velocityCooldownSeconds: number;
  }>;
}

/**
 * Handle all budget denial types. Returns a Response if denied, null if approved/skipped.
 * Dispatches the appropriate webhook in waitUntil for each denial type.
 */
export function handleBudgetDenials(
  outcome: BudgetCheckOutcome,
  ctx: RequestContext,
  env: Env,
  provider: Provider,
  requestModel: string,
  estimate: number,
  budgetEntities: BudgetEntity[],
): Response | null {
  const logPrefix = `[${provider}-route]`;

  if (outcome.status === "denied") {
    const reason = outcome.velocityDenied ? "velocity_exceeded"
      : outcome.sessionLimitDenied ? "session_limit_exceeded"
      : outcome.tagBudgetDenied ? "tag_budget_exceeded"
      : "budget_exceeded";
    emitMetric("budget_denied", { reason, provider, entityType: outcome.deniedEntityType ?? "unknown" });
  }

  // Velocity denial
  if (outcome.status === "denied" && outcome.velocityDenied) {
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildVelocityExceededPayload({
        budgetEntityType: outcome.deniedEntityType ?? "unknown",
        budgetEntityId: outcome.deniedEntityId ?? "unknown",
        velocityLimitMicrodollars: outcome.velocityDetails?.limitMicrodollars ?? 0,
        velocityWindowSeconds: outcome.velocityDetails?.windowSeconds ?? 60,
        velocityCurrentMicrodollars: outcome.velocityDetails?.currentMicrodollars ?? 0,
        cooldownSeconds: outcome.retryAfterSeconds ?? 60,
        model: requestModel,
        provider,
      }, ctx.auth.apiVersion),
    );
    return new Response(
      JSON.stringify({
        error: {
          code: "velocity_exceeded",
          message: "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.",
          details: outcome.velocityDetails ?? null,
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(outcome.retryAfterSeconds ?? 60),
          "X-NullSpend-Trace-Id": ctx.traceId,
        },
      },
    );
  }

  // Session limit denial
  if (outcome.status === "denied" && outcome.sessionLimitDenied) {
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildSessionLimitExceededPayload({
        budgetEntityType: outcome.deniedEntityType ?? "unknown",
        budgetEntityId: outcome.deniedEntityId ?? "unknown",
        sessionId: outcome.sessionId ?? "unknown",
        sessionSpendMicrodollars: outcome.sessionSpend ?? 0,
        sessionLimitMicrodollars: outcome.sessionLimit ?? 0,
        model: requestModel,
        provider,
      }, ctx.auth.apiVersion),
    );
    return new Response(
      JSON.stringify({
        error: {
          code: "session_limit_exceeded",
          message: "Request blocked: session spend exceeds session limit. Start a new session.",
          details: {
            session_id: outcome.sessionId ?? null,
            session_spend_microdollars: outcome.sessionSpend ?? 0,
            session_limit_microdollars: outcome.sessionLimit ?? 0,
          },
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Trace-Id": ctx.traceId,
        },
      },
    );
  }

  // Tag budget denial
  if (outcome.status === "denied" && outcome.tagBudgetDenied) {
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildTagBudgetExceededPayload({
        tagKey: outcome.tagKey ?? "unknown",
        tagValue: outcome.tagValue ?? "unknown",
        budgetEntityId: outcome.deniedEntityId ?? "unknown",
        budgetLimitMicrodollars: outcome.maxBudget ?? 0,
        budgetSpendMicrodollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
        estimatedRequestCostMicrodollars: estimate,
        model: requestModel,
        provider,
      }, ctx.auth.apiVersion),
    );
    return new Response(
      JSON.stringify({
        error: {
          code: "tag_budget_exceeded",
          message: "Request blocked: estimated cost exceeds tag budget limit.",
          details: {
            tag_key: outcome.tagKey ?? null,
            tag_value: outcome.tagValue ?? null,
            budget_limit_microdollars: outcome.maxBudget ?? 0,
            budget_spend_microdollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
          },
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Trace-Id": ctx.traceId,
        },
      },
    );
  }

  // Generic budget denial
  if (outcome.status === "denied") {
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildBudgetExceededPayload({
        budgetEntityType: outcome.deniedEntityType ?? budgetEntities[0]?.entityType ?? "unknown",
        budgetEntityId: outcome.deniedEntityId ?? budgetEntities[0]?.entityId ?? "unknown",
        budgetLimitMicrodollars: outcome.maxBudget ?? 0,
        budgetSpendMicrodollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
        estimatedRequestCostMicrodollars: estimate,
        model: requestModel,
        provider,
      }),
    );
    const resp = errorResponse("budget_exceeded", "Request blocked: estimated cost exceeds remaining budget", 429);
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    return resp;
  }

  return null; // Not denied — continue processing
}

/**
 * Dispatch velocity recovery webhooks when circuit breaker clears.
 */
export function dispatchVelocityRecoveryWebhooks(
  outcome: BudgetCheckOutcome,
  ctx: RequestContext,
  env: Env,
  provider: Provider,
): void {
  if (!outcome.velocityRecovered?.length || !ctx.webhookDispatcher || !ctx.auth.hasWebhooks) return;
  const logPrefix = `[${provider}-route]`;

  waitUntil((async () => {
    try {
      const cached = await getWebhookEndpoints(ctx.connectionString, ctx.ownerId, env.CACHE_KV);
      if (cached.length > 0) {
        const endpoints = await getWebhookEndpointsWithSecrets(ctx.connectionString, ctx.ownerId);
        for (const recovered of outcome.velocityRecovered!) {
          const event = buildVelocityRecoveredPayload({
            budgetEntityType: recovered.entityType,
            budgetEntityId: recovered.entityId,
            velocityLimitMicrodollars: recovered.velocityLimitMicrodollars,
            velocityWindowSeconds: recovered.velocityWindowSeconds,
            velocityCooldownSeconds: recovered.velocityCooldownSeconds,
          }, ctx.auth.apiVersion);
          await dispatchToEndpoints(ctx.webhookDispatcher!, endpoints, event);
        }
      }
    } catch (err) {
      console.error(`${logPrefix} Velocity recovery webhook dispatch failed:`, err);
    }
  })());
}

/**
 * Dispatch cost event webhooks + threshold crossing detection.
 * Called from waitUntil in both streaming and non-streaming paths.
 */
export async function dispatchCostEventWebhooks(
  ctx: RequestContext,
  env: Env,
  provider: Provider,
  costEvent: { requestId: string; provider: string; costMicrodollars: number; [key: string]: unknown },
  enrichment: { traceId: string; [key: string]: unknown },
  budgetEntities: BudgetEntity[],
  toolCallsRequested: unknown,
): Promise<void> {
  if (!ctx.webhookDispatcher || !ctx.auth.hasWebhooks) return;
  const logPrefix = `[${provider}-route]`;

  try {
    const cached = await getWebhookEndpoints(ctx.connectionString, ctx.ownerId, env.CACHE_KV);
    if (cached.length > 0) {
      const endpoints = await getWebhookEndpointsWithSecrets(ctx.connectionString, ctx.ownerId);
      const webhookData = {
        ...costEvent,
        ...enrichment,
        toolCallsRequested,
        createdAt: new Date().toISOString(),
        source: "proxy" as const,
      };
      for (const ep of endpoints) {
        if ((ep.payloadMode ?? "full") === "thin") {
          await ctx.webhookDispatcher!.dispatch(ep, buildThinCostEventPayload(webhookData.requestId, webhookData.provider as string, ep.apiVersion));
        } else {
          await ctx.webhookDispatcher!.dispatch(ep, buildCostEventPayload(webhookData, ep.apiVersion));
        }
      }

      if (budgetEntities.length > 0) {
        const epVersion = endpoints[0]?.apiVersion ?? CURRENT_API_VERSION;
        const thresholdEvents = detectThresholdCrossings(budgetEntities, costEvent.costMicrodollars, costEvent.requestId, epVersion);
        for (const te of thresholdEvents) {
          await dispatchToEndpoints(ctx.webhookDispatcher!, endpoints, te);
        }
      }
      expireRotatedSecrets(ctx.connectionString, endpoints).catch(() => {});
    }
  } catch (err) {
    console.error(`${logPrefix} Webhook dispatch failed:`, err);
  }
}

// ── Shared helpers ──────────────────────────────────────────────────

/**
 * Dispatch a denial webhook event in waitUntil.
 * Used by handleBudgetDenials internally and by MCP route for its custom response format.
 */
export function dispatchDenialWebhook(
  ctx: RequestContext,
  env: Env,
  logPrefix: string,
  buildEvent: () => ReturnType<typeof buildVelocityExceededPayload>,
): void {
  if (!ctx.webhookDispatcher || !ctx.auth.hasWebhooks) return;

  waitUntil((async () => {
    try {
      const cached = await getWebhookEndpoints(ctx.connectionString, ctx.ownerId, env.CACHE_KV);
      if (cached.length > 0) {
        const endpoints = await getWebhookEndpointsWithSecrets(ctx.connectionString, ctx.ownerId);
        const event = buildEvent();
        await dispatchToEndpoints(ctx.webhookDispatcher!, endpoints, event);
      }
    } catch (err) {
      console.error(`${logPrefix} Webhook dispatch failed:`, err);
    }
  })());
}
