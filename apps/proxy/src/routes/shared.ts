import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import { getWebhookEndpoints, getWebhookEndpointsWithSecrets } from "../lib/webhook-cache.js";
import { lookupCustomerUpgradeUrl, type BudgetEntity } from "../lib/budget-do-lookup.js";
import { resolveUpgradeUrl } from "../lib/upgrade-url.js";
import {
  buildVelocityExceededPayload,
  buildVelocityRecoveredPayload,
  buildSessionLimitExceededPayload,
  buildTagBudgetExceededPayload,
  buildCustomerBudgetExceededPayload,
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

/**
 * Build X-NullSpend-Budget-* response headers from the budget entities
 * checked during this request.
 *
 * Stripe-pattern "budget proximity" signal — lets clients monitor how close
 * they are to the wall without issuing separate API calls. Values are in
 * microdollars (matching internal storage + the *_microdollars fields
 * already present in denial response bodies).
 *
 * Returns an empty record when there are no budget entities. The absence
 * of headers signals "no budget enforcement" — NOT "unlimited." Clients
 * should treat missing headers as "the proxy has nothing to say about
 * budgets for this request."
 *
 * For multi-entity requests (e.g. user + org + customer + tag), picks the
 * single entity with the lowest remaining — the one that will bite first
 * on the next request. Adds X-NullSpend-Budget-Entity so clients know
 * which entity the triplet refers to. Ties broken deterministically by
 * (entityType, entityId) ASCII order.
 *
 * Snapshot semantics (documented limitation): values reflect the state
 * at the time the budget check ran. On streaming responses, headers are
 * flushed before the upstream completes, so `remaining` is the
 * post-reservation-pre-reconcile view — NOT guaranteed live after the
 * stream finishes. These headers are a proximity signal, not an
 * enforcement mechanism.
 *
 * @param budgetEntities entities checked during this request
 * @param reservedForThisRequest amount the request reserved against the
 *   DO — pass `estimate` on approved responses (the reservation landed),
 *   pass `0` on denied responses (no reservation landed).
 */
export function buildBudgetHeaders(
  budgetEntities: Pick<BudgetEntity, "entityType" | "entityId" | "maxBudget" | "spend" | "reserved">[],
  reservedForThisRequest: number,
): Record<string, string> {
  if (budgetEntities.length === 0) return {};

  // Pick the tightest entity — lowest remaining is the one that will bite
  // first. Tie-break deterministically so two clients hitting the same
  // state always see the same entity identifier.
  let tightest = budgetEntities[0];
  let tightestRemaining = Math.max(
    0,
    tightest.maxBudget - tightest.spend - tightest.reserved - reservedForThisRequest,
  );

  for (let i = 1; i < budgetEntities.length; i++) {
    const e = budgetEntities[i];
    const r = Math.max(0, e.maxBudget - e.spend - e.reserved - reservedForThisRequest);
    if (r < tightestRemaining) {
      tightest = e;
      tightestRemaining = r;
    } else if (r === tightestRemaining) {
      if (
        e.entityType < tightest.entityType ||
        (e.entityType === tightest.entityType && e.entityId < tightest.entityId)
      ) {
        tightest = e;
      }
    }
  }

  const spent = tightest.spend + tightest.reserved + reservedForThisRequest;
  return {
    "X-NullSpend-Budget-Limit": String(tightest.maxBudget),
    "X-NullSpend-Budget-Spent": String(spent),
    "X-NullSpend-Budget-Remaining": String(tightestRemaining),
    "X-NullSpend-Budget-Entity": `${tightest.entityType}:${tightest.entityId}`,
  };
}

export interface EnrichmentFields {
  upstreamDurationMs: number;
  sessionId: string | null;
  traceId: string;
  toolDefinitionTokens: number;
  tags: Record<string, string>;
  customerId: string | null;
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
 *
 * Async because `budget_exceeded` and `customer_budget_exceeded` denials
 * resolve an optional `upgrade_url` which may require a cold-path
 * Postgres lookup for per-customer overrides. Hot path (200 success)
 * never touches this code. On velocity/session/tag denials, no
 * upgrade_url lookup happens — per decision 5 of the plan, only
 * budget/customer-budget denials include upgrade_url.
 */
export async function handleBudgetDenials(
  outcome: BudgetCheckOutcome,
  ctx: RequestContext,
  env: Env,
  provider: Provider,
  requestModel: string,
  estimate: number,
  budgetEntities: BudgetEntity[],
): Promise<Response | null> {
  const logPrefix = `[${provider}-route]`;

  // Metric emission is deferred to the per-branch return points below so
  // the `upgradeUrlEmitted` tag reflects the actual response state (post
  // resolution + customer_settings lookup), not just the auth identity.
  // See E5 in the edge-case audit. The reason is computed here once so
  // all branches share a single source of truth.
  const reason = outcome.status === "denied"
    ? (outcome.velocityDenied ? "velocity_exceeded"
       : outcome.sessionLimitDenied ? "session_limit_exceeded"
       : outcome.tagBudgetDenied ? "tag_budget_exceeded"
       : outcome.deniedEntityType === "customer" ? "customer_budget_exceeded"
       : "budget_exceeded")
    : null;

  function emitDenialMetric(upgradeUrlEmitted: boolean, upgradeUrlSource: "per_customer" | "org" | "none"): void {
    emitMetric("budget_denied", {
      reason: reason ?? "unknown",
      provider,
      entityType: outcome.deniedEntityType ?? "unknown",
      upgradeUrlEmitted,
      upgradeUrlSource,
    });
  }

  // On denial the request did NOT reserve its estimate against the DO,
  // so pass 0 for reservedForThisRequest — headers reflect the state
  // right before this (rejected) request.
  const budgetHeaders = buildBudgetHeaders(budgetEntities, 0);

  // Velocity denial
  if (outcome.status === "denied" && outcome.velocityDenied) {
    emitDenialMetric(false, "none"); // upgrade_url never included on velocity
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
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

  // Session limit denial
  if (outcome.status === "denied" && outcome.sessionLimitDenied) {
    emitDenialMetric(false, "none");
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
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

  // Tag budget denial
  if (outcome.status === "denied" && outcome.tagBudgetDenied) {
    emitDenialMetric(false, "none"); // tag budgets don't carry upgrade_url
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
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

  // Customer budget denial
  if (outcome.status === "denied" && outcome.deniedEntityType === "customer") {
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildCustomerBudgetExceededPayload({
        customerId: outcome.deniedEntityId ?? "unknown",
        budgetLimitMicrodollars: outcome.maxBudget ?? 0,
        budgetSpendMicrodollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
        estimatedRequestCostMicrodollars: estimate,
        model: requestModel,
        provider,
      }, ctx.auth.apiVersion),
    );
    // Resolve upgrade_url: per-customer override (from customer_settings)
    // takes priority over org-level default. Cold-path Postgres query,
    // fails open on error. Only fires on customer denial paths.
    const customerId = outcome.deniedEntityId ?? ctx.customerId ?? null;
    // T7: defensive — the customer branch enter on deniedEntityType === "customer"
    // means the DO told us this is a customer denial. If deniedEntityId is
    // somehow missing, fall back to ctx.customerId. If BOTH are null we still
    // emit the denial (with customer_id: null in the body) but log a warning
    // + metric so the pathway is observable.
    if (customerId === null) {
      console.warn(
        `${logPrefix} customer_budget_exceeded denial with null customer_id (denied_entity=${outcome.deniedEntityId} ctx_customer=${ctx.customerId})`,
      );
      emitMetric("customer_denial_missing_id", {
        provider,
        orgId: ctx.auth.orgId ?? "unknown",
      });
    }
    const customerUrl = customerId && ctx.auth.orgId
      ? await lookupCustomerUpgradeUrl(ctx.connectionString, ctx.auth.orgId, customerId)
      : null;
    const upgradeUrl = resolveUpgradeUrl(ctx.auth.orgUpgradeUrl, customerUrl, customerId);
    // E5: emit the metric AFTER resolution so the source tag is accurate.
    // Per-customer override wins over org-level when both are set.
    const source: "per_customer" | "org" | "none" =
      customerUrl != null ? "per_customer"
      : (upgradeUrl != null ? "org" : "none");
    emitDenialMetric(upgradeUrl != null, source);

    return new Response(
      JSON.stringify({
        error: {
          code: "customer_budget_exceeded",
          message: "Request blocked: estimated cost exceeds customer budget limit.",
          ...(upgradeUrl ? { upgrade_url: upgradeUrl } : {}),
          details: {
            customer_id: outcome.deniedEntityId ?? null,
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
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

  // Generic budget denial
  if (outcome.status === "denied") {
    const entityType = outcome.deniedEntityType ?? budgetEntities?.[0]?.entityType ?? "unknown";
    const entityId = outcome.deniedEntityId ?? budgetEntities?.[0]?.entityId ?? "unknown";
    const budgetLimit = outcome.maxBudget ?? 0;
    const budgetSpend = (outcome.spend ?? 0) + (outcome.reserved ?? 0);
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildBudgetExceededPayload({
        budgetEntityType: entityType,
        budgetEntityId: entityId,
        budgetLimitMicrodollars: budgetLimit,
        budgetSpendMicrodollars: budgetSpend,
        estimatedRequestCostMicrodollars: estimate,
        model: requestModel,
        provider,
      }),
    );
    // Resolve upgrade_url for the generic budget_exceeded path. This
    // only considers the org-level default — per-customer overrides are
    // reserved for the customer_budget_exceeded branch above. No
    // Postgres lookup needed; the org URL came in on the auth identity.
    const genericUpgradeUrl = resolveUpgradeUrl(
      ctx.auth.orgUpgradeUrl,
      null,
      ctx.customerId ?? null,
    );
    emitDenialMetric(
      genericUpgradeUrl != null,
      genericUpgradeUrl != null ? "org" : "none",
    );

    return new Response(
      JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: "Request blocked: estimated cost exceeds remaining budget.",
          ...(genericUpgradeUrl ? { upgrade_url: genericUpgradeUrl } : {}),
          details: {
            entity_type: entityType,
            entity_id: entityId,
            budget_limit_microdollars: budgetLimit,
            budget_spend_microdollars: budgetSpend,
            estimated_cost_microdollars: estimate,
          },
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Trace-Id": ctx.traceId,
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
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
