import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import type { TierLabel } from "../lib/api-key-auth.js";
import { getPricingUrl, getSelfHostUrl } from "../lib/constants.js";
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
  buildPlanLimitExceededPayload,
  buildLoopDetectedPayload,
  buildCostEventPayload,
  buildThinCostEventPayload,
  buildThresholdPayload,
  CURRENT_API_VERSION,
} from "../lib/webhook-events.js";
import { dispatchToEndpoints } from "../lib/webhook-dispatch.js";
import { expireRotatedSecrets } from "../lib/webhook-expiry.js";
import { emitMetric } from "../lib/metrics.js";

export type Provider = "openai" | "anthropic" | "google" | "mcp";

export type Attribution = { userId: string | null; apiKeyId: string | null; actionId: string | null };

/** Machine-readable recovery hints on every 429 denial. */
export interface Recovery {
  retryable: boolean;
  owner_action_required: boolean;
  retry_after_seconds: number | null;
  docs: string | null;
}

/** Build a Recovery object for a given denial code. */
export function buildRecovery(
  code: string,
  retryAfterSeconds?: number | null,
): Recovery {
  const retryable = code === "velocity_exceeded" || code === "loop_detected";
  // PR-2c plan-audit F10: add "plan_limit_exceeded" to the ownerAction array
  // (NOT a switch refactor). Plan-limit is a hard block — owner must upgrade
  // or wait for period reset. retry_after_seconds stays null (not retryable).
  const ownerAction = ["budget_exceeded", "customer_budget_exceeded", "tag_budget_exceeded", "plan_limit_exceeded"].includes(code);
  return {
    retryable,
    owner_action_required: ownerAction,
    retry_after_seconds: retryable && retryAfterSeconds != null && Number.isFinite(retryAfterSeconds) && retryAfterSeconds >= 0
      ? retryAfterSeconds
      : null,
    docs: null, // populated when per-error docs pages ship
  };
}

/** Hardcoded backoff for loop detection denials (seconds). Used in both
 *  Retry-After header and recovery.retry_after_seconds body field. */
export const LOOP_RETRY_AFTER_SECONDS = 5;

/** Format microdollars as a dollar string like "$5.00".
 *  Display-only — all programmatic fields remain integer microdollars. */
export function fmtDollars(microdollars: number): string {
  return "$" + (microdollars / 1_000_000).toFixed(2);
}

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
  budgetEntities: Pick<BudgetEntity, "entityType" | "entityId" | "maxBudget" | "spend" | "reserved" | "finalizationReserve" | "avgRecentCost">[],
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

  const reserve = tightest.finalizationReserve ?? 0;
  const avgCost = tightest.avgRecentCost ?? 0;
  const spent = tightest.spend + tightest.reserved + reservedForThisRequest;
  const effectiveRemaining = Math.max(0, tightestRemaining - reserve);

  const headers: Record<string, string> = {
    "X-NullSpend-Budget-Limit": String(tightest.maxBudget),
    "X-NullSpend-Budget-Spent": String(spent),
    "X-NullSpend-Budget-Remaining": String(tightestRemaining),
    "X-NullSpend-Budget-Entity": `${tightest.entityType}:${tightest.entityId}`,
  };

  if (reserve > 0) {
    headers["X-NullSpend-Budget-Finalization-Reserve"] = String(reserve);
    headers["X-NullSpend-Budget-Effective-Remaining"] = String(effectiveRemaining);
  }

  if (avgCost > 0 && effectiveRemaining > 0) {
    headers["X-NullSpend-Budget-Requests-Remaining"] = `~${Math.floor(effectiveRemaining / avgCost)}`;
  }

  return headers;
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
  // PR-2d: billing period bounds resolved via resolvePeriodBounds(ctx.auth, now).
  // Stamped at ingest so late-arriving cost events land on the ORIGINAL period
  // row instead of the current period at reconcile time. Never null for new
  // writes — paid orgs use Stripe bounds, free/unpaid fall back to calendar month.
  periodStart: Date;
  periodEnd: Date;
}

interface BudgetCheckOutcome {
  status: "approved" | "denied" | "skipped";
  reservationId: string | null;
  budgetEntities: BudgetEntity[];
  /** Set when the denial is due to an invalid cost estimate (NaN/Infinity/negative). */
  invalidEstimate?: boolean;
  /** PR-2c: plan-limit denial (NullSpend-tier enforcement, fires BEFORE DO chain). */
  planLimitDenied?: boolean;
  planLimitCount?: number;
  planLimitBlockAt?: number;
  planLimitTier?: TierLabel;
  velocityDenied?: boolean;
  sessionLimitDenied?: boolean;
  tagBudgetDenied?: boolean;
  loopDetected?: boolean;
  loopDetails?: {
    type: "per_key" | "aggregate";
    model: string;
    provider: string;
    callCount: number;
    windowSeconds: number;
    maxCalls: number;
  };
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
  //
  // PR-2c codex-round-1 H3: `planLimitDenied` is the FIRST branch. Without
  // this ordering, metric tags `reason="budget_exceeded"` while body emits
  // `plan_limit_exceeded` — alerts + dashboards would lie.
  const reason = outcome.status === "denied"
    ? (outcome.planLimitDenied ? "plan_limit_exceeded"
       : outcome.velocityDenied ? "velocity_exceeded"
       : outcome.loopDetected ? "loop_detected"
       : outcome.sessionLimitDenied ? "session_limit_exceeded"
       : outcome.tagBudgetDenied ? "tag_budget_exceeded"
       : outcome.deniedEntityType === "customer" ? "customer_budget_exceeded"
       : "budget_exceeded")
    : null;

  // PR-2c codex-round-1 M9: added "platform" for plan-limit denials —
  // upgrade_url comes from NULLSPEND_PRICING_URL, NOT per-customer or org
  // settings. Mis-attributing would corrupt org-upgrade-URL adoption analytics.
  function emitDenialMetric(upgradeUrlEmitted: boolean, upgradeUrlSource: "per_customer" | "org" | "platform" | "none"): void {
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

  // PR-2c: plan-limit denial — fires BEFORE all other denial branches per
  // Decision #28 priority. Only reachable when `checkBudget` returned early
  // with `planLimitDenied: true` (Free-tier org at cap + PLAN_COUNTER_ENABLED=true).
  // Upgrade URL + self-host URL come from NullSpend defaults (env-overridable
  // for self-hosted / white-labeled deploys). No Retry-After header — plan-limit
  // is a hard block, owner-action-required (upgrade or wait for period reset).
  if (outcome.status === "denied" && outcome.planLimitDenied) {
    emitDenialMetric(true, "platform");
    const count = outcome.planLimitCount ?? 0;
    const blockAt = outcome.planLimitBlockAt ?? 0;
    const tier = outcome.planLimitTier ?? "free";
    const upgradeUrl = getPricingUrl(env);
    const selfHostUrl = getSelfHostUrl(env);
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildPlanLimitExceededPayload({
        count, blockAt, tier, upgradeUrl, selfHostUrl, model: requestModel, provider,
      }, ctx.auth.apiVersion),
    );
    return new Response(
      JSON.stringify({
        error: {
          code: "plan_limit_exceeded",
          message: `Plan limit reached: ${count} of ${blockAt} governed requests on ${tier} plan. Upgrade or wait for period reset.`,
          upgrade_url: upgradeUrl,
          self_host_url: selfHostUrl,
          details: { current_count: count, block_at: blockAt, tier },
          recovery: buildRecovery("plan_limit_exceeded"),
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Trace-Id": ctx.traceId,
          "X-NullSpend-Request-Id": ctx.nullspendRequestId,
          "X-NullSpend-Denied": "1",
          ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
          ...budgetHeaders,
        },
      },
    );
  }

  // NF-2: Invalid cost estimate — client/caller produced NaN / Infinity /
  // negative microdollars (see budget-orchestrator.ts validation-denial
  // branch). Fail closed with 422 Unprocessable Entity so the request
  // cannot bypass budget enforcement.
  //
  // Status code choice — 422, not 400: the body is well-formed JSON but
  // `max_tokens` / `max_completion_tokens` is semantically invalid. 400 is
  // reserved for "server can't parse the request" (malformed JSON, missing
  // required headers). 422 distinguishes "parsed fine, body is semantically
  // wrong" from 429 enforcement denials. Clients can branch on status alone
  // without inspecting the code field.
  //
  // With estimator sanitization (P0-4) this path should never fire on real
  // traffic; the metric `budget_check_invalid_estimate` emitted upstream in
  // the orchestrator makes any future regression visible.
  if (outcome.status === "denied" && outcome.invalidEstimate) {
    emitMetric("budget_denied", {
      reason: "invalid_estimate",
      provider,
      entityType: "n/a",
      upgradeUrlEmitted: false,
      upgradeUrlSource: "none",
    });
    return new Response(
      JSON.stringify({
        error: {
          code: "invalid_estimate",
          message: "Cost estimate could not be computed for this request. The request body may contain a semantically invalid max_tokens or max_completion_tokens value (e.g. NaN, Infinity, negative number).",
          details: null,
        },
      }),
      {
        status: 422,
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Trace-Id": ctx.traceId,
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

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
    const velRetryAfter = outcome.retryAfterSeconds ?? 60;
    const velDetails = outcome.velocityDetails;
    const velLimit = velDetails?.limitMicrodollars ?? 0;
    const velCurrent = velDetails?.currentMicrodollars ?? 0;
    const velWindow = velDetails?.windowSeconds ?? 60;
    const velMessage = velLimit > 0
      ? `Rate limit exceeded: ${fmtDollars(velCurrent)} spent in ${velWindow}s window (limit: ${fmtDollars(velLimit)}). Retry after ${velRetryAfter}s.`
      : "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.";
    return new Response(
      JSON.stringify({
        error: {
          code: "velocity_exceeded",
          message: velMessage,
          details: velDetails ?? null,
          recovery: buildRecovery("velocity_exceeded", velRetryAfter),
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(velRetryAfter),
          "X-NullSpend-Trace-Id": ctx.traceId,
          "X-NullSpend-Denied": "1",
          ...budgetHeaders,
        },
      },
    );
  }

  // Loop detection denial
  if (outcome.status === "denied" && outcome.loopDetected && outcome.loopDetails) {
    emitDenialMetric(false, "none");
    dispatchDenialWebhook(ctx, env, logPrefix, () =>
      buildLoopDetectedPayload({
        detectionType: outcome.loopDetails!.type,
        model: outcome.loopDetails!.model,
        provider: outcome.loopDetails!.provider,
        callCount: outcome.loopDetails!.callCount,
        windowSeconds: outcome.loopDetails!.windowSeconds,
        maxCalls: outcome.loopDetails!.maxCalls,
      }, ctx.auth.apiVersion),
    );

    const isAggregate = outcome.loopDetails.type === "aggregate";
    const message = isAggregate
      ? `Loop detected: ${outcome.loopDetails.callCount} distinct model patterns showed repeated calls in ${outcome.loopDetails.windowSeconds}s. This usually indicates a multi-model agent stuck in a loop. Adjust at https://nullspend.dev/app/budgets or set loop_max_calls=0 to disable.`
      : `Loop detected: ${outcome.loopDetails.model} called ${outcome.loopDetails.callCount} times with identical content in ${outcome.loopDetails.windowSeconds}s. Check for retry loops or stuck agent logic. Adjust at https://nullspend.dev/app/budgets or set loop_max_calls=0 to disable.`;

    return new Response(
      JSON.stringify({
        error: {
          code: "loop_detected",
          message,
          details: {
            type: outcome.loopDetails.type,
            model: isAggregate ? "aggregate" : outcome.loopDetails.model,
            provider: isAggregate ? "multiple" : outcome.loopDetails.provider,
            callCount: outcome.loopDetails.callCount,
            windowSeconds: outcome.loopDetails.windowSeconds,
            maxCalls: outcome.loopDetails.maxCalls,
          },
          recovery: buildRecovery("loop_detected", LOOP_RETRY_AFTER_SECONDS),
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(LOOP_RETRY_AFTER_SECONDS),
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
    const sessSpend = outcome.sessionSpend ?? 0;
    const sessLimit = outcome.sessionLimit ?? 0;
    const sessMessage = sessLimit > 0
      ? `Session limit reached: ${fmtDollars(sessSpend)} of ${fmtDollars(sessLimit)}. Start a new session.`
      : "Request blocked: session spend exceeds session limit. Start a new session.";
    return new Response(
      JSON.stringify({
        error: {
          code: "session_limit_exceeded",
          message: sessMessage,
          details: {
            session_id: outcome.sessionId ?? null,
            session_spend_microdollars: sessSpend,
            session_limit_microdollars: sessLimit,
          },
          recovery: buildRecovery("session_limit_exceeded"),
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
    const tagLimit = outcome.maxBudget ?? 0;
    const tagSpend = (outcome.spend ?? 0) + (outcome.reserved ?? 0);
    const tagRemaining = Math.max(0, tagLimit - tagSpend);
    const tagMessage = tagLimit > 0
      ? `Tag budget exceeded for ${outcome.tagKey ?? "unknown"}=${outcome.tagValue ?? "unknown"}: ${fmtDollars(tagRemaining)} of ${fmtDollars(tagLimit)} remaining.`
      : "Request blocked: estimated cost exceeds tag budget limit.";
    return new Response(
      JSON.stringify({
        error: {
          code: "tag_budget_exceeded",
          message: tagMessage,
          details: {
            tag_key: outcome.tagKey ?? null,
            tag_value: outcome.tagValue ?? null,
            budget_limit_microdollars: tagLimit,
            budget_spend_microdollars: tagSpend,
          },
          recovery: buildRecovery("tag_budget_exceeded"),
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

    const custLimit = outcome.maxBudget ?? 0;
    const custSpend = (outcome.spend ?? 0) + (outcome.reserved ?? 0);
    const custRemaining = Math.max(0, custLimit - custSpend);
    const custMessage = custLimit > 0
      ? `Customer budget exceeded for ${outcome.deniedEntityId ?? "unknown"}: ${fmtDollars(custRemaining)} of ${fmtDollars(custLimit)} remaining.`
      : "Request blocked: estimated cost exceeds customer budget limit.";
    return new Response(
      JSON.stringify({
        error: {
          code: "customer_budget_exceeded",
          message: custMessage,
          ...(upgradeUrl ? { upgrade_url: upgradeUrl } : {}),
          details: {
            customer_id: outcome.deniedEntityId ?? null,
            budget_limit_microdollars: custLimit,
            budget_spend_microdollars: custSpend,
            // CX-9: Include finalization reserve details on customer denials (parity with budget_exceeded)
            ...(outcome.finalizationReserve ? {
              finalization_reserve_microdollars: outcome.finalizationReserve,
              finalization_remaining_microdollars: Math.max(0, custLimit - custSpend - (outcome.finalizationReserve ?? 0)),
            } : {}),
          },
          recovery: buildRecovery("customer_budget_exceeded"),
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

    const budgetRemaining = Math.max(0, budgetLimit - budgetSpend);
    const budgetMessage = budgetLimit > 0
      ? `Budget exceeded: ${fmtDollars(budgetRemaining)} of ${fmtDollars(budgetLimit)} remaining. Request a budget increase or switch to a cheaper model.`
      : "Request blocked: estimated cost exceeds remaining budget.";
    return new Response(
      JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: budgetMessage,
          ...(genericUpgradeUrl ? { upgrade_url: genericUpgradeUrl } : {}),
          details: {
            entity_type: entityType,
            entity_id: entityId,
            budget_limit_microdollars: budgetLimit,
            budget_spend_microdollars: budgetSpend,
            estimated_cost_microdollars: estimate,
            ...(outcome.finalizationReserve ? {
              finalization_reserve_microdollars: outcome.finalizationReserve,
              finalization_remaining_microdollars: Math.max(0, budgetLimit - budgetSpend - (outcome.finalizationReserve ?? 0)),
            } : {}),
          },
          recovery: buildRecovery("budget_exceeded"),
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
  preComputedCrossings?: import("../durable-objects/user-budget.js").ThresholdCrossing[],
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

      // Use DO-deduped threshold crossings from reconcile. When undefined
      // (reconcile failed/skipped) or empty, don't fire — the DO is the
      // source of truth; stale local detection would emit duplicate or
      // false-positive webhooks when the spend never landed.
      if (preComputedCrossings && preComputedCrossings.length > 0) {
        const epVersion = endpoints[0]?.apiVersion ?? CURRENT_API_VERSION;
        for (const c of preComputedCrossings) {
          const event = buildThresholdPayload({
            budgetEntityType: c.entityType,
            budgetEntityId: c.entityId,
            budgetLimitMicrodollars: c.maxBudget,
            budgetSpendMicrodollars: c.spend,
            thresholdPercent: c.threshold,
            triggeredByRequestId: c.requestId,
            isCritical: c.isCritical,
          }, epVersion);
          await dispatchToEndpoints(ctx.webhookDispatcher!, endpoints, event);
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
