import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "./context.js";
import type { BudgetEntity } from "./budget-do-lookup.js";
import type { TierLabel } from "./api-key-auth.js";
import { doBudgetCheck, doBudgetReconcile, doIncrementPlanCounter } from "./budget-do-client.js";
import type { ThresholdCrossing } from "../durable-objects/user-budget.js";
import { resetBudgetPeriod } from "./budget-spend.js";
import { resolvePeriodBounds } from "./period-math.js";
import { emitMetric } from "./metrics.js";

export interface ReconcileOutcome {
  thresholdCrossings?: ThresholdCrossing[];
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
  deniedEntityType?: string;
  deniedEntityId?: string;
  remaining?: number;
  maxBudget?: number;
  spend?: number;
  reserved?: number;
  velocityDenied?: boolean;
  retryAfterSeconds?: number;
  velocityDetails?: {
    limitMicrodollars: number;
    windowSeconds: number;
    currentMicrodollars: number;
  };
  velocityRecovered?: Array<{
    entityType: string;
    entityId: string;
    velocityLimitMicrodollars: number;
    velocityWindowSeconds: number;
    velocityCooldownSeconds: number;
  }>;
  sessionLimitDenied?: boolean;
  sessionId?: string;
  sessionSpend?: number;
  sessionLimit?: number;
  tagBudgetDenied?: boolean;
  tagKey?: string;
  tagValue?: string;
  finalizationReserve?: number;
  /** Loop call count for warning header (set when count >= 80% of threshold) */
  loopCount?: number;
  loopMaxCalls?: number;
  loopDetected?: boolean;
  loopDetails?: {
    type: "per_key" | "aggregate";
    model: string;
    provider: string;
    callCount: number;
    windowSeconds: number;
    maxCalls: number;
  };
}

// ---------------------------------------------------------------------------
// checkBudget
// ---------------------------------------------------------------------------

export async function checkBudget(
  env: Env,
  ctx: RequestContext,
  estimateMicrodollars: number,
  finalize: boolean = false,
  loopContext: { provider: string; model: string; contentHash: string } | null = null,
): Promise<BudgetCheckOutcome> {
  // PR-2c build-audit F6: fail-loud if `ctx.nullspendRequestId` is missing.
  // index.fetch populates this before every RequestContext construction; any
  // test fixture or internal caller that forgets loses idempotency dedup
  // silently. Runtime check catches the regression at test time rather than
  // letting it slip into production as a silent double-count on retry.
  if (typeof ctx.nullspendRequestId !== "string" || ctx.nullspendRequestId.length === 0) {
    // edge-case-audit E2: metric has NO tags (cardinality-safe) — ownerId lives
    // in the structured log so forensic queries can find the offending source.
    emitMetric("nullspend_request_id_missing_on_ctx", {});
    console.error(
      "[budget-orchestrator] ctx.nullspendRequestId is missing or empty — idempotency dedup disabled for this request. Check index.fetch ingress setup.",
      { ownerId: ctx.ownerId, orgId: ctx.auth?.orgId ?? "unknown" },
    );
  }

  // PR-2c codex-round-1 C1: invalid-estimate guard at TOP of checkBudget, BEFORE
  // any plan-counter increment. Malformed requests (NaN, Infinity, negative)
  // must NOT burn governed-request quota. Existing checkBudgetDO guard stays as
  // defense-in-depth. Returns invalidEstimate outcome shape expected by the
  // existing 422 path in shared.ts handleBudgetDenials.
  if (!Number.isFinite(estimateMicrodollars) || estimateMicrodollars < 0) {
    emitMetric("budget_check_invalid_estimate", {
      ownerId: ctx.ownerId,
      orgId: ctx.auth.orgId ?? "unknown",
    });
    return {
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      invalidEstimate: true,
    };
  }

  // PR-2c: plan-limit check fires BEFORE the DO's velocity/loop/session/budget
  // chain (Decision #28 priority). Runs EVEN when hasBudgets=false — plan-limit
  // is NullSpend-tier gating, independent of org-configured budgets.
  //
  // Skipped when planLimitBlockAt is null (Enterprise + self-hosted — these tiers
  // never enforce).
  //
  // INTENTIONAL (codex-round-1 M7): plan-limit-denied requests do NOT update
  // loop_call_log or velocity windows downstream. A denied request has no LLM
  // spend, no reservation — loop state on never-forwarded requests would produce
  // misleading analytics on users who are already blocked. Do not extend loop
  // tracking to cover plan-limit-denied requests.
  // Use `!= null` (loose equality) so both `null` and `undefined` skip the check.
  // `undefined` can occur in test fixtures that pre-date PR-2a and don't populate
  // the field — treating them as "enforcement skipped" is the safe default.
  if (ctx.auth.planLimitBlockAt != null) {
    const { periodStart, periodEnd } = resolvePeriodBounds(ctx.auth, Date.now());
    const planResult = await doIncrementPlanCounter(env, {
      orgId: ctx.auth.orgId,
      planLimitBlockAt: ctx.auth.planLimitBlockAt,
      planLimitMode: ctx.auth.planLimitMode,
      tier: ctx.auth.tierLabel,
      periodStart,
      periodEnd,
      // PR-2c codex-round-1 C2: threading nullspendRequestId as idempotencyKey
      // so client retries (500/502/503 → retry) don't double-count governed
      // requests. DO's v11 plan_counter_idempotency persistence returns the
      // stored decision verbatim on replay (codex-round-3 C1).
      idempotencyKey: ctx.nullspendRequestId,
    });
    if (planResult.status === "denied") {
      return {
        status: "denied",
        reservationId: null,
        budgetEntities: [],
        planLimitDenied: true,
        planLimitCount: planResult.count,
        planLimitBlockAt: planResult.blockAt,
        planLimitTier: ctx.auth.tierLabel,
      };
    }
    // status === "approved" or "skipped" → fall through to budget check.
  }

  if (!ctx.auth.hasBudgets) {
    // P1 note: hasBudgets is cached for up to 120s (POSITIVE_TTL_MS) in the auth
    // cache. If an org adds its first budget and invalidation misses, requests
    // skip enforcement during the cache window. This metric makes the gap
    // observable — operators can correlate with budget creation timestamps.
    emitMetric("budget_check_skipped", { ownerId: ctx.ownerId, orgId: ctx.auth.orgId ?? "unknown" });
    return { status: "skipped", reservationId: null, budgetEntities: [] };
  }
  return checkBudgetDO(env, ctx.connectionString, ctx.auth.keyId, ctx.ownerId, ctx.auth.orgId, estimateMicrodollars, ctx.sessionId, ctx.tags, finalize, loopContext);
}

// ---------------------------------------------------------------------------
// reconcileBudget
// ---------------------------------------------------------------------------

export async function reconcileBudget(
  env: Env,
  ownerId: string | null,
  orgId: string | null,
  reservationId: string | null,
  actualCost: number,
  budgetEntities: BudgetEntity[],
  connectionString: string,
  options?: { throwOnError?: boolean },
): Promise<ReconcileOutcome> {
  try {
    if (reservationId && ownerId && orgId) {
      const entities = budgetEntities.map((e) => ({
        entityType: e.entityType,
        entityId: e.entityId,
      }));
      const result = await doBudgetReconcile(env, ownerId, orgId, reservationId, actualCost, entities);
      if (result.status !== "ok") {
        // P0 fix: make DO reconcile failures observable. Previously, non-ok
        // results were silently swallowed unless throwOnError was set (it never
        // was). The reservation hangs until alarm expiry, then gets reversed,
        // erasing the real spend. Now we always emit a metric so operators
        // can detect and investigate reconcile failures.
        emitMetric("reconcile_do_failure", {
          status: result.status,
          reservationId,
          actualCost,
          ownerId,
        });
        console.error(`[budget-orchestrator] DO reconcile returned non-ok: status=${result.status} reservation=${reservationId} cost=${actualCost}`);
        if (options?.throwOnError) {
          throw new Error(`Reconciliation failed with status: ${result.status}`);
        }
      }
      return { thresholdCrossings: result.thresholdCrossings };
    }
  } catch (err) {
    console.error("[budget-orchestrator] Reconciliation failed:", err);
    emitMetric("reconcile_exception", {
      reservationId: reservationId ?? "none",
      actualCost,
    });
    if (options?.throwOnError) throw err;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Internal: DO-first path — single RPC, no Postgres on hot path
// ---------------------------------------------------------------------------

async function checkBudgetDO(
  env: Env,
  connectionString: string,
  keyId: string | null,
  ownerId: string | null,
  orgId: string | null,
  estimateMicrodollars: number,
  sessionId: string | null = null,
  tags: Record<string, string>,
  finalize: boolean = false,
  loopContext: { provider: string; model: string; contentHash: string } | null = null,
): Promise<BudgetCheckOutcome> {
  if (!ownerId) {
    return { status: "skipped", reservationId: null, budgetEntities: [] };
  }

  // Convert tags to entity IDs for DO lookup
  const tagEntityIds = Object.entries(tags).map(([k, v]) => `${k}=${v}`);

  // Single DO RPC — no Postgres lookup, no cache
  const checkResult = await doBudgetCheck(env, ownerId, keyId, estimateMicrodollars, sessionId, tagEntityIds, orgId, finalize, loopContext);

  // NF-2: The DO returns `{ status: "denied", hasBudgets: false }` ONLY when
  // the incoming estimate is invalid (NaN / Infinity / negative).
  //
  // PR-2c codex-round-1 C1 moved the primary invalid-estimate guard to the TOP
  // of checkBudget above, so by the time we reach checkBudgetDO the estimate
  // is guaranteed finite + non-negative. This branch is therefore UNREACHABLE
  // via the normal checkBudget -> checkBudgetDO path but stays as defense-in-
  // depth for any future caller that bypasses the outer guard. Uses a distinct
  // metric name (`budget_check_invalid_estimate_defense`) so metric tag
  // schemas don't collide with the upstream emission (F3 from build-audit).
  if (checkResult.status === "denied" && !checkResult.hasBudgets) {
    emitMetric("budget_check_invalid_estimate_defense", {
      ownerId,
      estimateIsFinite: Number.isFinite(estimateMicrodollars),
      estimateIsNegative: estimateMicrodollars < 0,
    });
    return {
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      invalidEstimate: true,
    };
  }

  if (!checkResult.hasBudgets) {
    // Auth cache said hasBudgets=true (we got here), but DO says false — stale cache
    emitMetric("budget_cache_stale", { ownerId });
    return { status: "skipped", reservationId: null, budgetEntities: [] };
  }

  // Write back period resets to Postgres (registered with waitUntil to survive Worker lifecycle)
  if (checkResult.periodResets?.length && connectionString) {
    waitUntil(
      resetBudgetPeriod(connectionString, orgId!, checkResult.periodResets).catch((err) => {
        console.error("[budget-orchestrator] Period reset write-back failed:", err);
      }),
    );
  }

  // Build budgetEntities from DO response.
  // `reserved` is now the live value from CheckedEntity — exposes concurrent
  // in-flight reservations so downstream header computation reports accurate
  // remaining even under parallel load. Prior to 2026-04-08 this was hardcoded
  // to 0 because no caller needed it; the budget response headers feature
  // requires it.
  const budgetEntities: BudgetEntity[] = (checkResult.checkedEntities ?? []).map((e) => ({
    entityKey: `{budget}:${e.entityType}:${e.entityId}`,
    entityType: e.entityType,
    entityId: e.entityId,
    maxBudget: e.maxBudget,
    spend: e.spend,
    reserved: e.reserved,
    policy: e.policy,
    thresholdPercentages: e.thresholdPercentages ?? [50, 80, 90, 95],
    finalizationReserve: e.finalizationReserve ?? 0,
    avgRecentCost: e.avgRecentCost ?? 0,
  }));

  if (checkResult.status === "denied") {
    // Parse deniedEntity "type:id"
    let deniedEntityType: string | undefined;
    let deniedEntityId: string | undefined;
    if (checkResult.deniedEntity) {
      const sep = checkResult.deniedEntity.indexOf(":");
      deniedEntityType = checkResult.deniedEntity.slice(0, sep);
      deniedEntityId = checkResult.deniedEntity.slice(sep + 1);
    }

    // Velocity denial — separate from budget exhaustion
    if (checkResult.velocityDenied) {
      return {
        status: "denied",
        reservationId: null,
        budgetEntities,
        deniedEntityType,
        deniedEntityId,
        velocityDenied: true,
        retryAfterSeconds: checkResult.retryAfterSeconds,
        velocityDetails: checkResult.velocityDetails,
      };
    }

    // Loop detection denial — separate from velocity and budget exhaustion
    if (checkResult.loopDetected) {
      return {
        status: "denied",
        reservationId: null,
        budgetEntities,
        loopDetected: true,
        loopDetails: checkResult.loopDetails,
      };
    }

    // Session limit denial — separate from both velocity and budget exhaustion
    if (checkResult.sessionLimitDenied) {
      return {
        status: "denied",
        reservationId: null,
        budgetEntities,
        deniedEntityType,
        deniedEntityId,
        sessionLimitDenied: true,
        sessionId: checkResult.sessionId,
        sessionSpend: checkResult.sessionSpend,
        sessionLimit: checkResult.sessionLimit,
      };
    }

    // Tag budget denial — separate from generic budget_exceeded
    if (deniedEntityType === "tag" && deniedEntityId) {
      const eqIdx = deniedEntityId.indexOf("=");
      return {
        status: "denied",
        reservationId: null,
        budgetEntities,
        deniedEntityType,
        deniedEntityId,
        tagBudgetDenied: true,
        tagKey: eqIdx > 0 ? deniedEntityId.slice(0, eqIdx) : deniedEntityId,
        tagValue: eqIdx > 0 ? deniedEntityId.slice(eqIdx + 1) : "",
        remaining: checkResult.remaining,
        maxBudget: checkResult.maxBudget,
        spend: checkResult.spend,
        reserved: Math.max(0, (checkResult.maxBudget ?? 0) - (checkResult.spend ?? 0) - (checkResult.remaining ?? 0)),
      };
    }

    const reserved = Math.max(0, (checkResult.maxBudget ?? 0) - (checkResult.spend ?? 0) - (checkResult.remaining ?? 0));
    return {
      status: "denied",
      reservationId: null,
      budgetEntities,
      deniedEntityType,
      deniedEntityId,
      remaining: checkResult.remaining,
      maxBudget: checkResult.maxBudget,
      spend: checkResult.spend,
      reserved,
      finalizationReserve: checkResult.finalizationReserve,
    };
  }

  return {
    status: "approved",
    reservationId: checkResult.reservationId ?? null,
    budgetEntities,
    ...(checkResult.velocityRecovered?.length && { velocityRecovered: checkResult.velocityRecovered }),
    ...(checkResult.loopCount != null && { loopCount: checkResult.loopCount, loopMaxCalls: checkResult.loopMaxCalls }),
  };
}
