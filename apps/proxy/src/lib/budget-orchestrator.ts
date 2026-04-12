import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "./context.js";
import type { BudgetEntity } from "./budget-do-lookup.js";
import { doBudgetCheck, doBudgetReconcile } from "./budget-do-client.js";
import { resetBudgetPeriod } from "./budget-spend.js";
import { enqueueReconciliation } from "./reconciliation-queue.js";
import { emitMetric } from "./metrics.js";

interface BudgetCheckOutcome {
  status: "approved" | "denied" | "skipped";
  reservationId: string | null;
  budgetEntities: BudgetEntity[];
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
}

// ---------------------------------------------------------------------------
// checkBudget
// ---------------------------------------------------------------------------

export async function checkBudget(
  env: Env,
  ctx: RequestContext,
  estimateMicrodollars: number,
): Promise<BudgetCheckOutcome> {
  if (!ctx.auth.hasBudgets) {
    emitMetric("budget_check_skipped", { ownerId: ctx.ownerId });
    return { status: "skipped", reservationId: null, budgetEntities: [] };
  }
  return checkBudgetDO(env, ctx.connectionString, ctx.auth.keyId, ctx.ownerId, ctx.auth.orgId, estimateMicrodollars, ctx.sessionId, ctx.tags);
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
): Promise<void> {
  try {
    if (reservationId && ownerId && orgId) {
      const entities = budgetEntities.map((e) => ({
        entityType: e.entityType,
        entityId: e.entityId,
      }));
      const status = await doBudgetReconcile(env, ownerId, orgId, reservationId, actualCost, entities, connectionString);
      if (options?.throwOnError && status !== "ok") {
        throw new Error(`Reconciliation failed with status: ${status}`);
      }
    }
  } catch (err) {
    console.error("[budget-orchestrator] Reconciliation failed:", err);
    if (options?.throwOnError) throw err;
  }
}

/**
 * Extract the optional RECONCILE_QUEUE binding from env.
 * The binding is not in the generated Env type (optional infra).
 */
export function getReconcileQueue(env: Env): Queue | undefined {
  return (env as Record<string, unknown>).RECONCILE_QUEUE as Queue | undefined;
}

/**
 * Queue-aware reconciliation: attempts to enqueue to Cloudflare Queues first,
 * falls back to direct reconciliation if the queue binding is absent or send fails.
 *
 * When the queue is available, reconciliation is decoupled from the request
 * lifecycle (no 30s waitUntil limit). The consumer retries with DLQ.
 */
export async function reconcileBudgetQueued(
  queue: Queue | undefined,
  env: Env,
  ownerId: string | null,
  orgId: string | null,
  reservationId: string | null,
  actualCost: number,
  budgetEntities: BudgetEntity[],
  connectionString: string,
): Promise<void> {
  if (queue && reservationId) {
    try {
      await enqueueReconciliation(queue, {
        type: "reconcile",
        reservationId,
        actualCostMicrodollars: actualCost,
        budgetEntities: budgetEntities.map((e) => ({
          entityKey: e.entityKey,
          entityType: e.entityType,
          entityId: e.entityId,
        })),
        ownerId,
        orgId,
        enqueuedAt: Date.now(),
      });
      return;
    } catch (err) {
      console.error("[budget-orchestrator] Queue send failed, falling back to direct:", err);
    }
  }
  // Fallback: direct reconciliation (current behavior)
  await reconcileBudget(env, ownerId, orgId, reservationId, actualCost, budgetEntities, connectionString);
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
): Promise<BudgetCheckOutcome> {
  if (!ownerId) {
    return { status: "skipped", reservationId: null, budgetEntities: [] };
  }

  // Convert tags to entity IDs for DO lookup
  const tagEntityIds = Object.entries(tags).map(([k, v]) => `${k}=${v}`);

  // Single DO RPC — no Postgres lookup, no cache
  const checkResult = await doBudgetCheck(env, ownerId, keyId, estimateMicrodollars, sessionId, tagEntityIds, orgId);

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
    };
  }

  return {
    status: "approved",
    reservationId: checkResult.reservationId ?? null,
    budgetEntities,
    ...(checkResult.velocityRecovered?.length && { velocityRecovered: checkResult.velocityRecovered }),
  };
}
