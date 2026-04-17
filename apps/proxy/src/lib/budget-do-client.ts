import type { BudgetRow, CheckResult, VelocityState, ThresholdCrossing } from "../durable-objects/user-budget.js";
import type { DOBudgetEntity } from "./budget-do-lookup.js";
import { emitMetric } from "./metrics.js";

export interface DOReconcileOutcome {
  status: "ok" | "error";
  thresholdCrossings?: ThresholdCrossing[];
}

// PXY-2: Worker-side PG write removed entirely (Strategy E).
// The DO outbox + alarm handler owns Postgres sync.

/**
 * Check budget via the UserBudgetDO.
 * Throws on DO error (fail-closed).
 * Emits `do_budget_check` metric with latency, status, and hasBudgets.
 */
export async function doBudgetCheck(
  env: Env,
  ownerId: string,
  keyId: string | null,
  estimateMicrodollars: number,
  sessionId: string | null,
  tagEntityIds: string[],
  orgId: string | null = null,
  finalize: boolean = false,
  loopContext: { provider: string; model: string; contentHash: string } | null = null,
): Promise<CheckResult> {
  const startMs = Date.now();
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  const result = await stub.checkAndReserve(keyId, estimateMicrodollars, 30_000, sessionId, tagEntityIds, orgId, finalize, loopContext);
  emitMetric("do_budget_check", {
    status: result.status,
    hasBudgets: result.hasBudgets,
    durationMs: Date.now() - startMs,
    velocityDenied: result.velocityDenied ?? false,
    velocityRecovered: (result.velocityRecovered?.length ?? 0) > 0,
    sessionLimitDenied: result.sessionLimitDenied ?? false,
    tagBudgetDenied: result.status === "denied" && (result.deniedEntity?.startsWith("tag:") ?? false),
    loopDetected: result.loopDetected ?? false,
  });
  return result;
}

/**
 * Reconcile a reservation via the UserBudgetDO.
 * Never throws — errors are caught, logged, and metrics emitted.
 *
 * Returns the reconciliation status:
 * - `"ok"`: DO reconcile succeeded
 * - `"error"`: DO reconcile itself failed
 *
 * Strategy E: No optimistic PG write. The DO's PXY-2 outbox
 * (pg_sync_outbox table) + alarm handler owns Postgres sync entirely.
 * The alarm fires in ~1s after reconcile and writes idempotently via
 * the reconciled_requests dedup table. This reduces waitUntil I/O
 * from ~50ms (DO + PG) to ~10ms (DO only).
 *
 * Trade-off: PG spend visibility delays from ~instant to ~1-5s.
 * Acceptable — the DO is the source of truth for budget enforcement.
 */
export async function doBudgetReconcile(
  env: Env,
  ownerId: string,
  orgId: string,
  reservationId: string,
  actualCost: number,
  _entities: Array<{ entityType: string; entityId: string }>,
): Promise<DOReconcileOutcome> {
  const startMs = Date.now();
  let status: "ok" | "error" = "ok";
  let thresholdCrossings: ThresholdCrossing[] | undefined;

  try {
    const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
    const reconcileResult = await stub.reconcile(reservationId, actualCost);

    thresholdCrossings = reconcileResult.thresholdCrossings;

    if (reconcileResult.status === "not_found") {
      // C1: not_found means expired OR already reconciled.
      // If already reconciled: outbox entry exists in DO, alarm handles PG retry.
      // If expired: no spend to write, nothing to do.
      // Either way, the Worker should NOT attempt a PG write here.
      // P0-1: Return [] (not undefined) — no spend was applied, so no thresholds crossed.
      // undefined would trigger stale-data fallback in shared.ts, producing false alerts.
      emitMetric("reconcile_not_found", { reservationId, costMicrodollars: actualCost });
      return { status: "ok", thresholdCrossings: [] };
    }

    if (reconcileResult.budgetsMissing && reconcileResult.budgetsMissing.length > 0) {
      console.warn("[budget-do-client] Reconciled reservation has missing budgets", {
        reservationId,
        costMicrodollars: actualCost,
        budgetsMissing: reconcileResult.budgetsMissing,
      });
      emitMetric("reconcile_budget_missing", {
        reservationId,
        costMicrodollars: actualCost,
        budgetsMissing: reconcileResult.budgetsMissing,
      });
    }

    // Strategy E: PG sync handled entirely by DO outbox + alarm handler.
    // No optimistic PG write here — alarm fires in ~1s.
  } catch (err) {
    status = "error";
    // P0-1: Return [] on error — DO didn't run, no spend applied, no thresholds crossed.
    // undefined would trigger stale-data fallback producing false alerts.
    thresholdCrossings = [];
    console.error("[budget-do-client] Reconciliation failed:", err);
  } finally {
    emitMetric("do_reconciliation", {
      status,
      costMicrodollars: actualCost,
      durationMs: Date.now() - startMs,
    });
  }

  return { status, thresholdCrossings };
}

/**
 * Remove a budget entity from the UserBudgetDO.
 * Throws on DO error (fail-closed).
 */
export async function doBudgetRemove(
  env: Env,
  ownerId: string,
  entityType: string,
  entityId: string,
): Promise<void> {
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  await stub.removeBudget(entityType, entityId);
}

/**
 * Reset spend for a budget entity in the UserBudgetDO.
 * Throws on DO error (fail-closed).
 */
export async function doBudgetResetSpend(
  env: Env,
  ownerId: string,
  entityType: string,
  entityId: string,
): Promise<void> {
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  await stub.resetSpend(entityType, entityId);
}

/**
 * Read velocity state from the UserBudgetDO.
 * Returns all velocity_state rows for the user.
 */
export async function doBudgetGetVelocityState(
  env: Env,
  ownerId: string,
): Promise<VelocityState[]> {
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  return stub.getVelocityState();
}

/**
 * Read budget state from the UserBudgetDO without creating any reservation.
 * Returns all budget rows for the owner.
 */
export async function doBudgetGetState(
  env: Env,
  ownerId: string,
): Promise<BudgetRow[]> {
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  return stub.getBudgetState();
}

/**
 * Upsert individual budget entities into the DO via `populateIfEmpty`.
 * Does NOT purge other entities — safe for single-entity mutations
 * (budget create/update from dashboard POST).
 *
 * After upserting, verifies the DO has the entities by reading back its
 * budget state. If any entities are missing, retries once. This defends
 * against a race window where the sync response returns before the DO
 * has durably committed all entities (observed under concurrent stress).
 */
export async function doBudgetUpsertEntities(
  env: Env,
  ownerId: string,
  entities: DOBudgetEntity[],
): Promise<void> {
  if (entities.length === 0) return;

  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  for (const e of entities) {
    await stub.populateIfEmpty(
      e.entityType, e.entityId, e.maxBudget, e.spend,
      e.policy, e.resetInterval, e.periodStart,
      e.velocityLimit, e.velocityWindow, e.velocityCooldown,
      e.thresholdPercentages, e.sessionLimit, e.finalizationReserve,
      e.loopMaxCalls ?? null, e.loopWindowSeconds ?? null, e.loopAggregateMaxKeys ?? null,
    );
  }

  // Verification: confirm entities are in the DO's SQLite state.
  // getBudgetState() reads directly from SQLite, so this should
  // always reflect committed upserts. Retry is a safety net only.
  const state = await stub.getBudgetState();
  const stateKeys = new Set(state.map((s) => `${s.entity_type}:${s.entity_id}`));
  const missing = entities.filter((e) => !stateKeys.has(`${e.entityType}:${e.entityId}`));

  if (missing.length > 0) {
    console.error(
      `[budget-do-client] UNEXPECTED: ${missing.length}/${entities.length} entities missing from SQLite after upsert, retrying`,
      missing.map((e) => `${e.entityType}:${e.entityId}`),
    );
    for (const e of missing) {
      await stub.populateIfEmpty(
        e.entityType, e.entityId, e.maxBudget, e.spend,
        e.policy, e.resetInterval, e.periodStart,
        e.velocityLimit, e.velocityWindow, e.velocityCooldown,
        e.thresholdPercentages, e.sessionLimit, e.finalizationReserve,
      );
    }
    emitMetric("budget_sync_retry", { ownerId, missingCount: missing.length });
  }
}

