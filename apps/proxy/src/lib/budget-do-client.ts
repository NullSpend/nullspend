import type { BudgetRow, CheckResult, VelocityState } from "../durable-objects/user-budget.js";
import type { DOBudgetEntity } from "./budget-do-lookup.js";
import { updateBudgetSpend } from "./budget-spend.js";
import { emitMetric } from "./metrics.js";

const PG_MAX_RETRIES = 2;
const PG_RETRY_DELAYS = [200, 800];

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
): Promise<CheckResult> {
  const startMs = Date.now();
  const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
  const result = await stub.checkAndReserve(keyId, estimateMicrodollars, 30_000, sessionId, tagEntityIds);
  emitMetric("do_budget_check", {
    status: result.status,
    hasBudgets: result.hasBudgets,
    durationMs: Date.now() - startMs,
    velocityDenied: result.velocityDenied ?? false,
    velocityRecovered: (result.velocityRecovered?.length ?? 0) > 0,
    sessionLimitDenied: result.sessionLimitDenied ?? false,
    tagBudgetDenied: result.status === "denied" && (result.deniedEntity?.startsWith("tag:") ?? false),
  });
  return result;
}

/**
 * Reconcile a reservation via the UserBudgetDO + Postgres write-back.
 * Never throws — errors are caught, logged, and metrics emitted.
 *
 * Returns the reconciliation status:
 * - `"ok"`: DO reconcile + Postgres write both succeeded (or actualCost=0, no PG write needed)
 * - `"pg_failed"`: DO reconcile succeeded but Postgres write failed after retries (split-brain)
 * - `"error"`: DO reconcile itself failed
 *
 * Retries the Postgres write up to PG_MAX_RETRIES times with backoff
 * to prevent DO/Postgres split-brain on transient failures.
 */
export async function doBudgetReconcile(
  env: Env,
  ownerId: string,
  reservationId: string,
  actualCost: number,
  entities: Array<{ entityType: string; entityId: string }>,
  connectionString: string,
): Promise<"ok" | "pg_failed" | "error"> {
  const startMs = Date.now();
  let retries = 0;
  let status: "ok" | "pg_failed" | "error" = "ok";

  try {
    const stub = env.USER_BUDGET.get(env.USER_BUDGET.idFromName(ownerId));
    const reconcileResult = await stub.reconcile(reservationId, actualCost);

    if (reconcileResult.status === "not_found") {
      // Reservation already reconciled (expired by alarm or duplicate call).
      // Do NOT write to Postgres — the spend was already counted.
      console.warn("[budget-do-client] Reservation not found in DO (already reconciled?)", {
        reservationId,
        costMicrodollars: actualCost,
      });
      emitMetric("reconcile_not_found", { reservationId, costMicrodollars: actualCost });
      return "ok";
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

    if (actualCost > 0) {
      let pgSuccess = false;

      for (let attempt = 0; attempt <= PG_MAX_RETRIES; attempt++) {
        try {
          await updateBudgetSpend(connectionString, entities, actualCost);
          pgSuccess = true;
          retries = attempt;
          break;
        } catch (err) {
          if (attempt < PG_MAX_RETRIES) {
            await new Promise((r) => setTimeout(r, PG_RETRY_DELAYS[attempt]));
          } else {
            retries = attempt;
            console.error("[budget-do-client] Postgres write failed after retries", {
              reservationId,
              actualCost,
              entities: entities.map((e) => `${e.entityType}:${e.entityId}`),
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }
      }

      if (!pgSuccess) {
        status = "pg_failed";
        console.error("[budget-do-client] DO/Postgres split-brain: reservation", reservationId);
      }
    }
  } catch (err) {
    status = "error";
    console.error("[budget-do-client] Reconciliation failed:", err);
  } finally {
    emitMetric("do_reconciliation", {
      status,
      costMicrodollars: actualCost,
      durationMs: Date.now() - startMs,
      retries,
    });
  }

  return status;
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
      e.thresholdPercentages, e.sessionLimit,
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
        e.thresholdPercentages, e.sessionLimit,
      );
    }
    emitMetric("budget_sync_retry", { ownerId, missingCount: missing.length });
  }
}

