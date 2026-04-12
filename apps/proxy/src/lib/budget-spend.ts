import { getSql } from "./db.js";
import { emitMetric } from "./metrics.js";

/**
 * Idempotent spend increment on each budget entity in Postgres.
 *
 * PXY-2: Uses a `reconciled_requests` dedup table to prevent double-counting
 * on retry (queue retry, outbox alarm retry, manual recovery). Each
 * (requestId, entityType, entityId) tuple is written exactly once.
 *
 * Entities are sorted by (entityType, entityId) before the transaction
 * to prevent deadlocks when concurrent reconciliations overlap.
 */
export async function updateBudgetSpend(
  connectionString: string,
  orgId: string,
  requestId: string,
  entities: { entityType: string; entityId: string }[],
  actualCostMicrodollars: number,
  skipDbWrites = false,
): Promise<void> {
  if (actualCostMicrodollars <= 0 || entities.length === 0) return;

  if (skipDbWrites) {
    console.log("[budget-spend] Local dev — spend update (not persisted):", {
      entities,
      actualCostMicrodollars,
    });
    return;
  }

  const sql = getSql(connectionString);

  // Sort entities by (entityType, entityId) to prevent deadlocks
  // when concurrent reconciliations overlap on the same entities.
  const sorted = [...entities].sort((a, b) =>
    a.entityType.localeCompare(b.entityType) || a.entityId.localeCompare(b.entityId),
  );

  await sql.begin(async (tx) => {
    for (const entity of sorted) {
      // 1. Idempotent dedup: INSERT only if this (requestId, entity) hasn't been reconciled
      const inserted = await tx`
        INSERT INTO reconciled_requests (request_id, entity_type, entity_id, org_id, cost_microdollars)
        VALUES (${requestId}, ${entity.entityType}, ${entity.entityId}, ${orgId}, ${actualCostMicrodollars})
        ON CONFLICT (request_id, entity_type, entity_id) DO NOTHING
      `;

      // 2. Only increment spend if this is a new reconciliation (not a duplicate)
      if (inserted.count > 0) {
        const updated = await tx`
          UPDATE budgets
          SET spend_microdollars = spend_microdollars + ${actualCostMicrodollars},
              updated_at = NOW()
          WHERE entity_type = ${entity.entityType}
            AND entity_id = ${entity.entityId}
            AND org_id = ${orgId}
        `;
        // C2: Detect missing budget row (dedup succeeded but nothing to update).
        // Codex #2 fix: Delete the dedup record so the alarm can retry later
        // when the budget row may exist (race between budget creation and first request).
        if (updated.count === 0) {
          await tx`
            DELETE FROM reconciled_requests
            WHERE request_id = ${requestId} AND entity_type = ${entity.entityType} AND entity_id = ${entity.entityId}
          `;
          console.error("[budget-spend] Budget row missing — dedup record removed for retry", {
            requestId, entityType: entity.entityType, entityId: entity.entityId, orgId,
          });
          emitMetric("reconcile_budget_row_missing", {
            entityType: entity.entityType, entityId: entity.entityId,
          });
        }
      } else {
        // C10: Dedup hit — check for cost mismatch (corruption signal)
        const existing = await tx`
          SELECT cost_microdollars FROM reconciled_requests
          WHERE request_id = ${requestId} AND entity_type = ${entity.entityType} AND entity_id = ${entity.entityId}
        `;
        if (existing[0] && Number(existing[0].cost_microdollars) !== actualCostMicrodollars) {
          console.error("[budget-spend] DEDUP COST MISMATCH", {
            requestId, entityType: entity.entityType,
            stored: existing[0].cost_microdollars, attempted: actualCostMicrodollars,
          });
          emitMetric("reconcile_dedup_cost_mismatch", { requestId });
        }
      }
    }
  });
}

/**
 * Reset budget period in Postgres: set spend=0 and update currentPeriodStart.
 * Called when the DO detects an expired budget period.
 * Runs inside `waitUntil` — logs errors but never throws.
 */
export async function resetBudgetPeriod(
  connectionString: string,
  orgId: string,
  resets: Array<{ entityType: string; entityId: string; newPeriodStart: number }>,
  skipDbWrites = false,
): Promise<void> {
  if (resets.length === 0) return;

  if (skipDbWrites) {
    console.log("[budget-spend] Local dev — period reset (not persisted):", { resets });
    return;
  }

  try {
    const sql = getSql(connectionString);

    // Sort entities by (entityType, entityId) to prevent deadlocks
    const sorted = [...resets].sort((a, b) =>
      a.entityType.localeCompare(b.entityType) || a.entityId.localeCompare(b.entityId),
    );

    await sql.begin(async (tx) => {
      for (const reset of sorted) {
        await tx`
          UPDATE budgets
          SET spend_microdollars = 0,
              current_period_start = ${new Date(reset.newPeriodStart).toISOString()},
              updated_at = NOW()
          WHERE entity_type = ${reset.entityType}
            AND entity_id = ${reset.entityId}
            AND org_id = ${orgId}
        `;
      }
    });
  } catch (err) {
    console.error(
      "[budget-spend] Failed to reset period:",
      err instanceof Error ? err.message : "Unknown error",
    );
  }
}
