import { getSql } from "./db.js";

/**
 * Atomically increment `spend_microdollars` on each budget entity in Postgres.
 * Throws on failure so callers (reconcileReservation) can retry.
 *
 * Entities are sorted by (entityType, entityId) before the transaction
 * to prevent deadlocks when concurrent reconciliations overlap.
 *
 * Ensures Postgres spend stays current so DO sync starts from an accurate
 * baseline.
 */
export async function updateBudgetSpend(
  connectionString: string,
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
      await tx`
        UPDATE budgets
        SET spend_microdollars = spend_microdollars + ${actualCostMicrodollars},
            updated_at = NOW()
        WHERE entity_type = ${entity.entityType}
          AND entity_id = ${entity.entityId}
      `;
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
