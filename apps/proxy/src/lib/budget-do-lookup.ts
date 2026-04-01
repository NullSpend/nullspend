import { getSql } from "./db.js";

export interface BudgetEntity {
  entityKey: string;
  entityType: string;
  entityId: string;
  maxBudget: number;
  spend: number;
  reserved: number;
  policy: string;
  thresholdPercentages?: number[];
}

export interface DOBudgetEntity {
  entityType: string;
  entityId: string;
  maxBudget: number;
  spend: number;
  policy: string;
  resetInterval: string | null;
  periodStart: number; // epoch ms
  velocityLimit: number | null;
  velocityWindow: number; // ms (default 60000)
  velocityCooldown: number; // ms (default 60000)
  thresholdPercentages: number[];
  sessionLimit: number | null;
}

interface BudgetDbRow {
  entity_type: string;
  entity_id: string;
  max_budget_microdollars: number;
  spend_microdollars: number;
  policy: string;
  reset_interval: string | null;
  current_period_start: string | null;
  velocity_limit_microdollars: number | null;
  velocity_window_seconds: number | null;
  velocity_cooldown_seconds: number | null;
  threshold_percentages: number[] | null;
  session_limit_microdollars: number | null;
}

function mapRow(row: BudgetDbRow, entityType: string, entityId: string): DOBudgetEntity {
  return {
    entityType,
    entityId,
    maxBudget: Number(row.max_budget_microdollars),
    spend: Number(row.spend_microdollars),
    policy: row.policy,
    resetInterval: row.reset_interval ?? null,
    periodStart: row.current_period_start ? new Date(row.current_period_start).getTime() : 0,
    velocityLimit: row.velocity_limit_microdollars != null ? Number(row.velocity_limit_microdollars) : null,
    velocityWindow: (row.velocity_window_seconds ?? 60) * 1000,
    velocityCooldown: (row.velocity_cooldown_seconds ?? 60) * 1000,
    thresholdPercentages: row.threshold_percentages ?? [50, 80, 90, 95],
    sessionLimit: row.session_limit_microdollars != null ? Number(row.session_limit_microdollars) : null,
  };
}

/**
 * Query Postgres directly for budget entities with all DO-required fields.
 * Uses raw postgres.js tagged templates for minimal bundle size.
 * Throws on error (caller decides fail mode).
 */
export async function lookupBudgetsForDO(
  connectionString: string,
  identity: { keyId: string | null; orgId: string | null; userId: string | null; tags: Record<string, string> },
): Promise<DOBudgetEntity[]> {
  const { keyId, orgId, userId } = identity;
  const entities: { type: string; id: string }[] = [];

  if (keyId) {
    entities.push({ type: "api_key", id: keyId });
  }
  if (userId) {
    entities.push({ type: "user", id: userId });
  }

  if (entities.length === 0) return [];

  const result: DOBudgetEntity[] = [];

  try {
    const sql = getSql(connectionString);

    for (const entity of entities) {
      const rows = await sql<BudgetDbRow[]>`
        SELECT entity_type, entity_id, max_budget_microdollars, spend_microdollars,
               policy, reset_interval, current_period_start,
               velocity_limit_microdollars, velocity_window_seconds, velocity_cooldown_seconds,
               threshold_percentages, session_limit_microdollars
        FROM budgets
        WHERE org_id = ${orgId} AND entity_type = ${entity.type} AND entity_id = ${entity.id}
      `;

      if (rows.length > 0) {
        result.push(mapRow(rows[0], entity.type, entity.id));
      }
    }

    // Tag budget lookup: query by org_id + entity_type='tag' + entity_id IN (tag keys)
    const tags = identity.tags;
    if (Object.keys(tags).length > 0 && orgId) {
      const tagEntityIds = Object.entries(tags).map(([k, v]) => `${k}=${v}`);
      const tagRows = await sql<BudgetDbRow[]>`
        SELECT entity_type, entity_id, max_budget_microdollars, spend_microdollars,
               policy, reset_interval, current_period_start,
               velocity_limit_microdollars, velocity_window_seconds, velocity_cooldown_seconds,
               threshold_percentages, session_limit_microdollars
        FROM budgets
        WHERE org_id = ${orgId}
          AND entity_type = 'tag'
          AND entity_id = ANY(${tagEntityIds})
      `;

      for (const row of tagRows) {
        result.push(mapRow(row, "tag", row.entity_id));
      }
    }
  } catch (err) {
    console.error(
      "[budget-do-lookup] Postgres lookup failed:",
      err instanceof Error ? err.message : "Unknown error",
    );
    throw err;
  }

  return result;
}
