import { getSql } from "./db.js";
import { emitMetric } from "./metrics.js";

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

  // Note: do NOT early-return when entities is empty — tag/customer lookups
  // below may still find budgets scoped only by org_id (e.g., /internal sync
  // for a customer or tag budget arrives with no keyId/userId).

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

    // Tag budget lookup: one query per tag entity ID. Done sequentially rather
    // than via ANY(...) because postgres.js + Hyperdrive don't reliably bind
    // a JS string[] as a Postgres text[] (values with "=" and "-" in tag
    // strings get rejected as "malformed array literal"). The per-tag loop
    // is fine: entity counts are tiny (1-3 tags per request).
    const tags = identity.tags;
    if (Object.keys(tags).length > 0 && orgId) {
      const tagEntityIds = Object.entries(tags).map(([k, v]) => `${k}=${v}`);
      for (const tagEntityId of tagEntityIds) {
        const tagRows = await sql<BudgetDbRow[]>`
          SELECT entity_type, entity_id, max_budget_microdollars, spend_microdollars,
                 policy, reset_interval, current_period_start,
                 velocity_limit_microdollars, velocity_window_seconds, velocity_cooldown_seconds,
                 threshold_percentages, session_limit_microdollars
          FROM budgets
          WHERE org_id = ${orgId}
            AND entity_type = 'tag'
            AND entity_id = ${tagEntityId}
        `;
        for (const row of tagRows) {
          result.push(mapRow(row, "tag", row.entity_id));
        }
      }
    }

    // Customer budget lookup: query by org_id + entity_type='customer' + entity_id
    // Uses the auto-injected customer tag (from X-NullSpend-Customer header or tags["customer"]).
    const customerId = tags?.["customer"];
    if (customerId && orgId) {
      const customerRows = await sql<BudgetDbRow[]>`
        SELECT entity_type, entity_id, max_budget_microdollars, spend_microdollars,
               policy, reset_interval, current_period_start,
               velocity_limit_microdollars, velocity_window_seconds, velocity_cooldown_seconds,
               threshold_percentages, session_limit_microdollars
        FROM budgets
        WHERE org_id = ${orgId}
          AND entity_type = 'customer'
          AND entity_id = ${customerId}
      `;

      for (const row of customerRows) {
        result.push(mapRow(row, "customer", row.entity_id));
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

/**
 * Look up the per-customer upgrade URL from `customer_settings.upgrade_url`.
 *
 * Called ONLY from the denial branch of `handleBudgetDenials` when the
 * denying entity is a customer. The hot path (happy 200) never touches
 * this query — it's cold-path cost on an already-slow denial response.
 *
 * `customer_settings` is decoupled from `customer_mappings` (which is
 * Stripe-revenue-sync-specific) so orgs using per-customer budgets
 * WITHOUT Stripe integration can still configure per-customer overrides.
 *
 * Returns null when:
 *   - No row matches (customer has no settings row, or row has null upgrade_url)
 *   - The query fails for any reason (fail-open — the denial still ships,
 *     just without the upgrade_url field). Emits `customer_upgrade_url_lookup_failed`
 *     metric on failure so systematic issues show up in dashboards.
 *
 * Uses the shared `getSql` pool so this reuses the existing per-request
 * postgres.js instance.
 */
export async function lookupCustomerUpgradeUrl(
  connectionString: string,
  orgId: string,
  customerId: string,
): Promise<string | null> {
  try {
    const sql = getSql(connectionString);
    const rows = await sql<{ upgrade_url: string | null }[]>`
      SELECT upgrade_url
      FROM customer_settings
      WHERE org_id = ${orgId}
        AND customer_id = ${customerId}
      LIMIT 1
    `;
    if (rows.length === 0) return null;
    const url = rows[0].upgrade_url;
    return typeof url === "string" && url.length > 0 ? url : null;
  } catch (err) {
    console.warn(
      "[budget-do-lookup] customer upgrade_url lookup failed (fail-open):",
      err instanceof Error ? err.message : "Unknown error",
    );
    emitMetric("customer_upgrade_url_lookup_failed", {
      orgId,
      error: err instanceof Error ? err.message : "unknown",
    });
    return null;
  }
}
