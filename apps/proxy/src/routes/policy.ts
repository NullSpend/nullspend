import type { AuthResult } from "../lib/auth.js";
import { errorResponse } from "../lib/errors.js";
import { doBudgetGetState } from "../lib/budget-do-client.js";
import type { BudgetRow } from "../durable-objects/user-budget.js";
import { getAllPricing } from "@nullspend/cost-engine";
import { emitMetric } from "../lib/metrics.js";

interface PolicyBudget {
  remaining_microdollars: number;
  max_microdollars: number;
  spend_microdollars: number;
  period_end: string | null;
  entity_type: string;
  entity_id: string;
}

interface CheapestModel {
  model: string;
  provider?: string;
  input_per_mtok: number;
  output_per_mtok: number;
}

interface PolicyResponse {
  budget: PolicyBudget | null;
  allowed_models: string[] | null;
  allowed_providers: string[] | null;
  cheapest_per_provider: Record<string, CheapestModel> | null;
  cheapest_overall: CheapestModel | null;
  restrictions_active: boolean;
}

/**
 * Compute period_end from a budget row's reset_interval and period_start.
 * Returns ISO string or null if no reset interval.
 */
function computePeriodEnd(row: BudgetRow): string | null {
  if (!row.reset_interval || row.period_start === 0) return null;
  const start = new Date(row.period_start);
  switch (row.reset_interval) {
    case "daily":
      start.setUTCDate(start.getUTCDate() + 1);
      break;
    case "weekly":
      start.setUTCDate(start.getUTCDate() + 7);
      break;
    case "monthly":
      start.setUTCMonth(start.getUTCMonth() + 1);
      break;
    case "yearly":
      start.setUTCFullYear(start.getUTCFullYear() + 1);
      break;
    default:
      return null;
  }
  return start.toISOString();
}

/**
 * Find cheapest models from the pricing catalog, filtered by allowed models/providers.
 */
function findCheapestModels(
  allowedModels: string[] | null,
  allowedProviders: string[] | null,
): { perProvider: Record<string, CheapestModel>; overall: CheapestModel | null } {
  const catalog = getAllPricing();
  const perProvider: Record<string, CheapestModel> = {};
  let overall: CheapestModel | null = null;

  for (const [key, pricing] of Object.entries(catalog)) {
    const slashIdx = key.indexOf("/");
    if (slashIdx === -1) continue;
    const provider = key.slice(0, slashIdx);
    const model = key.slice(slashIdx + 1);

    // Filter by allowed providers
    if (allowedProviders && !allowedProviders.includes(provider)) continue;
    // Filter by allowed models
    if (allowedModels && !allowedModels.includes(model)) continue;

    const combinedCost = pricing.inputPerMTok + pricing.outputPerMTok;
    const entry: CheapestModel = {
      model,
      input_per_mtok: pricing.inputPerMTok,
      output_per_mtok: pricing.outputPerMTok,
    };

    // Track cheapest per provider
    const existing = perProvider[provider];
    if (!existing || combinedCost < existing.input_per_mtok + existing.output_per_mtok) {
      perProvider[provider] = entry;
    }

    // Track cheapest overall
    if (!overall || combinedCost < overall.input_per_mtok + overall.output_per_mtok) {
      overall = { ...entry, provider };
    }
  }

  return { perProvider, overall };
}

/**
 * GET /v1/policy — returns the key's budget state, restrictions, and cheapest models.
 * Read-only. No reservations, no side effects.
 */
export async function handlePolicy(
  request: Request,
  env: Env,
  auth: AuthResult,
  traceId: string,
): Promise<Response> {
  const startMs = performance.now();

  try {
    const ownerId = auth.orgId ?? auth.userId;

    // Budget state (from DO, read-only)
    let policyBudget: PolicyBudget | null = null;

    if (auth.hasBudgets) {
      try {
        const budgetRows = await doBudgetGetState(env, ownerId);
        if (budgetRows.length > 0) {
          // Find most restrictive (lowest remaining)
          let mostRestrictive: BudgetRow | null = null;
          let lowestRemaining = Infinity;

          for (const row of budgetRows) {
            const remaining = row.max_budget - row.spend - row.reserved;
            if (remaining < lowestRemaining) {
              lowestRemaining = remaining;
              mostRestrictive = row;
            }
          }

          if (mostRestrictive) {
            policyBudget = {
              remaining_microdollars: Math.max(0, mostRestrictive.max_budget - mostRestrictive.spend - mostRestrictive.reserved),
              max_microdollars: mostRestrictive.max_budget,
              spend_microdollars: mostRestrictive.spend,
              period_end: computePeriodEnd(mostRestrictive),
              entity_type: mostRestrictive.entity_type,
              entity_id: mostRestrictive.entity_id,
            };
          }
        }
      } catch (err) {
        // DO unavailable — return policy without budget data
        console.error("[policy] Budget DO unavailable:", err instanceof Error ? err.message : err);
        emitMetric("policy_budget_error", { ownerId });
      }
    }

    // Restrictions from auth
    const allowedModels = auth.allowedModels;
    const allowedProviders = auth.allowedProviders;
    const restrictionsActive = allowedModels !== null || allowedProviders !== null;

    // Cheapest models
    const { perProvider, overall } = findCheapestModels(allowedModels, allowedProviders);

    const response: PolicyResponse = {
      budget: policyBudget,
      allowed_models: allowedModels,
      allowed_providers: allowedProviders,
      cheapest_per_provider: Object.keys(perProvider).length > 0 ? perProvider : null,
      cheapest_overall: overall,
      restrictions_active: restrictionsActive,
    };

    const durationMs = Math.round(performance.now() - startMs);
    emitMetric("policy_request", { durationMs, hasBudget: policyBudget !== null, hasRestrictions: restrictionsActive });

    const resp = Response.json(response);
    resp.headers.set("X-NullSpend-Trace-Id", traceId);
    resp.headers.set("Cache-Control", "no-store");
    return resp;
  } catch (err) {
    console.error("[policy] Unhandled error:", err);
    const resp = errorResponse("internal_error", "Failed to retrieve policy", 500);
    resp.headers.set("X-NullSpend-Trace-Id", traceId);
    return resp;
  }
}
