import type { BudgetEntity } from "./budget-do-lookup.js";
import { buildThresholdPayload, type WebhookEvent } from "./webhook-events.js";

const DEFAULT_THRESHOLDS: readonly number[] = Object.freeze([50, 80, 90, 95]);

/**
 * Detect budget threshold crossings after a cost event.
 *
 * Compares the pre-request spend against the post-request spend to find
 * thresholds that were crossed by this specific request.
 *
 * Uses per-entity threshold percentages when available, falling back to
 * the default [50, 80, 90, 95]. The last threshold in the array is
 * classified as critical; all others are warning. Thresholds >= 90 are
 * also classified as critical (backward compat for default thresholds).
 *
 * // TODO: KV or DO-based dedup for threshold alerts
 * Currently, concurrent requests may both detect the same crossing.
 * Customer-side dedup by event type + entity is sufficient for launch.
 */
export function detectThresholdCrossings(
  budgetEntities: BudgetEntity[],
  costMicrodollars: number,
  requestId: string,
  apiVersion?: string,
): WebhookEvent[] {
  const events: WebhookEvent[] = [];

  for (const entity of budgetEntities) {
    if (entity.maxBudget <= 0) continue;

    const previousSpend = entity.spend;
    const newSpend = previousSpend + costMicrodollars;

    const previousPercent = Math.floor((previousSpend / entity.maxBudget) * 100);
    const newPercent = Math.floor((newSpend / entity.maxBudget) * 100);

    const thresholds = entity.thresholdPercentages ?? DEFAULT_THRESHOLDS;
    const lastThreshold = thresholds.length > 0 ? thresholds[thresholds.length - 1] : undefined;

    // Find thresholds crossed by this request (were below before, at or above now)
    for (const threshold of thresholds) {
      if (previousPercent < threshold && newPercent >= threshold) {
        // Critical if: last in the array OR >= 90 (backward compat)
        const isCritical = threshold === lastThreshold || threshold >= 90;
        events.push(
          buildThresholdPayload({
            budgetEntityType: entity.entityType,
            budgetEntityId: entity.entityId,
            budgetLimitMicrodollars: entity.maxBudget,
            budgetSpendMicrodollars: newSpend,
            thresholdPercent: threshold,
            triggeredByRequestId: requestId,
            isCritical,
          }, apiVersion),
        );
      }
    }
  }

  return events;
}
