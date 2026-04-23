import { emitMetric } from "./metrics.js";

/**
 * Resolve the billing period bounds for a request.
 *
 * - Paid org with valid subscription period → use subscription bounds verbatim.
 * - Free / unpaid org (both fields null) → fall back to UTC calendar month.
 * - Paid-but-corrupt (one field null, other set, OR end <= start) → fall back
 *   to calendar month AND emit `plan_counter_period_fallback{reason}` metric
 *   for observability (per Decision #36 / plan-audit A5 / codex PR-2a-R3-2).
 *
 * `orgId` is required on the input type (not optional) per codex PR-2a-N5 —
 * optional would collapse the paid-corrupt metric into an "unknown" bucket
 * and weaken the A5 observability fix. Null `orgId` (valid per the
 * `ApiKeyIdentity` type) is tagged as the literal string `"null"` so the
 * sentinel is distinguishable from a forgotten argument.
 *
 * The metric name is `plan_counter_period_fallback` — canonical across
 * proxy-side and DO-side invariant checks (per Decision #38 + codex
 * PR-2a-R3-2). Never `plan_counter_invalid_period` (deprecated alias).
 */
export function resolvePeriodBounds(
  identity: {
    orgId: string | null;
    subscriptionPeriodStart: number | null;
    subscriptionPeriodEnd: number | null;
  },
  now: number,
): { periodStart: number; periodEnd: number } {
  const s = identity.subscriptionPeriodStart;
  const e = identity.subscriptionPeriodEnd;

  // Happy path: valid subscription period with ordering invariant satisfied.
  if (s !== null && e !== null && e > s) {
    return { periodStart: s, periodEnd: e };
  }

  // Paid-but-corrupt: emit signal so ops can tell "normal Free user" from
  // "paid sub with bad Stripe data". Alert on any sustained rate > 0.
  if (s !== null || e !== null) {
    emitMetric("plan_counter_period_fallback", {
      reason: e !== null && s !== null && e <= s ? "paid_inverted" : "paid_partial",
      orgId: identity.orgId ?? "null",
    });
  }
  // else: both null → Free / unpaid / self-hosted — expected fallback, no metric.

  // Fallback: UTC calendar month.
  const d = new Date(now);
  const periodStart = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
  const periodEnd = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1);
  return { periodStart, periodEnd };
}
