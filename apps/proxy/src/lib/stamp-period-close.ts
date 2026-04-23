/**
 * PR-6a: DO period-close writer.
 *
 * Spec: `docs/plans/pricing-pr6a-overage-foundation.md` §2 (DO period-close
 * writer) + §5 (D1-D5 tests) + codex R3 P1 rationale (bridge elimination).
 *
 * Called from the DO alarm handler (`handlePlanCounterBoundaryFlush`) at the
 * same moment the expired `plan_counter` row would be deleted. Reads the
 * live subscription once, computes whether the closed period exceeded the
 * tier's included request cap, and stamps 3 snapshot columns on the matching
 * `org_period_usage` row in a single UPDATE.
 *
 * **Idempotency.** The UPDATE predicate includes `tier_at_period_end IS NULL`,
 * so a retry after a prior success is a no-op (the row already has snapshots).
 * PR-6b's cron flip to `disposition='billed'` is also protected — the same
 * predicate prevents re-stamping a row whose disposition has advanced past
 * `billable_pending`.
 *
 * **Failure mode.** If this helper throws (Stripe status unrecognized, DB
 * unavailable, UPDATE errored), the DO alarm handler leaves the `plan_counter`
 * row in place and the next alarm retries. The alarm wrapper catches the
 * throw + emits `stamp_period_close_failure`. Fail-closed invariant: we'd
 * rather retry a period-close indefinitely than delete the DO's period
 * counter without a matching PG snapshot.
 *
 * **Unknown status (R3 P0).** When `subscription.status` falls outside the
 * recognized enum, we still stamp `tier_at_period_end` + `status_at_period_end`
 * (so ops can see what Stripe returned) but LEAVE `disposition` NULL. Ops
 * triages the row manually; PR-6b's cron skips NULL-disposition rows because
 * the partial index (`disposition = 'billable_pending'`) excludes them.
 *
 * **Why TIERS caps are duplicated here.** The dashboard's `lib/stripe/tiers.ts`
 * is the authoritative TIERS map, but the proxy (Cloudflare Worker) can't
 * import from `@/lib/*`. Keeping the caps + rates duplicated here is the same
 * tradeoff the proxy already makes for `TierLabel` in `api-key-auth.ts` — a
 * shared `@nullspend/tiers` package is a reasonable future refactor once a
 * third consumer appears. If the dashboard TIERS moves, this table moves too
 * and the `tiers-matrix` test in `lib/stripe/tiers.test.ts` catches drift.
 */

import { getSql } from "./db.js";
import { emitMetric } from "./metrics.js";

export interface StampPeriodCloseArgs {
  orgId: string;
  /** ms-epoch period_start matching the opu row's primary key. */
  periodStart: number;
  /** ms-epoch period_end — stamped for structured logs; UPDATE predicate uses period_start only. */
  periodEnd: number;
}

export type StampDisposition = "billable_pending" | "evaluated_skipped" | null;

export interface StampPeriodCloseResult {
  /** True when this call wrote fresh snapshot columns (0 on idempotent re-run). */
  applied: boolean;
  /** Resolved tier snapshot (null when no subscription exists). */
  tier: string | null;
  /** Resolved status snapshot (null when no subscription exists). */
  status: string | null;
  /** Disposition written — or null if unknown-status fail-closed path. */
  disposition: StampDisposition;
  /** When true, the caller should retry on the next alarm tick (no sub row yet, etc). */
  deferred: boolean;
}

/**
 * Tier caps + per-request overage rates in microcents.
 * Mirror of `TIERS` in `lib/stripe/tiers.ts`.
 * - `free` / `enterprise`: no overage (null rate).
 * - `pro`: 500k included, 100 microcents per overage request ($0.01).
 * - `scale`: 2M included, 50 microcents per overage request ($0.005).
 */
const TIER_CAPS: Record<string, { included: number | null; rateMicroCents: number | null }> = {
  free: { included: 100_000, rateMicroCents: null },
  pro: { included: 500_000, rateMicroCents: 100 },
  scale: { included: 2_000_000, rateMicroCents: 50 },
  enterprise: { included: null, rateMicroCents: null },
};

/**
 * Stripe subscription statuses that produce a billable disposition.
 *
 * - `active`  : the customer is currently paying — obvious.
 * - `canceled`: PR-6b CX-R1-4. Cancel-mid-period customers still owe overage
 *               for usage BEFORE canceling; cancellation stops renewal, it
 *               does NOT forgive prior usage. Mirrors the dashboard's
 *               `computeOverage` which treats canceled billable.
 *
 * ANY other observed status — `past_due`, `trialing`, and unknowns — goes to
 * evaluated_skipped (known-non-billable) or NULL disposition (unknown_status
 * path). NULL forces ops to triage before PR-6b's cron silently starts or
 * stops billing on a status Stripe quietly introduced.
 */
const BILLABLE_STATUSES = new Set(["active", "canceled"]);

/**
 * Statuses whose disposition is deterministically `evaluated_skipped`.
 *
 * Kept deliberately narrow: only the two statuses `computeOverage` handles
 * explicitly by reason (`trialing`, `past_due`). `canceled` moved to
 * BILLABLE_STATUSES per PR-6b CX-R1-4. Other non-billable statuses —
 * `incomplete`, `incomplete_expired`, `unpaid`, `paused` — fall through to
 * the fail-closed NULL-disposition path per plan R3 P0 ("refuses to stamp
 * disposition='billable_pending' for unknown status, forcing manual
 * review"). Keeps proxy stamp semantics consistent with the dashboard's
 * `computeOverage` — a paused org gets the same ops-facing signal from
 * both sides.
 */
const KNOWN_NON_BILLABLE_STATUSES = new Set(["trialing", "past_due"]);

/**
 * Determine the disposition for a (tier, status, governedCount) snapshot.
 *
 * Fail-closed returns (`null`):
 *   - Unknown `status` (not in the recognized enum) — R3 P0 invariant, also
 *     explicit in the audit finding #2 resolution.
 *   - Unknown `tier` (not in TIER_CAPS) — R3 P0 invariant extended to tiers.
 *     `evaluated_skipped` would silently tell PR-6b's cron "don't bill this";
 *     NULL forces ops to triage the row manually.
 *
 * Distinguishes known-not-overageable tiers (`free`, `enterprise`) from
 * unknown tiers by checking TIER_CAPS membership first.
 */
export function computeStampDisposition(
  tier: string,
  status: string,
  governedRequestsCount: number,
): StampDisposition {
  if (!BILLABLE_STATUSES.has(status)) {
    // Known non-billable = evaluated_skipped; unknown = NULL (fail closed).
    return KNOWN_NON_BILLABLE_STATUSES.has(status) ? "evaluated_skipped" : null;
  }
  const caps = TIER_CAPS[tier];
  if (!caps) {
    // Unknown tier — fail-closed per R3 P0 (audit finding #2). Do NOT
    // silently mark as evaluated_skipped; that tells PR-6b's cron "don't
    // bill this" on what could be a misconfigured-but-billable row.
    return null;
  }
  if (caps.included === null || caps.rateMicroCents === null) {
    // Free (null rate) or Enterprise (null included + rate) — both are
    // known non-overageable by design. Different from unknown tier.
    return "evaluated_skipped";
  }
  const usage = Math.max(0, governedRequestsCount);
  return usage > caps.included ? "billable_pending" : "evaluated_skipped";
}

/**
 * Stamp the three snapshot columns on the latest closed period for `orgId`.
 *
 * Single DB transaction: SELECT subscription + opu row, compute disposition,
 * conditional UPDATE. The UPDATE predicate enforces idempotency:
 * `WHERE org_id = $1 AND period_start = $2 AND tier_at_period_end IS NULL`.
 *
 * Returns `deferred: true` (no throw) when no subscription row exists for
 * `orgId` — the DO retries on the next alarm tick. This is expected on
 * the first post-deploy period close for an org that was Free at
 * increment time but has since churned.
 */
export async function stampPeriodClose(
  connectionString: string,
  args: StampPeriodCloseArgs,
): Promise<StampPeriodCloseResult> {
  const sql = getSql(connectionString);

  return sql.begin(async (tx) => {
    // Read sub (tier + status) + opu row's governed count atomically.
    // Composite SELECT over subscriptions + org_period_usage so we don't
    // race a concurrent ingest update on the usage row.
    const subRows = await tx<{ tier: string; status: string }[]>`
      SELECT tier, status
      FROM subscriptions
      WHERE org_id = ${args.orgId}::uuid
      LIMIT 1
    `;

    const opuRows = await tx<{ governed_requests_count: number; tier_at_period_end: string | null; disposition: string | null }[]>`
      SELECT governed_requests_count, tier_at_period_end, disposition
      FROM org_period_usage
      WHERE org_id = ${args.orgId}::uuid
        AND period_start = to_timestamp(${args.periodStart} / 1000.0)
      LIMIT 1
    `;

    const opu = opuRows[0];
    if (!opu) {
      // No opu row (increment path never fired for this period). Nothing to stamp.
      return { applied: false, tier: null, status: null, disposition: null, deferred: false };
    }

    // Idempotency check: if the snapshot already exists, don't re-stamp.
    // (Also guards against 6b's `disposition='billed'` flip being overwritten
    // by a re-entry of this helper after a DO alarm retry — the UPDATE below
    // has the same guard, but checking here saves the second round trip.)
    if (opu.tier_at_period_end !== null) {
      return {
        applied: false,
        tier: opu.tier_at_period_end,
        status: null,
        disposition: (opu.disposition as StampDisposition) ?? null,
        deferred: false,
      };
    }

    // Audit finding #3: "no subscription row" is the NullSpend convention for
    // Free-tier orgs — paid orgs always have a subscriptions row. Prior
    // implementation returned `deferred: true` here, which is wrong for Free:
    // the alarm would retain the plan_counter row forever (no sub row to wait
    // for), the opu row would never get snapshot-stamped, and the preview
    // endpoint would emit `overage_fallback_snapshot_null` for every Free row
    // in perpetuity — defeating the metric's purpose as a migration signal.
    //
    // Correct behavior: synthesize a Free-tier snapshot. Free has no overage
    // billing by design (TIERS.free.overageMicroCentsPerRequest === null), so
    // disposition is a terminal 'evaluated_skipped'. Distinct metric from the
    // paid-bridge case so ops can still see Free-tier stamps vs. genuine
    // "sub lookup failed" anomalies.
    const sub = subRows[0] ?? { tier: "free", status: "active" };
    if (!subRows[0]) {
      emitMetric("stamp_period_close_free_synthesized", {
        orgId: args.orgId,
        periodStart: args.periodStart,
      });
    }

    const disposition = computeStampDisposition(
      sub.tier,
      sub.status,
      opu.governed_requests_count,
    );

    if (disposition === null) {
      // R3 P0 unknown-status OR unknown-tier path: stamp tier + status so
      // ops can see what Stripe returned, but leave disposition NULL so the
      // PR-6b cron's partial-index scan (`WHERE disposition =
      // 'billable_pending'`) skips the row. Manual triage required.
      //
      // Audit finding #2: emit two distinct metrics so ops can distinguish
      // "new Stripe status we don't recognize" (catch with docs update) from
      // "new tier we don't recognize" (probable config bug — the tier set is
      // under our control).
      const metricName =
        TIER_CAPS[sub.tier] === undefined
          ? "stamp_period_close_unknown_tier"
          : "stamp_period_close_unknown_status";
      emitMetric(metricName, {
        orgId: args.orgId,
        tier: sub.tier,
        status: sub.status,
      });
      await tx`
        UPDATE org_period_usage
        SET tier_at_period_end = ${sub.tier},
            status_at_period_end = ${sub.status},
            last_updated_at = now()
        WHERE org_id = ${args.orgId}::uuid
          AND period_start = to_timestamp(${args.periodStart} / 1000.0)
          AND tier_at_period_end IS NULL
      `;
      return {
        applied: true,
        tier: sub.tier,
        status: sub.status,
        disposition: null,
        deferred: false,
      };
    }

    await tx`
      UPDATE org_period_usage
      SET tier_at_period_end = ${sub.tier},
          status_at_period_end = ${sub.status},
          disposition = ${disposition},
          last_updated_at = now()
      WHERE org_id = ${args.orgId}::uuid
        AND period_start = to_timestamp(${args.periodStart} / 1000.0)
        AND tier_at_period_end IS NULL
    `;

    emitMetric("stamp_period_close_success", {
      orgId: args.orgId,
      tier: sub.tier,
      status: sub.status,
      disposition,
    });

    return {
      applied: true,
      tier: sub.tier,
      status: sub.status,
      disposition,
      deferred: false,
    };
  });
}
