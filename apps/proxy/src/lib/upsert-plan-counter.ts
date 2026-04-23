/**
 * PR-2b: Hyperdrive-backed upsert for `org_period_usage`.
 *
 * Called from the DO alarm's plan-counter outbox dispatch branch. Writes
 * an additive delta (ON CONFLICT ... DO UPDATE SET governed_requests_count =
 * EXISTING + INCOMING).
 *
 * **Retry-idempotency (per build-audit Finding #1):** the
 * `plan_counter_sync_requests` table acts as a Postgres-side dedup guard.
 * Each call first attempts `INSERT ... ON CONFLICT DO NOTHING` on the dedup
 * table keyed by `requestId`; the subsequent additive upsert to
 * `org_period_usage` runs ONLY when the dedup insert actually landed.
 *
 * This closes the race where the DO alarm sequence
 * `await upsertPlanCounterPeriod(...)` → `ackPlanCounterEntryById(...)` gets
 * interrupted between the Postgres commit and the DO-local ack — without the
 * dedup table, the next alarm retry would double-add to
 * `governed_requests_count`. Same pattern as `updateBudgetSpend`'s
 * `reconciled_requests` defense for the budget-spend path.
 *
 * Period bounds come from the outbox entry (which captured them at increment
 * time). We DO NOT re-resolve bounds here — a delayed delivery must land on
 * the ORIGINAL period row, not whichever period is live at retry time.
 *
 * Uses `fetch_types: false` `postgres.js` via the shared `getSql` helper
 * (I/O context isolation — per-request instance).
 */

import { getSql } from "./db.js";

export interface UpsertPlanCounterPeriodResult {
  applied: boolean;
}

/**
 * Upsert a governed-request delta into `org_period_usage`, dedup'd by
 * `requestId` via `plan_counter_sync_requests`. Returns `{ applied: true }`
 * if the delta was newly committed, `{ applied: false }` if the requestId
 * was already seen (retry of an earlier partial-commit) so no counter
 * change was needed.
 *
 * period_start / period_end are ms-epoch integers (matches DO SQLite
 * conventions). Converted to `to_timestamp(... / 1000.0)` for the
 * TIMESTAMPTZ column.
 *
 * The dedup INSERT + conditional upsert run inside `sql.begin()` so they
 * commit atomically — no window where the dedup row exists without the
 * usage row, or vice versa.
 */
export async function upsertPlanCounterPeriod(
  connectionString: string,
  entry: {
    requestId: string;
    orgId: string;
    periodStart: number;
    periodEnd: number;
    deltaCount: number;
  },
): Promise<UpsertPlanCounterPeriodResult> {
  if (entry.deltaCount <= 0) return { applied: false };

  const sql = getSql(connectionString);

  return sql.begin(async (tx) => {
    // Dedup: INSERT only if we haven't seen this (org_id, requestId, periodStart)
    // tuple before. Composite key across THREE columns (edge-case EC6 +
    // codex-final F2):
    //   - period_start: SDK caller reusing a deterministic idempotency key
    //     across a billing-period boundary gets fresh credit in each period.
    //   - org_id: two orgs using the same idempotency key in the same period
    //     don't collide — each org's dedup namespace is isolated.
    const dedup = await tx`
      INSERT INTO plan_counter_sync_requests (org_id, request_id, period_start)
      VALUES (${entry.orgId}::uuid, ${entry.requestId}, to_timestamp(${entry.periodStart} / 1000.0))
      ON CONFLICT (org_id, request_id, period_start) DO NOTHING
      RETURNING request_id
    `;

    if (dedup.count === 0) {
      // Already committed on a prior attempt — skip the usage upsert entirely.
      return { applied: false };
    }

    await tx`
      INSERT INTO org_period_usage (org_id, period_start, period_end, governed_requests_count, last_updated_at)
      VALUES (
        ${entry.orgId}::uuid,
        to_timestamp(${entry.periodStart} / 1000.0),
        to_timestamp(${entry.periodEnd} / 1000.0),
        ${entry.deltaCount},
        now()
      )
      ON CONFLICT (org_id, period_start) DO UPDATE SET
        governed_requests_count = org_period_usage.governed_requests_count + EXCLUDED.governed_requests_count,
        last_updated_at = now()
    `;

    return { applied: true };
  });
}
