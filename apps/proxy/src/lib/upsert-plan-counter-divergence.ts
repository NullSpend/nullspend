/**
 * PR-2e F2-partial: Hyperdrive-backed upsert for `plan_counter_divergences`.
 *
 * Called from the DO alarm's divergence outbox dispatch branch. Writes
 * one row per divergence event, guarded by a dedup table.
 *
 * **Retry-idempotency (PR-2e codex R7 Issue 1A):** the
 * `plan_counter_divergence_dedup` table acts as a Postgres-side dedup
 * guard. Each call first attempts `INSERT ... ON CONFLICT DO NOTHING` on
 * the dedup table keyed by (org_id, event_id); the subsequent divergence
 * INSERT runs ONLY when the dedup insert actually landed.
 *
 * Closes the race where the DO alarm sequence `upsertPlanCounterDivergence()`
 * -> `ackPlanCounterDivergenceEntryById()` gets interrupted between PG
 * commit and DO-local ack -- without the dedup table, the next alarm
 * retry would insert a SECOND divergence row, inflating the watcher's
 * COUNT(*) over the 15m window. Same pattern as `plan_counter_sync_requests`
 * defends `upsertPlanCounterPeriod`.
 *
 * **Event-time timestamps (PR-2e codex R7 C1):** `divergence_at_ms` from
 * the outbox entry is passed as an EXPLICIT column value via
 * `to_timestamp(... / 1000.0)`. DO NOT rely on PG's `DEFAULT now()` --
 * flush-time semantics land events in the wrong 15m window under
 * HYPERDRIVE lag or retry backoff, breaking the launch-watcher contract.
 *
 * Uses `fetch_types: false` `postgres.js` via the shared `getSql` helper
 * (I/O context isolation -- per-request instance).
 */

import { getSql } from "./db.js";

export interface UpsertPlanCounterDivergenceResult {
  applied: boolean;
}

/**
 * Upsert a divergence event, dedup'd by `(org_id, event_id)`. Returns
 * `{ applied: true }` if the event was newly committed, `{ applied: false }`
 * if the event_id was already seen (retry of an earlier partial-commit)
 * so no second row was written.
 *
 * All timestamps are ms-epoch integers (matches DO SQLite conventions).
 * Converted to `to_timestamp(... / 1000.0)` for TIMESTAMPTZ columns.
 */
export async function upsertPlanCounterDivergence(
  connectionString: string,
  entry: {
    eventId: string;
    orgId: string;
    tier: string;
    divergenceAtMs: number;
    storedPeriodStart: number;
    incomingPeriodStart: number;
  },
): Promise<UpsertPlanCounterDivergenceResult> {
  const sql = getSql(connectionString);

  return sql.begin(async (tx) => {
    // Dedup: INSERT only if (org_id, event_id) not already seen.
    const dedup = await tx`
      INSERT INTO plan_counter_divergence_dedup (org_id, event_id)
      VALUES (${entry.orgId}::uuid, ${entry.eventId})
      ON CONFLICT (org_id, event_id) DO NOTHING
      RETURNING event_id
    `;

    if (dedup.count === 0) {
      // Already committed on a prior attempt -- skip the divergence insert.
      return { applied: false };
    }

    // Explicit divergence_at -- DO NOT rely on PG DEFAULT now() per codex R7 C1.
    await tx`
      INSERT INTO plan_counter_divergences
        (org_id, divergence_at, tier, stored_period_start, incoming_period_start)
      VALUES (
        ${entry.orgId}::uuid,
        to_timestamp(${entry.divergenceAtMs} / 1000.0),
        ${entry.tier},
        to_timestamp(${entry.storedPeriodStart} / 1000.0),
        to_timestamp(${entry.incomingPeriodStart} / 1000.0)
      )
    `;

    return { applied: true };
  });
}
