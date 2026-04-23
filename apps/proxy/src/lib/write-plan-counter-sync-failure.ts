/**
 * PR-2e F2-partial: DO-side helper to write a row into `plan_counter_sync_failures`.
 *
 * Used by the divergence outbox abandoned-handler (codex R7 C4 observability
 * fix) to emit a durable signal when a divergence outbox entry exceeds max
 * retry attempts. Without this, an outbox stall (HYPERDRIVE down for hours,
 * or terminal-FK classification bug) produces only AE/console emissions
 * that the launch-watcher does NOT query — false-green on a real failure.
 *
 * Reason codes are OPEN-SET by design: this helper accepts any string reason.
 * The divergence-abandoned path uses `reason='divergence_outbox_abandoned'`;
 * the dashboard fail-open path (F1, next commit) uses `reason='batch_sync_failed'`.
 * `/api/internal/metrics-summary` aggregates by reason to give the watcher
 * per-signal counts.
 *
 * Best-effort write: callers MUST wrap in try/catch so a PG outage doesn't
 * propagate into the DO alarm dispatcher. Per `monitoring_hardening_can_introduce_failures`
 * rule (2026-04-20 cross-model learning): observability code MUST NOT take
 * down the path it's observing.
 */

import { getSql } from "./db.js";

export async function writePlanCounterSyncFailure(
  connectionString: string,
  entry: {
    orgId: string;
    reason: string;
    count: number;
  },
): Promise<void> {
  const sql = getSql(connectionString);
  await sql`
    INSERT INTO plan_counter_sync_failures (org_id, reason, count)
    VALUES (${entry.orgId}::uuid, ${entry.reason}, ${entry.count})
  `;
}
