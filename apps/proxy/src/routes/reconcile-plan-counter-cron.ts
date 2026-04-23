/**
 * Plan-counter drift reconciliation cron (PR-2d / Decision #35).
 *
 * Runs inside the proxy as a CF Worker cron trigger (every 15 minutes per
 * `wrangler.jsonc`). Picks up cost events that the dashboard's direct-mode
 * sync path couldn't successfully count at ingest — those events are tagged
 * `_ns_counter_sync_skipped: "true"` in their `cost_events.tags` JSONB column
 * (fail-open path in `app/api/cost-events/*`).
 *
 * Design points preserved here from the 5 codex rounds:
 *
 * - **Period override (codex P0 #1 / C59):** the DO increment uses the event's
 *   STORED `period_start` / `period_end` — NOT `Date.now()` bounds. A March
 *   event reconciled on April 15 must still land on the March plan-counter row.
 *
 * - **`id`-scoped tag clear (codex R2 #1 / C44):** the post-success `UPDATE`
 *   targets `cost_events.id` (UUID PK), not `request_id`. A concurrent event
 *   insert with the same request_id (different provider, different row) will
 *   not have its tag stripped by this run.
 *
 * - **Crash safety between DO call and tag clear (C43):** if the DO RPC
 *   succeeds but the subsequent `UPDATE` throws (Postgres blip, lock, etc.),
 *   we leave the tag in place. The next run re-submits the DO call with the
 *   same `reconcile-${eventId}` idempotency key — DO dedup returns the prior
 *   count unchanged. No drift, no double-count.
 *
 * - **LIMIT 500 per run (C44b):** scheduled handlers have a 30s default CPU
 *   cap. Capping the batch leaves headroom; future runs catch up on drift.
 *
 * - **Idempotency namespace UNIFIED with live path (build-audit F1):** the
 *   cron computes the SAME `sha256IdempotencyKey(request_id, provider)` that
 *   the dashboard's `/api/cost-events` route uses. Catches the partial-success
 *   failure mode where the proxy completed the live-path increment but the
 *   HTTP response was lost mid-flight — the dashboard sees timeout, fail-opens
 *   the skip tag, the cron picks the event up, and DO dedup recognizes the
 *   replay. Without a shared key, the cron would over-count under that race.
 *   The original "reconcile-${id}" prefix design assumed live + cron paths
 *   were mutually exclusive; that assumption broke under proxy timeouts.
 */

import { getSql } from "../lib/db.js";
import { doIncrementPlanCounter } from "../lib/budget-do-client.js";
import { emitMetric } from "../lib/metrics.js";
import { toFiniteNumber } from "../lib/number-coerce.js";
import { sha256IdempotencyKey } from "../lib/idempotency-key.js";
import { writePlanCounterSyncFailure } from "../lib/write-plan-counter-sync-failure.js";
import type { PlanLimitMode, TierLabel } from "../lib/api-key-auth.js";

/**
 * Per-run batch cap. See file-level docstring — scheduled handlers have a
 * 30s CPU default; ~500 DO round-trips at p95 ~50ms each gives 25s headroom
 * vs. the limit. If drift volume exceeds this sustainably, future work can
 * raise the cap OR shorten the cron interval.
 */
export const RECONCILE_BATCH_LIMIT = 500;

// Inlined (not imported from api-key-auth.ts) — the two `normalize*` helpers
// there are file-private. Duplicated here rather than exporting them because
// the cron path has exactly one caller and the helpers are 3 lines each.
function normalizePlanLimitMode(v: unknown): PlanLimitMode {
  return v === "hard" ? "hard" : "soft";
}

function normalizeTierLabel(v: unknown): TierLabel {
  if (v === "pro" || v === "scale" || v === "enterprise" || v === "free") return v;
  return "free";
}

/**
 * Entry point — dispatched by `index.ts::scheduled()` on the 15-minute cron.
 *
 * Never throws (errors surface as metrics). The cron handler must always
 * return cleanly so the CF scheduler doesn't mark the tick as failed — a
 * single bad row should not drop the whole batch.
 */
export async function handleReconcilePlanCounterCron(env: Env): Promise<void> {
  // PR-2e Sub-PR 16 + codex R5 #1: KV heartbeat write. MUST be the first
  // statement of the handler, BEFORE any Postgres access. A Postgres-resident
  // heartbeat is silenced by the same PG outage it monitors. The KV write
  // proves the cron handler ran, regardless of whether the actual
  // reconciliation work succeeded. Watcher reads `plan_counter_cron_last_tick_at`
  // from this KV key via cross-worker binding.
  //
  // Wrapped in its OWN try/catch and treated as best-effort. If KV throws,
  // emit a dedicated metric and CONTINUE — a KV outage MUST NOT take down the
  // cron itself (codex R5 #1: monitoring code that fails the path it's
  // monitoring is worse than no monitoring code).
  try {
    await env.CACHE_KV.put(
      "plan_counter_cron_last_tick_at",
      Date.now().toString(),
    );
  } catch (kvErr) {
    console.error(
      "[reconcile-plan-counter-cron] KV heartbeat write failed (continuing):",
      kvErr,
    );
    emitMetric("plan_counter_heartbeat_write_failed", {
      reason: kvErr instanceof Error ? kvErr.message.slice(0, 100) : "unknown",
    });
  }

  const connectionString = env.HYPERDRIVE.connectionString;
  const sql = getSql(connectionString);

  // Query + hydrate identity. Pattern mirrors `api-key-auth.ts::lookupKeyInDb`:
  // single LEFT JOIN LATERAL against subscriptions, hardcoded tier limit
  // CASE (matches 100K/500K/2M/∞ truth already codified there).
  //
  // Build-audit F1: also SELECT `request_id` + `provider` so the cron can
  // compute the SAME idempotency key the live-path used — DO dedup is then
  // the single source of truth for whether this event has already been
  // counted, regardless of which path got there first.
  let rows: ReadonlyArray<Record<string, unknown>>;
  try {
    rows = await sql`
      SELECT ce.id AS event_id,
        ce.request_id,
        ce.provider,
        ce.org_id,
        (EXTRACT(EPOCH FROM ce.period_start) * 1000)::BIGINT AS period_start,
        (EXTRACT(EPOCH FROM ce.period_end) * 1000)::BIGINT AS period_end,
        COALESCE(s.tier, 'free') AS tier_label,
        CASE
          WHEN s.tier IS NULL OR s.tier = 'free' THEN 100000
          WHEN s.tier = 'pro'   THEN 500000
          WHEN s.tier = 'scale' THEN 2000000
          ELSE NULL
        END AS plan_limit_block_at,
        CASE
          WHEN s.tier IS NULL OR s.tier = 'free' THEN 'hard'
          ELSE 'soft'
        END AS plan_limit_mode
      FROM cost_events ce
      LEFT JOIN LATERAL (
        SELECT tier
        FROM subscriptions
        WHERE org_id = ce.org_id AND status IN ('active', 'past_due', 'trialing')
        LIMIT 1
      ) s ON TRUE
      WHERE ce.tags ->> '_ns_counter_sync_skipped' = 'true'
        AND ce.period_start IS NOT NULL
        AND ce.period_end IS NOT NULL
        AND ce.org_id IS NOT NULL
      ORDER BY ce.created_at ASC
      LIMIT ${RECONCILE_BATCH_LIMIT}
    `;
  } catch (err) {
    console.error("[reconcile-plan-counter-cron] Query failed:", err);
    emitMetric("plan_counter_drift_query_failed", {
      reason: err instanceof Error ? err.message.slice(0, 100) : "unknown",
    });
    return;
  }

  emitMetric("plan_counter_drift_cron_run", { candidateCount: rows.length });

  let reconciled = 0;
  let failed = 0;

  for (const raw of rows) {
    const eventId = raw.event_id as string;
    const orgId = raw.org_id as string;
    const requestId = raw.request_id as string;
    const provider = raw.provider as string;

    const periodStart = toFiniteNumber(raw.period_start);
    const periodEnd = toFiniteNumber(raw.period_end);
    if (periodStart === null || periodEnd === null) {
      // SQL guard already filters out NULL periods, but defense-in-depth against
      // coercion surprises (BIGINT → string → NaN). Counts as a failure.
      emitMetric("plan_counter_drift_reconcile_failed", {
        reason: "period_coercion",
        orgId,
        eventId,
      });
      failed++;
      continue;
    }

    // Build-audit F1: identical hash to the dashboard live-path. If the
    // proxy already incremented for this (request_id, provider) tuple, the
    // DO's `plan_counter_idempotency` table will dedup this call — the
    // counter stays at the prior value and `count` reflects current state.
    const idempotencyKey = await sha256IdempotencyKey(requestId, provider);

    try {
      await doIncrementPlanCounter(env, {
        orgId,
        planLimitBlockAt: toFiniteNumber(raw.plan_limit_block_at),
        planLimitMode: normalizePlanLimitMode(raw.plan_limit_mode),
        tier: normalizeTierLabel(raw.tier_label),
        periodStart,
        periodEnd,
        idempotencyKey,
      });
    } catch (doErr) {
      console.error("[reconcile-plan-counter-cron] DO increment failed:", { orgId, eventId, error: doErr });
      emitMetric("plan_counter_drift_reconcile_failed", {
        reason: "do_error",
        orgId,
        eventId,
      });
      failed++;
      continue;
    }

    try {
      // jsonb `-` operator removes the key. Scoping by `id` (UUID PK) is
      // atomic vs. concurrent inserts of new events against the same request_id.
      await sql`
        UPDATE cost_events
        SET tags = tags - '_ns_counter_sync_skipped'
        WHERE id = ${eventId}
      `;
    } catch (updateErr) {
      // DO call already succeeded; tag clear failed. Leave tag in place — next
      // run retries, DO dedup returns same count. No double-count, no drift.
      console.error("[reconcile-plan-counter-cron] Tag clear failed (row will retry):", { eventId, error: updateErr });
      emitMetric("plan_counter_drift_tag_clear_failed", { orgId, eventId });
      // PR-2e audit Finding 1: persist the tag-clear failure to
      // `plan_counter_sync_failures{reason='tag_clear_failed'}` so the
      // launch-watcher's /api/internal/metrics-summary can aggregate it
      // and fire the `tag_clear_failing` alert. Without this PG write the
      // metric is console-only — operators don't see the precondition
      // for the DO's 24h idempotency-prune-triggered double-count race
      // until after it's already happened.
      //
      // Own try/catch: if PG is the cause of the UPDATE failure (pool
      // exhausted, network blip), this write likely fails too — emit a
      // secondary metric and continue, per the "observability code MUST
      // NOT take down the path it's observing" rule.
      try {
        await writePlanCounterSyncFailure(connectionString, {
          orgId,
          reason: "tag_clear_failed",
          count: 1,
        });
      } catch (writeErr) {
        emitMetric("plan_counter_tag_clear_write_failed", {
          reason:
            writeErr instanceof Error ? writeErr.message.slice(0, 100) : "unknown",
        });
      }
      continue;
    }

    emitMetric("plan_counter_drift_reconciled", { orgId, eventId });
    reconciled++;
  }

  emitMetric("plan_counter_drift_cron_summary", {
    candidateCount: rows.length,
    reconciled,
    failed,
    batchFull: rows.length === RECONCILE_BATCH_LIMIT,
  });

  // PR-2e Sub-PR 16: persist the cron tick summary to Postgres so the
  // launch-watcher's /api/internal/metrics-summary endpoint can aggregate
  // it over a sliding window. Console-only emit (above) is for log
  // forensics; the Postgres write is the queryable source of truth.
  //
  // Wrapped in own try/catch for the same reason as the KV heartbeat:
  // observability writes MUST NOT take down the cron itself. If this insert
  // fails the existing rows are still reconciled (those happened above),
  // we just lose ONE tick's row from the watcher's aggregate — the next
  // tick re-populates within 15 min. Emits a dedicated metric so the
  // failure is itself observable via console logs.
  try {
    await sql`
      INSERT INTO plan_counter_cron_history (batch_full, candidate_count, reconciled, failed)
      VALUES (
        ${rows.length === RECONCILE_BATCH_LIMIT},
        ${rows.length},
        ${reconciled},
        ${failed}
      )
    `;
  } catch (writeErr) {
    console.error(
      "[reconcile-plan-counter-cron] cron_history insert failed (continuing):",
      writeErr,
    );
    emitMetric("plan_counter_cron_history_write_failed", {
      reason:
        writeErr instanceof Error ? writeErr.message.slice(0, 100) : "unknown",
    });
  }

  // PR-2e follow-up: prune retry-idempotency rows older than 7 days.
  //
  // Both tables are internal retry state (NOT product data) per their schema
  // comments — the outbox retry window is minutes, a 7-day cutoff is generous
  // headroom. `written_at` has a dedicated index on each table specifically
  // for this DELETE. Post-index the query cost is O(rows-to-delete), so in
  // steady state after backlog catch-up each tick deletes only ~15 min of
  // just-aged-out rows — near zero.
  //
  // Run on every tick (not "daily"): simpler than time-predicate gating, no
  // state to coordinate, and resilient to cron ticks being skipped. Wrapped
  // in its own try/catch per the same "monitoring/maintenance MUST NOT take
  // down the path it's attached to" rule as the KV heartbeat and cron_history
  // insert above.
  try {
    // `RETURNING 1` makes postgres.js return one row per deleted row so we
    // can report the deleted count via metric without a second COUNT query.
    const syncResult = await sql`
      DELETE FROM plan_counter_sync_requests
      WHERE written_at < now() - INTERVAL '7 days'
      RETURNING 1
    `;
    const divergenceResult = await sql`
      DELETE FROM plan_counter_divergence_dedup
      WHERE written_at < now() - INTERVAL '7 days'
      RETURNING 1
    `;
    emitMetric("plan_counter_retry_state_pruned", {
      syncRequestsDeleted: Array.isArray(syncResult) ? syncResult.length : 0,
      divergenceDedupDeleted: Array.isArray(divergenceResult) ? divergenceResult.length : 0,
    });
  } catch (pruneErr) {
    console.error(
      "[reconcile-plan-counter-cron] retry-state prune failed (continuing):",
      pruneErr,
    );
    emitMetric("plan_counter_retry_state_prune_failed", {
      reason:
        pruneErr instanceof Error ? pruneErr.message.slice(0, 100) : "unknown",
    });
  }
}
