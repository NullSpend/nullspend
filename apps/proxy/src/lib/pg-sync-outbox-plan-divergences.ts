/**
 * PR-2e F2-partial: Transactional outbox for divergence-event sync.
 *
 * Mirrors `pg-sync-outbox-plan-counter.ts` (PR-2b) but with divergence-specific
 * entry shape. Standalone by design -- do NOT factor into a shared generic
 * (per PR-2e codex R7 C3). The two existing outboxes have subtly different
 * ack semantics (budget-spend acks by request group, plan-counter by row id)
 * and the plan-counter helper header explicitly says "PATH-SPECIFIC BY DESIGN.
 * Do NOT generalize this helper." A factored generic risks reintroducing the
 * F1 sibling-row deletion bug PR-2b closed.
 *
 * Pattern: `incrementPlanCounter()` writes entries atomically inside
 * `transactionSync()` at the divergence detection site. Alarm attempts
 * Hyperdrive write (`upsertPlanCounterDivergence`). On success -> ack by
 * row-id. On failure -> mark retry with exponential backoff. After max
 * attempts -> delete abandoned and INSERT a row into
 * `plan_counter_sync_failures` with `reason='divergence_outbox_abandoned'`
 * so the launch-watcher can detect stalled visibility (C4 requirement).
 *
 * **Event-time timestamps (PR-2e codex R7 C1):** `divergence_at_ms` is
 * captured by the DO at detection time (Date.now()) and stored on the
 * outbox row. `upsertPlanCounterDivergence` passes it as an EXPLICIT
 * column value, overriding Postgres's `DEFAULT now()`. Relying on the
 * server-default would land events in the wrong 15m window under
 * HYPERDRIVE lag or retry backoff, producing false-green in the event
 * window and false-alert in the flush window.
 */

import type { SqlStorage } from "./pg-sync-outbox.js";

// Backoff schedule matches budget-spend and plan-counter outboxes -- same
// retry characteristics. Keeps ops mental model consistent across the three.
const BACKOFF_MS = [5_000, 15_000, 45_000, 120_000, 300_000];

export interface PlanCounterDivergenceOutboxEntry {
  id: number;
  eventId: string;
  orgId: string;
  tier: string;
  divergenceAtMs: number;
  storedPeriodStart: number;
  incomingPeriodStart: number;
  attempts: number;
  nextAttemptAt: number;
  createdAt: number;
}

/**
 * Create the divergence outbox table + indexes. Called from DO's initSchema
 * v12 migration. Idempotent.
 */
export function createPlanCounterDivergenceOutboxTable(sql: SqlStorage): void {
  sql.exec(`
    CREATE TABLE IF NOT EXISTS pg_sync_outbox_plan_divergences (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      tier TEXT NOT NULL,
      divergence_at_ms INTEGER NOT NULL,
      stored_period_start INTEGER NOT NULL,
      incoming_period_start INTEGER NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_divergences_retry_idx
      ON pg_sync_outbox_plan_divergences(next_attempt_at, attempts);
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_divergences_event_id_idx
      ON pg_sync_outbox_plan_divergences(event_id);
  `);
}

/**
 * Write a divergence outbox entry. MUST be called INSIDE transactionSync()
 * so the outbox write is atomic with the DELETE FROM plan_counter_idempotency
 * call that the divergence-detection path executes next. Breaking atomicity
 * means a partial commit could lose the divergence event permanently.
 *
 * **Event ID (PR-2e codex R7 C1 + dedup Issue 1A):** caller generates the
 * `eventId` via `crypto.randomUUID()` at detection time. Used as the dedup
 * key by the downstream Postgres `plan_counter_divergence_dedup` table.
 *
 * **divergence_at_ms (PR-2e codex R7 C1):** caller captures `Date.now()` at
 * detection. Do NOT recompute in the upsert helper -- flush-time semantics
 * break the launch-watcher's 15m window contract.
 */
export function writePlanCounterDivergenceOutboxEntry(
  sql: SqlStorage,
  entry: {
    eventId: string;
    orgId: string;
    tier: string;
    divergenceAtMs: number;
    storedPeriodStart: number;
    incomingPeriodStart: number;
  },
): void {
  sql.exec(
    `INSERT INTO pg_sync_outbox_plan_divergences
       (event_id, org_id, tier, divergence_at_ms, stored_period_start, incoming_period_start, created_at, next_attempt_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, 0)`,
    entry.eventId,
    entry.orgId,
    entry.tier,
    entry.divergenceAtMs,
    entry.storedPeriodStart,
    entry.incomingPeriodStart,
    Date.now(),
  );
}

/**
 * Get entries eligible for retry: next_attempt_at <= now AND attempts < max.
 * Sorted by created_at (oldest first) for fairness.
 */
export function getRetryablePlanCounterDivergenceEntries(
  sql: SqlStorage,
  now: number,
  maxAttempts: number,
): PlanCounterDivergenceOutboxEntry[] {
  return sql.exec<PlanCounterDivergenceOutboxEntry>(
    `SELECT id, event_id AS eventId, org_id AS orgId, tier,
            divergence_at_ms AS divergenceAtMs,
            stored_period_start AS storedPeriodStart,
            incoming_period_start AS incomingPeriodStart,
            attempts, next_attempt_at AS nextAttemptAt, created_at AS createdAt
     FROM pg_sync_outbox_plan_divergences
     WHERE next_attempt_at <= ? AND attempts < ?
     ORDER BY created_at ASC`,
    now,
    maxAttempts,
  ).toArray();
}

/**
 * Delete a SPECIFIC outbox entry by its unique SQLite rowid. Called after a
 * successful Postgres upsert. Row-id scoped (not event_id scoped) for the
 * same reason as the plan-counter outbox: future schema changes may allow
 * event_id collisions across retries, and row-id is guaranteed unique.
 */
export function ackPlanCounterDivergenceEntryById(sql: SqlStorage, id: number): void {
  sql.exec("DELETE FROM pg_sync_outbox_plan_divergences WHERE id = ?", id);
}

/**
 * Mark a failed entry for retry with exponential backoff.
 * Identical schedule + semantics to plan-counter outbox's markRetryFailed.
 */
export function markPlanCounterDivergenceEntryRetryFailed(
  sql: SqlStorage,
  id: number,
  currentAttempt: number,
): void {
  const backoffIndex = Math.min(currentAttempt, BACKOFF_MS.length - 1);
  const nextAttemptAt = Date.now() + BACKOFF_MS[backoffIndex];
  sql.exec(
    "UPDATE pg_sync_outbox_plan_divergences SET attempts = attempts + 1, next_attempt_at = ? WHERE id = ?",
    nextAttemptAt,
    id,
  );
}

/**
 * Delete entries that have exceeded max attempts. Returns count of deleted rows.
 * Caller should INSERT a row into `plan_counter_sync_failures` with
 * `reason='divergence_outbox_abandoned'` so the launch-watcher surfaces the stall.
 */
export function deleteAbandonedPlanCounterDivergenceEntries(
  sql: SqlStorage,
  maxAttempts: number,
): number {
  const result = sql.exec(
    "DELETE FROM pg_sync_outbox_plan_divergences WHERE attempts >= ?",
    maxAttempts,
  );
  return result.rowsWritten;
}

/**
 * PR-2e codex R7 C2: dual-FK terminal classification.
 *
 * With the Issue 1A dedup pattern, TWO Postgres FK constraints can fire on
 * a deleted org:
 *   - `plan_counter_divergence_dedup_org_id_fkey` (dedup INSERT)
 *   - `plan_counter_divergences_org_id_fkey` (divergence INSERT)
 *
 * CASCADE on `organizations` removes the org's rows from BOTH tables, so
 * either constraint can fire depending on which INSERT runs first against
 * the deleted org. Only allowlisting one would leave outbox entries retrying
 * forever against the other until abandon -- false-red ops signal.
 *
 * **PATH-SPECIFIC BY DESIGN.** Do NOT generalize this helper. Each outbox
 * path's terminal classification matches its own constraint surface.
 *
 * Verify constraint names match drizzle-generated SQL via:
 *   rg -n 'ADD CONSTRAINT ".*org_id.*fkey"' drizzle
 */
const TERMINAL_PLAN_COUNTER_DIVERGENCE_CONSTRAINTS = new Set([
  "plan_counter_divergences_org_id_fkey",
  "plan_counter_divergence_dedup_org_id_fkey",
]);

export function isTerminalPlanCounterDivergenceFkError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { code?: unknown; constraint_name?: unknown; constraint?: unknown };
  if (e.code !== "23503") return false;
  const name = typeof e.constraint_name === "string"
    ? e.constraint_name
    : typeof e.constraint === "string" ? e.constraint : null;
  return name !== null && TERMINAL_PLAN_COUNTER_DIVERGENCE_CONSTRAINTS.has(name);
}

/**
 * Delete a divergence outbox entry permanently (TERMINAL path -- no retry).
 * Called from the DO alarm dispatch catch block when
 * `isTerminalPlanCounterDivergenceFkError(err)` returns true.
 */
export function deletePlanCounterDivergenceEntryTerminal(sql: SqlStorage, id: number): void {
  sql.exec("DELETE FROM pg_sync_outbox_plan_divergences WHERE id = ?", id);
}

/**
 * PR-2e codex R7 C4: outbox drain lag for the launch-watcher observability
 * surface. Same pattern as `computePlanCounterOutboxLagMs` -- MIN(created_at)
 * gives the oldest un-acked entry age.
 *
 * Returns `null` when the table is empty. Caller skips emit on null.
 * Clamps negative lag to 0 for clock-skew safety.
 */
export function computePlanCounterDivergenceOutboxLagMs(
  sql: SqlStorage,
  now: number,
): number | null {
  const row = sql.exec<{ oldest: number | null }>(
    "SELECT MIN(created_at) AS oldest FROM pg_sync_outbox_plan_divergences",
  ).toArray()[0];
  const oldest = row?.oldest;
  if (oldest === null || oldest === undefined) return null;
  return Math.max(0, now - oldest);
}
