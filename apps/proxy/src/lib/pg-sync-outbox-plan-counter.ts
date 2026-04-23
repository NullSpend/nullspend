/**
 * PR-2b: Transactional outbox for plan-counter (governed request count) sync.
 *
 * Mirrors `pg-sync-outbox.ts` (budget-spend path) but writes to a SEPARATE
 * table `pg_sync_outbox_plan_counter`. The separate table is a rolling-deploy
 * safety valve (per PR-2 Decision #25): during `wrangler deploy`, v_new and
 * v_old isolates coexist for ~60s. If plan-counter rows lived in the existing
 * `pg_sync_outbox`, v_old's alarm would read them, short-circuit on
 * `cost <= 0` in `updateBudgetSpend`, then DELETE them — governed-request
 * deltas would vanish every deploy. Separate table = v_old never sees them.
 *
 * Pattern: `incrementPlanCounter()` writes entries atomically inside
 * `transactionSync()`. Alarm attempts Hyperdrive write (`upsertPlanCounterPeriod`).
 * On success → `ackPlanCounterEntryById` (codex-final F1 — row-id scoped so
 * sibling rows sharing a `request_id` across periods don't get collateral-
 * deleted). On failure → `markPlanCounterEntryRetryFailed` with exponential
 * backoff. After max attempts → `deleteAbandonedPlanCounterEntries` and a
 * metric fires.
 */

import type { SqlStorage } from "./pg-sync-outbox.js";

// Backoff schedule matches budget-spend outbox — same retry characteristics
const BACKOFF_MS = [5_000, 15_000, 45_000, 120_000, 300_000];

export interface PlanCounterOutboxEntry {
  id: number;
  requestId: string;
  orgId: string;
  periodStart: number;
  periodEnd: number;
  deltaCount: number;
  attempts: number;
  nextAttemptAt: number;
  createdAt: number;
}

/**
 * Create the plan-counter outbox table + indexes. Called from DO's initSchema
 * v10 migration. Idempotent.
 */
export function createPlanCounterOutboxTable(sql: SqlStorage): void {
  sql.exec(`
    CREATE TABLE IF NOT EXISTS pg_sync_outbox_plan_counter (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      request_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      period_start INTEGER NOT NULL,
      period_end INTEGER NOT NULL,
      delta_count INTEGER NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_counter_retry_idx ON pg_sync_outbox_plan_counter(next_attempt_at, attempts);
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_counter_request_id_idx ON pg_sync_outbox_plan_counter(request_id);
    -- Build-audit F4: index for the lag-emit MIN(created_at) query so it stays
    -- O(log n) once the outbox grows past a few hundred rows under sustained
    -- write rates. Idempotent; DO-local SQLite — no migration drama.
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_counter_created_at_idx ON pg_sync_outbox_plan_counter(created_at);
  `);
}

/**
 * Write a plan-counter outbox entry. MUST be called INSIDE transactionSync()
 * so the outbox write is atomic with the plan_counter UPDATE / idempotency
 * INSERT. Breaking this invariant breaks the atomicity guarantee in §6 of
 * the PR-2b plan — a partial commit can double-count or drop a period.
 */
export function writePlanCounterOutboxEntry(
  sql: SqlStorage,
  entry: {
    requestId: string;
    orgId: string;
    periodStart: number;
    periodEnd: number;
    deltaCount: number;
  },
): void {
  sql.exec(
    `INSERT INTO pg_sync_outbox_plan_counter
       (request_id, org_id, period_start, period_end, delta_count, created_at, next_attempt_at)
     VALUES (?, ?, ?, ?, ?, ?, 0)`,
    entry.requestId,
    entry.orgId,
    entry.periodStart,
    entry.periodEnd,
    entry.deltaCount,
    Date.now(),
  );
}

/**
 * Get entries eligible for retry: next_attempt_at <= now AND attempts < max.
 * Sorted by created_at (oldest first) for fairness.
 */
export function getRetryablePlanCounterEntries(
  sql: SqlStorage,
  now: number,
  maxAttempts: number,
): PlanCounterOutboxEntry[] {
  return sql.exec<PlanCounterOutboxEntry>(
    `SELECT id, request_id AS requestId, org_id AS orgId,
            period_start AS periodStart, period_end AS periodEnd,
            delta_count AS deltaCount,
            attempts, next_attempt_at AS nextAttemptAt, created_at AS createdAt
     FROM pg_sync_outbox_plan_counter
     WHERE next_attempt_at <= ? AND attempts < ?
     ORDER BY created_at ASC`,
    now,
    maxAttempts,
  ).toArray();
}

/**
 * Delete a SPECIFIC outbox entry by its unique SQLite rowid. Called after a
 * successful Postgres upsert for THIS entry.
 *
 * **Scope (codex-final F1):** unlike budget-spend's `ackAllForRequest`, the
 * plan-counter outbox must ack by row `id`, not by `request_id`. The DO's
 * period-scoped dedup (codex G4 / R3-M1) allows the same caller-supplied
 * idempotency key to produce TWO outbox rows when a retry crosses a period
 * boundary — one for the old period, one for the new. Acking by request_id
 * would delete both rows the moment the first upsert succeeds, and if the
 * second upsert then fails, `markPlanCounterEntryRetryFailed` could not
 * find its target → silent data loss for that period's delta.
 */
export function ackPlanCounterEntryById(sql: SqlStorage, id: number): void {
  sql.exec("DELETE FROM pg_sync_outbox_plan_counter WHERE id = ?", id);
}

/**
 * Mark a failed entry for retry with exponential backoff.
 * Identical schedule + semantics to budget-spend outbox's markRetryFailed.
 */
export function markPlanCounterEntryRetryFailed(
  sql: SqlStorage,
  id: number,
  currentAttempt: number,
): void {
  const backoffIndex = Math.min(currentAttempt, BACKOFF_MS.length - 1);
  const nextAttemptAt = Date.now() + BACKOFF_MS[backoffIndex];
  sql.exec(
    "UPDATE pg_sync_outbox_plan_counter SET attempts = attempts + 1, next_attempt_at = ? WHERE id = ?",
    nextAttemptAt,
    id,
  );
}

/**
 * Delete entries that have exceeded max attempts. Returns count of deleted rows.
 */
export function deleteAbandonedPlanCounterEntries(sql: SqlStorage, maxAttempts: number): number {
  const result = sql.exec(
    "DELETE FROM pg_sync_outbox_plan_counter WHERE attempts >= ?",
    maxAttempts,
  );
  return result.rowsWritten;
}

/**
 * PR-2c plan-audit F2 + codex-round-1 H5 + codex-round-2 H5: path-specific
 * allowlist for FK-violation constraint names that indicate the org row has
 * been deleted. When a Postgres FK violation (error code `23503`) fires on
 * one of these constraints, the org cascade-dropped the referenced rows —
 * retries will never succeed, the outbox entry is TERMINAL.
 *
 * **PATH-SPECIFIC BY DESIGN.** Do NOT generalize this helper to cover other
 * outboxes. Each outbox path has its own FK constraints with distinct risk
 * profiles. Generic classification is the bug codex-round-2 H5 flagged.
 *
 * Verify exact constraint names match the drizzle-generated SQL via:
 *   rg -n 'ADD CONSTRAINT ".*org_id.*fkey"' drizzle
 * Update this Set when migrations rename or restructure FKs.
 */
const TERMINAL_PLAN_COUNTER_CONSTRAINTS = new Set([
  "org_period_usage_org_id_fkey",
  "plan_counter_sync_requests_org_id_fkey",
]);

/**
 * Classify a Postgres error as TERMINAL (org deleted) vs retryable. Returns
 * `true` only when: error has `code === "23503"` AND the FK constraint name
 * matches an allowlisted entry. Transient FK misses (replication lag), FK
 * violations on other tables (webhooks, api_keys), and any non-23503 error
 * stay on the retry path.
 *
 * **Field-name compatibility (edge-case-audit E1):** postgres.js exposes the
 * constraint via `.constraint_name` (underscore suffix — see
 * `node_modules/postgres/types/index.d.ts`). Other Postgres clients
 * (node-postgres, pg-native) use `.constraint` without underscore. Accept
 * BOTH for belt-and-braces across driver versions / direct-connection paths.
 */
export function isTerminalPlanCounterFkError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { code?: unknown; constraint_name?: unknown; constraint?: unknown };
  if (e.code !== "23503") return false;
  const name = typeof e.constraint_name === "string"
    ? e.constraint_name
    : typeof e.constraint === "string" ? e.constraint : null;
  return name !== null && TERMINAL_PLAN_COUNTER_CONSTRAINTS.has(name);
}

/**
 * Delete a plan-counter outbox entry permanently (TERMINAL path — no retry).
 * Called from the DO alarm dispatch catch block when
 * `isTerminalPlanCounterFkError(err)` returns true.
 */
export function deletePlanCounterEntryTerminal(sql: SqlStorage, id: number): void {
  sql.exec("DELETE FROM pg_sync_outbox_plan_counter WHERE id = ?", id);
}

/**
 * PR-2d (Decision #34, codex R2#2 + R3#5): outbox drain lag in milliseconds.
 *
 * `SELECT MIN(created_at)` gives the oldest un-acked entry. Success DELETEs a
 * row (→ gone from the table), retry failure bumps `attempts`/`next_attempt_at`
 * but leaves `created_at` frozen — so the MIN tracks the age of the oldest
 * still-outstanding row, which is the actual "how far behind is Postgres?"
 * signal. That's why there's NO `acked` column filter: the table represents
 * pending state by row existence alone.
 *
 * Returns `null` when the table is empty — caller skips the metric emit so a
 * quiet worker doesn't look like zero-lag (the shadow-mode alert distinguishes
 * "no signal" from "signal of zero" via metric cardinality, not value).
 *
 * Clamps negative lag to 0 — `created_at` in the future relative to `now` would
 * indicate clock skew between the DO write path and the caller. Returning a
 * negative number would be silently wrong in an alert on p99 > threshold.
 */
export function computePlanCounterOutboxLagMs(
  sql: SqlStorage,
  now: number,
): number | null {
  const row = sql.exec<{ oldest: number | null }>(
    "SELECT MIN(created_at) AS oldest FROM pg_sync_outbox_plan_counter",
  ).toArray()[0];
  const oldest = row?.oldest;
  if (oldest === null || oldest === undefined) return null;
  return Math.max(0, now - oldest);
}
