/**
 * PXY-2: Transactional outbox for Postgres budget spend sync.
 *
 * SQL helpers that operate on the DO's ctx.storage.sql (SQLite).
 * The outbox ensures failed PG writes are retried by the alarm handler.
 *
 * Pattern: reconcile() writes outbox entries atomically with spend
 * adjustment. Worker attempts PG write optimistically. On success,
 * ackAllForRequest() clears the outbox. On failure, alarm retries
 * with exponential backoff. After max attempts, entries are abandoned
 * and a metric is emitted.
 */

// Backoff schedule for alarm retries (milliseconds)
const BACKOFF_MS = [5_000, 15_000, 45_000, 120_000, 300_000];

export interface PgSyncEntry {
  id: number;
  requestId: string;
  orgId: string;
  entityType: string;
  entityId: string;
  costMicrodollars: number;
  attempts: number;
  nextAttemptAt: number;
  createdAt: number;
}

/**
 * SqlStorage interface matching Cloudflare DO's ctx.storage.sql.
 * Defined here to avoid importing cloudflare:workers in tests.
 */
export interface SqlStorage {
  exec<T = Record<string, unknown>>(query: string, ...bindings: unknown[]): {
    toArray(): T[];
    rowsWritten: number;
  };
}

/**
 * Create the outbox table. Called from initSchema() in the DO.
 * Idempotent — safe to call on every DO construction.
 */
export function createOutboxTable(sql: SqlStorage): void {
  sql.exec(`
    CREATE TABLE IF NOT EXISTS pg_sync_outbox (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      request_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      entity_type TEXT NOT NULL,
      entity_id TEXT NOT NULL,
      cost_microdollars INTEGER NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_retry_idx ON pg_sync_outbox(next_attempt_at, attempts);
    CREATE INDEX IF NOT EXISTS pg_sync_outbox_request_id_idx ON pg_sync_outbox(request_id);
  `);
}

/**
 * Write an outbox entry. Called INSIDE transactionSync() in reconcile()
 * so it's atomic with the spend adjustment.
 */
export function writeOutboxEntry(
  sql: SqlStorage,
  entry: { requestId: string; orgId: string; entityType: string; entityId: string; costMicrodollars: number },
): void {
  sql.exec(
    `INSERT INTO pg_sync_outbox (request_id, org_id, entity_type, entity_id, cost_microdollars, created_at, next_attempt_at)
     VALUES (?, ?, ?, ?, ?, ?, 0)`,
    entry.requestId,
    entry.orgId,
    entry.entityType,
    entry.entityId,
    entry.costMicrodollars,
    Date.now(),
  );
}

/**
 * Get entries eligible for retry: next_attempt_at <= now AND attempts < max.
 * Sorted by created_at (oldest first) for fairness.
 */
export function getRetryableEntries(
  sql: SqlStorage,
  now: number,
  maxAttempts: number,
): PgSyncEntry[] {
  return sql.exec<PgSyncEntry>(
    `SELECT id, request_id AS requestId, org_id AS orgId, entity_type AS entityType,
            entity_id AS entityId, cost_microdollars AS costMicrodollars,
            attempts, next_attempt_at AS nextAttemptAt, created_at AS createdAt
     FROM pg_sync_outbox
     WHERE next_attempt_at <= ? AND attempts < ?
     ORDER BY created_at ASC`,
    now,
    maxAttempts,
  ).toArray();
}

/**
 * Delete ALL outbox entries for a requestId. Called after successful PG write.
 * Covers all entities in the reservation (C5: all-or-nothing ack).
 */
export function ackAllForRequest(sql: SqlStorage, requestId: string): void {
  sql.exec("DELETE FROM pg_sync_outbox WHERE request_id = ?", requestId);
}

/**
 * Mark a failed entry for retry with exponential backoff.
 * Sets next_attempt_at based on the backoff schedule.
 */
export function markRetryFailed(sql: SqlStorage, id: number, currentAttempt: number): void {
  const backoffIndex = Math.min(currentAttempt, BACKOFF_MS.length - 1);
  const nextAttemptAt = Date.now() + BACKOFF_MS[backoffIndex];
  sql.exec(
    "UPDATE pg_sync_outbox SET attempts = attempts + 1, next_attempt_at = ? WHERE id = ?",
    nextAttemptAt,
    id,
  );
}

/**
 * Delete entries that have exceeded max attempts. Returns count of deleted rows.
 * Called from alarm handler after processing retryable entries.
 */
export function deleteAbandonedEntries(sql: SqlStorage, maxAttempts: number): number {
  const result = sql.exec(
    "DELETE FROM pg_sync_outbox WHERE attempts >= ?",
    maxAttempts,
  );
  return result.rowsWritten;
}
