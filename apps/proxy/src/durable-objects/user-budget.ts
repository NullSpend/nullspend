import { DurableObject } from "cloudflare:workers";
import {
  writeOutboxEntry,
  getRetryableEntries,
  ackAllForRequest,
  markRetryFailed,
  deleteAbandonedEntries,
} from "../lib/pg-sync-outbox.js";
import {
  createPlanCounterOutboxTable,
  writePlanCounterOutboxEntry,
  getRetryablePlanCounterEntries,
  ackPlanCounterEntryById,
  markPlanCounterEntryRetryFailed,
  deleteAbandonedPlanCounterEntries,
  deletePlanCounterEntryTerminal,
  isTerminalPlanCounterFkError,
  computePlanCounterOutboxLagMs,
} from "../lib/pg-sync-outbox-plan-counter.js";
import {
  createPlanCounterDivergenceOutboxTable,
  writePlanCounterDivergenceOutboxEntry,
  getRetryablePlanCounterDivergenceEntries,
  ackPlanCounterDivergenceEntryById,
  markPlanCounterDivergenceEntryRetryFailed,
  deleteAbandonedPlanCounterDivergenceEntries,
  deletePlanCounterDivergenceEntryTerminal,
  isTerminalPlanCounterDivergenceFkError,
  computePlanCounterDivergenceOutboxLagMs,
} from "../lib/pg-sync-outbox-plan-divergences.js";
import { updateBudgetSpend } from "../lib/budget-spend.js";
import { upsertPlanCounterPeriod } from "../lib/upsert-plan-counter.js";
import { upsertPlanCounterDivergence } from "../lib/upsert-plan-counter-divergence.js";
import { writePlanCounterSyncFailure } from "../lib/write-plan-counter-sync-failure.js";
import { getSql } from "../lib/db.js";
import type { TierLabel } from "../lib/api-key-auth.js";
import { optionalBinding } from "../lib/env.js";
import { emitMetric } from "../lib/metrics.js";
import { stampPeriodClose } from "../lib/stamp-period-close.js";

/**
 * PR-2b: emergency safety valve for `plan_counter_idempotency`. Module-scope
 * so callers / tests can pass a different bound to `compactIdempotencyTable`.
 * Normal ops: 24h TTL handles it; this only fires on pathological key reuse
 * or adversarial patterns.
 */
export const IDEMPOTENCY_EMERGENCY_BOUND = 1_000_000;

/**
 * PR-2b build-audit Finding #3: pure helper for the emergency compaction
 * path, extracted so it's directly unit-testable with a small bound + 101
 * rows. Previously the logic lived inline in `handleIdempotencyDedupPrune`
 * and C30c couldn't stub `IDEMPOTENCY_EMERGENCY_BOUND` (module binding was
 * already loaded by the time `vi.doMock` ran). Extraction also makes the
 * `seen_at` index dependency explicit — the query plan MUST use
 * `plan_counter_idempotency_seen_at_idx` or this becomes a full sort.
 *
 * Returns the number of rows removed (0 if under bound).
 */
export function compactIdempotencyTable(
  sql: import("../lib/pg-sync-outbox.js").SqlStorage,
  bound: number = IDEMPOTENCY_EMERGENCY_BOUND,
): { removed: number; remaining: number } {
  const cnt = sql.exec<{ cnt: number }>(
    "SELECT COUNT(*) as cnt FROM plan_counter_idempotency",
  ).toArray()[0]?.cnt ?? 0;
  if (cnt <= bound) return { removed: 0, remaining: cnt };

  const targetRemoved = Math.floor(cnt / 2);
  sql.exec(
    `DELETE FROM plan_counter_idempotency WHERE key IN (
       SELECT key FROM plan_counter_idempotency
       ORDER BY seen_at ASC
       LIMIT ?
     )`,
    targetRemoved,
  );
  return { removed: targetRemoved, remaining: cnt - targetRemoved };
}

/**
 * PR-2b: incrementPlanCounter return type. Discriminated union (per codex G3)
 * so callers don't mistake a null-org no-op for a successful increment to 0.
 * PR-2c's orchestrator + PR-2d's internal endpoint handle `skipped` by letting
 * the request through without counter gating (debug/test paths).
 *
 * `skipped.reason = "non_uuid_org"` added per build-audit Finding #2 — the
 * upstream DO routing key (`idFromName(ownerId)`) may legitimately be a
 * userId (Free-tier / self-hosted fallback) for the budget-spend path. For
 * plan-counter we MUST route on a UUID orgId because `upsertPlanCounterPeriod`
 * casts `::uuid` against `org_period_usage.org_id`. Returning skipped early
 * prevents a retry-loop that would eventually abandon the entry silently.
 */
export type IncrementPlanCounterResult =
  | { status: "approved"; count: number }
  | { status: "denied"; count: number; blockAt: number }
  | { status: "skipped"; reason: "null_org" | "non_uuid_org" | "idempotency_key_too_long" };

/**
 * PR-2c codex-round-1 M8 / D1 flipped: options-object signature for the DO
 * RPC. `orgId` is NOT here (DO derives from `this.ctx.id.name`). `tier` IS
 * here so shadow-mode metric tags carry it without a second RPC round-trip.
 */
export interface IncrementPlanCounterArgs {
  planLimitBlockAt: number | null;
  planLimitMode: "hard" | "soft";
  tier: TierLabel;
  periodStart: number;
  periodEnd: number;
  idempotencyKey?: string;
}

/**
 * PR-2b build-audit Finding #2: shared UUID validator for the DO-entry guard.
 * Matches the canonical 8-4-4-4-12 lowercase hex form that Postgres UUID
 * columns accept. Exported so tests can assert on it directly.
 */
export const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * PR-2b edge-case audit EC2: length cap on caller-supplied idempotency keys.
 * Stops a malicious or buggy SDK caller from bloating DO SQLite (each key
 * becomes a PRIMARY KEY row in `plan_counter_idempotency`) and the downstream
 * outbox `request_id` + Postgres `plan_counter_sync_requests.request_id`.
 * 256 chars fits UUID v4 / request-id ergonomics with headroom.
 */
export const MAX_IDEMPOTENCY_KEY_LENGTH = 256;

// ── Types ──────────────────────────────────────────────────────────

export interface BudgetRow {
  entity_type: string;
  entity_id: string;
  max_budget: number;
  spend: number;
  reserved: number;
  policy: string;
  reset_interval: string | null;
  period_start: number;
  velocity_limit: number | null;
  velocity_window: number;
  velocity_cooldown: number;
  threshold_percentages: string | null;
  session_limit: number | null;
  finalization_reserve: number;
  avg_recent_cost: number;
  last_alerted_threshold: number;
  loop_max_calls: number | null;
  loop_window_seconds: number | null;
  loop_aggregate_max_keys: number | null;
}

export interface CheckedEntity {
  entityType: string;
  entityId: string;
  maxBudget: number;
  spend: number;
  /**
   * Live reserved amount from the DO's budgets table — sum of all in-flight
   * reservations against this entity at check time. Used by the proxy routes
   * to compute accurate `X-NullSpend-Budget-Remaining` headers under
   * concurrent load (without this, two parallel requests would each see
   * the other's reservation as zero).
   */
  reserved: number;
  policy: string;
  thresholdPercentages: number[];
  sessionLimit: number | null;
  finalizationReserve: number;
  avgRecentCost: number;
}

export interface CheckResult {
  status: "approved" | "denied";
  hasBudgets: boolean;
  reservationId?: string;
  deniedEntity?: string;
  remaining?: number;
  maxBudget?: number;
  spend?: number;
  periodResets?: Array<{ entityType: string; entityId: string; newPeriodStart: number }>;
  checkedEntities?: CheckedEntity[];
  velocityDenied?: boolean;
  retryAfterSeconds?: number;
  velocityDetails?: {
    limitMicrodollars: number;
    windowSeconds: number;
    currentMicrodollars: number;
  };
  velocityRecovered?: Array<{
    entityType: string;
    entityId: string;
    velocityLimitMicrodollars: number;
    velocityWindowSeconds: number;
    velocityCooldownSeconds: number;
  }>;
  sessionLimitDenied?: boolean;
  sessionId?: string;
  sessionSpend?: number;
  sessionLimit?: number;
  finalizationReserve?: number;
  loopDetected?: boolean;
  loopDetails?: {
    type: "per_key" | "aggregate";
    model: string;
    provider: string;
    callCount: number;
    windowSeconds: number;
    maxCalls: number;
  };
  /** Loop call count for warning header (set when count >= 80% of threshold) */
  loopCount?: number;
  loopMaxCalls?: number;
}

export interface ThresholdCrossing {
  entityType: string;
  entityId: string;
  maxBudget: number;
  spend: number;
  threshold: number;
  isCritical: boolean;
  requestId: string;
}

export interface ReconcileResult {
  status: "reconciled" | "not_found";
  spends?: Record<string, number>;
  budgetsMissing?: string[];
  thresholdCrossings?: ThresholdCrossing[];
}

export interface VelocityState {
  entity_key: string;
  window_size_ms: number;
  window_start_ms: number;
  current_count: number;
  current_spend: number;
  prev_count: number;
  prev_spend: number;
  tripped_at: number | null;
}

// ── Helpers ─────────────────────────────────────────────────────────

/** Compute the start of the current budget period. */
export function currentPeriodStart(
  interval: string,
  periodStart: number,
  now: number,
): number {
  let start = periodStart;
  const msPerDay = 86_400_000;

  // Fast path for daily/weekly (fixed intervals)
  if (interval === "daily" || interval === "weekly") {
    const step = interval === "daily" ? msPerDay : 7 * msPerDay;
    while (start + step <= now) {
      start += step;
    }
    return start;
  }

  // Month-accurate for monthly
  if (interval === "monthly") {
    const d = new Date(start);
    while (true) {
      const next = new Date(d);
      next.setUTCMonth(next.getUTCMonth() + 1);
      if (next.getTime() > now) break;
      d.setUTCMonth(d.getUTCMonth() + 1);
    }
    return d.getTime();
  }

  // Year-accurate for yearly
  if (interval === "yearly") {
    const d = new Date(start);
    while (true) {
      const next = new Date(d);
      next.setUTCFullYear(next.getUTCFullYear() + 1);
      if (next.getTime() > now) break;
      d.setUTCFullYear(d.getUTCFullYear() + 1);
    }
    return d.getTime();
  }

  return start;
}

/** Parse an entity key safely (handles IDs containing colons). */
function parseEntityKey(key: string): [string, string] {
  const sep = key.indexOf(":");
  if (sep <= 0) throw new Error(`Invalid entity key (missing separator): ${key}`);
  return [key.slice(0, sep), key.slice(sep + 1)];
}

const DEFAULT_THRESHOLDS: readonly number[] = Object.freeze([50, 80, 90, 95]);

/** Safely parse threshold_percentages JSON TEXT from SQLite. */
export function parseThresholds(raw: string | null): number[] {
  if (!raw) return [...DEFAULT_THRESHOLDS];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((v: unknown) => typeof v === "number")) {
      // Deduplicate and sort to prevent duplicate events within a single reconcile
      return [...new Set(parsed as number[])].sort((a, b) => a - b);
    }
    return [...DEFAULT_THRESHOLDS];
  } catch {
    return [...DEFAULT_THRESHOLDS];
  }
}

/**
 * AUDIT-7 orphan sweep: compute which DO budget rows have no matching Postgres row
 * and are old enough to safely evict.
 *
 * A DO row is an orphan iff:
 *   - No row in `pgRows` matches its (entity_type, entity_id) pair
 *   - Its `synced_at` is older than `now - safetyMs` (guards against a Postgres
 *     commit-visibility race where a newly populated DO row could be misread
 *     as deleted upstream)
 *
 * Pure function — no side effects. The caller owns DELETE + metric emission.
 */
export function findOrphanedBudgets(
  doRows: ReadonlyArray<{ entity_type: string; entity_id: string; synced_at: number }>,
  pgRows: ReadonlyArray<{ entity_type: string; entity_id: string }>,
  now: number,
  safetyMs: number = 60_000,
): Array<{ entity_type: string; entity_id: string }> {
  const pgSet = new Set(pgRows.map((r) => `${r.entity_type}:${r.entity_id}`));
  const cutoff = now - safetyMs;
  const orphans: Array<{ entity_type: string; entity_id: string }> = [];
  for (const row of doRows) {
    if (pgSet.has(`${row.entity_type}:${row.entity_id}`)) continue;
    if (row.synced_at >= cutoff) continue;
    orphans.push({ entity_type: row.entity_type, entity_id: row.entity_id });
  }
  return orphans;
}

// ── Durable Object ──────────────────────────────────────────────────

/** Cached loop denial: stores the original details so cached responses are accurate. */
interface CachedLoopDenial {
  expiry: number;
  details: NonNullable<CheckResult["loopDetails"]>;
}

export class UserBudgetDO extends DurableObject {
  /** Loop denial backoff cache: key+hash → expiry + original details. Prevents the
   *  denial path from becoming a hot loop when a stuck agent retries after a 429.
   *  Lazily evicted on each check; hard cap of 1000 entries. */
  private loopDenialCache = new Map<string, CachedLoopDenial>();
  private static readonly LOOP_DENIAL_BACKOFF_MS = 5_000;
  private static readonly LOOP_DENIAL_CACHE_MAX = 1_000;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.initSchema();
      const count = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM budgets",
      ).toArray()[0]?.cnt ?? 0;
      console.log(`[UserBudgetDO] initialized, ${count} budgets loaded`);

      // PXY-2: Schedule alarm if pending outbox entries exist (cold rehydration).
      // PR-2b: include plan-counter outbox so a cold DO with only plan-counter
      // work still wakes for dispatch.
      const pendingOutbox = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM pg_sync_outbox",
      ).toArray()[0]?.cnt ?? 0;
      const pendingPlanCounterOutbox = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM pg_sync_outbox_plan_counter",
      ).toArray()[0]?.cnt ?? 0;
      // PR-2e F2-partial (Issue 3): include divergence outbox pending count so a
      // cold DO with only stalled divergence rows still wakes for dispatch.
      // Per codex R7 C4: divergence outbox stall is a launch-safety signal —
      // leaving it untracked means a silent visibility failure after a deploy.
      //
      // F2-partial codex-diff review (2026-04-19): no try/catch here. `initSchema()`
      // above runs the v12 migration BEFORE this COUNT executes (both are inside
      // the same `blockConcurrencyWhile`), so the table is guaranteed to exist.
      // The original guard was a phantom race — it would have masked real SQLite
      // corruption or migration bugs behind a silent 0 count.
      const pendingPlanCounterDivergenceOutbox = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM pg_sync_outbox_plan_divergences",
      ).toArray()[0]?.cnt ?? 0;
      // PR-6a edge-case E6: retained expired plan_counter rows also need a
      // cold-start wake. Previously only outboxes triggered the reconstruction
      // alarm, so a DO evicted between period-end and the first stampPeriodClose
      // attempt would stay silent until unrelated activity woke it. Now
      // reconstruction schedules the alarm for ANY work the alarm handler
      // cares about, including retained plan_counter rows — which closes the
      // pre-first-wake-after-eviction window the audit flagged.
      const retainedExpiredPlanCounterRows = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM plan_counter WHERE period_end < ?",
        Date.now(),
      ).toArray()[0]?.cnt ?? 0;
      const totalPending =
        pendingOutbox +
        pendingPlanCounterOutbox +
        pendingPlanCounterDivergenceOutbox +
        retainedExpiredPlanCounterRows;
      if (totalPending > 0) {
        const currentAlarm = await this.ctx.storage.getAlarm();
        if (!currentAlarm) {
          await this.ctx.storage.setAlarm(Date.now() + 1_000);
          console.log(`[UserBudgetDO] scheduled alarm for ${totalPending} pending entries (budget-spend=${pendingOutbox}, plan-counter=${pendingPlanCounterOutbox}, divergence=${pendingPlanCounterDivergenceOutbox}, expired-plan-counter=${retainedExpiredPlanCounterRows})`);
        }
      }
    });
  }

  private initSchema(): void {
    // v1 schema
    this.ctx.storage.sql.exec(`
      CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS budgets (
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        max_budget INTEGER NOT NULL DEFAULT 0,
        spend INTEGER NOT NULL DEFAULT 0,
        reserved INTEGER NOT NULL DEFAULT 0,
        policy TEXT NOT NULL DEFAULT 'strict_block',
        reset_interval TEXT,
        period_start INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (entity_type, entity_id)
      );
      CREATE TABLE IF NOT EXISTS reservations (
        id TEXT PRIMARY KEY,
        amount INTEGER NOT NULL,
        entity_keys TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL
      );
      INSERT OR IGNORE INTO _schema_version(version) VALUES (1);
    `);

    // Read schema version once — cascade through all applicable migrations
    const version = this.ctx.storage.sql.exec<{ version: number }>(
      "SELECT MAX(version) as version FROM _schema_version",
    ).toArray()[0]?.version ?? 1;

    // v2 migration: velocity limits
    if (version < 2) {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS velocity_state (
          entity_key TEXT PRIMARY KEY,
          window_size_ms INTEGER NOT NULL,
          window_start_ms INTEGER NOT NULL,
          current_count INTEGER NOT NULL DEFAULT 0,
          current_spend INTEGER NOT NULL DEFAULT 0,
          prev_count INTEGER NOT NULL DEFAULT 0,
          prev_spend INTEGER NOT NULL DEFAULT 0,
          tripped_at INTEGER
        );
      `);
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN velocity_limit INTEGER"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN velocity_window INTEGER DEFAULT 60000"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN velocity_cooldown INTEGER DEFAULT 60000"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (2)");
    }

    // v3 migration: configurable budget thresholds
    if (version < 3) {
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN threshold_percentages TEXT DEFAULT '[50,80,90,95]'"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (3)");
    }

    // v4 migration: session limits
    if (version < 4) {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS session_spend (
          entity_key TEXT NOT NULL,
          session_id TEXT NOT NULL,
          spend INTEGER NOT NULL DEFAULT 0,
          request_count INTEGER NOT NULL DEFAULT 0,
          last_seen INTEGER NOT NULL,
          PRIMARY KEY (entity_key, session_id)
        );
        CREATE INDEX IF NOT EXISTS session_spend_last_seen_idx ON session_spend(last_seen);
      `);
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN session_limit INTEGER"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE reservations ADD COLUMN session_id TEXT"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (4)");
    }

    // v5 migration: PXY-2 — self-describing reservations + PG sync outbox
    if (version < 5) {
      // Reservation enrichment — carries org context for outbox writes
      try { this.ctx.storage.sql.exec("ALTER TABLE reservations ADD COLUMN org_id TEXT"); } catch { /* already exists */ }

      // Transactional outbox for Postgres budget spend sync
      this.ctx.storage.sql.exec(`
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
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (5)");
    }

    // v6 migration: finalization reserve + avg recent cost for Requests-Remaining header
    if (version < 6) {
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN finalization_reserve INTEGER NOT NULL DEFAULT 0"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN avg_recent_cost INTEGER NOT NULL DEFAULT 0"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (6)");
    }

    // v7 migration: threshold alert dedup — tracks highest threshold % already alerted per entity
    if (version < 7) {
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN last_alerted_threshold INTEGER NOT NULL DEFAULT 0"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (7)");
    }

    // v8 migration: loop detection call log + budget config columns
    if (version < 8) {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS loop_call_log (
          key TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_loop_key_hash ON loop_call_log(key, content_hash);
        CREATE INDEX IF NOT EXISTS idx_loop_ts ON loop_call_log(ts);
      `);
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN loop_max_calls INTEGER"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN loop_window_seconds INTEGER"); } catch { /* already exists */ }
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN loop_aggregate_max_keys INTEGER"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (8)");
    }

    // v9 migration: AUDIT-7 orphan sweep — track last-synced-at for race-safe eviction
    if (version < 9) {
      try { this.ctx.storage.sql.exec("ALTER TABLE budgets ADD COLUMN synced_at INTEGER NOT NULL DEFAULT 0"); } catch { /* already exists */ }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (9)");
    }

    // v10 migration (PR-2b): plan-counter tables + period-scoped idempotency + dedicated outbox.
    // - `plan_counter`: 1-row-at-a-time invariant (lazy period reset in incrementPlanCounter).
    // - `plan_counter_idempotency`: period-scoped dedup per Decision #39. seen_at index
    //   makes TTL prune + emergency compaction feasible at 1M+ rows (R3-M2).
    // - `pg_sync_outbox_plan_counter`: dedicated table per Decision #25 — separate from
    //   `pg_sync_outbox` so v_old isolates during rolling deploy don't accidentally
    //   delete governed-request deltas via `updateBudgetSpend`'s `cost <= 0` short-circuit.
    if (version < 10) {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS plan_counter (
          period_start INTEGER PRIMARY KEY,
          period_end INTEGER NOT NULL,
          count INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plan_counter_idempotency (
          key TEXT PRIMARY KEY,
          period_start INTEGER NOT NULL,
          seen_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS plan_counter_idempotency_seen_at_idx ON plan_counter_idempotency(seen_at);
      `);
      createPlanCounterOutboxTable(this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage);
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (10)");
    }

    // v11 migration (PR-2c codex-round-3 C1 + codex-round-4 M1): persist the
    // original plan-limit decision per idempotency key so replay returns the
    // exact outcome of the first attempt (not a recompute from live counter).
    //
    // Columns are NULLABLE (NO default) — codex-round-4 M1: during rolling
    // deploy, a v_old isolate can still INSERT using the legacy 3-column shape.
    // NULL in any of the new columns is a SENTINEL for "pre-migration or
    // v_old-write row, do not trust". Replay path treats that as fresh and
    // falls through, overwriting the row with the full v_new shape.
    // codex-final review: ALTER TABLE ADD COLUMN is NOT idempotent on re-entry
    // — SQLite throws `duplicate column` if a column already exists. Without
    // a reentrancy guard, a DO that dies between the first ALTER and the
    // `_schema_version` INSERT would brick on next startup. We probe the
    // table's columns via `PRAGMA table_info` and add only the missing ones.
    // Final `INSERT OR IGNORE` records the migration as complete regardless
    // of how many columns were added this run.
    if (version < 11) {
      const existing = this.ctx.storage.sql.exec<{ name: string }>(
        "PRAGMA table_info(plan_counter_idempotency)",
      ).toArray().map((c) => c.name);
      if (!existing.includes("status")) {
        this.ctx.storage.sql.exec("ALTER TABLE plan_counter_idempotency ADD COLUMN status TEXT");
      }
      if (!existing.includes("decision_count")) {
        this.ctx.storage.sql.exec("ALTER TABLE plan_counter_idempotency ADD COLUMN decision_count INTEGER");
      }
      if (!existing.includes("decision_block_at")) {
        this.ctx.storage.sql.exec("ALTER TABLE plan_counter_idempotency ADD COLUMN decision_block_at INTEGER");
      }
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (11)");
    }

    // v12 migration (PR-2e F2-partial): divergence outbox. Captures
    // `plan_counter_period_divergence` events in a durable SQLite outbox
    // that drains to `plan_counter_divergences` Postgres via the alarm
    // handler. Without this, divergence events are emitMetric-only and the
    // launch-watcher can false-green on a known double-count vector (per
    // codex-adversarial-review 2026-04-19 R6 finding #2 / R7 C1/C2/C4).
    //
    // Helper is idempotent (CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS).
    if (version < 12) {
      createPlanCounterDivergenceOutboxTable(this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage);
      this.ctx.storage.sql.exec("INSERT OR IGNORE INTO _schema_version(version) VALUES (12)");
    }
  }

  // ── RPC Methods ────────────────────────────────────────────────────

  /**
   * Atomic budget check + reservation.
   * Queries SQLite for matching budgets (user-level + keyId-specific api_key).
   * Handles inline period resets. Only strict_block denies.
   */
  async checkAndReserve(
    keyId: string | null,
    estimateMicrodollars: number,
    reservationTtlMs: number = 30_000,
    sessionId: string | null = null,
    tagEntityIds: string[] = [],
    orgId: string | null = null,
    finalize: boolean = false,
    loopContext: { provider: string; model: string; contentHash: string } | null = null,
  ): Promise<CheckResult> {
    if (estimateMicrodollars < 0 || !Number.isFinite(estimateMicrodollars)) {
      return { status: "denied", hasBudgets: false };
    }

    const reservationId = crypto.randomUUID();
    const now = Date.now();

    let result: CheckResult = { status: "approved", hasBudgets: false };
    let reserved = false;
    let finalizeZoneDenied = false; // CX-1: tracks when finalize=true was denied because entity wasn't in zone
    let loopCallCount = 0;
    let loopMaxCallsForWarning = 0;
    let loopDenialCached = false;
    const periodResets: Array<{ entityType: string; entityId: string; newPeriodStart: number }> = [];
    const checkedEntities: CheckedEntity[] = [];
    const velocityRecovered: Array<{
      entityType: string;
      entityId: string;
      velocityLimitMicrodollars: number;
      velocityWindowSeconds: number;
      velocityCooldownSeconds: number;
    }> = [];

    this.ctx.storage.transactionSync(() => {
      // Phase 1: Query matching budgets from SQLite
      let query = "SELECT * FROM budgets WHERE entity_type = 'user'";
      const params: unknown[] = [];

      if (keyId) {
        query += " OR (entity_type = 'api_key' AND entity_id = ?)";
        params.push(keyId);
      }

      if (tagEntityIds.length > 0) {
        const placeholders = tagEntityIds.map(() => "?").join(",");
        query += ` OR (entity_type = 'tag' AND entity_id IN (${placeholders}))`;
        params.push(...tagEntityIds);
      }

      // Customer budget: extract customer ID from auto-injected "customer=<id>" tag entry
      const customerTag = tagEntityIds.find((t) => t.startsWith("customer="));
      if (customerTag) {
        const customerId = customerTag.slice("customer=".length);
        if (customerId) {
          query += " OR (entity_type = 'customer' AND entity_id = ?)";
          params.push(customerId);
        }
      }

      const rows: BudgetRow[] = this.ctx.storage.sql
        .exec<BudgetRow>(query, ...params)
        .toArray();

      if (rows.length === 0) {
        result = { status: "approved", hasBudgets: false };
        return;
      }

      // Phase 1.5: Period resets + collect checkedEntities
      for (const row of rows) {
        if (row.reset_interval && row.period_start > 0) {
          const newPeriodStart = currentPeriodStart(
            row.reset_interval,
            row.period_start,
            now,
          );
          if (newPeriodStart > row.period_start) {
            this.ctx.storage.sql.exec(
              `UPDATE budgets SET spend = 0, reserved = 0, last_alerted_threshold = 0, period_start = ?
               WHERE entity_type = ? AND entity_id = ?`,
              newPeriodStart,
              row.entity_type,
              row.entity_id,
            );
            row.spend = 0;
            row.reserved = 0;
            row.period_start = newPeriodStart;
            periodResets.push({ entityType: row.entity_type, entityId: row.entity_id, newPeriodStart });
          }
        }

        checkedEntities.push({
          entityType: row.entity_type,
          entityId: row.entity_id,
          maxBudget: row.max_budget,
          spend: row.spend,
          reserved: row.reserved,
          policy: row.policy,
          thresholdPercentages: parseThresholds(row.threshold_percentages),
          sessionLimit: row.session_limit ?? null,
          finalizationReserve: row.finalization_reserve ?? 0,
          avgRecentCost: row.avg_recent_cost ?? 0,
        });
      }

      // ── Phase 0: Velocity check (before budget check) ──────────────
      // Velocity increments are deferred to after Phase 2 (budget check)
      // to avoid phantom spend from budget-denied requests.
      interface VelocityIncrement {
        entityKey: string;
        windowMs: number;
        windowStart: number;
        prevCount: number;
        prevSpend: number;
        currCount: number;
        currSpend: number;
      }
      const pendingVelocityIncrements: VelocityIncrement[] = [];

      for (const row of rows) {
        if (row.velocity_limit == null) continue;

        const entityKey = `${row.entity_type}:${row.entity_id}`;
        const windowMs = row.velocity_window ?? 60_000;
        const cooldownMs = row.velocity_cooldown ?? 60_000;

        // Read velocity state
        const vs = this.ctx.storage.sql.exec<VelocityState>(
          "SELECT * FROM velocity_state WHERE entity_key = ?", entityKey,
        ).toArray()[0];

        // Circuit breaker: if tripped and still in cooldown, fast-deny
        if (vs?.tripped_at && (now - vs.tripped_at < cooldownMs)) {
          result = {
            status: "denied", hasBudgets: true,
            velocityDenied: true, deniedEntity: entityKey,
            retryAfterSeconds: Math.ceil((vs.tripped_at + cooldownMs - now) / 1000),
          };
          return; // exit transactionSync
        }

        // If circuit breaker expired, clear it and reset counters so the
        // agent gets a fresh window to prove it's no longer looping.
        // Skip the velocity check for this entity on the recovery request
        // (counters are zeroed — first post-recovery request always passes).
        if (vs?.tripped_at) {
          this.ctx.storage.sql.exec(
            `UPDATE velocity_state SET tripped_at = NULL,
              current_count = 0, current_spend = 0,
              prev_count = 0, prev_spend = 0,
              window_start_ms = ?
            WHERE entity_key = ?`, now, entityKey,
          );
          velocityRecovered.push({
            entityType: row.entity_type,
            entityId: row.entity_id,
            velocityLimitMicrodollars: row.velocity_limit!,
            velocityWindowSeconds: Math.round((row.velocity_window ?? 60_000) / 1000),
            velocityCooldownSeconds: Math.round((row.velocity_cooldown ?? 60_000) / 1000),
          });
          // Defer increment with fresh counters
          pendingVelocityIncrements.push({
            entityKey, windowMs, windowStart: now,
            prevCount: 0, prevSpend: 0, currCount: 0, currSpend: 0,
          });
          continue; // skip sliding window check — fresh start
        }

        if (!vs) {
          // Auto-initialize velocity_state so enforcement starts immediately
          this.ctx.storage.sql.exec(
            `INSERT OR IGNORE INTO velocity_state (entity_key, window_size_ms, window_start_ms)
             VALUES (?, ?, ?)`,
            entityKey, windowMs, now,
          );
          pendingVelocityIncrements.push({
            entityKey, windowMs, windowStart: now,
            prevCount: 0, prevSpend: 0, currCount: 0, currSpend: 0,
          });
          continue;
        }

        // Sliding window counter
        let windowStart = vs.window_start_ms;
        let prevCount = vs.prev_count, prevSpend = vs.prev_spend;
        let currCount = vs.current_count, currSpend = vs.current_spend;

        // Window rotation
        if (now >= windowStart + windowMs) {
          const newWindowStart = now - (now % windowMs);
          if (newWindowStart > windowStart + windowMs) {
            // More than 1 window elapsed — prev window is also stale
            prevCount = 0; prevSpend = 0;
          } else {
            prevCount = currCount; prevSpend = currSpend;
          }
          currCount = 0; currSpend = 0;
          windowStart = newWindowStart;
        }

        // Weighted estimation
        const elapsed = now - windowStart;
        const weight = Math.max(0, (windowMs - elapsed) / windowMs);
        const estimatedSpend = prevSpend * weight + currSpend;

        // Check: would this request push us over?
        if (estimatedSpend + estimateMicrodollars > row.velocity_limit) {
          // Trip circuit breaker
          this.ctx.storage.sql.exec(
            "UPDATE velocity_state SET tripped_at = ? WHERE entity_key = ?", now, entityKey,
          );
          result = {
            status: "denied", hasBudgets: true,
            velocityDenied: true, deniedEntity: entityKey,
            retryAfterSeconds: Math.ceil(cooldownMs / 1000),
            velocityDetails: {
              limitMicrodollars: row.velocity_limit,
              windowSeconds: Math.round(windowMs / 1000),
              currentMicrodollars: Math.round(estimatedSpend),
            },
          };
          return;
        }

        // Queue increment (applied after budget check passes)
        pendingVelocityIncrements.push({
          entityKey, windowMs, windowStart,
          prevCount, prevSpend, currCount, currSpend,
        });
      }

      // ── Phase 1: Loop detection (after velocity, before budget) ─────
      // Default-on: loopMaxCalls null → use default 50. Set to 0 to disable.
      // Prefers user-type entity for config (EC-3 fix). INSERT is DEFERRED to
      // Phase 2.5 (after budget check passes) to prevent budget-denied requests
      // from inflating the loop counter (BUG-1 fix).
      let pendingLoopInsert: { loopKey: string; contentHash: string; ts: number } | null = null;
      if (loopContext) {
        // EC-3: prefer user-type entity for loop config (deterministic)
        const loopEntity = rows.find((r) => r.entity_type === "user") ?? rows[0];
        const maxCalls = loopEntity.loop_max_calls ?? 50;

        if (maxCalls > 0) {
          const loopKey = `${loopContext.provider}:${loopContext.model}`;
          const windowSeconds = loopEntity.loop_window_seconds ?? 60;
          const windowMs = windowSeconds * 1000;
          const cacheKey = `${loopKey}:${loopContext.contentHash}`;

          // EC-1: Lazily evict expired denial cache entries + enforce size cap
          if (this.loopDenialCache.size > 0) {
            for (const [k, v] of this.loopDenialCache) {
              if (now >= v.expiry) this.loopDenialCache.delete(k);
            }
            if (this.loopDenialCache.size > UserBudgetDO.LOOP_DENIAL_CACHE_MAX) {
              this.loopDenialCache.clear();
            }
          }

          // Denial backoff: if this key+hash was denied within 5s, skip SQL entirely
          const cached = this.loopDenialCache.get(cacheKey);
          if (cached && now < cached.expiry) {
            // EC-6: return the original detection details, not hardcoded per_key
            loopDenialCached = true;
            result = {
              status: "denied",
              hasBudgets: true,
              loopDetected: true,
              loopDetails: cached.details,
            };
            return; // exit transactionSync — cached denial, no SQL
          }

          // Prune old entries + count existing (WITHOUT inserting yet)
          this.ctx.storage.sql.exec("DELETE FROM loop_call_log WHERE ts < ?", now - windowMs);
          // Count existing + 1 (the pending insert that will happen if budget passes)
          const existingCount = this.ctx.storage.sql.exec<{ cnt: number }>(
            "SELECT COUNT(*) as cnt FROM loop_call_log WHERE key = ? AND content_hash = ?",
            loopKey, loopContext.contentHash,
          ).toArray()[0]?.cnt ?? 0;
          const loopCount = existingCount + 1; // +1 for the pending insert

          if (loopCount >= maxCalls) {
            const details: NonNullable<CheckResult["loopDetails"]> = {
              type: "per_key",
              model: loopContext.model,
              provider: loopContext.provider,
              callCount: loopCount,
              windowSeconds,
              maxCalls,
            };
            this.loopDenialCache.set(cacheKey, { expiry: now + UserBudgetDO.LOOP_DENIAL_BACKOFF_MS, details });
            console.log(
              `[UserBudgetDO] loop denied: key=${loopKey} count=${loopCount}/${maxCalls} window=${windowSeconds}s type=per_key`,
            );
            result = { status: "denied", hasBudgets: true, loopDetected: true, loopDetails: details };
            return; // exit transactionSync
          }

          // Aggregate: count distinct keys with 3+ same-content repeats
          // Scoped to the configured window to prevent stale entries from
          // the alarm's wider 2-minute prune window from triggering aggregate.
          const aggregateMaxKeys = loopEntity.loop_aggregate_max_keys ?? 5;
          if (aggregateMaxKeys > 0) {
            const windowCutoff = now - windowMs;
            const qualifyingKeys = this.ctx.storage.sql.exec<{ cnt: number }>(`
              SELECT COUNT(DISTINCT key) as cnt
              FROM loop_call_log
              WHERE ts >= ? AND key IN (
                SELECT key FROM loop_call_log
                WHERE ts >= ?
                GROUP BY key, content_hash HAVING COUNT(*) >= 3
              )
            `, windowCutoff, windowCutoff).toArray()[0]?.cnt ?? 0;

            if (qualifyingKeys >= aggregateMaxKeys) {
              const details: NonNullable<CheckResult["loopDetails"]> = {
                type: "aggregate",
                model: "aggregate",
                provider: "multiple",
                callCount: qualifyingKeys,
                windowSeconds,
                maxCalls: aggregateMaxKeys,
              };
              this.loopDenialCache.set(cacheKey, { expiry: now + UserBudgetDO.LOOP_DENIAL_BACKOFF_MS, details });
              console.log(
                `[UserBudgetDO] loop denied: qualifyingKeys=${qualifyingKeys}/${aggregateMaxKeys} window=${windowSeconds}s type=aggregate`,
              );
              result = { status: "denied", hasBudgets: true, loopDetected: true, loopDetails: details };
              return; // exit transactionSync
            }
          }

          // Defer INSERT to Phase 2.5 (only if budget check passes)
          pendingLoopInsert = { loopKey, contentHash: loopContext.contentHash, ts: now };

          // Warning: set loopCount/loopMaxCalls on result when approaching threshold (>= 80%)
          loopCallCount = loopCount;
          loopMaxCallsForWarning = maxCalls;
        }
      }

      // ── Session limit check (after velocity + loop, before budget) ──
      // P1-2: Priority order is velocity > loop > session > budget. The
      // session check lives here so that a request hitting both a velocity
      // limit and a session limit surfaces `velocity_exceeded` (the
      // higher-priority denial code) rather than `session_limit_exceeded`.
      // Same rationale for loop > session. Session denial exits before
      // budget logic runs — denied requests should not affect budget
      // counters (same invariant as velocity and budget denials).
      if (sessionId) {
        for (const row of rows) {
          if (row.session_limit == null) continue;

          const entityKey = `${row.entity_type}:${row.entity_id}`;
          const sessionRow = this.ctx.storage.sql.exec<{ spend: number }>(
            "SELECT spend FROM session_spend WHERE entity_key = ? AND session_id = ?",
            entityKey, sessionId,
          ).toArray()[0];

          const currentSessionSpend = sessionRow?.spend ?? 0;
          if (currentSessionSpend + estimateMicrodollars > row.session_limit) {
            console.log(
              `[UserBudgetDO] session denied: entity=${entityKey} session=${sessionId} spend=${currentSessionSpend} limit=${row.session_limit} estimate=${estimateMicrodollars}`,
            );
            result = {
              status: "denied",
              hasBudgets: true,
              sessionLimitDenied: true,
              deniedEntity: entityKey,
              sessionId,
              sessionSpend: currentSessionSpend,
              sessionLimit: row.session_limit,
            };
            return; // exit transactionSync
          }
        }
      }

      // Phase 2: Check each entity's budget (with finalization reserve)
      for (const row of rows) {
        const remaining = row.max_budget - row.spend - row.reserved;
        // CX-1: finalize only skips reserve subtraction when the entity is already
        // in the reserve zone (spend+reserved >= limit-reserve). Prevents callers
        // from burning through the reserve before reaching the zone.
        const reserve = row.finalization_reserve ?? 0;
        const inReserveZone = reserve > 0 && (row.spend + row.reserved) >= (row.max_budget - reserve);
        const effectiveRemaining = (finalize && inReserveZone) ? remaining : remaining - reserve;

        if (
          row.policy === "strict_block" &&
          estimateMicrodollars > effectiveRemaining
        ) {
          result = {
            status: "denied",
            hasBudgets: true,
            deniedEntity: `${row.entity_type}:${row.entity_id}`,
            remaining: effectiveRemaining,
            maxBudget: row.max_budget,
            spend: row.spend,
            finalizationReserve: row.finalization_reserve ?? 0,
          };
          // CX-1 observability: flag when finalize=true was ignored because entity wasn't in zone
          if (finalize && reserve > 0 && !inReserveZone) {
            finalizeZoneDenied = true;
          }
          console.log(
            `[UserBudgetDO] denied: entity=${row.entity_type}:${row.entity_id} remaining=${effectiveRemaining} estimate=${estimateMicrodollars} finalize=${finalize} inReserveZone=${inReserveZone}`,
          );
          return; // Exit transactionSync — no reservation made
        }
      }

      // Phase 2.5: Apply deferred loop insert + velocity increments (only reached if budget check passed)
      if (pendingLoopInsert) {
        this.ctx.storage.sql.exec(
          "INSERT INTO loop_call_log (key, content_hash, ts) VALUES (?, ?, ?)",
          pendingLoopInsert.loopKey, pendingLoopInsert.contentHash, pendingLoopInsert.ts,
        );
      }
      for (const vi of pendingVelocityIncrements) {
        this.ctx.storage.sql.exec(
          `INSERT INTO velocity_state (entity_key, window_size_ms, window_start_ms, current_count, current_spend, prev_count, prev_spend)
           VALUES (?, ?, ?, 1, ?, ?, ?)
           ON CONFLICT(entity_key) DO UPDATE SET
             window_start_ms = ?, prev_count = ?, prev_spend = ?,
             current_count = ?, current_spend = ?`,
          vi.entityKey, vi.windowMs, vi.windowStart, estimateMicrodollars, vi.prevCount, vi.prevSpend,
          vi.windowStart, vi.prevCount, vi.prevSpend,
          vi.currCount + 1, vi.currSpend + estimateMicrodollars,
        );
      }

      // Phase 3: Reserve across all entities that have budgets
      const entityKeys: string[] = [];
      for (const row of rows) {
        const key = `${row.entity_type}:${row.entity_id}`;
        this.ctx.storage.sql.exec(
          "UPDATE budgets SET reserved = reserved + ? WHERE entity_type = ? AND entity_id = ?",
          estimateMicrodollars,
          row.entity_type,
          row.entity_id,
        );
        entityKeys.push(key);
      }

      // Store reservation for crash recovery (includes session_id for alarm reversal, org_id for outbox)
      this.ctx.storage.sql.exec(
        `INSERT INTO reservations (id, amount, entity_keys, created_at, expires_at, session_id, org_id)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        reservationId,
        estimateMicrodollars,
        JSON.stringify(entityKeys),
        now,
        now + reservationTtlMs,
        sessionId,
        orgId,
      );

      // Phase 3.5: Increment session spend for entities with session limits
      if (sessionId) {
        for (const row of rows) {
          if (row.session_limit == null) continue;
          const entityKey = `${row.entity_type}:${row.entity_id}`;
          this.ctx.storage.sql.exec(
            `INSERT INTO session_spend (entity_key, session_id, spend, request_count, last_seen)
             VALUES (?, ?, ?, 1, ?)
             ON CONFLICT(entity_key, session_id) DO UPDATE SET
               spend = spend + ?,
               request_count = request_count + 1,
               last_seen = ?`,
            entityKey, sessionId, estimateMicrodollars, now,
            estimateMicrodollars, now,
          );
        }
      }

      result = { status: "approved", hasBudgets: true, reservationId };
      reserved = true;
    });

    // CX-1 observability: emit metric when finalize=true was denied because entity wasn't in reserve zone
    if (finalizeZoneDenied) {
      emitMetric("finalization_zone_denied", {});
    }

    // Attach period resets and checkedEntities to result
    if (periodResets.length > 0) {
      result.periodResets = periodResets;
    }
    if (checkedEntities.length > 0) {
      result.checkedEntities = checkedEntities;
    }
    if (velocityRecovered.length > 0) {
      result.velocityRecovered = velocityRecovered;
    }

    // Loop detection metric (outside transaction, fire-and-forget)
    if (result.loopDetected && result.loopDetails) {
      emitMetric("loop_detected", {
        type: result.loopDetails.type,
        model: result.loopDetails.model,
        provider: result.loopDetails.provider,
        callCount: result.loopDetails.callCount,
        maxCalls: result.loopDetails.maxCalls,
        cached: loopDenialCached,
      });
    }

    // Loop count warning: attach when approaching threshold (>= 80%)
    if (loopCallCount > 0 && loopMaxCallsForWarning > 0 && loopCallCount >= Math.floor(loopMaxCallsForWarning * 0.8)) {
      result.loopCount = loopCallCount;
      result.loopMaxCalls = loopMaxCallsForWarning;
    }

    // Update in-memory cache

    // Schedule alarm for reservation expiry (session cleanup piggybacks on the same alarm).
    // Retry once on failure — a missed alarm means reservations are never cleaned up,
    // permanently holding budget capacity.
    if (reserved) {
      const nextExpiry = now + reservationTtlMs;
      try {
        const currentAlarm = await this.ctx.storage.getAlarm();
        if (!currentAlarm || currentAlarm > nextExpiry) {
          await this.ctx.storage.setAlarm(nextExpiry);
        }
      } catch {
        try {
          await this.ctx.storage.setAlarm(nextExpiry);
        } catch (retryErr) {
          console.error("[UserBudgetDO] Failed to set alarm after retry:", retryErr);
        }
      }
    }

    return result;
  }

  /**
   * Settle a reservation after actual cost is known.
   * Skips spend update when actualCost is 0.
   */
  async reconcile(
    reservationId: string,
    actualCostMicrodollars: number,
  ): Promise<ReconcileResult> {
    const row = this.ctx.storage.sql
      .exec<{ amount: number; entity_keys: string; session_id: string | null; org_id: string | null }>(
        "SELECT amount, entity_keys, session_id, org_id FROM reservations WHERE id = ?",
        reservationId,
      )
      .toArray()[0];

    if (!row) {
      console.log(`[UserBudgetDO] reconcile not_found: reservationId=${reservationId}`);
      return { status: "not_found" };
    }

    const entityKeys: string[] = JSON.parse(row.entity_keys);
    const spends: Record<string, number> = {};
    const budgetsMissing: string[] = [];
    const thresholdCrossings: ThresholdCrossing[] = [];

    this.ctx.storage.transactionSync(() => {
      for (const key of entityKeys) {
        const [entityType, entityId] = parseEntityKey(key);

        if (actualCostMicrodollars > 0) {
          const rows = this.ctx.storage.sql.exec<{
            spend: number;
            max_budget: number;
            threshold_percentages: string | null;
            last_alerted_threshold: number;
          }>(
            `UPDATE budgets SET
              spend = spend + ?,
              reserved = MAX(0, reserved - ?)
             WHERE entity_type = ? AND entity_id = ?
             RETURNING spend, max_budget, threshold_percentages, last_alerted_threshold`,
            actualCostMicrodollars,
            row.amount,
            entityType,
            entityId,
          ).toArray();
          if (rows.length > 0) {
            const { spend: newSpend, max_budget, threshold_percentages, last_alerted_threshold } = rows[0];
            spends[key] = newSpend;

            // P0-1: Atomic threshold crossing detection — dedup via last_alerted_threshold
            if (max_budget > 0) {
              const newPercent = Math.floor((newSpend / max_budget) * 100);
              const thresholds = parseThresholds(threshold_percentages);
              const lastAlerted = last_alerted_threshold;
              const lastThreshold = thresholds.length > 0 ? thresholds[thresholds.length - 1] : undefined;

              let highestCrossed = lastAlerted;
              for (const t of thresholds) {
                if (t > lastAlerted && t <= newPercent) {
                  const isCritical = t === lastThreshold || t >= 90;
                  thresholdCrossings.push({
                    entityType, entityId, maxBudget: max_budget, spend: newSpend,
                    threshold: t, isCritical, requestId: reservationId,
                  });
                  highestCrossed = Math.max(highestCrossed, t);
                }
              }

              if (highestCrossed > lastAlerted) {
                this.ctx.storage.sql.exec(
                  "UPDATE budgets SET last_alerted_threshold = ? WHERE entity_type = ? AND entity_id = ?",
                  highestCrossed, entityType, entityId,
                );
              }
            }

            // EMA update for avg_recent_cost (X-NullSpend-Budget-Requests-Remaining)
            const emaValue = Math.max(0, Math.round(actualCostMicrodollars * 0.2));
            if (Number.isFinite(emaValue)) {
              this.ctx.storage.sql.exec(
                `UPDATE budgets SET avg_recent_cost = CAST(ROUND(avg_recent_cost * 0.8 + ?) AS INTEGER)
                 WHERE entity_type = ? AND entity_id = ?`,
                emaValue,
                entityType,
                entityId,
              );
            }
          } else {
            budgetsMissing.push(key);
            console.warn(
              `[UserBudgetDO] reconcile: budget missing for entity=${key}, cost=${actualCostMicrodollars} untracked`,
            );
          }
        } else {
          this.ctx.storage.sql.exec(
            `UPDATE budgets SET
              reserved = MAX(0, reserved - ?)
             WHERE entity_type = ? AND entity_id = ?`,
            row.amount,
            entityType,
            entityId,
          );
          const rows = this.ctx.storage.sql.exec<{ spend: number }>(
            "SELECT spend FROM budgets WHERE entity_type = ? AND entity_id = ?",
            entityType,
            entityId,
          ).toArray();
          if (rows.length > 0) {
            spends[key] = rows[0].spend;
          } else {
            budgetsMissing.push(key);
            console.warn(
              `[UserBudgetDO] reconcile: budget missing for entity=${key}, cost=${actualCostMicrodollars} untracked`,
            );
          }
        }
      }

      // Session spend correction — runs regardless of actualCost (handles zero-cost case)
      if (row.session_id) {
        const delta = actualCostMicrodollars - row.amount; // negative if overestimated
        if (delta !== 0) {
          for (const key of entityKeys) {
            this.ctx.storage.sql.exec(
              "UPDATE session_spend SET spend = MAX(0, spend + ?) WHERE entity_key = ? AND session_id = ?",
              delta, key, row.session_id,
            );
          }
        }
      }

      // PXY-2: Write outbox entries for PG sync (atomic with spend adjustment).
      // Outbox presence = "reconciled, PG pending". No outbox = expired.
      if (actualCostMicrodollars > 0 && row.org_id) {
        for (const key of entityKeys) {
          const [entityType, entityId] = parseEntityKey(key);
          writeOutboxEntry(this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage, {
            requestId: reservationId,
            orgId: row.org_id,
            entityType,
            entityId,
            costMicrodollars: actualCostMicrodollars,
          });
        }
      }

      this.ctx.storage.sql.exec(
        "DELETE FROM reservations WHERE id = ?",
        reservationId,
      );
    });

    // Emit metric OUTSIDE transaction (fire-and-forget)
    if (actualCostMicrodollars > 0 && row.org_id) {
      emitMetric("pg_sync_outbox_written", { requestId: reservationId, entityCount: entityKeys.length });

      // Codex #5: Ensure alarm fires soon for outbox entries (don't wait for
      // reservation expiry ~30s). Outbox entries have next_attempt_at=0.
      try {
        const current = await this.ctx.storage.getAlarm();
        const soonAlarm = Date.now() + 1_000;
        if (!current || current > soonAlarm) {
          await this.ctx.storage.setAlarm(soonAlarm);
        }
      } catch { /* best-effort — reservation alarm is the fallback */ }
    }

    // P0-1: Emit metric for threshold crossings (outside transaction, fire-and-forget)
    for (const c of thresholdCrossings) {
      emitMetric("threshold_crossing_detected", {
        entityType: c.entityType, entityId: c.entityId,
        threshold: c.threshold, isCritical: c.isCritical,
      });
    }

    const result: ReconcileResult = { status: "reconciled", spends, thresholdCrossings };
    if (budgetsMissing.length > 0) {
      result.budgetsMissing = budgetsMissing;
    }
    return result;
  }

  /**
   * PXY-2: Acknowledge successful PG sync — delete all outbox entries for a requestId.
   * Called by doBudgetReconcile after updateBudgetSpend succeeds.
   * C5: All entities for the request are acked atomically.
   */
  async ackPgSync(requestId: string): Promise<void> {
    ackAllForRequest(this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage, requestId);
  }

  /**
   * Seed or refresh a budget entity from Postgres.
   * On first insert: uses all provided values.
   * On conflict: updates max_budget, policy, reset_interval from Postgres
   * but preserves the DO's authoritative spend, reserved, and period_start.
   *
   * Note: method name is an RPC method on the DO stub — do NOT rename
   * (would break rolling deploys).
   */
  async populateIfEmpty(
    entityType: string,
    entityId: string,
    maxBudget: number,
    spend: number,
    policy: string,
    resetInterval: string | null,
    periodStart: number,
    velocityLimit: number | null = null,
    velocityWindow: number = 60_000,
    velocityCooldown: number = 60_000,
    thresholdPercentages: number[] = [...DEFAULT_THRESHOLDS],
    sessionLimit: number | null = null,
    finalizationReserve: number = 0,
    loopMaxCalls: number | null = null,
    loopWindowSeconds: number | null = null,
    loopAggregateMaxKeys: number | null = null,
  ): Promise<boolean> {
    // Check if entity already exists (for return value)
    const existed = this.ctx.storage.sql.exec<{ cnt: number }>(
      "SELECT COUNT(*) as cnt FROM budgets WHERE entity_type = ? AND entity_id = ?",
      entityType, entityId,
    ).toArray()[0]?.cnt > 0;

    this.ctx.storage.transactionSync(() => {
      // Single UPSERT with all fields — avoids separate UPDATE statements
      // synced_at stamped on both INSERT and UPDATE so the orphan sweep can
      // tell recently-synced rows apart from stale ones (AUDIT-7).
      this.ctx.storage.sql.exec(
        `INSERT INTO budgets
         (entity_type, entity_id, max_budget, spend, reserved, policy, reset_interval, period_start,
          velocity_limit, velocity_window, velocity_cooldown, threshold_percentages, session_limit,
          finalization_reserve, loop_max_calls, loop_window_seconds, loop_aggregate_max_keys, synced_at)
         VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(entity_type, entity_id) DO UPDATE SET
           max_budget = excluded.max_budget,
           policy = excluded.policy,
           reset_interval = excluded.reset_interval,
           velocity_limit = excluded.velocity_limit,
           velocity_window = excluded.velocity_window,
           velocity_cooldown = excluded.velocity_cooldown,
           threshold_percentages = excluded.threshold_percentages,
           last_alerted_threshold = CASE
             WHEN budgets.threshold_percentages != excluded.threshold_percentages THEN 0
             ELSE budgets.last_alerted_threshold END,
           session_limit = excluded.session_limit,
           finalization_reserve = excluded.finalization_reserve,
           loop_max_calls = excluded.loop_max_calls,
           loop_window_seconds = excluded.loop_window_seconds,
           loop_aggregate_max_keys = excluded.loop_aggregate_max_keys,
           synced_at = excluded.synced_at`,
        entityType,
        entityId,
        maxBudget,
        spend,
        policy,
        resetInterval,
        periodStart,
        velocityLimit,
        velocityWindow,
        velocityCooldown,
        JSON.stringify(thresholdPercentages),
        sessionLimit,
        finalizationReserve,
        loopMaxCalls,
        loopWindowSeconds,
        loopAggregateMaxKeys,
        Date.now(),
      );

      // Create/update velocity_state row
      if (velocityLimit !== null) {
        const entityKey = `${entityType}:${entityId}`;
        this.ctx.storage.sql.exec(
          `INSERT INTO velocity_state (entity_key, window_size_ms, window_start_ms)
           VALUES (?, ?, ?)
           ON CONFLICT(entity_key) DO UPDATE SET
             window_start_ms = CASE WHEN velocity_state.window_size_ms != excluded.window_size_ms
               THEN excluded.window_start_ms ELSE velocity_state.window_start_ms END,
             current_count = CASE WHEN velocity_state.window_size_ms != excluded.window_size_ms
               THEN 0 ELSE velocity_state.current_count END,
             current_spend = CASE WHEN velocity_state.window_size_ms != excluded.window_size_ms
               THEN 0 ELSE velocity_state.current_spend END,
             prev_count = CASE WHEN velocity_state.window_size_ms != excluded.window_size_ms
               THEN 0 ELSE velocity_state.prev_count END,
             prev_spend = CASE WHEN velocity_state.window_size_ms != excluded.window_size_ms
               THEN 0 ELSE velocity_state.prev_spend END,
             window_size_ms = excluded.window_size_ms`,
          entityKey, velocityWindow, Date.now(),
        );
      } else {
        this.ctx.storage.sql.exec(
          "DELETE FROM velocity_state WHERE entity_key = ?",
          `${entityType}:${entityId}`,
        );
      }
    });

    return !existed;
  }

  /** Read-only budget state (for dashboard queries or debugging). */
  async getBudgetState(): Promise<BudgetRow[]> {
    return this.ctx.storage.sql.exec<BudgetRow>(
      "SELECT entity_type, entity_id, max_budget, spend, reserved, policy, reset_interval, period_start, velocity_limit, velocity_window, velocity_cooldown, threshold_percentages, session_limit, finalization_reserve, avg_recent_cost, last_alerted_threshold, loop_max_calls, loop_window_seconds, loop_aggregate_max_keys FROM budgets",
    ).toArray();
  }

  /** Read-only velocity state (for dashboard live status). */
  async getVelocityState(): Promise<VelocityState[]> {
    return this.ctx.storage.sql.exec<VelocityState>(
      "SELECT * FROM velocity_state",
    ).toArray();
  }

  /** Read-only outbox state (for PXY-2 observability and testing). */
  async getOutboxEntries(): Promise<Array<{
    id: number;
    request_id: string;
    org_id: string;
    entity_type: string;
    entity_id: string;
    cost_microdollars: number;
    attempts: number;
    next_attempt_at: number;
    created_at: number;
  }>> {
    return this.ctx.storage.sql.exec<{
      id: number;
      request_id: string;
      org_id: string;
      entity_type: string;
      entity_id: string;
      cost_microdollars: number;
      attempts: number;
      next_attempt_at: number;
      created_at: number;
    }>(
      "SELECT id, request_id, org_id, entity_type, entity_id, cost_microdollars, attempts, next_attempt_at, created_at FROM pg_sync_outbox ORDER BY id ASC",
    ).toArray();
  }

  /** PR-2b: read-only plan-counter outbox state. */
  async getPlanCounterOutboxEntries(): Promise<Array<{
    id: number;
    request_id: string;
    org_id: string;
    period_start: number;
    period_end: number;
    delta_count: number;
    attempts: number;
    next_attempt_at: number;
    created_at: number;
  }>> {
    return this.ctx.storage.sql.exec<{
      id: number;
      request_id: string;
      org_id: string;
      period_start: number;
      period_end: number;
      delta_count: number;
      attempts: number;
      next_attempt_at: number;
      created_at: number;
    }>(
      "SELECT id, request_id, org_id, period_start, period_end, delta_count, attempts, next_attempt_at, created_at FROM pg_sync_outbox_plan_counter ORDER BY id ASC",
    ).toArray();
  }

  /** PR-2b: read-only plan_counter state (1-row-at-a-time invariant). */
  async getPlanCounterRow(): Promise<{
    period_start: number;
    period_end: number;
    count: number;
    created_at: number;
  } | null> {
    const rows = this.ctx.storage.sql.exec<{
      period_start: number;
      period_end: number;
      count: number;
      created_at: number;
    }>("SELECT * FROM plan_counter LIMIT 1").toArray();
    return rows[0] ?? null;
  }

  /** PR-2b: read-only idempotency state. */
  async getPlanCounterIdempotencyRows(): Promise<Array<{
    key: string;
    period_start: number;
    seen_at: number;
  }>> {
    return this.ctx.storage.sql.exec<{
      key: string;
      period_start: number;
      seen_at: number;
    }>(
      "SELECT key, period_start, seen_at FROM plan_counter_idempotency ORDER BY seen_at ASC",
    ).toArray();
  }

  /** Read-only reservation state (for PXY-2 observability and testing). */
  async getReservations(): Promise<Array<{
    id: string;
    amount: number;
    entity_keys: string;
    created_at: number;
    expires_at: number;
    session_id: string | null;
    org_id: string | null;
  }>> {
    return this.ctx.storage.sql.exec<{
      id: string;
      amount: number;
      entity_keys: string;
      created_at: number;
      expires_at: number;
      session_id: string | null;
      org_id: string | null;
    }>(
      "SELECT id, amount, entity_keys, created_at, expires_at, session_id, org_id FROM reservations ORDER BY created_at ASC",
    ).toArray();
  }

  /** Remove a budget entity.
   *  Called via internal invalidation endpoint.
   *  Reservations are NOT deleted — reconcile() handles missing budgets
   *  gracefully (reports budgetsMissing, skips spend), and the alarm
   *  cleans up expired reservations. This preserves spend tracking
   *  for co-covered entities in multi-entity reservations. */
  async removeBudget(entityType: string, entityId: string): Promise<void> {
    const entityKey = `${entityType}:${entityId}`;
    this.ctx.storage.transactionSync(() => {
      // Delete the budget row
      this.ctx.storage.sql.exec(
        "DELETE FROM budgets WHERE entity_type = ? AND entity_id = ?",
        entityType,
        entityId,
      );

      // Delete velocity_state row
      this.ctx.storage.sql.exec(
        "DELETE FROM velocity_state WHERE entity_key = ?",
        entityKey,
      );

      // Delete session_spend rows for this entity
      this.ctx.storage.sql.exec(
        "DELETE FROM session_spend WHERE entity_key = ?",
        entityKey,
      );
    });
  }

  /** Reset spend for a budget entity (called via internal invalidation endpoint). */
  async resetSpend(entityType: string, entityId: string): Promise<void> {
    const entityKey = `${entityType}:${entityId}`;

    this.ctx.storage.transactionSync(() => {
      // 1. Find all reservations referencing this entity
      const matching = this.ctx.storage.sql
        .exec<{ id: string; amount: number; entity_keys: string }>(
          `SELECT r.id, r.amount, r.entity_keys
           FROM reservations r, json_each(r.entity_keys) j
           WHERE j.value = ?`,
          entityKey,
        )
        .toArray();

      // 2. Decrement reserved on all co-covered entities and delete reservations
      for (const rsv of matching) {
        const keys: string[] = JSON.parse(rsv.entity_keys);
        for (const key of keys) {
          const [eType, eId] = parseEntityKey(key);
          this.ctx.storage.sql.exec(
            "UPDATE budgets SET reserved = MAX(0, reserved - ?) WHERE entity_type = ? AND entity_id = ?",
            rsv.amount,
            eType,
            eId,
          );
        }
        this.ctx.storage.sql.exec("DELETE FROM reservations WHERE id = ?", rsv.id);
      }

      // 3. Reset the target entity (spend=0, reserved=0, last_alerted_threshold=0
      //    — reserved may already be 0 from step 2, but set explicitly for clean state)
      this.ctx.storage.sql.exec(
        `UPDATE budgets SET spend = 0, reserved = 0, last_alerted_threshold = 0, period_start = ?
         WHERE entity_type = ? AND entity_id = ?`,
        Date.now(),
        entityType,
        entityId,
      );

      // 4. Clear velocity state so circuit breaker resets on manual spend reset
      this.ctx.storage.sql.exec(
        `UPDATE velocity_state SET
          tripped_at = NULL, current_count = 0, current_spend = 0,
          prev_count = 0, prev_spend = 0
        WHERE entity_key = ?`,
        entityKey,
      );

      // 5. Clear session_spend for this entity
      this.ctx.storage.sql.exec(
        "DELETE FROM session_spend WHERE entity_key = ?",
        entityKey,
      );

      // Loop call log is NOT cleared here — entries are per-DO (not per-entity)
      // and are time-pruned by the 60s window. Clearing would affect unrelated
      // entities sharing this DO. The denial backoff cache is cleared so the
      // entity can immediately resume normal operation.
    });

    // Clear loop denial backoff cache on spend reset
    this.loopDenialCache.clear();

  }

  /**
   * PR-2b: increment the governed-request counter for the owning org.
   *
   * Wrapped in `transactionSync()` so the counter UPDATE + outbox INSERT +
   * idempotency INSERT commit atomically — a throw anywhere rolls back all
   * three. Atomicity proven by test C29.
   *
   * Null-orgId guard returns `{ status: "skipped", reason: "null_org" }` —
   * `idFromString`-created DO stubs have null `ctx.id.name`, and the outbox's
   * `org_id TEXT NOT NULL` would throw inside the transaction otherwise.
   * Callers MUST handle `skipped` distinctly from `approved`.
   *
   * Period-scoped idempotency dedup (Decision #39): if a retry arrives with
   * a DIFFERENT period than the stored dedup row, treat the stored row as
   * stale — delete it and count fresh. See plan §"Period-scoped dedup
   * invariant" for the Stripe-renewal edge + rationale.
   */
  async incrementPlanCounter(args: IncrementPlanCounterArgs): Promise<IncrementPlanCounterResult> {
    const { planLimitBlockAt, planLimitMode, tier, periodStart, periodEnd, idempotencyKey } = args;
    const now = Date.now();

    const orgId = this.ctx.id.name;
    if (!orgId) {
      emitMetric("plan_counter_skipped_null_org", {});
      return { status: "skipped", reason: "null_org" };
    }

    // Build-audit Finding #2: guard against non-UUID DO names. The existing
    // budget-spend path keys on `orgId ?? userId`, so a Free-tier / self-hosted
    // caller may legitimately route to a DO whose name is a userId. Trying to
    // upsert that into `org_period_usage(org_id UUID)` throws at Postgres level,
    // loops through MAX_ATTEMPTS, and silently abandons the entry. Fail fast
    // with a distinct reason so PR-2c / PR-2d can treat it as a pass-through.
    if (!UUID_RE.test(orgId)) {
      emitMetric("plan_counter_skipped_non_uuid_org", {});
      return { status: "skipped", reason: "non_uuid_org" };
    }

    // Edge-case EC2: cap idempotency key length at the DO boundary. Stops a
    // buggy/malicious SDK caller from writing multi-MB keys into DO SQLite.
    if (idempotencyKey !== undefined && idempotencyKey.length > MAX_IDEMPOTENCY_KEY_LENGTH) {
      emitMetric("plan_counter_skipped_idempotency_key_too_long", { orgId });
      return { status: "skipped", reason: "idempotency_key_too_long" };
    }

    // Period invariant (Decision #38 / codex N14): inverted/zero bounds → fall-soft
    // to calendar month. Metric emission uses unified name across proxy + DO so
    // `plan_counter_period_fallback{reason}` is a single queryable series.
    let effectivePeriodStart = periodStart;
    let effectivePeriodEnd = periodEnd;
    if (!(effectivePeriodEnd > effectivePeriodStart)) {
      emitMetric("plan_counter_period_fallback", { reason: "paid_inverted", orgId });
      const d = new Date(now);
      effectivePeriodStart = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
      effectivePeriodEnd = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1);
    }

    const sqlStorage = this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage;
    let result: IncrementPlanCounterResult | undefined;

    this.ctx.storage.transactionSync(() => {
      // (1) Period-scoped idempotency (Decision #39 + PR-2c codex-round-3 C1
      //     persisted decision replay). Read all 5 columns; NULL sentinel on
      //     status/decision_count → pre-migration or v_old-write row, treat
      //     as fresh (codex-round-4 M1).
      if (idempotencyKey) {
        const seen = this.ctx.storage.sql.exec<{
          period_start: number;
          status: string | null;
          decision_count: number | null;
          decision_block_at: number | null;
        }>(
          "SELECT period_start, status, decision_count, decision_block_at FROM plan_counter_idempotency WHERE key = ?",
          idempotencyKey,
        ).toArray()[0];
        if (seen) {
          if (seen.period_start !== effectivePeriodStart) {
            // Stale key from prior period — delete + fall through to fresh increment.
            //
            // Edge-audit E1: emit a divergence metric BEFORE the delete so ops can
            // chart subscription-renewal-driven over-counts. The intended use of this
            // path is the SDK Stripe-renewal retry case (Decision #39 / plan
            // §"Period-scoped dedup invariant"). Sustained rate > 0 in production
            // signals the F1 partial-success window crossing a billing boundary —
            // counter increments BOTH for the stored period (live-path) AND the
            // incoming period (cron replay), inflating usage by 1 per occurrence.
            // Tags include the period delta so operators can size the drift impact.
            emitMetric("plan_counter_period_divergence", {
              tier,
              orgId,
              storedPeriodStart: seen.period_start,
              incomingPeriodStart: effectivePeriodStart,
            });
            // PR-2e F2-partial (codex R7 C1/Issue 1A): write a durable outbox
            // entry for this divergence event. The emitMetric above goes only
            // to AE/console; the launch-watcher queries Postgres via
            // `/api/internal/metrics-summary` so we need a PG-side row. This
            // outbox write is INSIDE the same transactionSync as the DELETE
            // below — atomic per Decision #30.
            //
            // `eventId` (dedup key) and `divergenceAtMs` (event-time, NOT
            // flush-time) are captured HERE so retry-dup protection works and
            // the watcher's 15m window is aligned with when the event actually
            // happened rather than when the outbox drained (codex R7 C1).
            writePlanCounterDivergenceOutboxEntry(
              this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage,
              {
                eventId: crypto.randomUUID(),
                orgId,
                tier,
                divergenceAtMs: Date.now(),
                storedPeriodStart: seen.period_start,
                incomingPeriodStart: effectivePeriodStart,
              },
            );
            this.ctx.storage.sql.exec(
              "DELETE FROM plan_counter_idempotency WHERE key = ?",
              idempotencyKey,
            );
          } else if (seen.status === null || seen.decision_count === null) {
            // Pre-migration row (PR-2b shape) OR v_old-write row during rolling deploy.
            // Decision fields never populated — can't trust for replay. Delete + fall
            // through so fresh path writes a proper v_new-shape row.
            emitMetric("plan_counter_idempotency_pre_v11_upgrade", {});
            this.ctx.storage.sql.exec(
              "DELETE FROM plan_counter_idempotency WHERE key = ?",
              idempotencyKey,
            );
          } else {
            // Full v_new decision present — replay verbatim. Do NOT recompute from
            // live counter (codex-round-3 C1: same key must produce same outcome).
            // Do NOT emit plan_limit_would_block (codex-round-3 M4: fires only on
            // fresh increments, not on replay).
            if (seen.status === "denied") {
              // edge-case-audit E4: avoid non-null assertion on decision_block_at.
              // Write path invariant guarantees it's set on denied rows, but we
              // fall back to 0 to avoid a surprise runtime cast if the invariant
              // is ever violated (admin SQL UPDATE, future bug path, etc).
              result = { status: "denied", count: seen.decision_count, blockAt: seen.decision_block_at ?? 0 };
            } else {
              result = { status: "approved", count: seen.decision_count };
            }
            emitMetric("plan_counter_idempotency_hit", { tier, replayed_status: result.status });
            return; // exit transactionSync — replay path, no writes
          }
        }
      }

      // (2) Period boundary — lazy reset if caller's bounds differ from stored row.
      //
      // Per edge-case audit EC1: the OLD plan_counter row is a LOCAL enforcement
      // cache; each individual increment already wrote a `deltaCount: 1` outbox
      // entry with the correct `period_start` captured at write time. Postgres
      // therefore already has the per-period count via the per-increment entries.
      // We used to additionally flush `deltaCount: old.count` here, which added
      // a second copy of the accumulated count on top of the already-synced
      // +1 stream → 2x over-count at every boundary cross. The reset is now
      // purely local: delete the stale row and insert a fresh one for the new
      // period. The alarm-driven boundary reset in `handlePlanCounterBoundaryFlush`
      // follows the same rule.
      const current = this.ctx.storage.sql.exec<{ period_start: number; count: number }>(
        "SELECT period_start, count FROM plan_counter WHERE period_start = ?",
        effectivePeriodStart,
      ).toArray()[0];
      if (!current) {
        this.ctx.storage.sql.exec("DELETE FROM plan_counter");
        this.ctx.storage.sql.exec(
          "INSERT INTO plan_counter (period_start, period_end, count, created_at) VALUES (?, ?, 0, ?)",
          effectivePeriodStart, effectivePeriodEnd, now,
        );
      }

      // (3) Atomic trio: counter UPDATE + outbox INSERT + idempotency INSERT.
      const updated = this.ctx.storage.sql.exec<{ count: number }>(
        "UPDATE plan_counter SET count = count + 1 WHERE period_start = ? RETURNING count",
        effectivePeriodStart,
      ).toArray()[0];
      const count = updated.count;

      writePlanCounterOutboxEntry(sqlStorage, {
        requestId: idempotencyKey ?? crypto.randomUUID(),
        orgId,
        periodStart: effectivePeriodStart,
        periodEnd: effectivePeriodEnd,
        deltaCount: 1,
      });

      // PR-2c codex-round-1 F3 + H4 + codex-round-3 M4: compute denial decision
      // for this fresh increment. Helper is invoked ONLY on the fresh path so
      // the `plan_limit_would_block` shadow-mode metric fires once per original
      // request, not once per retry. CRITICAL: flag gates ONLY the denial
      // decision, NOT the counter increment — counter + outbox already wrote
      // above so shadow-mode observability stays honest.
      //
      // codex-round-3 H2: PLAN_COUNTER_ENABLED via optionalBinding() so the var
      // typechecks against the extended OptionalEnv interface.
      const planEnabled = optionalBinding(this.env, "PLAN_COUNTER_ENABLED") === "true";
      // Use `!= null` (loose equality) to match the orchestrator's style at
      // budget-orchestrator.ts:110. In practice the orchestrator only calls
      // with a defined blockAt, so both variants behave identically here — but
      // keeping the style aligned prevents future refactor drift (build-audit F4).
      if (planLimitBlockAt != null && count > planLimitBlockAt) {
        if (planLimitMode === "hard") {
          if (planEnabled) {
            result = { status: "denied", count, blockAt: planLimitBlockAt };
          } else {
            // Shadow mode — would-block observability signal without enforcement.
            // codex-round-1 H4: tags are {tier, mode} only. orgId lives in structured log.
            emitMetric("plan_limit_would_block", { tier, mode: planLimitMode });
            console.log("[UserBudgetDO] plan_limit_would_block (shadow):", {
              orgId, tier, count, blockAt: planLimitBlockAt,
            });
            result = { status: "approved", count };
          }
        } else {
          // PR-2e /review P1-3: soft-mode overflow observability. Pro/Scale
          // customers hitting their included-requests cap never deny — overage
          // is billed, not blocked. But without this metric, soft-mode had
          // ZERO observability signal (plan_limit_would_block only fires in
          // shadow mode). Overage-billing accuracy and Pro/Scale-conversion
          // funnel analytics both depend on charting this.
          //
          // NOT wired into launch-watcher alerts — this is a billing /
          // product metric, not an outage signal.
          emitMetric("plan_limit_would_warn", { tier, mode: planLimitMode });
          console.log("[UserBudgetDO] plan_limit_would_warn (soft overflow):", {
            orgId, tier, count, blockAt: planLimitBlockAt,
          });
          result = { status: "approved", count };
        }
      } else {
        result = { status: "approved", count };
      }

      // PR-2c codex-round-3 C1: persist the decision for idempotency replay
      // correctness. INSERT OR REPLACE handles the NULL-sentinel delete-then-
      // fresh path in the replay branch (DELETE already ran by the time we
      // get here, but OR REPLACE keeps us safe against any race). Write
      // status + decision_count for every idempotency-keyed request; blockAt
      // only when denied.
      if (idempotencyKey) {
        const decisionStatus = result.status; // "approved" | "denied" (never "skipped" — early-returned above)
        const decisionCount = result.count;
        const decisionBlockAt = result.status === "denied" ? result.blockAt : null;
        this.ctx.storage.sql.exec(
          "INSERT OR REPLACE INTO plan_counter_idempotency (key, period_start, seen_at, status, decision_count, decision_block_at) VALUES (?, ?, ?, ?, ?, ?)",
          idempotencyKey, effectivePeriodStart, now, decisionStatus, decisionCount, decisionBlockAt,
        );
      }
    });

    // Alarm scheduling OUTSIDE the transaction (matches existing reconcile pattern).
    try {
      const currentAlarm = await this.ctx.storage.getAlarm();
      const soonAlarm = Date.now() + 1_000;
      if (!currentAlarm || currentAlarm > soonAlarm) {
        await this.ctx.storage.setAlarm(soonAlarm);
      }
    } catch { /* best-effort */ }

    // transactionSync is synchronous — `result` is always assigned when the
    // block completes. Non-null assertion is safe.
    return result!;
  }

  /**
   * PR-2b: alarm sub-handler — reset a closed plan_counter period if any.
   *
   * Per edge-case audit EC1, this sub-handler now ONLY deletes the local
   * cache row. No outbox entry is written. Reason: Postgres already has
   * the correct per-period count from the per-increment `deltaCount: 1`
   * outbox entries; a flush with `deltaCount: row.count` would double-add
   * the accumulated count on top of the already-synced stream. The old
   * name "BoundaryFlush" reflected the original (incorrect) design that
   * wrote an outbox entry here; the current behavior is a pure local reset.
   *
   * Atomicity (per codex round-2 H2) is preserved for defense-in-depth even
   * though the only write is a DELETE — a future change that re-adds an
   * outbox write must stay transactional.
   */
  async handlePlanCounterBoundaryFlush(): Promise<void> {
    const orgId = this.ctx.id.name;
    if (!orgId) {
      // Build-audit Finding #4: observability parity with the increment path,
      // which also emits a skipped-null-org metric. Without this, a misrouted
      // stub (idFromString, newUniqueId) with a stranded plan_counter row
      // would be silently stuck — no signal for ops.
      emitMetric("plan_counter_flush_skipped_null_org", {});
      return;
    }

    // Step 1: read the local plan_counter row without holding a write lock.
    // Deletion happens later, conditionally on stampPeriodClose success.
    const row = this.ctx.storage.sql.exec<{
      period_start: number; period_end: number; count: number;
    }>("SELECT * FROM plan_counter LIMIT 1").toArray()[0];

    if (row && row.period_end < Date.now()) {
      // PR-6a R3 P1: stamp `org_period_usage` snapshots BEFORE deleting the
      // local counter. If stamp fails (HYPERDRIVE unavailable, PG error,
      // unknown Stripe status, deferred on missing sub row), leave the
      // counter in place so the next alarm tick retries. The alarm wrapper
      // (handlePlanCounterBoundaryFlush is called inside try/catch in
      // `alarm()`) converts throws into emitted metrics + continued alarm
      // composition; returning early here has the same practical effect
      // but avoids noise in the alarm's error path.
      let connectionString: string | undefined;
      try {
        connectionString = this.env.HYPERDRIVE.connectionString;
      } catch (err) {
        emitMetric("stamp_period_close_failure", {
          orgId,
          reason: "hyperdrive_unavailable",
        });
        console.error(
          "[UserBudgetDO] stampPeriodClose: HYPERDRIVE unavailable, plan_counter retained for retry:",
          err,
        );
        return; // skip DELETE; next alarm tick retries
      }

      try {
        const stampResult = await stampPeriodClose(connectionString, {
          orgId,
          periodStart: row.period_start,
          periodEnd: row.period_end,
        });
        if (stampResult.deferred) {
          // No sub row materialized yet for this org (e.g., unpaid→paid race).
          // Leave the counter in place and retry on the next alarm.
          return;
        }
      } catch (err) {
        emitMetric("stamp_period_close_failure", {
          orgId,
          reason: "stamp_threw",
          error: err instanceof Error ? err.message : String(err),
        });
        console.error(
          "[UserBudgetDO] stampPeriodClose failed, plan_counter retained for retry:",
          err,
        );
        return; // skip DELETE; next alarm tick retries
      }

      // Stamp succeeded (applied=true OR applied=false idempotent). The
      // expired period's snapshot is now persisted; safe to purge the local
      // counter. Re-check the row inside transactionSync because a fresh
      // increment could have rotated the period since our initial SELECT
      // (lazy reset in incrementPlanCounter replaces the row atomically).
      this.ctx.storage.transactionSync(() => {
        const current = this.ctx.storage.sql.exec<{
          period_start: number;
        }>("SELECT period_start FROM plan_counter LIMIT 1").toArray()[0];
        if (current && current.period_start === row.period_start) {
          this.ctx.storage.sql.exec("DELETE FROM plan_counter");
        }
      });
    }

    // PR-2d (Decision #34 / codex R2#2 / C60): emit outbox-drain lag for the
    // shadow-mode alert. Outside transactionSync so a read-only SELECT never
    // extends the write lock. Skip emit on empty outbox — quiet DOs shouldn't
    // look like zero-lag to the alert consumer (see helper docstring).
    const lagMs = computePlanCounterOutboxLagMs(
      this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage,
      Date.now(),
    );
    if (lagMs !== null) {
      emitMetric("plan_counter_outbox_lag_ms", { value: lagMs });
    }
  }

  /**
   * PR-2b: alarm sub-handler — prune `plan_counter_idempotency` by 24h TTL.
   *
   * Returns the next wake time (ms epoch) if the table still has rows
   * post-prune, or `null` if empty. Per codex G2 — sub-handler does NOT
   * call setAlarm itself; main alarm() composes Math.min across all sources.
   *
   * Emergency bound (codex round-2 M2): at extreme ingest rates the live 24h
   * working set could hit millions of rows. If post-TTL count exceeds
   * `IDEMPOTENCY_EMERGENCY_BOUND`, force-delete the oldest ~50% by `seen_at`
   * and emit `plan_counter_idempotency_emergency_compact`. Safety valve only.
   */
  handleIdempotencyDedupPrune(): number | null {
    const sql = this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage;
    const cutoff = Date.now() - 24 * 3600 * 1000;
    sql.exec("DELETE FROM plan_counter_idempotency WHERE seen_at < ?", cutoff);

    // Emergency compaction — safety valve. See `compactIdempotencyTable`
    // (build-audit Finding #3) for the direct-unit-testable primitive.
    const compaction = compactIdempotencyTable(sql);
    if (compaction.removed > 0) {
      emitMetric("plan_counter_idempotency_emergency_compact", {
        removed: compaction.removed, remaining: compaction.remaining,
      });
      console.error(
        `[UserBudgetDO] ALERT: plan_counter_idempotency exceeded ${IDEMPOTENCY_EMERGENCY_BOUND} rows — force-compacted oldest ${compaction.removed}`,
      );
    }
    return compaction.remaining > 0 ? Date.now() + 3600 * 1000 : null;
  }

  /**
   * PR-2e F2-partial: divergence outbox drain (codex R7 Issue 2A).
   *
   * Sibling to `handlePlanCounterBoundaryFlush`. Drains
   * `pg_sync_outbox_plan_divergences` by calling `upsertPlanCounterDivergence`
   * (dedup'd via `plan_counter_divergence_dedup`) per-entry. On success, ack
   * by row id. On terminal FK 23503 against EITHER `plan_counter_divergences`
   * OR `plan_counter_divergence_dedup` (codex R7 C2), delete the entry
   * permanently. On other errors, mark retry with exponential backoff.
   *
   * After the retry loop, abandoned entries (attempts >= max) are deleted AND
   * a row is written to `plan_counter_sync_failures` with
   * `reason='divergence_outbox_abandoned'` so the launch-watcher surfaces the
   * stall (codex R7 C4 observability fix). Best-effort — per
   * `monitoring_hardening_can_introduce_failures` learning, the PG write for
   * the stall signal is wrapped in try/catch so a PG outage cannot take down
   * the DO alarm dispatcher.
   *
   * Returns the next wake time (ms epoch) if entries remain with
   * `next_attempt_at > now`, or `null` if empty. Per codex G2 pattern, caller
   * composes with Math.min.
   */
  async handlePlanCounterDivergenceOutboxDrain(): Promise<number | null> {
    const sql = this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage;
    const now = Date.now();
    const MAX_ATTEMPTS = 5;

    // Per-tick lag emission for AE/log observability. Quiet DOs stay quiet
    // (helper returns null on empty table) to avoid false-zero readings.
    const lagMs = computePlanCounterDivergenceOutboxLagMs(sql, now);
    if (lagMs !== null) {
      emitMetric("plan_counter_divergence_outbox_lag_ms", { value: lagMs });
    }

    // F2-partial build-audit Finding 3 fix: check for ANY work BEFORE touching
    // HYPERDRIVE so idle DOs (the 99% case) don't emit HYPERDRIVE-unavailable
    // stderr on every alarm tick. `pending` = retryable rows; `initialAbandoned`
    // = rows already at max attempts from a prior tick. If both are empty, no
    // work to do — return without resolving HYPERDRIVE.
    //
    // We re-query the abandoned set AFTER the retry loop below (entries may
    // have been bumped from attempts=4 to 5 in this tick). This initial query
    // is a pre-check only.
    const pending = getRetryablePlanCounterDivergenceEntries(sql, now, MAX_ATTEMPTS);
    const initialAbandoned = sql.exec<{ id: number; orgId: string }>(
      "SELECT id, org_id AS orgId FROM pg_sync_outbox_plan_divergences WHERE attempts >= ?",
      MAX_ATTEMPTS,
    ).toArray();

    if (pending.length === 0 && initialAbandoned.length === 0) {
      return null;
    }

    // There's work — resolve HYPERDRIVE once for BOTH branches (codex-diff
    // review Bug 1+2 fix). Both the retry loop AND the abandoned-cleanup
    // block must handle its absence gracefully.
    //   - Bug 1: HYPERDRIVE-unavailable early-return skipped abandoned cleanup
    //     entirely — stranded rows with attempts=5 in SQLite forever.
    //   - Bug 2: alarm with ONLY abandoned rows (pending.length === 0) lost the
    //     PG signal because connectionString was never populated.
    let connectionString: string | undefined;
    try {
      connectionString = this.env.HYPERDRIVE.connectionString;
    } catch (err) {
      console.error("[UserBudgetDO] divergence drain: HYPERDRIVE unavailable this tick:", err);
      // Do NOT early-return — we still need to: (a) mark pending rows as failed
      // so they retry on the next alarm, and (b) delete any already-abandoned
      // rows from SQLite so they don't leak.
    }

    if (pending.length > 0) {
      if (!connectionString) {
        // HYPERDRIVE down — mark all pending entries as failed and let the
        // next alarm retry them after backoff. Same behavior as the PR-2b
        // plan-counter outbox drain block.
        for (const entry of pending) {
          markPlanCounterDivergenceEntryRetryFailed(sql, entry.id, entry.attempts);
        }
      } else {
        for (const entry of pending) {
          try {
            await upsertPlanCounterDivergence(connectionString, {
              eventId: entry.eventId,
              orgId: entry.orgId,
              tier: entry.tier,
              divergenceAtMs: entry.divergenceAtMs,
              storedPeriodStart: entry.storedPeriodStart,
              incomingPeriodStart: entry.incomingPeriodStart,
            });
            ackPlanCounterDivergenceEntryById(sql, entry.id);
          } catch (err) {
            if (isTerminalPlanCounterDivergenceFkError(err)) {
              deletePlanCounterDivergenceEntryTerminal(sql, entry.id);
              emitMetric("plan_counter_divergence_outbox_terminal_fk_violation", {});
            } else {
              markPlanCounterDivergenceEntryRetryFailed(sql, entry.id, entry.attempts);
            }
          }
        }
      }
    }

    // Drain abandoned entries AFTER the retry loop so any that JUST hit
    // max-attempts in this tick are caught. Query BEFORE delete so we can
    // write the stall signal per-org. Runs REGARDLESS of HYPERDRIVE status
    // per Bug 1 fix — the SQLite cleanup doesn't need HYPERDRIVE.
    const abandonedRows = sql.exec<{ id: number; orgId: string }>(
      "SELECT id, org_id AS orgId FROM pg_sync_outbox_plan_divergences WHERE attempts >= ?",
      MAX_ATTEMPTS,
    ).toArray();
    if (abandonedRows.length > 0) {
      const removed = deleteAbandonedPlanCounterDivergenceEntries(sql, MAX_ATTEMPTS);
      emitMetric("plan_counter_divergence_outbox_abandoned", { count: removed });

      // C4 fix: durable PG signal for the launch-watcher. Best-effort —
      // per `monitoring_hardening_can_introduce_failures`, PG outage must
      // NOT propagate into the DO alarm dispatcher.
      if (connectionString) {
        for (const row of abandonedRows) {
          try {
            await writePlanCounterSyncFailure(connectionString, {
              orgId: row.orgId,
              reason: "divergence_outbox_abandoned",
              count: 1,
            });
          } catch (writeErr) {
            emitMetric("plan_counter_divergence_abandoned_signal_write_failed", {});
            console.error("[UserBudgetDO] failed to write divergence_outbox_abandoned signal:", writeErr);
          }
        }
      } else {
        // F2-partial codex-diff Bug 1 fix: HYPERDRIVE was unavailable this tick
        // so the PG signal is lost for these abandoned rows. Emit a secondary
        // metric so ops can see the signal pipeline failed this round. Rows
        // are still deleted from SQLite (no leak) but the watcher can't count
        // them for 15 minutes.
        emitMetric("plan_counter_divergence_abandoned_signal_skipped_no_hyperdrive", {
          count: removed,
        });
      }
    }

    return this.computeNextDivergenceRetry(sql, now, MAX_ATTEMPTS);
  }

  /**
   * Compute the earliest next-wake time for the divergence outbox.
   * Extracted for readability; mirrors the inline `nextPlanCounterOutbox`
   * SELECT pattern in `alarm()` for the plan-counter outbox.
   */
  private computeNextDivergenceRetry(
    sql: import("../lib/pg-sync-outbox.js").SqlStorage,
    _now: number,
    maxAttempts: number,
  ): number | null {
    const next = sql.exec<{ next: number | null }>(
      "SELECT MIN(next_attempt_at) as next FROM pg_sync_outbox_plan_divergences WHERE attempts < ?",
      maxAttempts,
    ).toArray()[0]?.next;
    return next === null || next === undefined ? null : next;

  }

  /**
   * Alarm handler: clean up expired reservations.
   * Cleans up expired reservations and stale session spend entries.
   */
  async alarm(): Promise<void> {
    const now = Date.now();
    const expired = this.ctx.storage.sql
      .exec<{ id: string; amount: number; entity_keys: string; session_id: string | null }>(
        "SELECT id, amount, entity_keys, session_id FROM reservations WHERE expires_at <= ?",
        now,
      )
      .toArray();

    if (expired.length > 0) {
      console.log(`[UserBudgetDO] alarm: cleaning up ${expired.length} expired reservation(s)`);

      this.ctx.storage.transactionSync(() => {
        for (const rsv of expired) {
          try {
            const keys: string[] = JSON.parse(rsv.entity_keys);
            for (const key of keys) {
              const [entityType, entityId] = parseEntityKey(key);
              this.ctx.storage.sql.exec(
                "UPDATE budgets SET reserved = MAX(0, reserved - ?) WHERE entity_type = ? AND entity_id = ?",
                rsv.amount,
                entityType,
                entityId,
              );
            }
            // Reverse session spend for expired reservations
            if (rsv.session_id) {
              for (const key of keys) {
                this.ctx.storage.sql.exec(
                  "UPDATE session_spend SET spend = MAX(0, spend - ?) WHERE entity_key = ? AND session_id = ?",
                  rsv.amount, key, rsv.session_id,
                );
              }
            }
          } catch (err) {
            console.error(`[UserBudgetDO] alarm: failed to clean reservation ${rsv.id}, skipping:`, err);
          }
          this.ctx.storage.sql.exec(
            "DELETE FROM reservations WHERE id = ?",
            rsv.id,
          );
        }
      });

      }

    // Loop call log cleanup: prune entries older than the max configured window.
    // Use the largest loop_window_seconds across all budget entities (or 120s minimum)
    // to avoid deleting entries that are still within a user's configured detection window.
    const maxLoopWindowRow = this.ctx.storage.sql.exec<{ max_win: number | null }>(
      "SELECT MAX(loop_window_seconds) as max_win FROM budgets",
    ).toArray()[0];
    const loopPruneMs = Math.max(120_000, ((maxLoopWindowRow?.max_win ?? 60) + 60) * 1000);
    this.ctx.storage.sql.exec("DELETE FROM loop_call_log WHERE ts < ?", now - loopPruneMs);
    // Safety cap: if table exceeds 5000 rows after pruning, truncate to most recent 1000
    const loopRowCount = this.ctx.storage.sql.exec<{ cnt: number }>(
      "SELECT COUNT(*) as cnt FROM loop_call_log",
    ).toArray()[0]?.cnt ?? 0;
    if (loopRowCount > 5000) {
      this.ctx.storage.sql.exec(`
        DELETE FROM loop_call_log WHERE rowid NOT IN (
          SELECT rowid FROM loop_call_log ORDER BY ts DESC LIMIT 1000
        )
      `);
    }

    // Session cleanup: delete stale sessions (last_seen > 24h)
    const SESSION_TTL_MS = 24 * 60 * 60 * 1000;
    // AUDIT-7: hourly orphan sweep cadence for idle DOs with budget rows.
    const ORPHAN_SWEEP_INTERVAL_MS = 60 * 60 * 1000;
    const cutoff = now - SESSION_TTL_MS;
    const deleted = this.ctx.storage.sql.exec(
      "DELETE FROM session_spend WHERE last_seen < ?",
      cutoff,
    );
    if (deleted.rowsWritten > 0) {
      console.log(`[UserBudgetDO] alarm: cleaned up ${deleted.rowsWritten} stale session(s)`);
    }

    // ── PR-2b: plan-counter sub-handlers ──
    // Per codex round-2 H2: catch flush errors independently so they don't abort
    // the rest of alarm composition. The flush is best-effort per-alarm; a failure
    // just leaves the closed period in place for the next alarm to retry.
    try {
      await this.handlePlanCounterBoundaryFlush();
    } catch (err) {
      emitMetric("plan_counter_boundary_flush_error", {});
      console.error("[UserBudgetDO] alarm: handlePlanCounterBoundaryFlush failed:", err);
    }

    // PR-2e F2-partial (codex R7 Issue 2A): drain the divergence outbox.
    // Wrapped in try/catch at the alarm() level so a handler failure doesn't
    // take down the rest of alarm composition (mirrors boundary flush).
    let nextDivergenceOutbox: number | null = null;
    try {
      nextDivergenceOutbox = await this.handlePlanCounterDivergenceOutboxDrain();
    } catch (err) {
      emitMetric("plan_counter_divergence_outbox_drain_error", {});
      console.error("[UserBudgetDO] alarm: handlePlanCounterDivergenceOutboxDrain failed:", err);
    }

    // Per codex G2: sub-handler returns its desired next-wake time; we compose
    // with Math.min below rather than letting it setAlarm internally.
    const nextIdempotencyPrune = this.handleIdempotencyDedupPrune();

    // ── PXY-2: Process pending PG sync outbox entries ──
    const MAX_OUTBOX_ATTEMPTS = 5;
    const sqlStorage = this.ctx.storage.sql as import("../lib/pg-sync-outbox.js").SqlStorage;
    const pending = getRetryableEntries(sqlStorage, now, MAX_OUTBOX_ATTEMPTS);

    // Hoist connectionString so pruning can reuse it (avoids a second HYPERDRIVE access)
    let connectionString: string | undefined;

    if (pending.length > 0) {
      try {
        connectionString = this.env.HYPERDRIVE.connectionString;
      } catch (err) {
        // HYPERDRIVE unavailable — mark all entries as failed
        console.error("[UserBudgetDO] alarm: HYPERDRIVE unavailable, marking outbox entries failed:", err);
        for (const entry of pending) {
          markRetryFailed(sqlStorage, entry.id, entry.attempts);
        }
      }

      if (connectionString) {
        // C5: Group by requestId — all entities for one request are written + acked together
        const byRequest = new Map<string, typeof pending>();
        for (const entry of pending) {
          const group = byRequest.get(entry.requestId) ?? [];
          group.push(entry);
          byRequest.set(entry.requestId, group);
        }

        for (const [requestId, entries] of byRequest) {
          try {
            await updateBudgetSpend(
              connectionString,
              entries[0].orgId,
              requestId,
              entries.map((e) => ({ entityType: e.entityType, entityId: e.entityId })),
              entries[0].costMicrodollars,
            );
            // All entities succeeded — ack all
            ackAllForRequest(sqlStorage, requestId);
            emitMetric("pg_sync_alarm_success", { requestId, entityCount: entries.length });
          } catch (err) {
            // Mark all entries in this group as failed with backoff
            for (const entry of entries) {
              markRetryFailed(sqlStorage, entry.id, entry.attempts);
            }
            console.error("[UserBudgetDO] outbox PG sync failed:", {
              requestId, attempt: entries[0].attempts + 1,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }

      }

      // Abandon entries that exceeded max attempts (runs regardless of
      // connectionString — entries may have been marked failed by the
      // HYPERDRIVE-unavailable path above and now need cleanup)
      const abandoned = deleteAbandonedEntries(sqlStorage, MAX_OUTBOX_ATTEMPTS);
      if (abandoned > 0) {
        emitMetric("pg_sync_abandoned", { count: abandoned });
        console.error(`[UserBudgetDO] ALERT: ${abandoned} outbox entries abandoned after ${MAX_OUTBOX_ATTEMPTS} attempts`);
      }
    }

    // ── PR-2b: Process pending plan-counter outbox entries ──
    // Independent from the budget-spend path above: one table failing doesn't
    // block the other. Per codex G1: mirror the HYPERDRIVE-unavailable mark-failed
    // semantics so rows with `next_attempt_at = 0` get a positive backoff stamp —
    // otherwise the rescheduler's null-check would still fire, but the table would
    // churn uselessly each alarm. Per codex round-2 H1: acquire Hyperdrive locally
    // if the budget-spend branch didn't populate `connectionString` (plan-counter-
    // only alarm — common case when budget-spend outbox is empty).
    {
      const planCounterEntries = getRetryablePlanCounterEntries(sqlStorage, now, MAX_OUTBOX_ATTEMPTS);

      if (planCounterEntries.length > 0) {
        if (!connectionString) {
          try { connectionString = this.env.HYPERDRIVE.connectionString; }
          catch { /* genuinely unavailable — fall through to mark-failed */ }
        }

        if (connectionString) {
          for (const entry of planCounterEntries) {
            try {
              const { applied } = await upsertPlanCounterPeriod(connectionString, {
                requestId: entry.requestId,
                orgId: entry.orgId,
                periodStart: entry.periodStart,
                periodEnd: entry.periodEnd,
                deltaCount: entry.deltaCount,
              });
              // Codex-final F1: ack by outbox row id, NOT by request_id. Same
              // request_id can span two outbox rows when a cross-period retry
              // exists; request_id ack would delete the sibling row and risk
              // silent data loss if that row's own upsert later failed.
              ackPlanCounterEntryById(sqlStorage, entry.id);
              // `applied=false` means Postgres had already recorded this
              // (org, requestId, periodStart) from a prior alarm that wrote
              // but couldn't ack. Metric stays low-cardinality; structured log
              // on dedup hit carries (orgId, periodStart, requestId) for
              // post-hoc debugging without cardinality explosion — per codex
              // R5 observability note.
              emitMetric(
                applied ? "plan_counter_sync_success" : "plan_counter_sync_dedup_hit",
                { requestId: entry.requestId },
              );
              if (!applied) {
                console.log(
                  `[UserBudgetDO] plan_counter_sync_dedup_hit orgId=${entry.orgId} periodStart=${entry.periodStart} requestId=${entry.requestId}`,
                );
              }
            } catch (err) {
              // PR-2c plan-audit F2 + codex-round-1 H5 + codex-round-2 H5:
              // classify FK-violation errors on allowlisted constraints as
              // TERMINAL (org has been deleted; FK cascade dropped
              // org_period_usage / plan_counter_sync_requests rows; retries
              // will never succeed). Delete the outbox entry immediately + emit
              // metric. Other errors (23503 on non-allowlisted constraints,
              // transient connection failures, etc) go through the existing
              // retry path — same behavior as before.
              if (isTerminalPlanCounterFkError(err)) {
                deletePlanCounterEntryTerminal(sqlStorage, entry.id);
                // codex-round-1 H4: metric has NO tags — orgId + constraint live
                // in the structured log below for forensic reconstruction.
                emitMetric("plan_counter_outbox_terminal_fk_violation", {});
                // edge-case-audit E1: postgres.js uses .constraint_name; other
                // pg clients use .constraint. Read both for resilience.
                const pgErr = err as { constraint_name?: string; constraint?: string } | null;
                const constraint = pgErr?.constraint_name ?? pgErr?.constraint;
                console.warn("[UserBudgetDO] plan_counter outbox TERMINAL (FK violation — org likely deleted):", {
                  requestId: entry.requestId,
                  orgId: entry.orgId,
                  constraint,
                  error: err instanceof Error ? err.message : String(err),
                });
              } else {
                markPlanCounterEntryRetryFailed(sqlStorage, entry.id, entry.attempts);
                console.error("[UserBudgetDO] plan_counter outbox PG sync failed:", {
                  requestId: entry.requestId, attempt: entry.attempts + 1,
                  error: err instanceof Error ? err.message : String(err),
                });
              }
            }
          }
        } else {
          console.error("[UserBudgetDO] alarm: HYPERDRIVE unavailable, marking plan_counter outbox entries failed");
          for (const entry of planCounterEntries) {
            markPlanCounterEntryRetryFailed(sqlStorage, entry.id, entry.attempts);
          }
        }

        const abandonedPlanCounter = deleteAbandonedPlanCounterEntries(sqlStorage, MAX_OUTBOX_ATTEMPTS);
        if (abandonedPlanCounter > 0) {
          emitMetric("plan_counter_outbox_abandoned", { count: abandonedPlanCounter });
          console.error(`[UserBudgetDO] ALERT: ${abandonedPlanCounter} plan_counter outbox entries abandoned after ${MAX_OUTBOX_ATTEMPTS} attempts`);
        }
      }
    }

    // C9: Prune old reconciled_requests (best-effort).
    // Only attempt when we successfully obtained a connectionString above,
    // to avoid opening a new HYPERDRIVE connection just for pruning.
    if (connectionString) {
      try {
        const pruneSql = getSql(connectionString);
        await pruneSql`DELETE FROM reconciled_requests WHERE reconciled_at < NOW() - INTERVAL '7 days'`;
        await pruneSql.end({ timeout: 0 }).catch(() => {});
      } catch { /* best-effort pruning — connection may have gone stale */ }
    }

    // AUDIT-7: Sweep DO budget rows whose Postgres counterpart has been deleted.
    // Catches cases where DELETE FROM budgets happened without a paired
    // /internal/budget/invalidate action=remove — prevents permanent orphans
    // that mislead /v1/policy and break test isolation. ownerId comes from the
    // DO ID's name (idFromName). Safe skips: no ownerId, no DO budgets, or
    // HYPERDRIVE unavailable. findOrphanedBudgets enforces a 60s safety window
    // so a just-synced row that Postgres can't yet see is never evicted.
    const sweepOwnerId = this.ctx.id.name;
    if (sweepOwnerId) {
      const sweepDoRows = this.ctx.storage.sql
        .exec<{ entity_type: string; entity_id: string; synced_at: number }>(
          "SELECT entity_type, entity_id, synced_at FROM budgets",
        )
        .toArray();
      if (sweepDoRows.length > 0) {
        if (!connectionString) {
          try { connectionString = this.env.HYPERDRIVE.connectionString; } catch { /* unavailable — skip sweep */ }
        }
        if (connectionString) {
          try {
            const sweepSql = getSql(connectionString);
            const sweepPgRows = await sweepSql<{ entity_type: string; entity_id: string }[]>`
              SELECT entity_type, entity_id FROM budgets WHERE org_id = ${sweepOwnerId}
            `;
            const orphans = findOrphanedBudgets(sweepDoRows, sweepPgRows, now);
            for (const orphan of orphans) {
              await this.removeBudget(orphan.entity_type, orphan.entity_id);
              emitMetric("do_budget_orphan_evicted", {
                ownerId: sweepOwnerId,
                entityType: orphan.entity_type,
                entityId: orphan.entity_id,
              });
              console.warn(
                `[UserBudgetDO] AUDIT-7: evicted orphan budget ${orphan.entity_type}:${orphan.entity_id} (owner=${sweepOwnerId})`,
              );
            }
          } catch (err) {
            console.error("[UserBudgetDO] alarm: orphan sweep failed:", err);
            emitMetric("do_budget_orphan_sweep_error", {
              ownerId: sweepOwnerId,
              error: err instanceof Error ? err.message : "unknown",
            });
          }
        }
      }
    }

    // Reschedule: next reservation expiry OR session cleanup OR outbox retry
    const next = this.ctx.storage.sql
      .exec<{ next_exp: number | null }>(
        "SELECT MIN(expires_at) as next_exp FROM reservations",
      )
      .toArray()[0];

    const hasSessionRows = this.ctx.storage.sql
      .exec("SELECT 1 FROM session_spend LIMIT 1")
      .toArray().length > 0;

    let nextAlarm: number | null = null;
    if (next?.next_exp) nextAlarm = next.next_exp;
    if (hasSessionRows) {
      const sessionCleanup = now + SESSION_TTL_MS;
      nextAlarm = nextAlarm ? Math.min(nextAlarm, sessionCleanup) : sessionCleanup;
    }

    // C6: Schedule alarm for next outbox retry (uses persisted next_attempt_at).
    // Per codex round-2 M1: null-check explicitly instead of truthy. A
    // `next_attempt_at = 0` is the "ready to fire now" sentinel — the previous
    // `if (nextOutbox)` treated 0 as falsy and silently dropped it, leaving
    // ready-to-retry entries unscheduled.
    const nextOutbox = this.ctx.storage.sql
      .exec<{ next: number | null }>(
        "SELECT MIN(next_attempt_at) as next FROM pg_sync_outbox WHERE attempts < ?",
        MAX_OUTBOX_ATTEMPTS,
      ).toArray()[0]?.next;
    if (nextOutbox !== null && nextOutbox !== undefined) {
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, nextOutbox) : nextOutbox;
    }

    // PR-2b: same null-check semantics for the plan-counter outbox.
    const nextPlanCounterOutbox = this.ctx.storage.sql
      .exec<{ next: number | null }>(
        "SELECT MIN(next_attempt_at) as next FROM pg_sync_outbox_plan_counter WHERE attempts < ?",
        MAX_OUTBOX_ATTEMPTS,
      ).toArray()[0]?.next;
    if (nextPlanCounterOutbox !== null && nextPlanCounterOutbox !== undefined) {
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, nextPlanCounterOutbox) : nextPlanCounterOutbox;
    }

    // PR-2e F2-partial: compose divergence outbox next-wake time (codex G2 pattern).
    if (nextDivergenceOutbox !== null) {
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, nextDivergenceOutbox) : nextDivergenceOutbox;
    }

    // PR-2b (codex G2): compose idempotency-prune desired wake time.
    if (nextIdempotencyPrune !== null) {
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, nextIdempotencyPrune) : nextIdempotencyPrune;
    }

    // AUDIT-7: Housekeeping wake so idle DOs with budgets still run the orphan
    // sweep. Without this, a DO whose activity died before its Postgres peer
    // was deleted would keep the phantom budget forever.
    const hasBudgetRows = this.ctx.storage.sql
      .exec<{ cnt: number }>("SELECT COUNT(*) as cnt FROM budgets")
      .toArray()[0]?.cnt ?? 0;
    if (hasBudgetRows > 0) {
      const housekeeping = now + ORPHAN_SWEEP_INTERVAL_MS;
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, housekeeping) : housekeeping;
    }

    // PR-6a audit finding #4: if an expired plan_counter row is retained
    // (stampPeriodClose failed or hit HYPERDRIVE-unavailable above), we MUST
    // explicitly schedule a retry — otherwise the DO only wakes on unrelated
    // activity (reservations, outbox, idempotency prune). A quiet DO with a
    // retained row could sit forever. 5-min retry matches the watcher's
    // polling cadence so the retry loop is bounded.
    //
    // Edge-case audit E6 residual risk (narrowed):
    //   - Cold-start reconstruction (constructor `blockConcurrencyWhile`) now
    //     also wakes on retained expired plan_counter rows — so any request
    //     to the evicted DO schedules the alarm within 1s.
    //   - This reschedule branch handles the "alarm fired, stamp failed, DO
    //     still alive" case — retry on next 5-min tick.
    //   - The remaining narrow window: DO is evicted AND receives zero
    //     requests until the period_end boundary passes by >7 days (CF
    //     eviction policy). PR-6b's Postgres-side recovery sweep closes
    //     that last gap — see TODOS.md `PR-6b missed-snapshot recovery`.
    const retainedExpiredPlanCounter = this.ctx.storage.sql
      .exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM plan_counter WHERE period_end < ?",
        now,
      )
      .toArray()[0]?.cnt ?? 0;
    if (retainedExpiredPlanCounter > 0) {
      const planCounterRetry = now + 5 * 60 * 1000;
      nextAlarm = nextAlarm !== null ? Math.min(nextAlarm, planCounterRetry) : planCounterRetry;
    }

    // Per codex round-2 M1: null-check, not truthy. `nextAlarm = 0` is a valid
    // "fire now" signal and must set the alarm.
    if (nextAlarm !== null) {
      await this.ctx.storage.setAlarm(nextAlarm);
    }
  }
}
