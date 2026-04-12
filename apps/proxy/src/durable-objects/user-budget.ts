import { DurableObject } from "cloudflare:workers";
import {
  writeOutboxEntry,
  getRetryableEntries,
  ackAllForRequest,
  markRetryFailed,
  deleteAbandonedEntries,
} from "../lib/pg-sync-outbox.js";
import { updateBudgetSpend } from "../lib/budget-spend.js";
import { getSql } from "../lib/db.js";
import { emitMetric } from "../lib/metrics.js";

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
}

export interface ReconcileResult {
  status: "reconciled" | "not_found";
  spends?: Record<string, number>;
  budgetsMissing?: string[];
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
      return parsed;
    }
    return [...DEFAULT_THRESHOLDS];
  } catch {
    return [...DEFAULT_THRESHOLDS];
  }
}

// ── Durable Object ──────────────────────────────────────────────────

export class UserBudgetDO extends DurableObject {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.initSchema();
      const count = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM budgets",
      ).toArray()[0]?.cnt ?? 0;
      console.log(`[UserBudgetDO] initialized, ${count} budgets loaded`);

      // PXY-2: Schedule alarm if pending outbox entries exist (cold rehydration)
      const pendingOutbox = this.ctx.storage.sql.exec<{ cnt: number }>(
        "SELECT COUNT(*) as cnt FROM pg_sync_outbox",
      ).toArray()[0]?.cnt ?? 0;
      if (pendingOutbox > 0) {
        const currentAlarm = await this.ctx.storage.getAlarm();
        if (!currentAlarm) {
          await this.ctx.storage.setAlarm(Date.now() + 1_000);
          console.log(`[UserBudgetDO] scheduled alarm for ${pendingOutbox} pending outbox entries`);
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
  ): Promise<CheckResult> {
    if (estimateMicrodollars < 0 || !Number.isFinite(estimateMicrodollars)) {
      return { status: "denied", hasBudgets: false };
    }

    const reservationId = crypto.randomUUID();
    const now = Date.now();

    let result: CheckResult = { status: "approved", hasBudgets: false };
    let reserved = false;
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
              `UPDATE budgets SET spend = 0, reserved = 0, period_start = ?
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
        });
      }

      // ── Session limit check (before velocity) ──────────────────────
      // Session denial exits before velocity logic runs — denied requests
      // should not affect velocity counters (same as budget denial).
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

      // Phase 2: Check each entity's budget
      for (const row of rows) {
        const remaining = row.max_budget - row.spend - row.reserved;

        if (
          row.policy === "strict_block" &&
          estimateMicrodollars > remaining
        ) {
          result = {
            status: "denied",
            hasBudgets: true,
            deniedEntity: `${row.entity_type}:${row.entity_id}`,
            remaining,
            maxBudget: row.max_budget,
            spend: row.spend,
          };
          console.log(
            `[UserBudgetDO] denied: entity=${row.entity_type}:${row.entity_id} remaining=${remaining} estimate=${estimateMicrodollars}`,
          );
          return; // Exit transactionSync — no reservation made
        }
      }

      // Phase 2.5: Apply deferred velocity increments (only reached if budget check passed)
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

    this.ctx.storage.transactionSync(() => {
      for (const key of entityKeys) {
        const [entityType, entityId] = parseEntityKey(key);

        if (actualCostMicrodollars > 0) {
          const rows = this.ctx.storage.sql.exec<{ spend: number }>(
            `UPDATE budgets SET
              spend = spend + ?,
              reserved = MAX(0, reserved - ?)
             WHERE entity_type = ? AND entity_id = ?
             RETURNING spend`,
            actualCostMicrodollars,
            row.amount,
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

    const result: ReconcileResult = { status: "reconciled", spends };
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
  ): Promise<boolean> {
    // Check if entity already exists (for return value)
    const existed = this.ctx.storage.sql.exec<{ cnt: number }>(
      "SELECT COUNT(*) as cnt FROM budgets WHERE entity_type = ? AND entity_id = ?",
      entityType, entityId,
    ).toArray()[0]?.cnt > 0;

    this.ctx.storage.transactionSync(() => {
      // Single UPSERT with all fields — avoids separate UPDATE statements
      this.ctx.storage.sql.exec(
        `INSERT INTO budgets
         (entity_type, entity_id, max_budget, spend, reserved, policy, reset_interval, period_start,
          velocity_limit, velocity_window, velocity_cooldown, threshold_percentages, session_limit)
         VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(entity_type, entity_id) DO UPDATE SET
           max_budget = excluded.max_budget,
           policy = excluded.policy,
           reset_interval = excluded.reset_interval,
           velocity_limit = excluded.velocity_limit,
           velocity_window = excluded.velocity_window,
           velocity_cooldown = excluded.velocity_cooldown,
           threshold_percentages = excluded.threshold_percentages,
           session_limit = excluded.session_limit`,
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
      "SELECT entity_type, entity_id, max_budget, spend, reserved, policy, reset_interval, period_start, velocity_limit, velocity_window, velocity_cooldown, threshold_percentages, session_limit FROM budgets",
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

      // 3. Reset the target entity (spend=0, reserved=0 — reserved may already be 0
      //    from step 2, but we set it explicitly to ensure clean state)
      this.ctx.storage.sql.exec(
        `UPDATE budgets SET spend = 0, reserved = 0, period_start = ?
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
    });

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

    // Session cleanup: delete stale sessions (last_seen > 24h)
    const SESSION_TTL_MS = 24 * 60 * 60 * 1000;
    const cutoff = now - SESSION_TTL_MS;
    const deleted = this.ctx.storage.sql.exec(
      "DELETE FROM session_spend WHERE last_seen < ?",
      cutoff,
    );
    if (deleted.rowsWritten > 0) {
      console.log(`[UserBudgetDO] alarm: cleaned up ${deleted.rowsWritten} stale session(s)`);
    }

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

    // C6: Schedule alarm for next outbox retry (uses persisted next_attempt_at)
    const nextOutbox = this.ctx.storage.sql
      .exec<{ next: number | null }>(
        "SELECT MIN(next_attempt_at) as next FROM pg_sync_outbox WHERE attempts < ?",
        MAX_OUTBOX_ATTEMPTS,
      ).toArray()[0]?.next;
    if (nextOutbox) {
      nextAlarm = nextAlarm ? Math.min(nextAlarm, nextOutbox) : nextOutbox;
    }

    if (nextAlarm) {
      await this.ctx.storage.setAlarm(nextAlarm);
    }
  }
}
