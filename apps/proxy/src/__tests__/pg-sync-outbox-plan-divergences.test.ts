/**
 * PR-2e F2-partial: unit tests for pg-sync-outbox-plan-divergences module.
 *
 * Mirrors `pg-sync-outbox-plan-counter.test.ts` pattern — in-memory SQL mock
 * to verify SQL shape, parameter binding, and separate-table invariant
 * (every statement targets `pg_sync_outbox_plan_divergences`, never the
 * other two outbox tables).
 *
 * Covers codex R7 C1 (event_id + divergence_at_ms captured at write time)
 * and C2 (dual-FK terminal classification).
 */
import { describe, it, expect, beforeEach } from "vitest";
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
import type { SqlStorage } from "../lib/pg-sync-outbox.js";

function createMockSql() {
  const calls: Array<{ query: string; bindings: unknown[] }> = [];
  let nextResult: { toArray: () => unknown[]; rowsWritten: number } = {
    toArray: () => [],
    rowsWritten: 0,
  };

  const sql: SqlStorage & {
    calls: typeof calls;
    setNextResult: (result: { toArray?: () => unknown[]; rowsWritten?: number }) => void;
  } = {
    calls,
    setNextResult(result) {
      nextResult = {
        toArray: result.toArray ?? (() => []),
        rowsWritten: result.rowsWritten ?? 0,
      };
    },
    exec(query: string, ...bindings: unknown[]) {
      calls.push({ query, bindings });
      return nextResult;
    },
  };

  return sql;
}

describe("pg-sync-outbox-plan-divergences", () => {
  let sql: ReturnType<typeof createMockSql>;

  beforeEach(() => {
    sql = createMockSql();
  });

  it("createPlanCounterDivergenceOutboxTable creates table + indexes on the dedicated table", () => {
    createPlanCounterDivergenceOutboxTable(sql);
    expect(sql.calls).toHaveLength(1);
    const q = sql.calls[0].query;
    expect(q).toContain("CREATE TABLE IF NOT EXISTS pg_sync_outbox_plan_divergences");
    expect(q).toContain("event_id TEXT NOT NULL");
    expect(q).toContain("org_id TEXT NOT NULL");
    expect(q).toContain("tier TEXT NOT NULL");
    expect(q).toContain("divergence_at_ms INTEGER NOT NULL");
    expect(q).toContain("stored_period_start INTEGER NOT NULL");
    expect(q).toContain("incoming_period_start INTEGER NOT NULL");
    expect(q).toContain("CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_divergences_retry_idx");
    expect(q).toContain("CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_divergences_event_id_idx");
    // Separate-table invariant — no reference to the OTHER outbox table names.
    expect(q).not.toMatch(/\bpg_sync_outbox\b(?!_plan_counter|_plan_divergences)/);
    expect(q).not.toContain("pg_sync_outbox_plan_counter");
  });

  it("writePlanCounterDivergenceOutboxEntry captures event_id + divergence_at_ms as binding params (codex R7 C1)", () => {
    writePlanCounterDivergenceOutboxEntry(sql, {
      eventId: "evt-abc-123",
      orgId: "00000000-0000-0000-0000-0000000000aa",
      tier: "free",
      divergenceAtMs: 1_700_000_000_000,
      storedPeriodStart: 1_690_000_000_000,
      incomingPeriodStart: 1_700_000_000_000,
    });
    expect(sql.calls).toHaveLength(1);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("INSERT INTO pg_sync_outbox_plan_divergences");
    expect(query).toContain("VALUES (?, ?, ?, ?, ?, ?, ?, 0)");
    expect(bindings[0]).toBe("evt-abc-123");
    expect(bindings[1]).toBe("00000000-0000-0000-0000-0000000000aa");
    expect(bindings[2]).toBe("free");
    expect(bindings[3]).toBe(1_700_000_000_000); // divergence_at_ms captured at write time
    expect(bindings[4]).toBe(1_690_000_000_000);
    expect(bindings[5]).toBe(1_700_000_000_000);
    expect(typeof bindings[6]).toBe("number"); // created_at
  });

  it("getRetryablePlanCounterDivergenceEntries filters by next_attempt_at AND attempts", () => {
    const now = 1_700_000_000_000;
    const maxAttempts = 5;
    sql.setNextResult({
      toArray: () => [{
        id: 1, eventId: "evt-1", orgId: "org-1", tier: "free",
        divergenceAtMs: 100, storedPeriodStart: 0, incomingPeriodStart: 200,
        attempts: 0, nextAttemptAt: 0, createdAt: 100,
      }],
    });
    const entries = getRetryablePlanCounterDivergenceEntries(sql, now, maxAttempts);
    expect(entries).toHaveLength(1);
    expect(entries[0].eventId).toBe("evt-1");
    expect(entries[0].divergenceAtMs).toBe(100);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("FROM pg_sync_outbox_plan_divergences");
    expect(query).toContain("WHERE next_attempt_at <= ? AND attempts < ?");
    expect(query).toContain("ORDER BY created_at ASC");
    expect(bindings).toEqual([now, maxAttempts]);
  });

  it("ackPlanCounterDivergenceEntryById deletes by row id only (row-id scoped per codex R7)", () => {
    ackPlanCounterDivergenceEntryById(sql, 42);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("DELETE FROM pg_sync_outbox_plan_divergences WHERE id = ?");
    // Importantly, NOT `WHERE event_id = ?` — row-id is guaranteed unique
    // even if a future retry path reuses an event_id.
    expect(query).not.toContain("WHERE event_id");
    expect(bindings).toEqual([42]);
  });

  it("markPlanCounterDivergenceEntryRetryFailed uses exponential backoff schedule", () => {
    markPlanCounterDivergenceEntryRetryFailed(sql, 42, 0);
    let { query, bindings } = sql.calls[0];
    expect(query).toContain("UPDATE pg_sync_outbox_plan_divergences SET attempts = attempts + 1");
    const nextAt0 = bindings[0] as number;
    expect(nextAt0).toBeGreaterThan(Date.now() + 4_000);
    expect(nextAt0).toBeLessThan(Date.now() + 6_000);
    expect(bindings[1]).toBe(42);

    sql.calls.length = 0;
    markPlanCounterDivergenceEntryRetryFailed(sql, 43, 10);
    ({ query, bindings } = sql.calls[0]);
    const nextAt5 = bindings[0] as number;
    expect(nextAt5).toBeGreaterThan(Date.now() + 299_000);
    expect(nextAt5).toBeLessThan(Date.now() + 301_000);
  });

  it("deleteAbandonedPlanCounterDivergenceEntries deletes by attempts >= max and returns count", () => {
    sql.setNextResult({ rowsWritten: 3 });
    const count = deleteAbandonedPlanCounterDivergenceEntries(sql, 5);
    expect(count).toBe(3);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("DELETE FROM pg_sync_outbox_plan_divergences WHERE attempts >= ?");
    expect(bindings).toEqual([5]);
  });

  it("deletePlanCounterDivergenceEntryTerminal deletes by row id (terminal path)", () => {
    deletePlanCounterDivergenceEntryTerminal(sql, 99);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("DELETE FROM pg_sync_outbox_plan_divergences WHERE id = ?");
    expect(bindings).toEqual([99]);
  });
});

describe("isTerminalPlanCounterDivergenceFkError (codex R7 C2 dual-FK classification)", () => {
  it("returns true for 23503 on plan_counter_divergences_org_id_fkey", () => {
    expect(isTerminalPlanCounterDivergenceFkError({
      code: "23503",
      constraint_name: "plan_counter_divergences_org_id_fkey",
    })).toBe(true);
  });

  it("returns true for 23503 on plan_counter_divergence_dedup_org_id_fkey (codex R7 C2 — second FK)", () => {
    expect(isTerminalPlanCounterDivergenceFkError({
      code: "23503",
      constraint_name: "plan_counter_divergence_dedup_org_id_fkey",
    })).toBe(true);
  });

  it("accepts node-postgres `constraint` field (no underscore) for driver compat", () => {
    expect(isTerminalPlanCounterDivergenceFkError({
      code: "23503",
      constraint: "plan_counter_divergences_org_id_fkey",
    })).toBe(true);
  });

  it("returns false for 23503 on a non-allowlisted constraint (e.g., a different table's FK)", () => {
    expect(isTerminalPlanCounterDivergenceFkError({
      code: "23503",
      constraint_name: "some_other_table_org_id_fkey",
    })).toBe(false);
  });

  it("returns false for non-23503 errors", () => {
    expect(isTerminalPlanCounterDivergenceFkError({ code: "57P01" })).toBe(false);
    expect(isTerminalPlanCounterDivergenceFkError({ code: "23505" })).toBe(false);
  });

  it("returns false for non-object errors", () => {
    expect(isTerminalPlanCounterDivergenceFkError(null)).toBe(false);
    expect(isTerminalPlanCounterDivergenceFkError(undefined)).toBe(false);
    expect(isTerminalPlanCounterDivergenceFkError("string error")).toBe(false);
    expect(isTerminalPlanCounterDivergenceFkError(new Error("plain"))).toBe(false);
  });
});

describe("computePlanCounterDivergenceOutboxLagMs", () => {
  it("returns null when outbox is empty", () => {
    const sql = createMockSql();
    sql.setNextResult({ toArray: () => [{ oldest: null }] });
    expect(computePlanCounterDivergenceOutboxLagMs(sql, Date.now())).toBeNull();
  });

  it("returns now - oldest when rows exist", () => {
    const sql = createMockSql();
    const now = 1_700_000_000_000;
    sql.setNextResult({ toArray: () => [{ oldest: now - 5_000 }] });
    expect(computePlanCounterDivergenceOutboxLagMs(sql, now)).toBe(5_000);
  });

  it("clamps negative lag to 0 (clock-skew safety)", () => {
    const sql = createMockSql();
    const now = 1_700_000_000_000;
    sql.setNextResult({ toArray: () => [{ oldest: now + 1_000 }] });
    expect(computePlanCounterDivergenceOutboxLagMs(sql, now)).toBe(0);
  });

  it("queries MIN(created_at) from the correct table", () => {
    const sql = createMockSql();
    sql.setNextResult({ toArray: () => [{ oldest: null }] });
    computePlanCounterDivergenceOutboxLagMs(sql, Date.now());
    expect(sql.calls[0].query).toContain("SELECT MIN(created_at) AS oldest FROM pg_sync_outbox_plan_divergences");
  });
});
