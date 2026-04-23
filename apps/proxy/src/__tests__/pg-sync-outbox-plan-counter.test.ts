/**
 * PR-2b: Tests for the pg-sync-outbox-plan-counter module.
 * Mirrors `pg-sync-outbox.test.ts` — uses an in-memory SQLite mock to verify
 * SQL shape, parameter binding, and the separate-table invariant (every SQL
 * statement targets `pg_sync_outbox_plan_counter`, never `pg_sync_outbox`).
 */
import { describe, it, expect, beforeEach } from "vitest";
import {
  createPlanCounterOutboxTable,
  writePlanCounterOutboxEntry,
  getRetryablePlanCounterEntries,
  ackPlanCounterEntryById,
  markPlanCounterEntryRetryFailed,
  deleteAbandonedPlanCounterEntries,
} from "../lib/pg-sync-outbox-plan-counter.js";
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

describe("pg-sync-outbox-plan-counter", () => {
  let sql: ReturnType<typeof createMockSql>;

  beforeEach(() => {
    sql = createMockSql();
  });

  it("createPlanCounterOutboxTable creates table + indexes on the dedicated table", () => {
    createPlanCounterOutboxTable(sql);
    expect(sql.calls).toHaveLength(1);
    const q = sql.calls[0].query;
    expect(q).toContain("CREATE TABLE IF NOT EXISTS pg_sync_outbox_plan_counter");
    expect(q).toContain("period_start INTEGER NOT NULL");
    expect(q).toContain("period_end INTEGER NOT NULL");
    expect(q).toContain("delta_count INTEGER NOT NULL");
    expect(q).toContain("CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_counter_retry_idx");
    expect(q).toContain("CREATE INDEX IF NOT EXISTS pg_sync_outbox_plan_counter_request_id_idx");
    // Separate-table invariant — no reference to the budget-spend table name.
    expect(q).not.toMatch(/\bpg_sync_outbox\b(?!_plan_counter)/);
  });

  it("writePlanCounterOutboxEntry inserts with next_attempt_at=0 and all fields", () => {
    writePlanCounterOutboxEntry(sql, {
      requestId: "req-123",
      orgId: "org-abc",
      periodStart: 1_700_000_000_000,
      periodEnd: 1_702_000_000_000,
      deltaCount: 1,
    });
    expect(sql.calls).toHaveLength(1);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("INSERT INTO pg_sync_outbox_plan_counter");
    expect(query).toContain("VALUES (?, ?, ?, ?, ?, ?, 0)");
    expect(bindings[0]).toBe("req-123");
    expect(bindings[1]).toBe("org-abc");
    expect(bindings[2]).toBe(1_700_000_000_000);
    expect(bindings[3]).toBe(1_702_000_000_000);
    expect(bindings[4]).toBe(1);
    expect(typeof bindings[5]).toBe("number"); // created_at (Date.now())
  });

  it("getRetryablePlanCounterEntries filters by next_attempt_at AND attempts", () => {
    const now = 1_700_000_000_000;
    const maxAttempts = 5;
    sql.setNextResult({
      toArray: () => [{
        id: 1, requestId: "r1", orgId: "o1",
        periodStart: 1, periodEnd: 2, deltaCount: 3,
        attempts: 0, nextAttemptAt: 0, createdAt: 100,
      }],
    });
    const entries = getRetryablePlanCounterEntries(sql, now, maxAttempts);
    expect(entries).toHaveLength(1);
    expect(entries[0].deltaCount).toBe(3);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("FROM pg_sync_outbox_plan_counter");
    expect(query).toContain("WHERE next_attempt_at <= ? AND attempts < ?");
    expect(query).toContain("ORDER BY created_at ASC");
    expect(bindings).toEqual([now, maxAttempts]);
  });

  it("ackPlanCounterEntryById deletes by row id only (codex-final F1 — no collateral sibling deletion)", () => {
    ackPlanCounterEntryById(sql, 42);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("DELETE FROM pg_sync_outbox_plan_counter WHERE id = ?");
    // Importantly, NOT `WHERE request_id = ?` — two rows can share request_id
    // when a cross-period retry exists; ack must target exactly one row.
    expect(query).not.toContain("WHERE request_id");
    expect(bindings).toEqual([42]);
  });

  it("markPlanCounterEntryRetryFailed uses exponential backoff schedule", () => {
    // attempt 0 → 5s backoff
    markPlanCounterEntryRetryFailed(sql, 42, 0);
    let { query, bindings } = sql.calls[0];
    expect(query).toContain("UPDATE pg_sync_outbox_plan_counter SET attempts = attempts + 1");
    const nextAt0 = bindings[0] as number;
    expect(nextAt0).toBeGreaterThan(Date.now() + 4_000);
    expect(nextAt0).toBeLessThan(Date.now() + 6_000);
    expect(bindings[1]).toBe(42);

    // attempt 5 (beyond schedule) clamps to last element (300s)
    sql.calls.length = 0;
    markPlanCounterEntryRetryFailed(sql, 43, 10);
    ({ query, bindings } = sql.calls[0]);
    const nextAt5 = bindings[0] as number;
    expect(nextAt5).toBeGreaterThan(Date.now() + 299_000);
    expect(nextAt5).toBeLessThan(Date.now() + 301_000);
  });

  it("deleteAbandonedPlanCounterEntries deletes by attempts >= max and returns count", () => {
    sql.setNextResult({ rowsWritten: 3 });
    const count = deleteAbandonedPlanCounterEntries(sql, 5);
    expect(count).toBe(3);
    const { query, bindings } = sql.calls[0];
    expect(query).toContain("DELETE FROM pg_sync_outbox_plan_counter WHERE attempts >= ?");
    expect(bindings).toEqual([5]);
  });
});
