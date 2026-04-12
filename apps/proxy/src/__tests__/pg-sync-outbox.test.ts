/**
 * PXY-2: Tests for the pg-sync-outbox module.
 * Uses an in-memory SQLite mock to verify outbox table operations.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  createOutboxTable,
  writeOutboxEntry,
  getRetryableEntries,
  ackAllForRequest,
  markRetryFailed,
  deleteAbandonedEntries,
  type SqlStorage,
} from "../lib/pg-sync-outbox.js";

// ---------------------------------------------------------------------------
// In-memory SQLite mock
// ---------------------------------------------------------------------------

function createMockSql(): SqlStorage {
  // Simple mock that tracks calls and returns configurable results
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("pg-sync-outbox", () => {
  let sql: ReturnType<typeof createMockSql>;

  beforeEach(() => {
    sql = createMockSql();
  });

  // T1
  it("createOutboxTable creates table idempotently", () => {
    createOutboxTable(sql);
    expect(sql.calls).toHaveLength(1);
    expect(sql.calls[0].query).toContain("CREATE TABLE IF NOT EXISTS pg_sync_outbox");
    expect(sql.calls[0].query).toContain("next_attempt_at");

    // Calling again should not throw
    createOutboxTable(sql);
    expect(sql.calls).toHaveLength(2);
  });

  // T2
  it("writeOutboxEntry inserts with correct fields and defaults", () => {
    const now = Date.now();
    vi.spyOn(Date, "now").mockReturnValue(now);

    writeOutboxEntry(sql, {
      requestId: "rsv-123",
      orgId: "org-abc",
      entityType: "api_key",
      entityId: "key-1",
      costMicrodollars: 500_000,
    });

    expect(sql.calls).toHaveLength(1);
    expect(sql.calls[0].query).toContain("INSERT INTO pg_sync_outbox");
    expect(sql.calls[0].bindings).toEqual([
      "rsv-123",    // request_id
      "org-abc",    // org_id
      "api_key",    // entity_type
      "key-1",      // entity_id
      500_000,      // cost_microdollars
      now,          // created_at
    ]);

    vi.restoreAllMocks();
  });

  // T3
  it("getRetryableEntries returns only entries where next_attempt_at <= now AND attempts < max", () => {
    const now = 1_000_000;
    const maxAttempts = 5;

    sql.setNextResult({
      toArray: () => [
        {
          id: 1,
          requestId: "rsv-1",
          orgId: "org-1",
          entityType: "api_key",
          entityId: "key-1",
          costMicrodollars: 100,
          attempts: 2,
          nextAttemptAt: 999_000,
          createdAt: 900_000,
        },
      ],
    });

    const entries = getRetryableEntries(sql, now, maxAttempts);

    expect(entries).toHaveLength(1);
    expect(entries[0].requestId).toBe("rsv-1");
    expect(sql.calls[0].query).toContain("next_attempt_at <= ?");
    expect(sql.calls[0].query).toContain("attempts < ?");
    expect(sql.calls[0].bindings).toEqual([now, maxAttempts]);
  });

  // T4
  it("ackAllForRequest deletes ALL entries for a requestId (multi-entity)", () => {
    ackAllForRequest(sql, "rsv-123");

    expect(sql.calls).toHaveLength(1);
    expect(sql.calls[0].query).toContain("DELETE FROM pg_sync_outbox WHERE request_id = ?");
    expect(sql.calls[0].bindings).toEqual(["rsv-123"]);
  });

  // T5
  it("markRetryFailed increments attempts and sets next_attempt_at with backoff", () => {
    const now = 1_000_000;
    vi.spyOn(Date, "now").mockReturnValue(now);

    // First failure: backoff[0] = 5_000ms
    markRetryFailed(sql, 42, 0);
    expect(sql.calls[0].query).toContain("attempts = attempts + 1");
    expect(sql.calls[0].query).toContain("next_attempt_at = ?");
    expect(sql.calls[0].bindings[0]).toBe(now + 5_000); // first backoff
    expect(sql.calls[0].bindings[1]).toBe(42); // id

    // Third failure: backoff[2] = 45_000ms
    markRetryFailed(sql, 42, 2);
    expect(sql.calls[1].bindings[0]).toBe(now + 45_000);

    // Beyond schedule: caps at last backoff = 300_000ms
    markRetryFailed(sql, 42, 10);
    expect(sql.calls[2].bindings[0]).toBe(now + 300_000);

    vi.restoreAllMocks();
  });

  // T6
  it("deleteAbandonedEntries removes entries at max attempts and returns count", () => {
    sql.setNextResult({ rowsWritten: 3 });

    const count = deleteAbandonedEntries(sql, 5);

    expect(count).toBe(3);
    expect(sql.calls[0].query).toContain("DELETE FROM pg_sync_outbox WHERE attempts >= ?");
    expect(sql.calls[0].bindings).toEqual([5]);
  });
});
