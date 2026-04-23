/**
 * PR-2d C60 — `computePlanCounterOutboxLagMs` pure-helper test.
 *
 * The DO integration side of C60 (metric actually emits during a flush tick)
 * lives in `plan-counter.do.test.ts`. This file pins the math + null-handling
 * at a pure-unit granularity so future changes have a tight failure signal.
 */
import { describe, it, expect } from "vitest";
import { computePlanCounterOutboxLagMs } from "../lib/pg-sync-outbox-plan-counter.js";
import type { SqlStorage } from "../lib/pg-sync-outbox.js";

/**
 * In-memory `SqlStorage` stand-in. Responds to the single query shape the
 * helper issues — `SELECT MIN(created_at) AS oldest FROM pg_sync_outbox_plan_counter`
 * — by returning the pre-seeded rows' minimum. Any other query throws so
 * query-shape regressions show up as a test failure instead of a false green.
 */
function makeMockSql(rows: Array<{ created_at: number }>): SqlStorage {
  return {
    exec: (query: string) => {
      if (query.includes("MIN(created_at)") && query.includes("pg_sync_outbox_plan_counter")) {
        const oldest = rows.length === 0
          ? null
          : rows.reduce((min, r) => (r.created_at < min ? r.created_at : min), rows[0].created_at);
        return {
          toArray: () => [{ oldest }],
          rowsWritten: 0,
        };
      }
      throw new Error(`Unexpected query shape: ${query}`);
    },
  } as unknown as SqlStorage;
}

describe("computePlanCounterOutboxLagMs (C60)", () => {
  it("returns null when the table is empty", () => {
    const sql = makeMockSql([]);
    expect(computePlanCounterOutboxLagMs(sql, Date.now())).toBeNull();
  });

  it("returns (now - oldest) when a single row exists", () => {
    const now = 1_700_000_000_000;
    const oldest = now - 5_000;
    const sql = makeMockSql([{ created_at: oldest }]);
    expect(computePlanCounterOutboxLagMs(sql, now)).toBe(5_000);
  });

  it("returns (now - MIN(created_at)) across multiple rows (oldest wins)", () => {
    const now = 1_700_000_000_000;
    const sql = makeMockSql([
      { created_at: now - 1_000 },
      { created_at: now - 60_000 },  // oldest
      { created_at: now - 5_000 },
    ]);
    expect(computePlanCounterOutboxLagMs(sql, now)).toBe(60_000);
  });

  it("clamps negative lag to 0 when the oldest row's created_at is in the future (clock skew)", () => {
    const now = 1_700_000_000_000;
    const futureCreatedAt = now + 3_000;  // DO wrote +3s "ahead" of this call
    const sql = makeMockSql([{ created_at: futureCreatedAt }]);
    expect(computePlanCounterOutboxLagMs(sql, now)).toBe(0);
  });

  it("returns 0 (not null) when created_at equals now", () => {
    const now = 1_700_000_000_000;
    const sql = makeMockSql([{ created_at: now }]);
    expect(computePlanCounterOutboxLagMs(sql, now)).toBe(0);
  });

  it("returns null when the query result row shape is defensive (oldest undefined)", () => {
    // Covers the `row?.oldest === undefined` branch explicitly.
    const sql = {
      exec: () => ({
        toArray: () => [{}],
        rowsWritten: 0,
      }),
    } as unknown as SqlStorage;
    expect(computePlanCounterOutboxLagMs(sql, Date.now())).toBeNull();
  });

  it("returns null when the query returns zero rows", () => {
    const sql = {
      exec: () => ({
        toArray: () => [],
        rowsWritten: 0,
      }),
    } as unknown as SqlStorage;
    expect(computePlanCounterOutboxLagMs(sql, Date.now())).toBeNull();
  });
});
