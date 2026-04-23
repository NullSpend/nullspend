/**
 * PR-2b: Tests for upsertPlanCounterPeriod.
 * Mocks `getSql` so we verify the SQL template and binding shape without
 * touching a live Postgres instance.
 *
 * Covers build-audit Finding #1 — dedup via `plan_counter_sync_requests`
 * prevents retry-after-partial-commit from double-adding `governed_requests_count`.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the db helper BEFORE importing the SUT.
const sqlSpy = vi.fn();
// `sql.begin(fn)` runs the transaction fn with a `tx` tagged-template that
// forwards to `sqlSpy`. The mock spy captures every template + value set.
sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);

vi.mock("../lib/db.js", () => ({
  getSql: vi.fn(() => sqlSpy),
}));

import { upsertPlanCounterPeriod } from "../lib/upsert-plan-counter.js";

function dedupInserted() {
  // First call in the transaction is the dedup INSERT — return `{ count: 1 }`
  // shape that the SUT's `if (dedup.count === 0)` check depends on.
  sqlSpy.mockResolvedValueOnce({ count: 1 });
}

function dedupHit() {
  sqlSpy.mockResolvedValueOnce({ count: 0 });
}

describe("upsertPlanCounterPeriod", () => {
  beforeEach(() => {
    sqlSpy.mockReset();
    sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);
  });

  it("emits composite dedup INSERT then additive upsert on a first-time (request_id, period_start)", async () => {
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 }); // the usage upsert returns a row count

    const result = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "req-123",
      orgId: "00000000-0000-0000-0000-000000000001",
      periodStart: 1_700_000_000_000,
      periodEnd: 1_702_000_000_000,
      deltaCount: 5,
    });

    expect(result.applied).toBe(true);
    expect(sqlSpy).toHaveBeenCalledTimes(2);

    // Call 1 — three-column dedup INSERT (org_id, request_id, period_start).
    const dedupSql: string[] = sqlSpy.mock.calls[0][0];
    const dedupJoined = dedupSql.join(" ");
    expect(dedupJoined).toContain("INSERT INTO plan_counter_sync_requests (org_id, request_id, period_start)");
    expect(dedupJoined).toContain("ON CONFLICT (org_id, request_id, period_start) DO NOTHING");
    // Dedup INSERT carries org_id, request_id, period_start as bound values.
    expect(sqlSpy.mock.calls[0][1]).toBe("00000000-0000-0000-0000-000000000001");
    expect(sqlSpy.mock.calls[0][2]).toBe("req-123");
    expect(sqlSpy.mock.calls[0][3]).toBe(1_700_000_000_000);

    // Call 2 — usage upsert.
    const usageSql: string[] = sqlSpy.mock.calls[1][0];
    const usageJoined = usageSql.join(" ");
    expect(usageJoined).toContain("INSERT INTO org_period_usage");
    expect(usageJoined).toContain("ON CONFLICT (org_id, period_start) DO UPDATE");
    expect(usageJoined).toContain("governed_requests_count + EXCLUDED.governed_requests_count");
    expect(sqlSpy.mock.calls[1][1]).toBe("00000000-0000-0000-0000-000000000001");
    expect(sqlSpy.mock.calls[1][2]).toBe(1_700_000_000_000);
    expect(sqlSpy.mock.calls[1][3]).toBe(1_702_000_000_000);
    expect(sqlSpy.mock.calls[1][4]).toBe(5);
  });

  it("EC6: same request_id across DIFFERENT periods → both upserts apply (cross-period retry)", async () => {
    // April period — first attempt inserts dedup + upserts.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rApril = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "retry-key-abc",
      orgId: "00000000-0000-0000-0000-000000000001",
      periodStart: 1_704_067_200_000, // April 1
      periodEnd: 1_706_745_600_000,
      deltaCount: 1,
    });
    expect(rApril.applied).toBe(true);

    // May period — SAME request_id, DIFFERENT period_start. Composite key means
    // the dedup does NOT conflict (prior pass: rejected under single-column PK).
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rMay = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "retry-key-abc",
      orgId: "00000000-0000-0000-0000-000000000001",
      periodStart: 1_706_745_600_000, // May 1
      periodEnd: 1_709_251_200_000,
      deltaCount: 1,
    });
    expect(rMay.applied).toBe(true); // CRITICAL: would be false under single-column PK.
  });

  it("codex-final F2: same request_id + same period + DIFFERENT orgs → both upserts apply (cross-org)", async () => {
    // Org A — first attempt inserts dedup + upserts.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rA = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "shared-idem-key",
      orgId: "00000000-0000-0000-0000-00000000000a",
      periodStart: 1_704_067_200_000,
      periodEnd: 1_706_745_600_000,
      deltaCount: 1,
    });
    expect(rA.applied).toBe(true);

    // Org B — SAME request_id + SAME period, DIFFERENT org. With a two-column
    // PK on (request_id, period_start), Org B would have been silently
    // rejected. Three-column PK on (org_id, request_id, period_start) lets
    // each org's dedup namespace stand alone.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rB = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "shared-idem-key",
      orgId: "00000000-0000-0000-0000-00000000000b",
      periodStart: 1_704_067_200_000,
      periodEnd: 1_706_745_600_000,
      deltaCount: 1,
    });
    expect(rB.applied).toBe(true); // CRITICAL: would be false under two-column PK.
  });

  it("retry-after-partial-commit (dedup hit) does NOT upsert — applied=false", async () => {
    dedupHit();

    const result = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "req-duplicate",
      orgId: "00000000-0000-0000-0000-000000000002",
      periodStart: 1, periodEnd: 2, deltaCount: 42,
    });

    expect(result.applied).toBe(false);
    // Only the dedup INSERT ran — no usage upsert.
    expect(sqlSpy).toHaveBeenCalledTimes(1);
    const dedupSql: string[] = sqlSpy.mock.calls[0][0];
    expect(dedupSql.join(" ")).toContain("INSERT INTO plan_counter_sync_requests");
  });

  it("no-ops when deltaCount <= 0 — never even attempts the dedup insert", async () => {
    await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "r", orgId: "o", periodStart: 1, periodEnd: 2, deltaCount: 0,
    });
    await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "r", orgId: "o", periodStart: 1, periodEnd: 2, deltaCount: -3,
    });
    expect(sqlSpy).not.toHaveBeenCalled();
  });

  it("retry-idempotency: double-call with same requestId only applies once", async () => {
    // First call — inserts + upserts.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const r1 = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "req-idempotent",
      orgId: "00000000-0000-0000-0000-000000000003",
      periodStart: 1, periodEnd: 2, deltaCount: 7,
    });
    expect(r1.applied).toBe(true);
    expect(sqlSpy).toHaveBeenCalledTimes(2);

    // Second call — dedup hit, skips upsert.
    dedupHit();
    const r2 = await upsertPlanCounterPeriod("postgres://fake", {
      requestId: "req-idempotent",
      orgId: "00000000-0000-0000-0000-000000000003",
      periodStart: 1, periodEnd: 2, deltaCount: 7,
    });
    expect(r2.applied).toBe(false);
    // Total calls = 2 (first pair) + 1 (second dedup only) = 3 — NOT 4.
    expect(sqlSpy).toHaveBeenCalledTimes(3);
  });
});
