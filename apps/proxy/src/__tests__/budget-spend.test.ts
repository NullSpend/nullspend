/**
 * Budget Spend Unit Tests
 *
 * Tests updateBudgetSpend and resetBudgetPeriod with mocked postgres.js.
 * Validates defensive behavior: early returns on zero/negative cost or
 * empty entities, local dev bypass, transaction ordering, error handling,
 * and idempotent dedup (PXY-2).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock result that simulates a successful INSERT (count > 0)
const insertedResult = Object.assign([] as unknown[], { count: 1 });
// Mock result that simulates a duplicate INSERT (count = 0, dedup hit)
const dedupResult = Object.assign([] as unknown[], { count: 0 });
// Mock result that simulates a successful UPDATE
const updatedResult = Object.assign([] as unknown[], { count: 1 });

const mockTx = vi.fn().mockResolvedValue(insertedResult);
const mockBegin = vi.fn(async (cb: (tx: any) => Promise<void>) => {
  await cb(mockTx);
});
const mockSql = Object.assign(vi.fn().mockResolvedValue([]), { begin: mockBegin });

const mockEmitMetric = vi.fn();

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { updateBudgetSpend, resetBudgetPeriod } from "../lib/budget-spend.js";

const REMOTE_CONN = "postgresql://postgres:postgres@db.example.com:5432/postgres";

describe("updateBudgetSpend", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: INSERT returns count=1 (new row), UPDATE returns count=1 (row found)
    mockTx.mockResolvedValue(insertedResult);
    mockBegin.mockImplementation(async (cb: (tx: any) => Promise<void>) => {
      await cb(mockTx);
    });
  });

  it("early returns when actualCostMicrodollars is 0", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [{ entityType: "api_key", entityId: "key-1" }], 0);
    expect(mockBegin).not.toHaveBeenCalled();
  });

  it("early returns when actualCostMicrodollars is negative", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [{ entityType: "api_key", entityId: "key-1" }], -100);
    expect(mockBegin).not.toHaveBeenCalled();
  });

  it("early returns when entities array is empty", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [], 500_000);
    expect(mockBegin).not.toHaveBeenCalled();
  });

  it("skips DB write when skipDbWrites is true", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [{ entityType: "api_key", entityId: "key-1" }], 500_000, true);
    expect(mockBegin).not.toHaveBeenCalled();
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining("[budget-spend]"), expect.anything());
  });

  it("writes to DB by default (skipDbWrites defaults to false)", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [{ entityType: "user", entityId: "user-1" }], 100_000);
    expect(mockBegin).toHaveBeenCalledOnce();
  });

  it("calls tx for INSERT + UPDATE for each entity inside transaction", async () => {
    await updateBudgetSpend(
      REMOTE_CONN,
      "org-test",
      "req-test",
      [{ entityType: "api_key", entityId: "key-1" }, { entityType: "user", entityId: "user-1" }],
      500_000,
    );

    expect(mockBegin).toHaveBeenCalledOnce();
    // 2 entities × 2 calls each (INSERT + UPDATE) = 4 calls
    expect(mockTx).toHaveBeenCalledTimes(4);
  });

  it("sorts entities by (entityType, entityId) before transaction", async () => {
    await updateBudgetSpend(
      REMOTE_CONN,
      "org-test",
      "req-test",
      [
        { entityType: "user", entityId: "user-1" },
        { entityType: "api_key", entityId: "key-2" },
        { entityType: "api_key", entityId: "key-1" },
      ],
      500_000,
    );

    // 3 entities × 2 calls each (INSERT + UPDATE) = 6 calls
    expect(mockTx).toHaveBeenCalledTimes(6);

    // Verify sort order via INSERT calls (odd-indexed: 0, 2, 4)
    // INSERT tagged template params: [1]=requestId, [2]=entityType, [3]=entityId, [4]=orgId, [5]=cost
    const calls = mockTx.mock.calls;
    // 1st entity: api_key:key-1 (INSERT at index 0)
    expect(calls[0][2]).toBe("api_key");
    expect(calls[0][3]).toBe("key-1");
    // 2nd entity: api_key:key-2 (INSERT at index 2)
    expect(calls[2][2]).toBe("api_key");
    expect(calls[2][3]).toBe("key-2");
    // 3rd entity: user:user-1 (INSERT at index 4)
    expect(calls[4][2]).toBe("user");
    expect(calls[4][3]).toBe("user-1");
  });

  it("throws when transaction fails (for retry by caller)", async () => {
    mockBegin.mockRejectedValueOnce(new Error("connection failed"));

    await expect(
      updateBudgetSpend(REMOTE_CONN, "org-test", "req-test", [{ entityType: "api_key", entityId: "key-1" }], 500_000),
    ).rejects.toThrow("connection failed");
  });

  // Regression: Codex audit P0 #1 — cross-tenant budget corruption.
  // The UPDATE WHERE clause MUST include org_id so two orgs with the same
  // customer entity don't increment each other's budgets.
  it("passes orgId as SQL parameter for tenant isolation", async () => {
    await updateBudgetSpend(
      REMOTE_CONN,
      "org-alpha",
      "req-test",
      [{ entityType: "customer", entityId: "acme-corp" }],
      100_000,
    );

    // 1 entity: INSERT (call 0) + UPDATE (call 1) = 2 calls
    expect(mockTx).toHaveBeenCalledTimes(2);

    // INSERT call (index 0) params: [1]=requestId, [2]=entityType, [3]=entityId, [4]=orgId, [5]=cost
    const insertCall = mockTx.mock.calls[0];
    expect(insertCall[1]).toBe("req-test"); // requestId
    expect(insertCall[2]).toBe("customer"); // entityType
    expect(insertCall[3]).toBe("acme-corp"); // entityId
    expect(insertCall[4]).toBe("org-alpha"); // orgId
    expect(insertCall[5]).toBe(100_000); // cost

    // UPDATE call (index 1) params: [1]=cost, [2]=entityType, [3]=entityId, [4]=orgId
    const updateCall = mockTx.mock.calls[1];
    expect(updateCall[1]).toBe(100_000); // cost
    expect(updateCall[2]).toBe("customer"); // entityType
    expect(updateCall[3]).toBe("acme-corp"); // entityId
    expect(updateCall[4]).toBe("org-alpha"); // orgId
  });

  it("different orgIds produce different SQL parameters (cross-tenant isolation)", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-alpha", "req-test", [{ entityType: "customer", entityId: "acme-corp" }], 50_000);
    await updateBudgetSpend(REMOTE_CONN, "org-beta", "req-test-2", [{ entityType: "customer", entityId: "acme-corp" }], 75_000);

    // 2 calls × 2 tx calls each = 4 total
    expect(mockTx).toHaveBeenCalledTimes(4);

    // INSERT call orgId: call 0 (org-alpha INSERT), call 2 (org-beta INSERT)
    expect(mockTx.mock.calls[0][4]).toBe("org-alpha");
    expect(mockTx.mock.calls[2][4]).toBe("org-beta");

    // UPDATE call orgId: call 1 (org-alpha UPDATE), call 3 (org-beta UPDATE)
    expect(mockTx.mock.calls[1][4]).toBe("org-alpha");
    expect(mockTx.mock.calls[3][4]).toBe("org-beta");
  });

  // --- T7-T11: Idempotent dedup tests (PXY-2) ---

  it("T7: INSERT returns count=1 → UPDATE runs (new reconciliation)", async () => {
    mockTx
      .mockResolvedValueOnce(insertedResult) // INSERT count=1
      .mockResolvedValueOnce(updatedResult); // UPDATE count=1

    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-new", [{ entityType: "user", entityId: "u1" }], 100_000);

    expect(mockTx).toHaveBeenCalledTimes(2); // INSERT + UPDATE
    // No dedup mismatch metric
    expect(mockEmitMetric).not.toHaveBeenCalledWith("reconcile_dedup_cost_mismatch", expect.anything());
  });

  it("T8: INSERT returns count=0 → UPDATE skipped (dedup hit, same cost)", async () => {
    mockTx
      .mockResolvedValueOnce(dedupResult) // INSERT count=0 (dedup)
      .mockResolvedValueOnce([{ cost_microdollars: 100_000 }]); // SELECT returns matching cost

    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-dup", [{ entityType: "user", entityId: "u1" }], 100_000);

    expect(mockTx).toHaveBeenCalledTimes(2); // INSERT + SELECT (no UPDATE)
    // No mismatch metric — costs match
    expect(mockEmitMetric).not.toHaveBeenCalledWith("reconcile_dedup_cost_mismatch", expect.anything());
  });

  it("T9: dedup hit with cost mismatch emits reconcile_dedup_cost_mismatch metric", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockTx
      .mockResolvedValueOnce(dedupResult) // INSERT count=0 (dedup)
      .mockResolvedValueOnce([{ cost_microdollars: 50_000 }]); // SELECT returns DIFFERENT cost

    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-mismatch", [{ entityType: "user", entityId: "u1" }], 100_000);

    expect(mockTx).toHaveBeenCalledTimes(2); // INSERT + SELECT
    expect(console.error).toHaveBeenCalledWith(
      "[budget-spend] DEDUP COST MISMATCH",
      expect.objectContaining({ requestId: "req-mismatch", stored: 50_000, attempted: 100_000 }),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("reconcile_dedup_cost_mismatch", { requestId: "req-mismatch" });
  });

  it("T10: INSERT count=1 but UPDATE count=0 — deletes dedup record for retry + emits metric", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const noRowsUpdated = Object.assign([] as unknown[], { count: 0 });
    mockTx
      .mockResolvedValueOnce(insertedResult) // INSERT count=1
      .mockResolvedValueOnce(noRowsUpdated) // UPDATE count=0 (missing budget row)
      .mockResolvedValueOnce([]); // DELETE dedup record (Codex #2 fix)

    await updateBudgetSpend(REMOTE_CONN, "org-test", "req-orphan", [{ entityType: "user", entityId: "u1" }], 100_000);

    expect(mockTx).toHaveBeenCalledTimes(3); // INSERT + UPDATE + DELETE
    expect(console.error).toHaveBeenCalledWith(
      "[budget-spend] Budget row missing — dedup record removed for retry",
      expect.objectContaining({ requestId: "req-orphan", entityType: "user", entityId: "u1", orgId: "org-test" }),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("reconcile_budget_row_missing", {
      entityType: "user", entityId: "u1",
    });
  });

  it("T11: requestId is passed to INSERT for dedup table", async () => {
    await updateBudgetSpend(REMOTE_CONN, "org-test", "rsv-abc-123", [{ entityType: "api_key", entityId: "k1" }], 200_000);

    // INSERT call (index 0): tagged template params [1]=requestId
    expect(mockTx.mock.calls[0][1]).toBe("rsv-abc-123");
  });
});

describe("resetBudgetPeriod", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTx.mockResolvedValue([]);
    mockBegin.mockImplementation(async (cb: (tx: any) => Promise<void>) => {
      await cb(mockTx);
    });
  });

  it("early returns on empty array", async () => {
    await resetBudgetPeriod(REMOTE_CONN, "org-test", []);
    expect(mockBegin).not.toHaveBeenCalled();
  });

  it("calls tx for each reset inside transaction", async () => {
    await resetBudgetPeriod(REMOTE_CONN, "org-test", [
      { entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 },
      { entityType: "api_key", entityId: "key-1", newPeriodStart: 1_710_000_000_000 },
    ]);

    expect(mockBegin).toHaveBeenCalledOnce();
    expect(mockTx).toHaveBeenCalledTimes(2);
  });

  it("never throws on Postgres error", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockBegin.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    await expect(
      resetBudgetPeriod(REMOTE_CONN, "org-test", [
        { entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 },
      ]),
    ).resolves.toBeUndefined();

    expect(console.error).toHaveBeenCalledWith(
      expect.stringContaining("[budget-spend]"),
      expect.any(String),
    );
  });

  it("skips when skipDbWrites is true", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    await resetBudgetPeriod(REMOTE_CONN, "org-test", [
      { entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 },
    ], true);

    expect(mockBegin).not.toHaveBeenCalled();
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining("[budget-spend]"), expect.anything());
  });

  it("sorts entities before transaction to prevent deadlocks", async () => {
    await resetBudgetPeriod(REMOTE_CONN, "org-test", [
      { entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 },
      { entityType: "api_key", entityId: "key-1", newPeriodStart: 1_710_000_000_000 },
    ]);

    expect(mockTx).toHaveBeenCalledTimes(2);
    const calls = mockTx.mock.calls;
    // First call: api_key:key-1 (sorted)
    expect(calls[0][2]).toBe("api_key");
    expect(calls[0][3]).toBe("key-1");
    // Second call: user:user-1
    expect(calls[1][2]).toBe("user");
    expect(calls[1][3]).toBe("user-1");
  });

  // Regression: Codex P0 #1 — orgId must appear in resetBudgetPeriod SQL
  it("passes orgId as SQL parameter for tenant isolation", async () => {
    await resetBudgetPeriod(REMOTE_CONN, "org-gamma", [
      { entityType: "tag", entityId: "team=eng", newPeriodStart: 1_710_000_000_000 },
    ]);

    expect(mockTx).toHaveBeenCalledTimes(1);
    const call = mockTx.mock.calls[0];
    // Tagged template params: [1]=periodStart, [2]=entityType, [3]=entityId, [4]=orgId
    expect(call[2]).toBe("tag");
    expect(call[3]).toBe("team=eng");
    expect(call[4]).toBe("org-gamma");
  });
});
