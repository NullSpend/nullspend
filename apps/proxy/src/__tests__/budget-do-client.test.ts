import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockUpdateBudgetSpend, mockEmitMetric } = vi.hoisted(() => ({
  mockUpdateBudgetSpend: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: (...args: unknown[]) => mockUpdateBudgetSpend(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../durable-objects/user-budget.js", () => ({}));

import { doBudgetCheck, doBudgetReconcile, doBudgetUpsertEntities, doBudgetRemove, doBudgetResetSpend } from "../lib/budget-do-client.js";

function makeStub(overrides: Record<string, unknown> = {}) {
  const stub: Record<string, unknown> = {
    checkAndReserve: vi.fn().mockResolvedValue({ status: "approved", reservationId: "rsv-1" }),
    reconcile: vi.fn().mockResolvedValue({ status: "reconciled" }),
    populateIfEmpty: vi.fn().mockResolvedValue(true),
    removeBudget: vi.fn().mockResolvedValue(undefined),
    resetSpend: vi.fn().mockResolvedValue(undefined),
    getBudgetState: vi.fn().mockResolvedValue([]),
    ...overrides,
  };
  // Track upserted entities so getBudgetState returns them for verification
  const upserted: Array<{ entity_type: string; entity_id: string }> = [];
  (stub.populateIfEmpty as ReturnType<typeof vi.fn>).mockImplementation(
    (entityType: string, entityId: string) => {
      upserted.push({ entity_type: entityType, entity_id: entityId });
      (stub.getBudgetState as ReturnType<typeof vi.fn>).mockResolvedValue([...upserted]);
      return Promise.resolve(true);
    },
  );
  return stub;
}

function makeEnv(stub: ReturnType<typeof makeStub>): Env {
  return {
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue(stub),
    },
  } as unknown as Env;
}

describe("doBudgetCheck", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls stub.checkAndReserve with keyId and returns CheckResult", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    const result = await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, []);

    expect(stub.checkAndReserve).toHaveBeenCalledWith("key-1", 5_000_000, 30_000, null, []);
    expect(result).toEqual({ status: "approved", reservationId: "rsv-1" });
  });

  it("passes null keyId when no API key", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", null, 5_000_000, null, []);

    expect(stub.checkAndReserve).toHaveBeenCalledWith(null, 5_000_000, 30_000, null, []);
  });

  it("returns denied result from DO", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockResolvedValue({
        status: "denied",
        hasBudgets: true,
        deniedEntity: "user:user-1",
        remaining: 100,
        maxBudget: 50_000_000,
        spend: 49_999_900,
      }),
    });
    const env = makeEnv(stub);

    const result = await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, []);

    expect(result.status).toBe("denied");
    expect(result.deniedEntity).toBe("user:user-1");
  });

  it("throws on DO error (fail-closed)", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockRejectedValue(new Error("DO unavailable")),
    });
    const env = makeEnv(stub);

    await expect(
      doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, []),
    ).rejects.toThrow("DO unavailable");
  });

  it("emits do_budget_check metric with status, hasBudgets, durationMs", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockResolvedValue({
        status: "approved",
        hasBudgets: true,
        reservationId: "rsv-1",
      }),
    });
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, []);

    expect(mockEmitMetric).toHaveBeenCalledWith("do_budget_check", {
      status: "approved",
      hasBudgets: true,
      durationMs: expect.any(Number),
      velocityDenied: false,
      velocityRecovered: false,
      sessionLimitDenied: false,
      tagBudgetDenied: false,
    });
  });

  it("emits do_budget_check metric with hasBudgets=false for tracking-only users", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockResolvedValue({
        status: "approved",
        hasBudgets: false,
      }),
    });
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", null, 5_000_000, null, []);

    expect(mockEmitMetric).toHaveBeenCalledWith("do_budget_check", {
      status: "approved",
      hasBudgets: false,
      durationMs: expect.any(Number),
      velocityDenied: false,
      velocityRecovered: false,
      sessionLimitDenied: false,
      tagBudgetDenied: false,
    });
  });

  it("passes tagEntityIds to stub.checkAndReserve", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, ["project=openclaw", "env=prod"]);

    expect(stub.checkAndReserve).toHaveBeenCalledWith(
      "key-1", 5_000_000, 30_000, null, ["project=openclaw", "env=prod"],
    );
  });

  it("passes empty tagEntityIds with session", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", "key-1", 5_000_000, "sess-1", []);

    expect(stub.checkAndReserve).toHaveBeenCalledWith("key-1", 5_000_000, 30_000, "sess-1", []);
  });

  it("emits tagBudgetDenied=true when deniedEntity starts with tag:", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockResolvedValue({
        status: "denied",
        hasBudgets: true,
        deniedEntity: "tag:project=openclaw",
        remaining: 0,
        maxBudget: 50_000_000,
        spend: 50_000_000,
      }),
    });
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, ["project=openclaw"]);

    expect(mockEmitMetric).toHaveBeenCalledWith("do_budget_check", expect.objectContaining({
      tagBudgetDenied: true,
    }));
  });

  it("emits tagBudgetDenied=false for non-tag denials", async () => {
    const stub = makeStub({
      checkAndReserve: vi.fn().mockResolvedValue({
        status: "denied",
        hasBudgets: true,
        deniedEntity: "user:user-1",
        remaining: 0,
        maxBudget: 50_000_000,
        spend: 50_000_000,
      }),
    });
    const env = makeEnv(stub);

    await doBudgetCheck(env, "user-1", "key-1", 5_000_000, null, []);

    expect(mockEmitMetric).toHaveBeenCalledWith("do_budget_check", expect.objectContaining({
      tagBudgetDenied: false,
    }));
  });
});

describe("doBudgetReconcile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdateBudgetSpend.mockResolvedValue(undefined);
  });

  it("calls stub.reconcile + updateBudgetSpend", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env,
      "user-1",
      "rsv-1",
      1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(stub.reconcile).toHaveBeenCalledWith("rsv-1", 1_000);
    expect(mockUpdateBudgetSpend).toHaveBeenCalledWith(
      "postgres://test",
      [{ entityType: "user", entityId: "user-1" }],
      1_000,
    );
  });

  it("skips updateBudgetSpend when actualCost=0", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env, "user-1", "rsv-1", 0,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(stub.reconcile).toHaveBeenCalledWith("rsv-1", 0);
    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
  });

  it("returns 'error' on DO error", async () => {
    const stub = makeStub({
      reconcile: vi.fn().mockRejectedValue(new Error("DO error")),
    });
    const env = makeEnv(stub);

    await expect(
      doBudgetReconcile(env, "user-1", "rsv-1", 1_000, [{ entityType: "user", entityId: "user-1" }], "postgres://test"),
    ).resolves.toBe("error");
  });

  it("returns 'pg_failed' on Postgres error (retries exhausted)", async () => {
    vi.useFakeTimers();
    const stub = makeStub();
    const env = makeEnv(stub);
    mockUpdateBudgetSpend.mockRejectedValue(new Error("PG error"));

    const promise = doBudgetReconcile(env, "user-1", "rsv-1", 500, [{ entityType: "user", entityId: "user-1" }], "postgres://test");
    await vi.advanceTimersByTimeAsync(1_000);
    await expect(promise).resolves.toBe("pg_failed");
    vi.useRealTimers();
  });

  it("retries Postgres write and succeeds on second attempt", async () => {
    vi.useFakeTimers();
    const stub = makeStub();
    const env = makeEnv(stub);
    mockUpdateBudgetSpend
      .mockRejectedValueOnce(new Error("PG transient"))
      .mockResolvedValueOnce(undefined);

    const promise = doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );
    await vi.advanceTimersByTimeAsync(1_000);
    await promise;

    expect(mockUpdateBudgetSpend).toHaveBeenCalledTimes(2);
    expect(mockEmitMetric).toHaveBeenCalledWith("do_reconciliation", expect.objectContaining({
      status: "ok",
      retries: 1,
    }));
    vi.useRealTimers();
  });

  it("all retries exhausted → structured error + split-brain warning + metric", async () => {
    vi.useFakeTimers();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const stub = makeStub();
    const env = makeEnv(stub);
    mockUpdateBudgetSpend.mockRejectedValue(new Error("PG persistent"));

    const promise = doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );
    await vi.advanceTimersByTimeAsync(1_000);
    await promise;

    // 3 attempts total (0, 1, 2)
    expect(mockUpdateBudgetSpend).toHaveBeenCalledTimes(3);

    // Structured error logged
    expect(errorSpy).toHaveBeenCalledWith(
      "[budget-do-client] Postgres write failed after retries",
      expect.objectContaining({ reservationId: "rsv-1" }),
    );

    // Split-brain warning logged
    expect(errorSpy).toHaveBeenCalledWith(
      "[budget-do-client] DO/Postgres split-brain: reservation",
      "rsv-1",
    );

    // Metric emitted with pg_failed status
    expect(mockEmitMetric).toHaveBeenCalledWith("do_reconciliation", expect.objectContaining({
      status: "pg_failed",
      retries: 2,
    }));

    errorSpy.mockRestore();
    vi.useRealTimers();
  });

  it("skips Postgres write when DO returns not_found (already reconciled)", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const stub = makeStub({
      reconcile: vi.fn().mockResolvedValue({ status: "not_found" }),
    });
    const env = makeEnv(stub);

    const result = await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(result).toBe("ok");
    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledWith(
      "[budget-do-client] Reservation not found in DO (already reconciled?)",
      expect.objectContaining({ reservationId: "rsv-1" }),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("reconcile_not_found", expect.objectContaining({
      reservationId: "rsv-1",
      costMicrodollars: 1_000,
    }));
    warnSpy.mockRestore();
  });

  it("DO stub.reconcile failure → error status returned, Postgres write skipped", async () => {
    const stub = makeStub({
      reconcile: vi.fn().mockRejectedValue(new Error("DO error")),
    });
    const env = makeEnv(stub);

    const result = await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(result).toBe("error");
    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
    expect(mockEmitMetric).toHaveBeenCalledWith("do_reconciliation", expect.objectContaining({
      status: "error",
    }));
  });

  it("emits do_reconciliation metric on every call", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(mockEmitMetric).toHaveBeenCalledWith("do_reconciliation", expect.objectContaining({
      status: "ok",
      costMicrodollars: 1_000,
      durationMs: expect.any(Number),
      retries: 0,
    }));
  });

  it("emits metric even when actualCost=0 (no Postgres write)", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env, "user-1", "rsv-1", 0,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(mockEmitMetric).toHaveBeenCalledWith("do_reconciliation", expect.objectContaining({
      status: "ok",
      costMicrodollars: 0,
    }));
  });

  it("returns 'ok' on successful reconciliation", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    const result = await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(result).toBe("ok");
  });

  it("emits reconcile_budget_missing metric when DO reports missing budgets", async () => {
    const stub = makeStub({
      reconcile: vi.fn().mockResolvedValue({
        status: "reconciled",
        spends: {},
        budgetsMissing: ["user:u1"],
      }),
    });
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(mockEmitMetric).toHaveBeenCalledWith("reconcile_budget_missing", {
      reservationId: "rsv-1",
      costMicrodollars: 1_000,
      budgetsMissing: ["user:u1"],
    });
  });

  it("no reconcile_budget_missing metric when budgetsMissing is absent", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetReconcile(
      env, "user-1", "rsv-1", 1_000,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(mockEmitMetric).not.toHaveBeenCalledWith(
      "reconcile_budget_missing",
      expect.anything(),
    );
  });

  it("returns 'ok' when actualCost=0", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    const result = await doBudgetReconcile(
      env, "user-1", "rsv-1", 0,
      [{ entityType: "user", entityId: "user-1" }],
      "postgres://test",
    );

    expect(result).toBe("ok");
  });
});

describe("doBudgetRemove", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls stub.removeBudget with correct args", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetRemove(env, "user-1", "api_key", "key-1");

    expect(stub.removeBudget).toHaveBeenCalledWith("api_key", "key-1");
  });

  it("throws on DO error (fail-closed)", async () => {
    const stub = makeStub({
      removeBudget: vi.fn().mockRejectedValue(new Error("DO error")),
    });
    const env = makeEnv(stub);

    await expect(
      doBudgetRemove(env, "user-1", "api_key", "key-1"),
    ).rejects.toThrow("DO error");
  });
});

describe("doBudgetResetSpend", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls stub.resetSpend with correct args", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetResetSpend(env, "user-1", "user", "user-1");

    expect(stub.resetSpend).toHaveBeenCalledWith("user", "user-1");
  });

  it("throws on DO error (fail-closed)", async () => {
    const stub = makeStub({
      resetSpend: vi.fn().mockRejectedValue(new Error("DO error")),
    });
    const env = makeEnv(stub);

    await expect(
      doBudgetResetSpend(env, "user-1", "user", "user-1"),
    ).rejects.toThrow("DO error");
  });
});

describe("doBudgetUpsertEntities", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls populateIfEmpty for each entity", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetUpsertEntities(env, "user-1", [
      {
        entityType: "user",
        entityId: "user-1",
        maxBudget: 50_000_000,
        spend: 10_000_000,
        policy: "strict_block",
        resetInterval: "monthly",
        periodStart: 1_700_000_000_000,
        velocityLimit: null,
        velocityWindow: 60_000,
        velocityCooldown: 60_000,
        thresholdPercentages: [50, 80, 90, 95],
        sessionLimit: null,
      },
    ]);

    expect(stub.populateIfEmpty).toHaveBeenCalledTimes(1);
    expect(stub.populateIfEmpty).toHaveBeenCalledWith(
      "user", "user-1", 50_000_000, 10_000_000,
      "strict_block", "monthly", 1_700_000_000_000,
      null, 60_000, 60_000, [50, 80, 90, 95], null,
    );
  });

  it("handles multiple entities", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetUpsertEntities(env, "user-1", [
      { entityType: "user", entityId: "user-1", maxBudget: 50_000_000, spend: 0, policy: "strict_block", resetInterval: null, periodStart: 0, velocityLimit: null, velocityWindow: 60_000, velocityCooldown: 60_000, thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
      { entityType: "api_key", entityId: "key-1", maxBudget: 10_000_000, spend: 0, policy: "strict_block", resetInterval: null, periodStart: 0, velocityLimit: null, velocityWindow: 60_000, velocityCooldown: 60_000, thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
    ]);

    expect(stub.populateIfEmpty).toHaveBeenCalledTimes(2);
  });

  it("handles empty entity list", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetUpsertEntities(env, "user-1", []);

    expect(stub.populateIfEmpty).not.toHaveBeenCalled();
    expect(stub.getBudgetState).not.toHaveBeenCalled();
  });

  it("verifies entities after upsert and calls getBudgetState", async () => {
    const stub = makeStub();
    const env = makeEnv(stub);

    await doBudgetUpsertEntities(env, "user-1", [
      { entityType: "user", entityId: "user-1", maxBudget: 50_000_000, spend: 0, policy: "strict_block", resetInterval: null, periodStart: 0, velocityLimit: null, velocityWindow: 60_000, velocityCooldown: 60_000, thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
    ]);

    expect(stub.getBudgetState).toHaveBeenCalledTimes(1);
  });

  it("retries populateIfEmpty if entity is missing after upsert", async () => {
    const stub = makeStub();
    // Override: first upsert "succeeds" but getBudgetState returns empty,
    // simulating the race condition where DO hasn't committed yet
    let callCount = 0;
    (stub.populateIfEmpty as ReturnType<typeof vi.fn>).mockImplementation(
      (entityType: string, entityId: string) => {
        callCount++;
        // First call: don't update getBudgetState (simulate race)
        // Second call (retry): update it
        if (callCount >= 2) {
          (stub.getBudgetState as ReturnType<typeof vi.fn>).mockResolvedValue([
            { entity_type: entityType, entity_id: entityId },
          ]);
        }
        return Promise.resolve(true);
      },
    );
    (stub.getBudgetState as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const env = makeEnv(stub);

    await doBudgetUpsertEntities(env, "user-1", [
      { entityType: "user", entityId: "user-1", maxBudget: 50_000_000, spend: 0, policy: "strict_block", resetInterval: null, periodStart: 0, velocityLimit: null, velocityWindow: 60_000, velocityCooldown: 60_000, thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
    ]);

    // Called twice: once for initial upsert, once for retry
    expect(stub.populateIfEmpty).toHaveBeenCalledTimes(2);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_sync_retry", {
      ownerId: "user-1",
      missingCount: 1,
    });
  });
});
