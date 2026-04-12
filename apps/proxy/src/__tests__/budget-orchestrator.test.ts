import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  mockWaitUntil,
  mockDoBudgetCheck,
  mockDoBudgetReconcile,
  mockResetBudgetPeriod,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((p: Promise<unknown>) => { p.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockResetBudgetPeriod: vi.fn(),
}));

vi.mock("cloudflare:workers", () => ({
  waitUntil: mockWaitUntil,
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: (...args: unknown[]) => mockDoBudgetCheck(...args),
  doBudgetReconcile: (...args: unknown[]) => mockDoBudgetReconcile(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: vi.fn().mockResolvedValue(undefined),
  resetBudgetPeriod: (...args: unknown[]) => mockResetBudgetPeriod(...args),
}));

import { checkBudget, reconcileBudget } from "../lib/budget-orchestrator.js";
import type { RequestContext } from "../lib/context.js";

function makeCtx(overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    body: {},
    bodyText: "{}",
    auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-1",
    connectionString: "postgres://test",
    sessionId: null,
    traceId: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    tags: {},
    customerId: null,
    customerWarning: null,
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    ...overrides,
  };
}

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
    ...overrides,
  } as unknown as Env;
}

const keyEntity = {
  entityKey: "{budget}:api_key:key-1",
  entityType: "api_key",
  entityId: "key-1",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  reserved: 0,
  policy: "strict_block",
};

const checkedEntity = {
  entityType: "user",
  entityId: "user-1",
  maxBudget: 100_000_000,
  spend: 20_000_000,
  reserved: 0,
  policy: "strict_block",
};

describe("checkBudget — DO-first mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockResetBudgetPeriod.mockResolvedValue(undefined);
  });

  it("single DO RPC → correct outcome", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(result.status).toBe("approved");
    expect(result.reservationId).toBe("rsv-do-1");
    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(), "user-1", "key-1", 5_000_000, null, [], null,
    );
  });

  it("builds budgetEntities from checkedEntities in DO response", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(result.budgetEntities[0].entityKey).toBe("{budget}:user:user-1");
    expect(result.budgetEntities[0].reserved).toBe(0);
    expect(result.budgetEntities[0].maxBudget).toBe(100_000_000);
    expect(result.budgetEntities[0].spend).toBe(20_000_000);
    expect(result.budgetEntities[0].policy).toBe("strict_block");
  });

  it("denied with parsed deniedEntity (type + id)", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      remaining: 500,
      maxBudget: 100_000_000,
      spend: 99_999_500,
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(result.status).toBe("denied");
    expect(result.deniedEntityType).toBe("user");
    expect(result.deniedEntityId).toBe("user-1");
  });

  it("skipped when hasBudgets=false (no budgets in DO)", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: false,
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(result.status).toBe("skipped");
    expect(result.budgetEntities).toEqual([]);
  });

  it("skipped when no userId", async () => {
    const ctx = makeCtx({ auth: { userId: null, keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} }, ownerId: null as any });

    const result = await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(result.status).toBe("skipped");
    expect(mockDoBudgetCheck).not.toHaveBeenCalled();
  });

  it("throws on DO error (fail-closed)", async () => {
    mockDoBudgetCheck.mockRejectedValue(new Error("DO unavailable"));

    await expect(
      checkBudget(makeEnv(), makeCtx(), 5_000_000),
    ).rejects.toThrow("DO unavailable");
  });

  it("calls resetBudgetPeriod when periodResets present", async () => {
    const resets = [{ entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 }];
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      periodResets: resets,
      checkedEntities: [checkedEntity],
    });

    await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(mockResetBudgetPeriod).toHaveBeenCalledWith("postgres://test", null, resets);
  });

  it("skips resetBudgetPeriod when no periodResets", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      checkedEntities: [checkedEntity],
    });

    await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(mockResetBudgetPeriod).not.toHaveBeenCalled();
  });

  it("resetBudgetPeriod failure does not crash checkBudgetDO", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const resets = [{ entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 }];
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      periodResets: resets,
      checkedEntities: [checkedEntity],
    });
    mockResetBudgetPeriod.mockRejectedValue(new Error("Postgres down"));

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    // waitUntil catches the error — wait for microtask
    await new Promise((r) => setTimeout(r, 10));

    expect(result.status).toBe("approved");
    expect(errorSpy).toHaveBeenCalledWith(
      "[budget-orchestrator] Period reset write-back failed:",
      expect.any(Error),
    );
    errorSpy.mockRestore();
  });

  it("skips resetBudgetPeriod when connectionString is falsy", async () => {
    const resets = [{ entityType: "user", entityId: "user-1", newPeriodStart: 1_710_000_000_000 }];
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-do-1",
      periodResets: resets,
      checkedEntities: [checkedEntity],
    });

    await checkBudget(makeEnv(), makeCtx({ connectionString: "" }), 5_000_000);

    expect(mockResetBudgetPeriod).not.toHaveBeenCalled();
  });

  it("passes keyId=null to DO when auth has no keyId", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: false,
    });

    const ctx = makeCtx({ auth: { userId: "user-1", keyId: null, hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(), "user-1", null, 5_000_000, null, [], null,
    );
  });

  it("checkedEntities populated correctly in budgetEntities on approved", async () => {
    const entities = [
      { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 20_000_000, reserved: 0, policy: "strict_block" },
      { entityType: "api_key", entityId: "key-1", maxBudget: 50_000_000, spend: 5_000_000, reserved: 0, policy: "strict_block" },
    ];
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-multi",
      checkedEntities: entities,
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 5_000_000);

    expect(result.budgetEntities).toHaveLength(2);
    expect(result.budgetEntities[0].entityKey).toBe("{budget}:user:user-1");
    expect(result.budgetEntities[1].entityKey).toBe("{budget}:api_key:key-1");
    expect(result.budgetEntities[1].maxBudget).toBe(50_000_000);
  });

  it("budgetEntities carry live reserved from checkedEntities (not hardcoded 0)", async () => {
    // Regression guard: prior to 2026-04-08 the orchestrator hardcoded reserved:0
    // on the approved path, which made concurrent-request "remaining" calculations
    // inaccurate. The budget response headers feature depends on this being live.
    const entities = [
      { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 20_000_000, reserved: 7_500_000, policy: "strict_block" },
      { entityType: "org", entityId: "org-1", maxBudget: 500_000_000, spend: 100_000_000, reserved: 25_000_000, policy: "strict_block" },
    ];
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-reserved",
      checkedEntities: entities,
    });

    const result = await checkBudget(makeEnv(), makeCtx(), 1_000_000);

    expect(result.budgetEntities).toHaveLength(2);
    expect(result.budgetEntities[0].reserved).toBe(7_500_000);
    expect(result.budgetEntities[1].reserved).toBe(25_000_000);
  });

  it("T26: checkBudget passes ctx.auth.orgId through to doBudgetCheck", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-org",
      checkedEntities: [checkedEntity],
    });

    const ctx = makeCtx({
      auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: "org-xyz", apiVersion: "2026-04-01", defaultTags: {} },
    });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(), "user-1", "key-1", 5_000_000, null, [], "org-xyz",
    );
  });
});

describe("reconcileBudget — durable-objects mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDoBudgetReconcile.mockResolvedValue("ok");
  });

  it("calls doBudgetReconcile with userId", async () => {
    await reconcileBudget(
      makeEnv(), "user-1", "org-test", "rsv-1", 1_000,
      [keyEntity], "postgres://test",
    );

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(), "user-1", "org-test", "rsv-1", 1_000,
      [{ entityType: "api_key", entityId: "key-1" }],
      "postgres://test",
    );
  });

  it("handles actualCost=0 (upstream error path)", async () => {
    await reconcileBudget(
      makeEnv(), "user-1", "org-test", "rsv-1", 0,
      [keyEntity], "postgres://test",
    );

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(), "user-1", "org-test", "rsv-1", 0,
      expect.any(Array),
      "postgres://test",
    );
  });

  it("never throws by default", async () => {
    mockDoBudgetReconcile.mockRejectedValue(new Error("DO fail"));

    await expect(
      reconcileBudget(makeEnv(), "user-1", "org-test", "rsv-1", 1_000, [keyEntity], "postgres://test"),
    ).resolves.toBeUndefined();
  });

  it("throwOnError: throws when status is not ok", async () => {
    mockDoBudgetReconcile.mockResolvedValue("pg_failed");
    vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      reconcileBudget(makeEnv(), "user-1", "org-test", "rsv-1", 1_000, [keyEntity], "postgres://test", { throwOnError: true }),
    ).rejects.toThrow("Reconciliation failed with status: pg_failed");
  });

  it("throwOnError: does not throw when status is ok", async () => {
    mockDoBudgetReconcile.mockResolvedValue("ok");

    await expect(
      reconcileBudget(makeEnv(), "user-1", "org-test", "rsv-1", 1_000, [keyEntity], "postgres://test", { throwOnError: true }),
    ).resolves.toBeUndefined();
  });

  it("throwOnError: re-throws doBudgetReconcile rejection", async () => {
    mockDoBudgetReconcile.mockRejectedValue(new Error("DO exploded"));
    vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      reconcileBudget(makeEnv(), "user-1", "org-test", "rsv-1", 1_000, [keyEntity], "postgres://test", { throwOnError: true }),
    ).rejects.toThrow("DO exploded");
  });
});

// ── Tag budget orchestrator tests ─────────────────────────────────

describe("checkBudget — tag budget", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockResetBudgetPeriod.mockResolvedValue(undefined);
  });

  it("converts ctx.tags to tagEntityIds", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    const ctx = makeCtx({ tags: { project: "openclaw", env: "prod" } });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(), "user-1", "key-1", 5_000_000, null,
      ["project=openclaw", "env=prod"], null,
    );
  });

  it("empty tags → null tagEntityIds", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    const ctx = makeCtx({ tags: {} });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(), "user-1", "key-1", 5_000_000, null, [], null,
    );
  });

  it("tag denial returns tagBudgetDenied=true with tagKey/tagValue", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "tag:project=openclaw",
      remaining: 0,
      maxBudget: 50_000_000,
      spend: 50_000_000,
      checkedEntities: [],
    });

    const ctx = makeCtx({ tags: { project: "openclaw" } });
    const result = await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(result.status).toBe("denied");
    expect(result.tagBudgetDenied).toBe(true);
    expect(result.tagKey).toBe("project");
    expect(result.tagValue).toBe("openclaw");
    expect(result.deniedEntityType).toBe("tag");
    expect(result.deniedEntityId).toBe("project=openclaw");
  });

  it("edge case: tag value with = (e.g., env=a=b) → tagKey=env, tagValue=a=b", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "tag:env=a=b",
      remaining: 0,
      maxBudget: 50_000_000,
      spend: 50_000_000,
      checkedEntities: [],
    });

    const ctx = makeCtx({ tags: { env: "a=b" } });
    const result = await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(result.tagKey).toBe("env");
    expect(result.tagValue).toBe("a=b");
  });

  it("non-tag denial still returns generic budget_exceeded outcome", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      remaining: 0,
      maxBudget: 100_000_000,
      spend: 100_000_000,
      checkedEntities: [],
    });

    const ctx = makeCtx({ tags: { project: "openclaw" } });
    const result = await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(result.status).toBe("denied");
    expect(result.tagBudgetDenied).toBeUndefined();
    expect(result.deniedEntityType).toBe("user");
  });
});

// ── hasBudgets auth flag tests ────────────────────────────────────

describe("checkBudget — hasBudgets auth flag", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockResetBudgetPeriod.mockResolvedValue(undefined);
  });

  it("skips DO RPC when auth.hasBudgets is false", async () => {
    const ctx = makeCtx({ auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } });
    const result = await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(result.status).toBe("skipped");
    expect(result.reservationId).toBeNull();
    expect(result.budgetEntities).toEqual([]);
    expect(mockDoBudgetCheck).not.toHaveBeenCalled();
  });

  it("calls DO when auth.hasBudgets is true", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    const ctx = makeCtx({ auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    expect(mockDoBudgetCheck).toHaveBeenCalled();
  });

  it("emits budget_check_skipped metric when hasBudgets is false", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    const ctx = makeCtx({ auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } });
    await checkBudget(makeEnv(), ctx, 5_000_000);

    const metricCall = logSpy.mock.calls.find(([msg]) => {
      try {
        const parsed = JSON.parse(msg as string);
        return parsed._metric === "budget_check_skipped";
      } catch { return false; }
    });
    expect(metricCall).toBeTruthy();

    logSpy.mockRestore();
  });
});
