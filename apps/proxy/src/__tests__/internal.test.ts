import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

const { mockDoBudgetRemove, mockDoBudgetResetSpend, mockDoBudgetUpsertEntities, mockLookupBudgetsForDO, mockInvalidateAuthCacheForUser, mockEmitMetric } = vi.hoisted(() => ({
  mockDoBudgetRemove: vi.fn(),
  mockDoBudgetResetSpend: vi.fn(),
  mockDoBudgetUpsertEntities: vi.fn(),
  mockLookupBudgetsForDO: vi.fn(),
  mockInvalidateAuthCacheForUser: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetRemove: (...args: unknown[]) => mockDoBudgetRemove(...args),
  doBudgetResetSpend: (...args: unknown[]) => mockDoBudgetResetSpend(...args),
  doBudgetUpsertEntities: (...args: unknown[]) => mockDoBudgetUpsertEntities(...args),
}));

vi.mock("../lib/budget-do-lookup.js", () => ({
  lookupBudgetsForDO: (...args: unknown[]) => mockLookupBudgetsForDO(...args),
}));

vi.mock("../lib/api-key-auth.js", () => ({
  invalidateAuthCacheForOwner: (...args: unknown[]) => mockInvalidateAuthCacheForUser(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../lib/errors.js", () => ({
  errorResponse: (code: string, message: string, status: number) =>
    Response.json({ error: { code, message, details: null } }, { status }),
}));

import { handleBudgetInvalidation } from "../routes/internal.js";

const INTERNAL_SECRET = "test-secret-abc123";

beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as any).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.byteLength !== viewB.byteLength) return false;
      let result = 0;
      for (let i = 0; i < viewA.byteLength; i++) result |= viewA[i] ^ viewB[i];
      return result === 0;
    };
  }
});

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    INTERNAL_SECRET,
    HYPERDRIVE: { connectionString: "postgres://test" },
    ...overrides,
  } as unknown as Env;
}

function makeRequest(body: unknown, token: string = INTERNAL_SECRET): Request {
  return new Request("http://localhost/internal/budget/invalidate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  mockDoBudgetRemove.mockResolvedValue(undefined);
  mockDoBudgetResetSpend.mockResolvedValue(undefined);
  mockDoBudgetUpsertEntities.mockResolvedValue(undefined);
  mockLookupBudgetsForDO.mockResolvedValue([]);
});

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — auth", () => {
  it("returns 401 when Authorization header is missing", async () => {
    const req = new Request("http://localhost/internal/budget/invalidate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "remove", ownerId: "u1", entityType: "user", entityId: "u1" }),
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 401 when token is wrong", async () => {
    const req = makeRequest(
      { action: "remove", ownerId: "u1", entityType: "user", entityId: "u1" },
      "wrong-secret",
    );

    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 500 when INTERNAL_SECRET is not configured", async () => {
    const req = makeRequest({ action: "remove", ownerId: "u1", entityType: "user", entityId: "u1" });
    const env = makeEnv({ INTERNAL_SECRET: undefined } as any);

    const res = await handleBudgetInvalidation(req, env);
    expect(res.status).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// Body validation
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — body validation", () => {
  it("returns 400 for invalid JSON", async () => {
    const req = new Request("http://localhost/internal/budget/invalidate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${INTERNAL_SECRET}`,
      },
      body: "not json",
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("returns 400 for missing action", async () => {
    const req = makeRequest({ ownerId: "u1", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("returns 400 for invalid action", async () => {
    const req = makeRequest({ action: "nuke", ownerId: "u1", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("returns 400 for missing ownerId", async () => {
    const req = makeRequest({ action: "remove", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("returns 400 for empty entityType", async () => {
    const req = makeRequest({ action: "remove", ownerId: "u1", entityType: "", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// Remove action
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — remove", () => {
  it("calls doBudgetRemove with correct args", async () => {
    const req = makeRequest({ action: "remove", ownerId: "u1", entityType: "api_key", entityId: "k1" });
    const env = makeEnv();

    const res = await handleBudgetInvalidation(req, env);
    expect(res.status).toBe(200);

    expect(mockDoBudgetRemove).toHaveBeenCalledWith(env, "u1", "api_key", "k1");
    expect(mockInvalidateAuthCacheForUser).toHaveBeenCalledWith("u1");
  });

  it("emits budget_invalidation metric with action=remove", async () => {
    const req = makeRequest({ action: "remove", ownerId: "u1", entityType: "user", entityId: "u1" });
    await handleBudgetInvalidation(req, makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      action: "remove",
      status: "ok",
    }));
  });
});

// ---------------------------------------------------------------------------
// Reset spend action
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — reset_spend", () => {
  it("calls doBudgetResetSpend with correct args", async () => {
    const req = makeRequest({ action: "reset_spend", ownerId: "u1", entityType: "user", entityId: "u1" });
    const env = makeEnv();

    const res = await handleBudgetInvalidation(req, env);
    expect(res.status).toBe(200);

    expect(mockDoBudgetResetSpend).toHaveBeenCalledWith(env, "u1", "user", "u1");
  });
});

// ---------------------------------------------------------------------------
// Sync action
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — sync", () => {
  it("looks up entities from Postgres and upserts into DO", async () => {
    const entities = [{
      entityType: "user",
      entityId: "u1",
      maxBudget: 50_000_000,
      spend: 10_000_000,
      policy: "strict_block",
      resetInterval: "monthly",
      periodStart: 1_700_000_000_000,
      velocityLimit: null,
      velocityWindow: 60_000,
      velocityCooldown: 60_000,
    }];
    mockLookupBudgetsForDO.mockResolvedValue(entities);

    const req = makeRequest({ action: "sync", ownerId: "u1", entityType: "user", entityId: "u1" });
    const env = makeEnv();

    const res = await handleBudgetInvalidation(req, env);
    expect(res.status).toBe(200);

    expect(mockLookupBudgetsForDO).toHaveBeenCalledWith("postgres://test", { keyId: null, orgId: "u1", userId: "u1", tags: {} });
    expect(mockDoBudgetUpsertEntities).toHaveBeenCalledWith(env, "u1", entities);
    expect(mockEmitMetric).not.toHaveBeenCalledWith("budget_sync_empty", expect.anything());
  });

  it("syncs velocity fields through the full flow", async () => {
    const entities = [{
      entityType: "user",
      entityId: "u1",
      maxBudget: 50_000_000,
      spend: 0,
      policy: "strict_block",
      resetInterval: null,
      periodStart: 0,
      velocityLimit: 5_000_000,
      velocityWindow: 120_000,
      velocityCooldown: 90_000,
    }];
    mockLookupBudgetsForDO.mockResolvedValue(entities);

    const req = makeRequest({ action: "sync", ownerId: "u1", entityType: "user", entityId: "u1" });
    await handleBudgetInvalidation(req, makeEnv());

    const upsertedEntities = mockDoBudgetUpsertEntities.mock.calls[0][2];
    expect(upsertedEntities[0].velocityLimit).toBe(5_000_000);
    expect(upsertedEntities[0].velocityWindow).toBe(120_000);
    expect(upsertedEntities[0].velocityCooldown).toBe(90_000);
  });

  it("handles empty lookup result — emits budget_sync_empty metric and console.warn", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    mockLookupBudgetsForDO.mockResolvedValue([]);

    const req = makeRequest({ action: "sync", ownerId: "u1", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());

    expect(res.status).toBe(200);
    expect(mockDoBudgetUpsertEntities).toHaveBeenCalledWith(expect.anything(), "u1", []);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_sync_empty", {
      ownerId: "u1",
      entityType: "user",
      entityId: "u1",
    });
    expect(warnSpy).toHaveBeenCalledWith(
      "[internal] sync returned 0 entities from Postgres",
      { ownerId: "u1", entityType: "user", entityId: "u1" },
    );

    warnSpy.mockRestore();
  });

  it("invalidates auth cache after sync", async () => {
    mockLookupBudgetsForDO.mockResolvedValue([]);

    const req = makeRequest({ action: "sync", ownerId: "u1", entityType: "user", entityId: "u1" });
    await handleBudgetInvalidation(req, makeEnv());

    expect(mockInvalidateAuthCacheForUser).toHaveBeenCalledWith("u1");
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — error handling", () => {
  it("returns 500 and emits error metric when DO operation fails", async () => {
    mockDoBudgetRemove.mockRejectedValue(new Error("DO unavailable"));

    const req = makeRequest({ action: "remove", ownerId: "u1", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());

    expect(res.status).toBe(500);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      action: "remove",
      status: "error",
    }));
  });

  it("returns 500 when Postgres lookup fails during sync", async () => {
    mockLookupBudgetsForDO.mockRejectedValue(new Error("PG connection refused"));

    const req = makeRequest({ action: "sync", ownerId: "u1", entityType: "user", entityId: "u1" });
    const res = await handleBudgetInvalidation(req, makeEnv());

    expect(res.status).toBe(500);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      action: "sync",
      status: "error",
    }));
  });
});

// ---------------------------------------------------------------------------
// Tag budget sync
// ---------------------------------------------------------------------------

describe("handleBudgetInvalidation — tag sync", () => {
  it("sync with entityType=tag passes tags to lookupBudgetsForDO", async () => {
    const tagEntity = {
      entityType: "tag",
      entityId: "project=openclaw",
      maxBudget: 50_000_000,
      spend: 5_000_000,
      policy: "strict_block",
      resetInterval: null,
      periodStart: 0,
      velocityLimit: null,
      velocityWindow: 60_000,
      velocityCooldown: 60_000,
    };
    mockLookupBudgetsForDO.mockResolvedValue([tagEntity]);

    const req = makeRequest({
      action: "sync",
      ownerId: "u1",
      entityType: "tag",
      entityId: "project=openclaw",
    });
    const env = makeEnv();
    const res = await handleBudgetInvalidation(req, env);

    expect(res.status).toBe(200);
    expect(mockLookupBudgetsForDO).toHaveBeenCalledWith("postgres://test", {
      keyId: null,
      orgId: "u1",
      userId: null,
      tags: { project: "openclaw" },
    });
    expect(mockDoBudgetUpsertEntities).toHaveBeenCalledWith(env, "u1", [tagEntity]);
  });

  it("sync with entityType=api_key passes keyId and orgId", async () => {
    mockLookupBudgetsForDO.mockResolvedValue([]);

    const req = makeRequest({
      action: "sync",
      ownerId: "u1",
      entityType: "api_key",
      entityId: "key-1",
    });
    await handleBudgetInvalidation(req, makeEnv());

    expect(mockLookupBudgetsForDO).toHaveBeenCalledWith("postgres://test", {
      keyId: "key-1",
      orgId: "u1",
      userId: null,
      tags: {},
    });
  });
});
