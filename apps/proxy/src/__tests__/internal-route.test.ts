import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

const { mockDoBudgetRemove, mockDoBudgetResetSpend, mockEmitMetric } = vi.hoisted(() => ({
  mockDoBudgetRemove: vi.fn(),
  mockDoBudgetResetSpend: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetRemove: (...args: unknown[]) => mockDoBudgetRemove(...args),
  doBudgetResetSpend: (...args: unknown[]) => mockDoBudgetResetSpend(...args),
  doBudgetUpsertEntities: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../durable-objects/user-budget.js", () => ({}));

import { handleBudgetInvalidation } from "../routes/internal.js";

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    INTERNAL_SECRET: "test-secret-value",
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
    ...overrides,
  } as unknown as Env;
}

function makeRequest(options: {
  auth?: string;
  body?: unknown;
} = {}): Request {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.auth !== undefined) {
    headers["Authorization"] = options.auth;
  }

  return new Request("https://proxy.test/internal/budget/invalidate", {
    method: "POST",
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : "{}",
  });
}

const validBody = {
  action: "remove",
  ownerId: "user-1",
  entityType: "api_key",
  entityId: "key-1",
};

// Polyfill timingSafeEqual for test environment
beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as Record<string, unknown>).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.length !== viewB.length) return false;
      let result = 0;
      for (let i = 0; i < viewA.length; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

describe("handleBudgetInvalidation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDoBudgetRemove.mockResolvedValue(undefined);
    mockDoBudgetResetSpend.mockResolvedValue(undefined);
  });

  it("401: missing Authorization header", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: wrong token value", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer wrong-secret", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: malformed Authorization header (no Bearer prefix)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Basic test-secret-value", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("400: invalid JSON body", async () => {
    const req = new Request("https://proxy.test/internal/budget/invalidate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-secret-value",
      },
      body: "not-json{{{",
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    expect(res.status).toBe(400);
    const json = await res.json() as { error: { code: string } };
    expect(json.error.code).toBe("bad_request");
  });

  it("400: missing required fields", async () => {
    for (const missingField of ["action", "ownerId", "entityType", "entityId"]) {
      const body = { ...validBody, [missingField]: undefined };
      const res = await handleBudgetInvalidation(
        makeRequest({ auth: "Bearer test-secret-value", body }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    }
  });

  it("400: whitespace-only fields rejected", async () => {
    for (const field of ["ownerId", "entityType", "entityId"]) {
      const body = { ...validBody, [field]: "   " };
      const res = await handleBudgetInvalidation(
        makeRequest({ auth: "Bearer test-secret-value", body }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    }
  });

  it("400: fields exceeding 256 chars rejected", async () => {
    const longStr = "a".repeat(257);
    for (const field of ["ownerId", "entityType", "entityId"]) {
      const body = { ...validBody, [field]: longStr };
      const res = await handleBudgetInvalidation(
        makeRequest({ auth: "Bearer test-secret-value", body }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    }
  });

  it("200: fields at exactly 256 chars accepted", async () => {
    const maxStr = "a".repeat(256);
    const body = { ...validBody, ownerId: maxStr };
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body }),
      makeEnv(),
    );
    expect(res.status).toBe(200);
  });

  it("200: leading/trailing whitespace is trimmed", async () => {
    const body = { ...validBody, ownerId: "  user-1  ", entityType: " api_key ", entityId: " key-1 " };
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body }),
      makeEnv(),
    );
    expect(res.status).toBe(200);
    expect(mockDoBudgetRemove).toHaveBeenCalledWith(
      expect.anything(), "user-1", "api_key", "key-1",
    );
  });

  it("400: invalid action value", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body: { ...validBody, action: "invalid" } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("200: remove action calls doBudgetRemove + invalidateCache + emits metric", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body: validBody }),
      makeEnv(),
    );

    expect(res.status).toBe(200);
    const json = await res.json() as Record<string, unknown>;
    expect(json.ok).toBe(true);

    expect(mockDoBudgetRemove).toHaveBeenCalledWith(
      expect.anything(), "user-1", "api_key", "key-1",
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      action: "remove",
      status: "ok",
    }));
  });

  it("200: reset_spend action calls doBudgetResetSpend + invalidateCache + emits metric", async () => {
    const body = { ...validBody, action: "reset_spend" };
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body }),
      makeEnv(),
    );

    expect(res.status).toBe(200);
    expect(mockDoBudgetResetSpend).toHaveBeenCalledWith(
      expect.anything(), "user-1", "api_key", "key-1",
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      action: "reset_spend",
      status: "ok",
    }));
  });

  it("500: DO error returns 500 + error metric", async () => {
    mockDoBudgetRemove.mockRejectedValue(new Error("DO unavailable"));
    vi.spyOn(console, "error").mockImplementation(() => {});

    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body: validBody }),
      makeEnv(),
    );

    expect(res.status).toBe(500);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_invalidation", expect.objectContaining({
      status: "error",
    }));
  });

  it("500: INTERNAL_SECRET not configured", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body: validBody }),
      makeEnv({ INTERNAL_SECRET: "" }),
    );

    expect(res.status).toBe(500);
  });
});
