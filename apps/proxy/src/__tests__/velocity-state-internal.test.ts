/**
 * Velocity State Internal Endpoint Tests
 *
 * Tests GET /internal/budget/velocity-state:
 * 1. Auth: 401 without token, 401 with bad token
 * 2. Validation: 400 without ownerId
 * 3. Happy path: returns velocity state data
 * 4. Empty state: returns empty array
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

const { mockDoBudgetGetVelocityState, mockEmitMetric } = vi.hoisted(() => ({
  mockDoBudgetGetVelocityState: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetRemove: vi.fn(),
  doBudgetResetSpend: vi.fn(),
  doBudgetUpsertEntities: vi.fn(),
  doBudgetGetVelocityState: (...args: unknown[]) => mockDoBudgetGetVelocityState(...args),
}));

vi.mock("../lib/budget-do-lookup.js", () => ({
  lookupBudgetsForDO: vi.fn().mockResolvedValue([]),
}));

vi.mock("../lib/api-key-auth.js", () => ({
  invalidateAuthCacheForOwner: vi.fn(),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../lib/errors.js", () => ({
  errorResponse: (code: string, message: string, status: number) =>
    Response.json({ error: { code, message, details: null } }, { status }),
}));

import { handleVelocityState } from "../routes/internal.js";

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
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({
        getVelocityState: vi.fn(),
      }),
    },
    ...overrides,
  } as unknown as Env;
}

function makeRequest(ownerId?: string, token: string = INTERNAL_SECRET): Request {
  const url = ownerId
    ? `http://localhost/internal/budget/velocity-state?ownerId=${ownerId}`
    : "http://localhost/internal/budget/velocity-state";
  return new Request(url, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => {});
  mockDoBudgetGetVelocityState.mockResolvedValue([]);
});

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

describe("GET /internal/budget/velocity-state — auth", () => {
  it("returns 401 without Authorization header", async () => {
    const req = new Request("http://localhost/internal/budget/velocity-state?ownerId=user-1", {
      method: "GET",
    });
    const res = await handleVelocityState(req, makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 401 with invalid token", async () => {
    const res = await handleVelocityState(makeRequest("user-1", "wrong-token"), makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 500 when INTERNAL_SECRET not configured", async () => {
    const res = await handleVelocityState(
      makeRequest("user-1"),
      makeEnv({ INTERNAL_SECRET: undefined } as any),
    );
    expect(res.status).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

describe("GET /internal/budget/velocity-state — validation", () => {
  it("returns 400 without ownerId query param", async () => {
    const res = await handleVelocityState(makeRequest(), makeEnv());
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe("bad_request");
  });
});

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

describe("GET /internal/budget/velocity-state — happy path", () => {
  it("returns empty array when no velocity state", async () => {
    mockDoBudgetGetVelocityState.mockResolvedValue([]);

    const res = await handleVelocityState(makeRequest("user-1"), makeEnv());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({ velocityState: [] });
  });

  it("returns velocity state data for valid request", async () => {
    const velocityData = [
      {
        entity_key: "user:user-1",
        window_size_ms: 60_000,
        window_start_ms: Date.now() - 30_000,
        current_count: 5,
        current_spend: 2_500_000,
        prev_count: 3,
        prev_spend: 1_500_000,
        tripped_at: null,
      },
      {
        entity_key: "api_key:key-1",
        window_size_ms: 300_000,
        window_start_ms: Date.now() - 100_000,
        current_count: 10,
        current_spend: 5_000_000,
        prev_count: 8,
        prev_spend: 4_000_000,
        tripped_at: Date.now() - 10_000,
      },
    ];
    mockDoBudgetGetVelocityState.mockResolvedValue(velocityData);

    const res = await handleVelocityState(makeRequest("user-1"), makeEnv());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.velocityState).toHaveLength(2);
    expect(body.velocityState[0].entity_key).toBe("user:user-1");
    expect(body.velocityState[1].tripped_at).not.toBeNull();
  });

  it("passes correct ownerId to doBudgetGetVelocityState", async () => {
    mockDoBudgetGetVelocityState.mockResolvedValue([]);

    await handleVelocityState(makeRequest("specific-user-id"), makeEnv());

    expect(mockDoBudgetGetVelocityState).toHaveBeenCalledWith(
      expect.anything(),
      "specific-user-id",
    );
  });

  it("returns 500 and emits error metric when DO throws", async () => {
    mockDoBudgetGetVelocityState.mockRejectedValue(new Error("DO unavailable"));

    const res = await handleVelocityState(makeRequest("user-1"), makeEnv());
    expect(res.status).toBe(500);
    const body = await res.json();
    expect(body.error.code).toBe("internal_error");

    expect(mockEmitMetric).toHaveBeenCalledWith("velocity_state_lookup", {
      ownerId: "user-1",
      status: "error",
    });
  });

  it("emits success metric with count on happy path", async () => {
    mockDoBudgetGetVelocityState.mockResolvedValue([
      { entity_key: "user:u1", window_size_ms: 60000, window_start_ms: 0, current_count: 0, current_spend: 0, prev_count: 0, prev_spend: 0, tripped_at: null },
    ]);

    const res = await handleVelocityState(makeRequest("user-1"), makeEnv());
    expect(res.status).toBe(200);

    expect(mockEmitMetric).toHaveBeenCalledWith("velocity_state_lookup", {
      ownerId: "user-1",
      count: 1,
      status: "ok",
    });
  });
});
