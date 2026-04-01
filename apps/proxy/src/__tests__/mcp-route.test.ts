import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";

// crypto.subtle.timingSafeEqual is a CF Workers API; polyfill for Node.js tests
beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as any).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.byteLength !== viewB.byteLength) return false;
      let result = 0;
      for (let i = 0; i < viewA.byteLength; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

const mockLookupBudgetsForDO = vi.fn();
vi.mock("../lib/budget-do-lookup.js", () => ({
  lookupBudgetsForDO: (...args: unknown[]) => mockLookupBudgetsForDO(...args),
}));

const mockDoBudgetCheck = vi.fn();
const mockDoBudgetReconcile = vi.fn();
vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: (...args: unknown[]) => mockDoBudgetCheck(...args),
  doBudgetReconcile: (...args: unknown[]) => mockDoBudgetReconcile(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

const mockLogCostEventsBatch = vi.fn();
vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventsBatchQueued: (...args: unknown[]) => mockLogCostEventsBatch(...args),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));
import { handleMcpBudgetCheck, handleMcpEvents } from "../routes/mcp.js";
import type { RequestContext } from "../lib/context.js";

function makeRequest(
  path: string,
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Request {
  return new Request(`http://localhost${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    HYPERDRIVE: {
      connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
    ...overrides,
  } as Env;
}

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-1",
    connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    sessionId: null,
    traceId: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    tags: {},
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    ...overrides,
  };
}

const doEntity = {
  entityType: "user",
  entityId: "user-1",
  maxBudget: 1_000_000,
  spend: 100_000,
  policy: "strict_block",
  resetInterval: null,
  periodStart: 0,
};

describe("handleMcpBudgetCheck", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockLookupBudgetsForDO.mockReset();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns 400 when body is missing toolName", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(400);
    const json = await response.json();
    expect(json.error.code).toBe("bad_request");
  });

  it("includes X-NullSpend-Trace-Id on error responses", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(400);
    expect(response.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("includes X-NullSpend-Trace-Id on allowed responses", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(200);
    expect(response.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("returns 400 when toolName is empty string", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when serverName is empty string", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when estimateMicrodollars is NaN", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: NaN,
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when estimateMicrodollars is Infinity", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: Infinity,
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when estimateMicrodollars is negative", async () => {
    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: -100,
    }));

    expect(response.status).toBe(400);
  });

  it("returns allowed: true when no budget entities exist", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.allowed).toBe(true);
  });

  it("returns allowed: true with reservationId when budget check passes", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-123",
      checkedEntities: [{ entityType: doEntity.entityType, entityId: doEntity.entityId, maxBudget: doEntity.maxBudget, spend: doEntity.spend, policy: doEntity.policy }],
    });

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.allowed).toBe(true);
    expect(json.reservationId).toBe("rsv-123");
  });

  it("returns denied when budget is exceeded", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      remaining: 10,
      maxBudget: 100,
      spend: 90,
      checkedEntities: [{ entityType: "user", entityId: "user-1", maxBudget: 100, spend: 90, policy: "strict_block" }],
    });

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "expensive_call",
      serverName: "github",
      estimateMicrodollars: 100000,
    }));

    expect(response.status).toBe(429);
    const json = await response.json();
    expect(json.allowed).toBe(false);
    expect(json.denied).toBe(true);
    expect(json.remaining).toBe(10);
  });

  it("returns 503 when budget lookup fails", async () => {
    mockDoBudgetCheck.mockRejectedValue(new Error("DO down"));

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(503);
    const json = await response.json();
    expect(json.error.code).toBe("budget_unavailable");
  });

  it("returns 503 when doBudgetCheck fails", async () => {
    mockDoBudgetCheck.mockRejectedValue(new Error("DO unavailable"));

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const response = await handleMcpBudgetCheck(request, env, makeCtx({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10000,
    }));

    expect(response.status).toBe(503);
  });

  it("passes userId and keyId from auth result to doBudgetCheck", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });

    const request = makeRequest("/v1/mcp/budget/check", {});
    const env = makeEnv();

    const ctx = makeCtx(
      { toolName: "t", serverName: "s", estimateMicrodollars: 0 },
      { auth: { userId: "user-abc", keyId: "key-xyz", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} }, ownerId: "user-abc" },
    );
    await handleMcpBudgetCheck(request, env, ctx);

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(),
      "user-abc",
      "key-xyz",
      0,
      null,
      [],
    );
  });
});

describe("handleMcpEvents", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockLogCostEventsBatch.mockReset();
    mockDoBudgetReconcile.mockReset().mockResolvedValue(undefined);
    mockLookupBudgetsForDO.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("includes X-NullSpend-Trace-Id on error responses", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({}));

    expect(response.status).toBe(400);
    expect(response.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("includes X-NullSpend-Trace-Id on accepted responses", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "run_query", serverName: "supabase", durationMs: 150, costMicrodollars: 10000, status: "success" },
    ];

    const response = await handleMcpEvents(request, env, makeCtx({ events }));

    expect(response.status).toBe(200);
    expect(response.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("returns 400 when events array is missing", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({}));

    expect(response.status).toBe(400);
    const json = await response.json();
    expect(json.error.code).toBe("bad_request");
  });

  it("returns 400 when events array is empty", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({ events: [] }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when events exceed 50", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = Array.from({ length: 51 }, (_, i) => ({
      toolName: `t${i}`,
      serverName: "s",
      durationMs: 100,
      costMicrodollars: 10000,
      status: "success",
    }));

    const response = await handleMcpEvents(request, env, makeCtx({ events }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event has empty toolName", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "", serverName: "s", durationMs: 100, costMicrodollars: 10000, status: "success" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event has empty serverName", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "t", serverName: "", durationMs: 100, costMicrodollars: 10000, status: "success" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event durationMs is NaN", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "t", serverName: "s", durationMs: NaN, costMicrodollars: 10000, status: "success" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event costMicrodollars is negative", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "t", serverName: "s", durationMs: 100, costMicrodollars: -1, status: "success" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event durationMs is Infinity", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "t", serverName: "s", durationMs: Infinity, costMicrodollars: 10000, status: "success" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns 400 when event is missing required fields", async () => {
    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const response = await handleMcpEvents(request, env, makeCtx({
      events: [{ toolName: "t" }],
    }));

    expect(response.status).toBe(400);
  });

  it("returns accepted count for valid events", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "run_query", serverName: "supabase", durationMs: 150, costMicrodollars: 10000, status: "success" },
      { toolName: "list_files", serverName: "github", durationMs: 200, costMicrodollars: 10000, status: "success" },
    ];

    const response = await handleMcpEvents(request, env, makeCtx({ events }));

    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.accepted).toBe(2);

    await new Promise((r) => setTimeout(r, 10));
    expect(mockLogCostEventsBatch).toHaveBeenCalledTimes(1);
    expect(mockLogCostEventsBatch.mock.calls[0][2]).toHaveLength(2);
  });

  it("maps events to cost_events with provider=mcp and model=server/tool", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "run_query", serverName: "supabase", durationMs: 150, costMicrodollars: 10000, status: "success" },
    ];

    const ctx = makeCtx(
      { events },
      { auth: { userId: "user-1", keyId: "550e8400-e29b-41d4-a716-446655440000", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } },
    );
    await handleMcpEvents(request, env, ctx);

    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.any(String),
      expect.arrayContaining([
        expect.objectContaining({
          provider: "mcp",
          model: "supabase/run_query",
          inputTokens: 0,
          outputTokens: 0,
          cachedInputTokens: 0,
          reasoningTokens: 0,
          costMicrodollars: 10000,
          durationMs: 150,
          userId: "user-1",
          apiKeyId: "550e8400-e29b-41d4-a716-446655440000",
          toolName: "run_query",
          toolServer: "supabase",
          sessionId: null,
          source: "mcp",
        }),
      ]),
    );
    expect(mockLogCostEventsBatch.mock.calls[0][2]).toHaveLength(1);
  });

  it("includes sessionId when set in context", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "run_query", serverName: "supabase", durationMs: 100, costMicrodollars: 5000, status: "success" },
    ];

    const ctx = makeCtx(
      { events },
      { sessionId: "sess-mcp-1" },
    );
    await handleMcpEvents(request, env, ctx);
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.arrayContaining([
        expect.objectContaining({
          sessionId: "sess-mcp-1",
          toolName: "run_query",
          toolServer: "supabase",
        }),
      ]),
    );
  });

  it("apiKeyId comes from auth result directly (always valid UUID from DB)", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "t", serverName: "s", durationMs: 100, costMicrodollars: 10000, status: "success" },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.arrayContaining([
        expect.objectContaining({ apiKeyId: "key-1" }),
      ]),
    );
  });

  it("nulls out invalid actionId to prevent FK constraint failure", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      {
        toolName: "t",
        serverName: "s",
        durationMs: 100,
        costMicrodollars: 5000,
        status: "success",
        actionId: "not-a-uuid",
      },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.arrayContaining([
        expect.objectContaining({ actionId: null }),
      ]),
    );
  });

  it("preserves valid UUID actionId", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      {
        toolName: "t",
        serverName: "s",
        durationMs: 100,
        costMicrodollars: 5000,
        status: "success",
        actionId: "550e8400-e29b-41d4-a716-446655440000",
      },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.arrayContaining([
        expect.objectContaining({ actionId: "550e8400-e29b-41d4-a716-446655440000" }),
      ]),
    );
  });

  it("reconciles reservation when reservationId is present", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);
    mockLookupBudgetsForDO.mockResolvedValue([doEntity]);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      {
        toolName: "run_query",
        serverName: "supabase",
        durationMs: 150,
        costMicrodollars: 10000,
        status: "success",
        reservationId: "rsv-123",
      },
    ];

    const ctx = makeCtx(
      { events },
      { auth: { userId: "user-1", keyId: "550e8400-e29b-41d4-a716-446655440000", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} } },
    );
    await handleMcpEvents(request, env, ctx);
    await new Promise((r) => setTimeout(r, 10));

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-1",
      "rsv-123",
      10000,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("does not throw when logCostEventsBatch fails", async () => {
    mockLogCostEventsBatch.mockRejectedValue(new Error("DB down"));

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      {
        toolName: "t", serverName: "s", durationMs: 100, costMicrodollars: 10000, status: "success",
        reservationId: "rsv-fail-test",
      },
    ];

    mockLookupBudgetsForDO.mockResolvedValue([doEntity]);

    const response = await handleMcpEvents(request, env, makeCtx({ events }));

    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.accepted).toBe(1);

    // Reconciliation should still run even though batch insert failed
    await new Promise((r) => setTimeout(r, 10));
    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-1",
      "rsv-fail-test",
      10000,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("accepts events with valid UUID actionId", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      {
        toolName: "t",
        serverName: "s",
        durationMs: 100,
        costMicrodollars: 5000,
        status: "success",
        actionId: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.arrayContaining([
        expect.objectContaining({ actionId: "a1b2c3d4-e5f6-7890-abcd-ef1234567890" }),
      ]),
    );
  });

  it("calls lookupBudgetsForDO only once for multiple events with reservations", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);
    mockLookupBudgetsForDO.mockResolvedValue([doEntity]);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "t1", serverName: "s", durationMs: 100, costMicrodollars: 5000, status: "success", reservationId: "rsv-1" },
      { toolName: "t2", serverName: "s", durationMs: 200, costMicrodollars: 8000, status: "success", reservationId: "rsv-2" },
      { toolName: "t3", serverName: "s", durationMs: 50, costMicrodollars: 2000, status: "success" },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    // One batch insert for all 3 events
    expect(mockLogCostEventsBatch).toHaveBeenCalledTimes(1);
    expect(mockLogCostEventsBatch.mock.calls[0][2]).toHaveLength(3);

    // lookupBudgetsForDO called exactly once (not per-event)
    expect(mockLookupBudgetsForDO).toHaveBeenCalledTimes(1);

    // doBudgetReconcile called once per event WITH a reservationId (2, not 3)
    expect(mockDoBudgetReconcile).toHaveBeenCalledTimes(2);
  });

  it("includes NullSpend-Version header on successful response", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "run_query", serverName: "supabase", durationMs: 100, costMicrodollars: 5000, status: "success" },
    ];

    const response = await handleMcpEvents(request, env, makeCtx({ events }));

    expect(response.status).toBe(200);
    expect(response.headers.get("NullSpend-Version")).toBe("2026-04-01");
    await response.text();
  });

  it("does not call lookupBudgetsForDO when no events have reservations", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const request = makeRequest("/v1/mcp/events", {});
    const env = makeEnv();

    const events = [
      { toolName: "t1", serverName: "s", durationMs: 100, costMicrodollars: 5000, status: "success" },
      { toolName: "t2", serverName: "s", durationMs: 200, costMicrodollars: 8000, status: "success" },
    ];

    await handleMcpEvents(request, env, makeCtx({ events }));
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEventsBatch).toHaveBeenCalledTimes(1);
    expect(mockLookupBudgetsForDO).not.toHaveBeenCalled();
    expect(mockDoBudgetReconcile).not.toHaveBeenCalled();
  });
});
