/**
 * Budget Edge Case Tests
 *
 * Tests numeric edge cases, attribution nulls, updateBudgetSpend conditional
 * invocation, 429 response body structure, and sensitive data leakage through
 * the full handleChatCompletions flow.
 */
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";

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

const {
  mockWaitUntil,
  mockDoBudgetCheck,
  mockDoBudgetReconcile,
  mockEstimateMaxCost,
  mockUpdateBudgetSpend,
  mockCalculateOpenAICost,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((promise: Promise<unknown>) => { promise.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockEstimateMaxCost: vi.fn(),
  mockUpdateBudgetSpend: vi.fn(),
  mockCalculateOpenAICost: vi.fn(),
}));

vi.mock("cloudflare:workers", () => ({
  waitUntil: mockWaitUntil,
}));

vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn().mockReturnValue({
    inputPerMTok: 0.15,
    cachedInputPerMTok: 0.075,
    outputPerMTok: 0.60,
  }),
  costComponent: vi.fn((tokens: number, rate: number) => {
    if (tokens <= 0 || rate <= 0) return 0;
    return tokens * rate;
  }),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: (...args: unknown[]) => mockDoBudgetCheck(...args),
  doBudgetReconcile: (...args: unknown[]) => mockDoBudgetReconcile(...args),
}));

vi.mock("../lib/cost-estimator.js", () => ({
  estimateMaxCost: (...args: unknown[]) => mockEstimateMaxCost(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: (...args: unknown[]) => mockUpdateBudgetSpend(...args),
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/cost-calculator.js", () => ({
  calculateOpenAICost: (...args: unknown[]) => mockCalculateOpenAICost(...args),
}));
import { handleChatCompletions } from "../routes/openai.js";
import type { RequestContext } from "../lib/context.js";

function makeRequest(
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-test-key",
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

function makeEnv(): Env {
  return {
    HYPERDRIVE: {
      connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
  } as Env;
}

function makeSuccessResponse(usage = { prompt_tokens: 10, completion_tokens: 5 }) {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-test",
      model: "gpt-4o-mini-2024-07-18",
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
      usage,
    }),
    {
      status: 200,
      headers: { "content-type": "application/json", "x-request-id": "req-test" },
    },
  );
}

async function drainWaitUntil() {
  await Promise.all(
    mockWaitUntil.mock.calls.map(([p]: [Promise<unknown>]) => p.catch(() => {})),
  );
}

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    bodyByteLength: JSON.stringify(body).length,
    auth: { userId: "user-uuid-456", keyId: "a0a0a0a0-b1b1-c2c2-d3d3-e4e4e4e40001", hasWebhooks: false, hasBudgets: true, orgId: "org-test", apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-uuid-456",
    connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    skipDbWrites: false,
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

const defaultBody = {
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "hi" }],
};

const checkedEntity = {
  entityType: "api_key",
  entityId: "key-uuid-123",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  reserved: 0,
  policy: "strict_block",
};

describe("Budget Edge Cases", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset();
    mockEstimateMaxCost.mockReset();
    mockUpdateBudgetSpend.mockReset();
    mockCalculateOpenAICost.mockReset();
    mockDoBudgetReconcile.mockResolvedValue(undefined);
    mockUpdateBudgetSpend.mockResolvedValue(undefined);
    mockEstimateMaxCost.mockReturnValue(500_000);
    mockCalculateOpenAICost.mockReturnValue({ costMicrodollars: 42_000 });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  // --- Attribution nulls ---

  it("passes keyId=null to doBudgetCheck when auth result has no keyId", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody, {
      auth: { userId: "user-uuid-456", keyId: null as any, hasWebhooks: false, hasBudgets: true, orgId: "org-test", apiVersion: "2026-04-01", defaultTags: {} },
    }));

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      null,
      expect.any(Number),
      null,
      [],
      "org-test",
    );
  });

  it("passes { keyId, userId: null } when auth result has no userId", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    // userId=null means DO path skips (returns skipped) so no lookupBudgetsForDO call
    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody, {
      auth: { userId: null as any, keyId: "a0a0a0a0-b1b1-c2c2-d3d3-e4e4e4e40001", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      ownerId: null as any,
    }));

    // DO path skips when no userId — no lookup call
    expect(mockDoBudgetCheck).not.toHaveBeenCalled();
  });

  it("skips budget when auth has no userId and no keyId", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody, {
      auth: { userId: null as any, keyId: null as any, hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      ownerId: null as any,
    }));

    expect(mockDoBudgetCheck).not.toHaveBeenCalled();
  });

  // --- Zero estimate ---

  it("zero estimate still calls doBudgetCheck with estimate=0", async () => {
    mockEstimateMaxCost.mockReturnValue(0);
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-zero", checkedEntities: [checkedEntity] });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));

    expect(mockDoBudgetCheck).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      expect.anything(),
      0,
      null,
      [],
      "org-test",
    );
  });

  // --- updateBudgetSpend called with correct args ---

  it("doBudgetReconcile called with correct cost on non-streaming success", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-spend", checkedEntities: [checkedEntity] });
    mockCalculateOpenAICost.mockReturnValue({ costMicrodollars: 123_456 });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "org-test",
      "rsv-spend",
      123_456,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("updateBudgetSpend NOT called when upstream returns error (actualCost=0)", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-err", checkedEntities: [checkedEntity] });
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: "bad" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      }),
    );

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    await drainWaitUntil();

    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
  });

  it("updateBudgetSpend NOT called when no budget configured", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    await drainWaitUntil();

    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
  });

  // --- 429 response body structure ---

  it("429 response body contains all required fields", async () => {
    mockEstimateMaxCost.mockReturnValue(999_999);
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-uuid-123",
      remaining: 100_000,
      maxBudget: 50_000_000,
      spend: 49_400_000,
      checkedEntities: [checkedEntity],
    });

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    expect(res.status).toBe(429);

    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
    expect(body.error.message).toContain("budget");
    expect(body.error.details).toEqual(expect.objectContaining({
      entity_type: expect.any(String),
      entity_id: expect.any(String),
      budget_limit_microdollars: expect.any(Number),
      budget_spend_microdollars: expect.any(Number),
      estimated_cost_microdollars: expect.any(Number),
    }));
  });

  it("429 response does not contain sensitive data", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-uuid-123",
      remaining: 0,
      maxBudget: 50_000_000,
      spend: 50_000_000,
      checkedEntities: [checkedEntity],
    });

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const text = await res.text();

    expect(text).not.toContain("x-nullspend-key");
    expect(text).not.toContain("ns_live_sk_");
    expect(text).not.toContain("postgresql://");
    expect(text).not.toContain("upstash.io");
    expect(text).not.toContain("fake-token");
  });

  // --- Model passed correctly ---

  it("estimateMaxCost receives correct model string from body", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-model", checkedEntities: [checkedEntity] });
    globalThis.fetch = vi.fn().mockResolvedValue(makeSuccessResponse());

    const body = { model: "o3-mini", messages: [{ role: "user", content: "think" }] };
    await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(mockEstimateMaxCost).toHaveBeenCalledWith("o3-mini", body, expect.any(Number));
  });
});
