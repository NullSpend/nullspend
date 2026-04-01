/**
 * Streaming + Budget Integration Tests
 *
 * Verifies that streaming requests properly interact with the budget
 * lifecycle: reservation, reconciliation with actual cost, and
 * updateBudgetSpend assertions.
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

function makeSSEStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
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
    auth: { userId: "user-uuid-456", keyId: "a0a0a0a0-b1b1-c2c2-d3d3-e4e4e4e40001", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-uuid-456",
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

const streamBody = {
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "hi" }],
  stream: true,
};

const checkedEntity = {
  entityType: "api_key",
  entityId: "key-uuid-123",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  policy: "strict_block",
};

const sseWithUsage = [
  'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
  'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
  "data: [DONE]\n\n",
];

const sseWithoutUsage = [
  'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
  "data: [DONE]\n\n",
];

describe("Streaming + Budget Integration", () => {
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
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("streaming success with usage reconciles with actual cost > 0", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-stream-1", checkedEntities: [checkedEntity] });
    mockCalculateOpenAICost.mockReturnValue({ costMicrodollars: 42_000 });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s1" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "rsv-stream-1",
      42_000,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("streaming success with usage triggers doBudgetReconcile", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-stream-spend", checkedEntities: [checkedEntity] });
    mockCalculateOpenAICost.mockReturnValue({ costMicrodollars: 75_000 });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s2" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "rsv-stream-spend",
      75_000,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("streaming success without usage reconciles with actualCost=0", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-no-usage", checkedEntities: [checkedEntity] });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithoutUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s3" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "rsv-no-usage",
      0,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("streaming without usage does NOT call updateBudgetSpend", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-no-spend", checkedEntities: [checkedEntity] });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithoutUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s4" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
  });

  it("streaming with null upstream body returns 502 and reconciles with 0", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-null-body", checkedEntities: [checkedEntity] });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s5" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(502);
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "rsv-null-body",
      0,
      expect.any(Array),
      expect.any(String),
    );
  });

  it("streaming cost calculation failure still reconciles with 0 (last-resort)", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "rsv-cost-err", checkedEntities: [checkedEntity] });
    mockCalculateOpenAICost.mockImplementation(() => {
      throw new Error("cost calc explosion");
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s6" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    // Should have reconciled with 0 on last-resort
    expect(mockDoBudgetReconcile).toHaveBeenCalled();
    const lastCall = mockDoBudgetReconcile.mock.calls[mockDoBudgetReconcile.mock.calls.length - 1];
    expect(lastCall[3]).toBe(0);
  });

  it("streaming + budget denied returns 429", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-uuid-123",
      remaining: 100_000,
      maxBudget: 50_000_000,
      spend: 49_400_000,
      checkedEntities: [checkedEntity],
    });

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("budget_exceeded");
  });

  it("streaming + no budget configured passes through without budget calls", async () => {
    mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: false });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseWithUsage), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-s8" },
      }),
    );

    const res = await handleChatCompletions(makeRequest(streamBody), makeEnv(), makeCtx(streamBody));
    expect(res.status).toBe(200);
    await res.text();

    expect(mockDoBudgetReconcile).not.toHaveBeenCalled();
    expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
  });
});
