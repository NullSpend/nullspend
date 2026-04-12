import { cloudflareWorkersMock } from "./test-helpers.js";
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

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn().mockReturnValue({
    inputPerMTok: 0.15,
    outputPerMTok: 0.60,
    cachedInputPerMTok: 0.075,
  }),
  costComponent: vi.fn().mockReturnValue(100),
}));

const { mockDoBudgetCheck, mockDoBudgetReconcile } = vi.hoisted(() => ({
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: (...args: unknown[]) => mockDoBudgetCheck(...args),
  doBudgetReconcile: (...args: unknown[]) => mockDoBudgetReconcile(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

const { mockEstimateMaxCost } = vi.hoisted(() => {
  const mockEstimateMaxCost = vi.fn().mockReturnValue(500_000);
  return { mockEstimateMaxCost };
});
vi.mock("../lib/cost-estimator.js", () => ({
  estimateMaxCost: mockEstimateMaxCost,
}));
import { handleChatCompletions } from "../routes/openai.js";
import type { RequestContext } from "../lib/context.js";

const DO_CHECKED_ENTITY = {
  entityType: "api_key",
  entityId: "test-key-id",
  maxBudget: 10_000_000,
  spend: 1_000_000,
  reserved: 0,
  policy: "hard",
};

const OPENAI_RESPONSE = {
  id: "chatcmpl-test",
  model: "gpt-4o-mini-2024-07-18",
  choices: [{ index: 0, message: { role: "assistant", content: "Hello!" }, finish_reason: "stop" }],
  usage: {
    prompt_tokens: 25,
    completion_tokens: 10,
  },
};

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
    CACHE_KV: { get: vi.fn(), put: vi.fn(), delete: vi.fn() },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue({ toString: () => "do-id" }),
      get: vi.fn().mockReturnValue({
        fetch: vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "approved" }))),
      }),
    },
  } as Env;
}

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    auth: { userId: "user-1", keyId: "a0a0a0a0-b1b1-c2c2-d3d3-e4e4e4e40001", hasWebhooks: false, hasBudgets: true, orgId: "org-test", apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-1",
    connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
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

describe("OpenAI budget enforcement", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Optimistic execution: fetch starts in parallel with budget check.
    // Default mock returns a pending promise (aborted on denial).
    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset().mockResolvedValue(undefined);
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("budget denial returns 429 with budget_exceeded error shape", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:test-key-id",
      remaining: 100_000,
      maxBudget: 10_000_000,
      spend: 9_900_000,
      checkedEntities: [DO_CHECKED_ENTITY],
    });

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(429);
    const json = await res.json();
    expect(json.error.code).toBe("budget_exceeded");
    expect(json.error.message).toContain("budget");
    expect(json.error.details).toEqual(expect.objectContaining({
      entity_type: expect.any(String),
      entity_id: expect.any(String),
      budget_limit_microdollars: expect.any(Number),
      budget_spend_microdollars: expect.any(Number),
      estimated_cost_microdollars: expect.any(Number),
    }));

    // X-NullSpend-Budget-* headers stamped on 429 denial (reservedForThisRequest=0 — request was rejected)
    // DO_CHECKED_ENTITY: maxBudget=10M, spend=1M, reserved=0
    // Expected: limit=10M, spent=1M, remaining=9M (budget headers reflect PRE-denied-request state)
    expect(res.headers.get("X-NullSpend-Budget-Limit")).toBe("10000000");
    expect(res.headers.get("X-NullSpend-Budget-Spent")).toBe("1000000");
    expect(res.headers.get("X-NullSpend-Budget-Remaining")).toBe("9000000");
    expect(res.headers.get("X-NullSpend-Budget-Entity")).toBe("api_key:test-key-id");
  });

  it("successful non-streaming request reconciles with actual cost", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv_test_123",
      checkedEntities: [DO_CHECKED_ENTITY],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(OPENAI_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req_budget_test",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);

    // X-NullSpend-Budget-* headers stamped on 200 success path (post-reservation snapshot)
    // limit = 10M, spend = 1M, reserved = 0, estimate = 500k
    // spent = 1M + 0 + 500k = 1.5M, remaining = 10M - 1.5M = 8.5M
    expect(res.headers.get("X-NullSpend-Budget-Limit")).toBe("10000000");
    expect(res.headers.get("X-NullSpend-Budget-Spent")).toBe("1500000");
    expect(res.headers.get("X-NullSpend-Budget-Remaining")).toBe("8500000");
    expect(res.headers.get("X-NullSpend-Budget-Entity")).toBe("api_key:test-key-id");

    await vi.waitFor(() => {
      expect(mockDoBudgetReconcile).toHaveBeenCalled();
    });
    const callArgs = mockDoBudgetReconcile.mock.calls[0];
    expect(callArgs[3]).toBe("rsv_test_123");
    expect(callArgs[4]).toBeGreaterThan(0);
  });

  it("upstream 4xx error reconciles reservation with 0", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv_test_err",
      checkedEntities: [DO_CHECKED_ENTITY],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ error: { type: "invalid_request_error", message: "bad" } }),
        {
          status: 400,
          headers: { "content-type": "application/json", "x-request-id": "req_err" },
        },
      ),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(400);

    await vi.waitFor(() => {
      expect(mockDoBudgetReconcile).toHaveBeenCalled();
    });
    const callArgs = mockDoBudgetReconcile.mock.calls[0];
    expect(callArgs[3]).toBe("rsv_test_err");
    expect(callArgs[4]).toBe(0);
  });

  it("budget lookup failure returns 503 budget_unavailable", async () => {
    mockDoBudgetCheck.mockRejectedValue(new Error("DO connection failed"));

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(503);
    const json = await res.json();
    expect(json.error.code).toBe("budget_unavailable");
  });

  it("no budget entities skips enforcement entirely", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: false,
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(OPENAI_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req_no_budget",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);
    expect(mockDoBudgetReconcile).not.toHaveBeenCalled();
  });

  it("streaming request reconciles after stream completes", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv_stream_test",
      checkedEntities: [DO_CHECKED_ENTITY],
    });

    const sseChunks = [
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hello"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":10}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req_stream_budget",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);
    await res.text();

    await vi.waitFor(() => {
      expect(mockDoBudgetReconcile).toHaveBeenCalled();
    });
    const callArgs = mockDoBudgetReconcile.mock.calls[0];
    expect(callArgs[3]).toBe("rsv_stream_test");
    expect(callArgs[4]).toBeGreaterThan(0);
  });

  it("timeout/error reconciles reservation with 0 via outer catch", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv_timeout_test",
      checkedEntities: [DO_CHECKED_ENTITY],
    });

    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fetch timeout"));

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };

    await expect(
      handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body)),
    ).rejects.toThrow("fetch timeout");

    await vi.waitFor(() => {
      expect(mockDoBudgetReconcile).toHaveBeenCalled();
    });
    const callArgs = mockDoBudgetReconcile.mock.calls[0];
    expect(callArgs[3]).toBe("rsv_timeout_test");
    expect(callArgs[4]).toBe(0);
  });
});
