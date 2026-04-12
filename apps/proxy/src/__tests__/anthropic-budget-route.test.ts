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
    inputPerMTok: 3,
    outputPerMTok: 15,
    cachedInputPerMTok: 0.3,
  }),
  costComponent: vi.fn().mockReturnValue(100),
}));

// checkBudget is mocked in budget-orchestrator.js mock below

const { mockReconcileBudgetQueued } = vi.hoisted(() => ({
  mockReconcileBudgetQueued: vi.fn().mockResolvedValue(undefined),
}));

const { mockEstimateAnthropicMaxCost } = vi.hoisted(() => ({
  mockEstimateAnthropicMaxCost: vi.fn().mockReturnValue(500_000),
}));
vi.mock("../lib/anthropic-cost-estimator.js", () => ({
  estimateAnthropicMaxCost: mockEstimateAnthropicMaxCost,
}));

// Mock the entire budget-orchestrator to avoid deep transitive dependency chain
vi.mock("../lib/budget-orchestrator.js", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../lib/budget-orchestrator.js")>();
  return {
    ...orig,
    checkBudget: vi.fn(),
    reconcileBudget: vi.fn().mockResolvedValue(undefined),
    reconcileBudgetQueued: (...args: unknown[]) => mockReconcileBudgetQueued(...args),
    getReconcileQueue: vi.fn().mockReturnValue(undefined),
  };
});

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));
import { handleAnthropicMessages } from "../routes/anthropic.js";
import { checkBudget } from "../lib/budget-orchestrator.js";
import type { RequestContext } from "../lib/context.js";

const mockCheckBudget = checkBudget as ReturnType<typeof vi.fn>;

const BUDGET_ENTITY = {
  entityKey: "{budget}:api_key:test-key-id",
  entityType: "api_key",
  entityId: "test-key-id",
  maxBudget: 10_000_000,
  spend: 1_000_000,
  reserved: 0,
  policy: "hard",
};

const ANTHROPIC_RESPONSE = {
  id: "msg_01XFDUDYJgAACzvnptvVoYEL",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-20250514",
  content: [{ type: "text", text: "Hello!" }],
  stop_reason: "end_turn",
  usage: {
    input_tokens: 25,
    output_tokens: 10,
  },
};

function makeRequest(
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Request {
  return new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-ant-api03-test",
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

describe("Anthropic budget enforcement", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Optimistic execution: fetch starts in parallel with budget check.
    // Default mock returns a pending promise (aborted on denial).
    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockCheckBudget.mockReset();
    mockReconcileBudgetQueued.mockReset().mockResolvedValue(undefined);
    mockEstimateAnthropicMaxCost.mockReset().mockReturnValue(500_000);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("budget denial returns 429 with budget_exceeded error shape", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "denied",
      reservationId: null,
      budgetEntities: [BUDGET_ENTITY],
      remaining: 100_000,
      maxBudget: BUDGET_ENTITY.maxBudget,
      spend: 9_900_000,
    });

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(429);
    const json = await res.json();
    expect(json.error.code).toBe("budget_exceeded");
    expect(json.error.message).toContain("budget");
    expect(json.error.details).toEqual({
      entity_type: "api_key",
      entity_id: "test-key-id",
      budget_limit_microdollars: 10_000_000,
      budget_spend_microdollars: 9_900_000,
      estimated_cost_microdollars: 500_000,
    });

    // X-NullSpend-Budget-* headers on 429 denial (reservedForThisRequest=0).
    // BUDGET_ENTITY: maxBudget=10M, spend=1M, reserved=0 → spent=1M, remaining=9M
    expect(res.headers.get("X-NullSpend-Budget-Limit")).toBe("10000000");
    expect(res.headers.get("X-NullSpend-Budget-Spent")).toBe("1000000");
    expect(res.headers.get("X-NullSpend-Budget-Remaining")).toBe("9000000");
    expect(res.headers.get("X-NullSpend-Budget-Entity")).toBe("api_key:test-key-id");
  });

  it("successful non-streaming request reconciles with actual cost", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "approved",
      reservationId: "rsv_test_123",
      budgetEntities: [BUDGET_ENTITY],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_budget_test",
        },
      }),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);

    // X-NullSpend-Budget-* headers stamped on 200 success (reservedForThisRequest=estimate=500k)
    // limit=10M, spend=1M, reserved=0, estimate=500k → spent=1.5M, remaining=8.5M
    expect(res.headers.get("X-NullSpend-Budget-Limit")).toBe("10000000");
    expect(res.headers.get("X-NullSpend-Budget-Spent")).toBe("1500000");
    expect(res.headers.get("X-NullSpend-Budget-Remaining")).toBe("8500000");
    expect(res.headers.get("X-NullSpend-Budget-Entity")).toBe("api_key:test-key-id");

    // waitUntil fires reconciliation asynchronously; verify it was called
    await vi.waitFor(() => {
      expect(mockReconcileBudgetQueued).toHaveBeenCalled();
    });
    const callArgs = mockReconcileBudgetQueued.mock.calls[0];
    expect(callArgs[4]).toBe("rsv_test_123");
    expect(callArgs[5]).toBeGreaterThan(0);
  });

  it("upstream 4xx error reconciles reservation with 0", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "approved",
      reservationId: "rsv_test_err",
      budgetEntities: [BUDGET_ENTITY],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "bad" } }),
        {
          status: 400,
          headers: { "content-type": "application/json", "request-id": "req_err" },
        },
      ),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(400);

    await vi.waitFor(() => {
      expect(mockReconcileBudgetQueued).toHaveBeenCalled();
    });
    const callArgs = mockReconcileBudgetQueued.mock.calls[0];
    expect(callArgs[4]).toBe("rsv_test_err");
    expect(callArgs[5]).toBe(0);
  });

  it("budget lookup failure returns 503 budget_unavailable", async () => {
    mockCheckBudget.mockRejectedValue(new Error("Budget service unavailable"));

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(503);
    const json = await res.json();
    expect(json.error.code).toBe("budget_unavailable");
  });

  it("no budget entities skips enforcement entirely", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "skipped",
      reservationId: null,
      budgetEntities: [],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_no_budget",
        },
      }),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);
    expect(mockReconcileBudgetQueued).not.toHaveBeenCalled();
  });

  it("streaming request reconciles after stream completes", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "approved",
      reservationId: "rsv_stream_test",
      budgetEntities: [BUDGET_ENTITY],
    });

    const sseChunks = [
      'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","usage":{"input_tokens":25,"output_tokens":0}}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "req_stream_budget",
        },
      }),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(200);
    // Consume the stream to trigger the waitUntil callback
    await res.text();

    await vi.waitFor(() => {
      expect(mockReconcileBudgetQueued).toHaveBeenCalled();
    });
    const callArgs = mockReconcileBudgetQueued.mock.calls[0];
    expect(callArgs[4]).toBe("rsv_stream_test");
    expect(callArgs[5]).toBeGreaterThan(0);
  });

  it("timeout/error reconciles reservation with 0 via outer catch", async () => {
    mockCheckBudget.mockResolvedValue({
      status: "approved",
      reservationId: "rsv_timeout_test",
      budgetEntities: [BUDGET_ENTITY],
    });

    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fetch timeout"));

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };

    await expect(
      handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body)),
    ).rejects.toThrow("fetch timeout");

    await vi.waitFor(() => {
      expect(mockReconcileBudgetQueued).toHaveBeenCalled();
    });
    const callArgs = mockReconcileBudgetQueued.mock.calls[0];
    expect(callArgs[4]).toBe("rsv_timeout_test");
    expect(callArgs[5]).toBe(0);
  });
});
