/**
 * Stream Cancellation Cost Event Tests
 *
 * Verifies that cancelled streams write estimated cost events to the DB,
 * with correct tags, error isolation, and no impact on non-cancelled paths.
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

// ── Hoisted mocks ──────────────────────────────────────────────────

const {
  mockWaitUntil,
  mockDoBudgetCheck,
  mockDoBudgetReconcile,
  mockEstimateMaxCost,
  mockUpdateBudgetSpend,
  mockCalculateOpenAICost,
  mockLogCostEventQueued,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((promise: Promise<unknown>) => { promise.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockEstimateMaxCost: vi.fn(),
  mockUpdateBudgetSpend: vi.fn(),
  mockCalculateOpenAICost: vi.fn(),
  mockLogCostEventQueued: vi.fn(),
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

vi.mock("../lib/anthropic-cost-estimator.js", () => ({
  estimateAnthropicMaxCost: (...args: unknown[]) => mockEstimateMaxCost(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: (...args: unknown[]) => mockUpdateBudgetSpend(...args),
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: (...args: unknown[]) => mockLogCostEventQueued(...args),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/cost-calculator.js", () => ({
  calculateOpenAICost: (...args: unknown[]) => mockCalculateOpenAICost(...args),
}));

vi.mock("../lib/anthropic-cost-calculator.js", () => ({
  calculateAnthropicCost: vi.fn().mockReturnValue({ costMicrodollars: 42_000 }),
}));
vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpoints: vi.fn().mockResolvedValue([]),
  getWebhookEndpointsWithSecrets: vi.fn().mockResolvedValue([]),
  invalidateWebhookCache: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/webhook-thresholds.js", () => ({
  detectThresholdCrossings: vi.fn().mockReturnValue([]),
}));

vi.mock("../lib/webhook-dispatch.js", () => ({
  dispatchToEndpoints: vi.fn().mockResolvedValue(undefined),
  createWebhookDispatcher: vi.fn().mockReturnValue({ dispatch: vi.fn() }),
}));

vi.mock("../lib/webhook-expiry.js", () => ({
  expireRotatedSecrets: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/reconciliation-queue.js", () => ({
  enqueueReconciliation: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/sanitize-upstream-error.js", () => ({
  sanitizeUpstreamError: vi.fn().mockResolvedValue('{"error":"upstream"}'),
}));

// ── Imports ────────────────────────────────────────────────────────

import { handleChatCompletions } from "../routes/openai.js";
import { handleAnthropicMessages } from "../routes/anthropic.js";
import type { RequestContext } from "../lib/context.js";

// ── Helpers ────────────────────────────────────────────────────────

function makeRequest(
  url: string,
  body: Record<string, unknown>,
): Request {
  return new Request(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-test-key",
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
    auth: { userId: "user-uuid-456", keyId: "key-uuid-001", hasWebhooks: false, hasBudgets: true, orgId: "org-test", apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-uuid-456",
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

/**
 * Make a cancelled SSE stream — sends one chunk, then the readable side
 * gets cancelled when the consumer calls reader.cancel().
 */
function makeCancelledSSEStream(): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode('data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n'),
      );
      // Never close — simulates an ongoing stream that the client cancels
    },
  });
}

function makeAnthropicCancelledSSEStream(): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode('event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'),
      );
      // Never close — simulates an ongoing stream before message_start
    },
  });
}

function makeNonCancelledSSEStream(): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode('data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n'),
      );
      // Close without [DONE] — simulates stream ending without usage (not cancelled)
      controller.close();
    },
  });
}

/**
 * Make an SSE stream that errors mid-flight (P1-5 repro). Emits one chunk
 * via a `pull()` callback so the error fires AFTER the reader has consumed
 * the initial chunk — this matches the real upstream-disconnect scenario
 * where bytes arrived then the connection was severed.
 */
function makeErroringSSEStream(): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let pullCount = 0;
  return new ReadableStream({
    pull(controller) {
      pullCount++;
      if (pullCount === 1) {
        controller.enqueue(
          encoder.encode(
            'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
          ),
        );
      } else {
        controller.error(new Error("upstream connection reset mid-stream"));
      }
    },
  });
}

function makeAnthropicErroringSSEStream(): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let pullCount = 0;
  return new ReadableStream({
    pull(controller) {
      pullCount++;
      if (pullCount === 1) {
        controller.enqueue(
          encoder.encode(
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
          ),
        );
      } else {
        controller.error(new Error("upstream connection reset mid-stream"));
      }
    },
  });
}

async function drainWaitUntil() {
  await Promise.all(
    mockWaitUntil.mock.calls.map(([p]: [Promise<unknown>]) => p.catch(() => {})),
  );
}

const checkedEntity = {
  entityType: "api_key",
  entityId: "key-uuid-123",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  reserved: 0,
  policy: "strict_block",
};

const openaiStreamBody = {
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "hi" }],
  stream: true,
};

const anthropicStreamBody = {
  model: "claude-sonnet-4-20250514",
  messages: [{ role: "user", content: "hi" }],
  stream: true,
  max_tokens: 100,
};

// ── Tests ──────────────────────────────────────────────────────────

describe("Stream Cancellation Cost Event", () => {
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
    mockLogCostEventQueued.mockReset();
    mockDoBudgetReconcile.mockResolvedValue(undefined);
    mockUpdateBudgetSpend.mockResolvedValue(undefined);
    mockEstimateMaxCost.mockReturnValue(500_000);
    mockLogCostEventQueued.mockResolvedValue(undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  // ── OpenAI ─────────────────────────────────────────────────────

  it("OpenAI cancelled stream writes estimated cost event", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-cancel-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-cancel-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );
    expect(res.status).toBe(200);

    // Read one chunk then cancel to trigger the cancelled path
    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costEventArgs = mockLogCostEventQueued.mock.calls[0];
    const costEvent = costEventArgs[2]; // 3rd arg is the event object
    expect(costEvent.provider).toBe("openai");
    expect(costEvent.costMicrodollars).toBe(500_000);
    expect(costEvent.inputTokens).toBe(0);
    expect(costEvent.outputTokens).toBe(0);
    expect(costEvent.tags._ns_estimated).toBe("true");
    expect(costEvent.tags._ns_cancelled).toBe("true");
    expect(costEvent.eventType).toBe("llm");
  });

  it("PXY-4: OpenAI non-cancelled no-usage stream writes estimated cost event with _ns_no_usage tag", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-nocancel-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeNonCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-nocancel-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costArg = mockLogCostEventQueued.mock.calls[0][2];
    expect(costArg.tags._ns_no_usage).toBe("true");
    expect(costArg.tags._ns_estimated).toBe("true");
    expect(costArg.tags._ns_cancelled).toBeUndefined();
    expect(costArg.costMicrodollars).toBeGreaterThan(0);
    expect(costArg.eventType).toBe("llm");
  });

  it("cancelled stream preserves user tags alongside system tags", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-tags-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-tags-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody, { tags: { env: "prod", team: "infra" } }),
    );
    expect(res.status).toBe(200);

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.tags).toEqual({
      env: "prod",
      team: "infra",
      _ns_estimated: "true",
      _ns_cancelled: "true",
    });
  });

  it("cancelled stream cost event uses estimate as costMicrodollars", async () => {
    mockEstimateMaxCost.mockReturnValue(1_250_000);
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-cost-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-cost-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.costMicrodollars).toBe(1_250_000);
  });

  // ── Anthropic ──────────────────────────────────────────────────

  it("Anthropic cancelled stream writes estimated cost event", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-anth-cancel-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "request-id": "req-anth-cancel-1" },
      }),
    );

    const res = await handleAnthropicMessages(
      makeRequest("http://localhost/v1/messages", anthropicStreamBody),
      makeEnv(),
      makeCtx(anthropicStreamBody),
    );
    expect(res.status).toBe(200);

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.provider).toBe("anthropic");
    expect(costEvent.costMicrodollars).toBe(500_000);
    expect(costEvent.tags._ns_estimated).toBe("true");
    expect(costEvent.tags._ns_cancelled).toBe("true");
  });

  // ── Error isolation ────────────────────────────────────────────

  it("budget reconciliation still runs after cost event write", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-reconcile-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-reconcile-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    // Both cost event AND reconciliation should fire
    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "org-test",
      "rsv-reconcile-1",
      500_000, // estimate
      expect.any(Array),
    );
  });

  it("cost event write failure does not block reconciliation", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-fail-1", checkedEntities: [checkedEntity],
    });
    // Make logCostEventQueued reject
    mockLogCostEventQueued.mockRejectedValue(new Error("queue write failed"));

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-fail-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    // Cost event failed, but reconciliation must still run with estimate (not $0 from outer catch)
    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "org-test",
      "rsv-fail-1",
      500_000, // estimate, NOT 0
      expect.any(Array),
    );
  });

  it("Anthropic cost event write failure does not block reconciliation", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-anth-fail-1", checkedEntities: [checkedEntity],
    });
    mockLogCostEventQueued.mockRejectedValue(new Error("queue write failed"));

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "request-id": "req-anth-fail-1" },
      }),
    );

    const res = await handleAnthropicMessages(
      makeRequest("http://localhost/v1/messages", anthropicStreamBody),
      makeEnv(),
      makeCtx(anthropicStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).toHaveBeenCalledWith(
      expect.anything(),
      "user-uuid-456",
      "org-test",
      "rsv-anth-fail-1",
      500_000, // estimate, NOT 0
      expect.any(Array),
    );
  });

  it("cancelled stream uses requestModel when parser has no model", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-model-1", checkedEntities: [checkedEntity],
    });

    // Stream that never sends a model field — cancel immediately
    const encoder = new TextEncoder();
    const noModelStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"}}]}\n\n'));
      },
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(noModelStream, {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-model-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    // result.model is null (no model in chunk), so requestModel ("gpt-4o-mini") is used
    expect(costEvent.model).toBe("gpt-4o-mini");
  });

  it("OpenAI console.warn message does not say 'cost event not recorded'", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-warn-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-warn-1" },
      }),
    );

    const warnSpy = vi.spyOn(console, "warn");

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    const warnMessages = warnSpy.mock.calls.map(([msg]) => String(msg));
    const routeWarns = warnMessages.filter((m) => m.includes("[openai-route]"));
    for (const msg of routeWarns) {
      expect(msg).not.toContain("cost event not recorded");
    }
  });

  it("Anthropic console.warn message does not say 'cost event not recorded'", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-anth-warn-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "request-id": "req-anth-warn-1" },
      }),
    );

    const warnSpy = vi.spyOn(console, "warn");

    const res = await handleAnthropicMessages(
      makeRequest("http://localhost/v1/messages", anthropicStreamBody),
      makeEnv(),
      makeCtx(anthropicStreamBody),
    );

    const reader = res.body!.getReader();
    await reader.read();
    await reader.cancel();
    await drainWaitUntil();

    const warnMessages = warnSpy.mock.calls.map(([msg]) => String(msg));
    const routeWarns = warnMessages.filter((m) => m.includes("[anthropic-route]"));
    for (const msg of routeWarns) {
      expect(msg).not.toContain("cost event not recorded");
    }
  });

  // ── P1-5 (stream-error resultPromise hang) ─────────────────────

  it("P1-5: OpenAI upstream stream error mid-flight → cost event written via estimate (no hang)", async () => {
    // REGRESSION GUARD: before the P1-5 fix (2026-04-16), if the upstream
    // Response body's ReadableStream errored mid-stream (server disconnected,
    // network reset, etc), the SSE parser's flush/cancel hooks would not
    // fire reliably in workerd, leaving resultPromise hung. waitUntil
    // would time out at ~30s with the reservation never reconciled — real
    // spend got erased when the DO alarm eventually reclaimed it.
    //
    // The fix wraps upstreamBody in a ReadableStream whose start() reads
    // via getReader() and catches errors, resolving a separate promise
    // that races against resultPromise. This test constructs an upstream
    // that errors on the 2nd pull() — simulating a mid-stream connection
    // reset — and verifies: (1) the cost event is still logged
    // (Promise.race resolved, no hang), (2) the event carries both
    // _ns_cancelled and _ns_estimated tags, (3) cost is the estimate.
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-p15-1", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeErroringSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-p15-1" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );
    expect(res.status).toBe(200);

    // Drain the response body — it will error on the 2nd pull as the
    // upstream throws. The client-side stream errors; the proxy's error-
    // tracking wrapper catches that and resolves upstreamErrorPromise.
    try {
      await res.text();
    } catch {
      // Expected: body read fails due to upstream error.
    }
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.provider).toBe("openai");
    expect(costEvent.costMicrodollars).toBe(500_000);
    expect(costEvent.inputTokens).toBe(0);
    expect(costEvent.outputTokens).toBe(0);
    expect(costEvent.tags._ns_estimated).toBe("true");
    expect(costEvent.tags._ns_cancelled).toBe("true");
  });

  it("P1-5: Anthropic upstream stream error mid-flight → cost event written via estimate (no hang)", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-p15-2", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicErroringSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "request-id": "req-p15-2" },
      }),
    );

    const res = await handleAnthropicMessages(
      makeRequest("http://localhost/v1/messages", anthropicStreamBody),
      makeEnv(),
      makeCtx(anthropicStreamBody),
    );
    expect(res.status).toBe(200);

    try {
      await res.text();
    } catch {
      // Expected.
    }
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.provider).toBe("anthropic");
    expect(costEvent.tags._ns_estimated).toBe("true");
    expect(costEvent.tags._ns_cancelled).toBe("true");
    expect(costEvent.costMicrodollars).toBe(500_000);
  });

  it("P1-5: clean stream completion path unchanged by the error wrapper (no false cancellations)", async () => {
    // Defense against the fix introducing false cancellations on clean
    // streams. makeNonCancelledSSEStream closes normally without error;
    // the cost event should have _ns_no_usage (not _ns_cancelled).
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved", hasBudgets: true, reservationId: "rsv-p15-3", checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeNonCancelledSSEStream(), {
        status: 200,
        headers: { "content-type": "text/event-stream", "x-request-id": "req-p15-3" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest("http://localhost/v1/chat/completions", openaiStreamBody),
      makeEnv(),
      makeCtx(openaiStreamBody),
    );
    expect(res.status).toBe(200);
    await res.text();
    await drainWaitUntil();

    expect(mockLogCostEventQueued).toHaveBeenCalledTimes(1);
    const costEvent = mockLogCostEventQueued.mock.calls[0][2];
    expect(costEvent.tags._ns_no_usage).toBe("true");
    expect(costEvent.tags._ns_cancelled).toBeUndefined();
  });
});
