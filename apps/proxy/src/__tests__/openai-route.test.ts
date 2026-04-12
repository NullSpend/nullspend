/**
 * Unit tests for the OpenAI route handler logic.
 * Tests the handleChatCompletions function behavior with mocked
 * dependencies to cover internal error paths, cost calculation
 * failures, and response handling edge cases.
 *
 * These tests verify the route handler's resilience without
 * requiring a live OpenAI API key or running proxy server.
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import type { RequestContext } from "../lib/context.js";

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

const { mockIsKnownModel, mockLogCostEvent } = vi.hoisted(() => {
  const mockIsKnownModel = vi.fn().mockReturnValue(true);
  const mockLogCostEvent = vi.fn().mockResolvedValue(undefined);
  return { mockIsKnownModel, mockLogCostEvent };
});
vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: mockIsKnownModel,
  getModelPricing: vi.fn().mockReturnValue(null),
  costComponent: vi.fn().mockReturnValue(0),
}));
vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: (...args: unknown[]) => mockLogCostEvent(...args),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));
vi.mock("../lib/budget-orchestrator.js", () => ({
  checkBudget: vi.fn().mockResolvedValue({ status: "skipped", reservationId: null, budgetEntities: [] }),
  reconcileBudgetQueued: vi.fn().mockResolvedValue(undefined),
  getReconcileQueue: vi.fn().mockReturnValue(undefined),
}));

const { mockGetWebhookEndpoints, mockGetWebhookEndpointsWithSecrets } = vi.hoisted(() => {
  const mockGetWebhookEndpoints = vi.fn().mockResolvedValue([]);
  const mockGetWebhookEndpointsWithSecrets = vi.fn().mockResolvedValue([]);
  return { mockGetWebhookEndpoints, mockGetWebhookEndpointsWithSecrets };
});
vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpoints: (...args: unknown[]) => mockGetWebhookEndpoints(...args),
  getWebhookEndpointsWithSecrets: (...args: unknown[]) => mockGetWebhookEndpointsWithSecrets(...args),
  invalidateWebhookCache: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/webhook-thresholds.js", () => ({
  detectThresholdCrossings: vi.fn().mockReturnValue([]),
}));

const { mockStoreRequestBody, mockStoreStreamingResponseBody } = vi.hoisted(() => ({
  mockStoreRequestBody: vi.fn().mockResolvedValue(undefined),
  mockStoreStreamingResponseBody: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../lib/body-storage.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/body-storage.js")>();
  return {
    ...actual,
    storeRequestBody: (...args: unknown[]) => mockStoreRequestBody(...args),
    storeStreamingResponseBody: (...args: unknown[]) => mockStoreStreamingResponseBody(...args),
  };
});

import { handleChatCompletions } from "../routes/openai.js";

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

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
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

describe("handleChatCompletions", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("passes through unknown models with $0 cost (no hard reject)", async () => {
    mockIsKnownModel.mockReturnValueOnce(false);
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ model: "unknown-model", choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-unknown" },
      }),
    );
    const body = { model: "unknown-model", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
  });

  it("includes X-NullSpend-Trace-Id on upstream error responses", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: "bad" } }), {
        status: 500,
        headers: { "content-type": "application/json" },
      }),
    );
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("includes X-NullSpend-Trace-Id on successful non-streaming response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        id: "chatcmpl-test",
        model: "gpt-4o-mini",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        choices: [{ message: { role: "assistant", content: "hi" } }],
      }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-trace" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );
    expect(res.status).toBe(200);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
    await res.text();
  });

  it("forwards upstream error responses with correct status", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: "Invalid model", type: "invalid_request_error" } }), {
        status: 404,
        headers: { "content-type": "application/json", "x-request-id": "req-test-404" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "fake-model", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "fake-model", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error.message).toBe("Invalid model");
  });

  it("returns 502 when upstream streaming response has no body", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-no-body",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    );

    expect(res.status).toBe(502);
    const text = await res.text();
    expect(text).toContain("No response body");
  });

  it("handles non-streaming response with valid usage", async () => {
    const mockResponse = {
      id: "chatcmpl-test",
      model: "gpt-4o-mini-2024-07-18",
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-non-stream",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    );

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.usage.prompt_tokens).toBe(10);
    expect(body.usage.completion_tokens).toBe(5);
    expect(body.model).toBe("gpt-4o-mini-2024-07-18");
  });

  it("handles non-streaming response with unparseable body gracefully", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("this is not json at all!", {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-bad-json",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    );

    // Should still return the raw text to the client
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toBe("this is not json at all!");
  });

  it("handles non-streaming response with no usage object", async () => {
    const mockResponse = {
      id: "chatcmpl-test",
      model: "gpt-4o-mini-2024-07-18",
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-no-usage",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
      }),
    );

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).not.toHaveProperty("usage");
  });

  it("handles streaming response with proper SSE passthrough", async () => {
    const sseChunks = [
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-stream",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    );

    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("data:");
    expect(text).toContain("[DONE]");
    expect(text).toContain('"content":"hi"');
  });

  it("force-merges stream_options.include_usage for streaming requests", async () => {
    let capturedBody: string | null = null;

    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
      capturedBody = init.body as string;
      const sseChunks = [
        'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
        "data: [DONE]\n\n",
      ];
      return new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-merge",
        },
      });
    });

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: false },
    };

    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
    await res.text();

    expect(capturedBody).toBeTruthy();
    const sentBody = JSON.parse(capturedBody!);
    expect(sentBody.stream_options.include_usage).toBe(true);
  });

  it("extracts model from body as 'unknown' when model is missing", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: "model required" } }), {
        status: 400,
        headers: { "content-type": "application/json" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ messages: [{ role: "user", content: "hi" }] }),
    );

    // OpenAI will reject this, proxy forwards the error
    expect(res.status).toBe(400);
  });

  it("forwards x-request-id header from upstream to client", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-unique-id-abc",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(res.headers.get("x-request-id")).toBe("req-unique-id-abc");
    await res.text();
  });

  it("forwards x-ratelimit headers from upstream to client", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-rl",
          "x-ratelimit-limit-requests": "500",
          "x-ratelimit-remaining-requests": "499",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(res.headers.get("x-ratelimit-limit-requests")).toBe("500");
    expect(res.headers.get("x-ratelimit-remaining-requests")).toBe("499");
    await res.text();
  });

  it("strips x-nullspend-key header before sending to OpenAI", async () => {
    let capturedHeaders: Headers | null = null;

    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
      capturedHeaders = init.headers as Headers;
      return new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [] }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-strip" },
      });
    });

    await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(capturedHeaders).toBeTruthy();
    expect(capturedHeaders!.get("x-nullspend-key")).toBeNull();
  });

  it("streaming response includes anti-buffering headers", async () => {
    const sseChunks = [
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-stream-headers",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toBe("no-cache, no-transform");
    expect(res.headers.get("x-accel-buffering")).toBe("no");
    expect(res.headers.get("connection")).toBe("keep-alive");
    await res.text();
  });

  it("non-streaming response does NOT include anti-buffering headers", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-non-stream-headers" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toBeNull();
    expect(res.headers.get("x-accel-buffering")).toBeNull();
    await res.text();
  });

  it("passes AbortSignal.timeout to upstream fetch", async () => {
    let capturedSignal: AbortSignal | null = null;

    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
      capturedSignal = init.signal as AbortSignal;
      return new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [] }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-timeout" },
      });
    });

    await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(capturedSignal).toBeTruthy();
    expect(capturedSignal!.aborted).toBe(false);
  });

  it("includes enrichment fields in non-streaming cost event", async () => {
    mockLogCostEvent.mockClear();

    const mockResponse = {
      id: "chatcmpl-enrich",
      model: "gpt-4o-mini-2024-07-18",
      choices: [{
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_abc", type: "function", function: { name: "get_weather", arguments: '{"city":"SF"}' } },
          ],
        },
        finish_reason: "tool_calls",
      }],
      usage: { prompt_tokens: 50, completion_tokens: 20 },
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-enrich",
        },
      }),
    );

    const tools = [{ type: "function", function: { name: "get_weather", parameters: { type: "object" } } }];
    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "weather?" }],
      tools,
    };

    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { sessionId: "sess-123" }),
    );

    expect(res.status).toBe(200);
    await res.text();

    // Give waitUntil microtask a tick
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEvent).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.objectContaining({
        source: "proxy",
        sessionId: "sess-123",
        upstreamDurationMs: expect.any(Number),
        toolDefinitionTokens: expect.any(Number),
        toolCallsRequested: [{ name: "get_weather", id: "call_abc" }],
      }),
    );
    const callArgs = mockLogCostEvent.mock.calls[0][2];
    expect(callArgs.toolDefinitionTokens).toBeGreaterThan(0);
    expect(callArgs.upstreamDurationMs).toBeGreaterThanOrEqual(0);
  });

  it("includes tags from context in cost event", async () => {
    mockLogCostEvent.mockClear();

    const mockResponse = {
      id: "chatcmpl-tags",
      model: "gpt-4o-mini-2024-07-18",
      choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "x-request-id": "req-tags",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    };

    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { tags: { project: "alpha", env: "prod" } }),
    );

    expect(res.status).toBe(200);
    await res.text();
    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEvent).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.objectContaining({
        tags: expect.objectContaining({ project: "alpha", env: "prod" }),
      }),
    );
  });

  it("includes toolCallsRequested from streaming response", async () => {
    mockLogCostEvent.mockClear();

    const sseChunks = [
      'data: {"id":"chatcmpl-stc","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-stc","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"test\\"}"}}]},"finish_reason":null}]}\n\n',
      'data: {"id":"chatcmpl-stc","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":10}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-stream-tc",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "search" }],
      stream: true,
    };

    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv(),
      makeCtx(body),
    );

    expect(res.status).toBe(200);
    await res.text();

    // Give waitUntil + SSE parser time to finish
    await new Promise((r) => setTimeout(r, 50));

    expect(mockLogCostEvent).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.objectContaining({
        source: "proxy",
        toolCallsRequested: [{ name: "search", id: "call_s1" }],
      }),
    );
  });

  describe("upstream routing", () => {
    it("routes to custom upstream when x-nullspend-upstream is set", async () => {
      let capturedUrl: string | null = null;

      globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
        capturedUrl = url;
        return new Response(JSON.stringify({ model: "llama-3.1-70b-versatile", choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-groq" },
        });
      });

      const body = { model: "llama-3.1-70b-versatile", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://api.groq.com/openai" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
      expect(capturedUrl).toBe("https://api.groq.com/openai/v1/chat/completions");
      await res.text();
    });

    it("returns 400 for disallowed upstream URL", async () => {
      const fetchSpy = vi.fn();
      globalThis.fetch = fetchSpy;

      const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://evil.example.com" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(400);
      const json = await res.json();
      expect(json.error.code).toBe("invalid_upstream");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("allows unknown models when custom upstream is set", async () => {
      mockIsKnownModel.mockClear();

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ model: "llama-3.1-70b-versatile", choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-unknown-model" },
        }),
      );

      const body = { model: "llama-3.1-70b-versatile", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://api.groq.com/openai" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
      await res.text();
    });

    it("PXY-3: logs estimated cost for unknown model on custom upstream (not $0)", async () => {
      mockLogCostEvent.mockClear();

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ model: "llama-3.1-70b-versatile", choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-zero-cost" },
        }),
      );

      const body = { model: "llama-3.1-70b-versatile", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://api.groq.com/openai" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
      await res.text();
      await new Promise((r) => setTimeout(r, 10));

      expect(mockLogCostEvent).toHaveBeenCalledWith(
        undefined,
        expect.anything(),
        expect.objectContaining({
          source: "proxy",
          model: "llama-3.1-70b-versatile",
          tags: expect.objectContaining({ _ns_unpriced: "true", _ns_estimated: "true" }),
        }),
      );
      // PXY-3: cost should be the estimate (> 0), not $0
      const costArg = mockLogCostEvent.mock.calls[0][2] as { costMicrodollars: number };
      expect(costArg.costMicrodollars).toBeGreaterThan(0);
    });

    it("defaults to OpenAI when no upstream header is set", async () => {
      let capturedUrl: string | null = null;

      globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
        capturedUrl = url;
        return new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-default" },
        });
      });

      const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
      expect(capturedUrl).toBe("https://api.openai.com/v1/chat/completions");
      await res.text();
    });

    it("normalizes trailing slash in upstream URL", async () => {
      let capturedUrl: string | null = null;

      globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
        capturedUrl = url;
        return new Response(JSON.stringify({ model: "llama-3.1-70b-versatile", choices: [] }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-slash" },
        });
      });

      const body = { model: "llama-3.1-70b-versatile", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://api.groq.com/openai/" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
      expect(capturedUrl).toBe("https://api.groq.com/openai/v1/chat/completions");
      await res.text();
    });

    it("does not forward x-nullspend-upstream header to upstream", async () => {
      let capturedHeaders: Headers | null = null;

      globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
        capturedHeaders = init.headers as Headers;
        return new Response(JSON.stringify({ model: "llama-3.1-70b-versatile", choices: [] }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-no-fwd" },
        });
      });

      const body = { model: "llama-3.1-70b-versatile", messages: [{ role: "user", content: "hi" }] };
      await handleChatCompletions(
        makeRequest(body, { "x-nullspend-upstream": "https://api.groq.com/openai" }),
        makeEnv(),
        makeCtx(body),
      );

      expect(capturedHeaders).toBeTruthy();
      expect(capturedHeaders!.get("x-nullspend-upstream")).toBeNull();
    });

    it("passes through unknown models without upstream header (no hard reject)", async () => {
      mockIsKnownModel.mockReturnValueOnce(false);
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ model: "unknown-model", choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-pass" },
        }),
      );

      const body = { model: "unknown-model", messages: [{ role: "user", content: "hi" }] };
      const res = await handleChatCompletions(
        makeRequest(body),
        makeEnv(),
        makeCtx(body),
      );

      expect(res.status).toBe(200);
    });
  });

  it("includes NullSpend-Version header on successful non-streaming response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-version-header" },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      makeEnv(),
      makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("NullSpend-Version")).toBe("2026-04-01");
    await res.text();
  });

  describe("webhook dispatch", () => {
    beforeEach(() => {
      mockGetWebhookEndpoints.mockReset();
      mockGetWebhookEndpointsWithSecrets.mockReset();
    });

    it("dispatches cost_event with per-endpoint apiVersion for each endpoint", async () => {
      const mockResponse = {
        id: "chatcmpl-wh-version",
        model: "gpt-4o-mini-2024-07-18",
        choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      };

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: {
            "content-type": "application/json",
            "x-request-id": "req-wh-version",
          },
        }),
      );

      // Two endpoints with distinct apiVersions
      const endpointV1 = { id: "ep-1", url: "https://hooks.example.com/1", signingSecret: "sec1", eventTypes: [], apiVersion: "2026-04-01", defaultTags: {} };
      const endpointV2 = { id: "ep-2", url: "https://hooks.example.com/2", signingSecret: "sec2", eventTypes: [], apiVersion: "2099-01-01" };

      // getWebhookEndpoints returns non-empty (cache hit) so the secrets path is entered
      mockGetWebhookEndpoints.mockResolvedValue([
        { id: "ep-1", url: "https://hooks.example.com/1", eventTypes: [], apiVersion: "2026-04-01", defaultTags: {} },
        { id: "ep-2", url: "https://hooks.example.com/2", eventTypes: [], apiVersion: "2099-01-01" },
      ]);
      mockGetWebhookEndpointsWithSecrets.mockResolvedValue([endpointV1, endpointV2]);

      const dispatchSpy = vi.fn().mockResolvedValue(undefined);

      const body = {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      };

      const res = await handleChatCompletions(
        makeRequest(body),
        makeEnv(),
        makeCtx(body, {
          auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
          webhookDispatcher: { dispatch: dispatchSpy },
        }),
      );

      expect(res.status).toBe(200);
      await res.text();

      // Allow the waitUntil microtask to complete
      await new Promise((r) => setTimeout(r, 20));

      expect(dispatchSpy).toHaveBeenCalledTimes(2);

      // First call: endpoint ep-1 with apiVersion "2026-04-01"
      const [ep1Arg, event1Arg] = dispatchSpy.mock.calls[0];
      expect(ep1Arg.id).toBe("ep-1");
      expect(event1Arg.api_version).toBe("2026-04-01");

      // Second call: endpoint ep-2 with apiVersion "2099-01-01"
      const [ep2Arg, event2Arg] = dispatchSpy.mock.calls[1];
      expect(ep2Arg.id).toBe("ep-2");
      expect(event2Arg.api_version).toBe("2099-01-01");
    });

    it("dispatches ThinWebhookEvent (related_object, no data) for thin endpoint on non-streaming response", async () => {
      const mockResponse = {
        id: "chatcmpl-thin",
        model: "gpt-4o-mini-2024-07-18",
        choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      };

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: {
            "content-type": "application/json",
            "x-request-id": "req-thin-test",
          },
        }),
      );

      // Thin endpoint
      const thinEndpoint = {
        id: "ep-thin",
        url: "https://hooks.example.com/thin",
        signingSecret: "sec-thin",
        previousSigningSecret: null,
        secretRotatedAt: null,
        eventTypes: [],
        apiVersion: "2026-04-01", defaultTags: {},
        payloadMode: "thin" as const,
      };

      mockGetWebhookEndpoints.mockResolvedValue([
        { id: "ep-thin", url: "https://hooks.example.com/thin", eventTypes: [] },
      ]);
      mockGetWebhookEndpointsWithSecrets.mockResolvedValue([thinEndpoint]);

      const dispatchSpy = vi.fn().mockResolvedValue(undefined);

      const body = {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      };

      const res = await handleChatCompletions(
        makeRequest(body),
        makeEnv(),
        makeCtx(body, {
          auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
          webhookDispatcher: { dispatch: dispatchSpy },
        }),
      );

      expect(res.status).toBe(200);
      await res.text();

      // Allow the waitUntil microtask to complete
      await new Promise((r) => setTimeout(r, 20));

      expect(dispatchSpy).toHaveBeenCalledTimes(1);

      const [epArg, eventArg] = dispatchSpy.mock.calls[0];
      expect(epArg.id).toBe("ep-thin");

      // ThinWebhookEvent shape: has related_object, no data
      expect(eventArg.type).toBe("cost_event.created");
      expect(eventArg).toHaveProperty("related_object");
      expect(eventArg.related_object.id).toBe("req-thin-test");
      expect(eventArg.related_object.type).toBe("cost_event");
      expect(eventArg.related_object.url).toContain("requestId=req-thin-test");
      expect(eventArg.related_object.url).toContain("provider=openai");
      expect(eventArg).not.toHaveProperty("data");
      expect(eventArg.api_version).toBe("2026-04-01");
    });
  });

  describe("latency timing headers", () => {
    it("non-streaming response includes x-nullspend-overhead-ms header", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-timing" },
        }),
      );

      const res = await handleChatCompletions(
        makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
        makeEnv(),
        makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      );

      expect(res.status).toBe(200);
      expect(res.headers.get("x-nullspend-overhead-ms")).toMatch(/^\d+$/);
      const serverTiming = res.headers.get("Server-Timing")!;
      expect(serverTiming).toContain("overhead;dur=");
      expect(serverTiming).toContain("upstream;dur=");
      expect(serverTiming).toContain("total;dur=");
      await res.text();
    });

    it("streaming response includes timing headers", async () => {
      const sseChunks = [
        'data: {"id":"chatcmpl-t","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
        'data: {"id":"chatcmpl-t","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
        "data: [DONE]\n\n",
      ];

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(makeSSEStream(sseChunks), {
          status: 200,
          headers: { "content-type": "text/event-stream", "x-request-id": "req-stream-timing" },
        }),
      );

      const res = await handleChatCompletions(
        makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }], stream: true }),
        makeEnv(),
        makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }], stream: true }),
      );

      expect(res.status).toBe(200);
      expect(res.headers.get("x-nullspend-overhead-ms")).toMatch(/^\d+$/);
      expect(res.headers.get("Server-Timing")).toContain("overhead;dur=");
      await res.text();
    });

    it("upstream error response includes timing headers", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { message: "Server error", type: "server_error" } }), {
          status: 500,
          headers: { "content-type": "application/json", "x-request-id": "req-err-timing" },
        }),
      );

      const res = await handleChatCompletions(
        makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
        makeEnv(),
        makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      );

      expect(res.status).toBe(500);
      expect(res.headers.get("x-nullspend-overhead-ms")).toMatch(/^\d+$/);
      expect(res.headers.get("Server-Timing")).toContain("upstream;dur=");
    });

    it("emits proxy_latency metric on non-streaming response", async () => {
      const logSpy = vi.spyOn(console, "log");

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ model: "gpt-4o-mini", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
          status: 200,
          headers: { "content-type": "application/json", "x-request-id": "req-metric" },
        }),
      );

      const res = await handleChatCompletions(
        makeRequest({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
        makeEnv(),
        makeCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      );

      expect(res.status).toBe(200);
      await res.text();

      const metricCall = logSpy.mock.calls.find(
        (call) => typeof call[0] === "string" && call[0].includes('"_metric":"proxy_latency"'),
      );
      expect(metricCall).toBeTruthy();
      const parsed = JSON.parse(metricCall![0] as string);
      expect(parsed.provider).toBe("openai");
      expect(parsed.model).toBe("gpt-4o-mini");
      expect(typeof parsed.overheadMs).toBe("number");
      expect(typeof parsed.upstreamMs).toBe("number");
      expect(typeof parsed.totalMs).toBe("number");
      expect(parsed.streaming).toBe(false);
    });
  });

  it("cost calculation failure in streaming waitUntil does not affect response", async () => {
    const sseChunks = [
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":"NaN","completion_tokens":-1}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-bad-usage",
        },
      }),
    );

    const res = await handleChatCompletions(
      makeRequest({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      makeEnv(),
      makeCtx({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    );

    // Response should still be delivered regardless of cost calc issues
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("[DONE]");
  });

  it("stores streaming response body in R2 when requestLoggingEnabled", async () => {
    const sseChunks = [
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"delta":{"content":"hi"}}]}\n\n',
      'data: {"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
      "data: [DONE]\n\n",
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-request-id": "req-body-log",
        },
      }),
    );

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    };

    const mockBucket = { put: vi.fn().mockResolvedValue(undefined), get: vi.fn().mockResolvedValue(null) };
    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv({ BODY_STORAGE: mockBucket }),
      makeCtx(body, { requestLoggingEnabled: true }),
    );

    expect(res.status).toBe(200);
    // Consume the stream so the waitUntil callback fires
    await res.text();
    // Allow waitUntil microtasks to settle
    await new Promise((r) => setTimeout(r, 50));

    expect(mockStoreStreamingResponseBody).toHaveBeenCalledWith(
      mockBucket,
      "user-1",
      "req-body-log",
      expect.stringContaining("data:"),
    );
    expect(mockStoreRequestBody).toHaveBeenCalledWith(
      mockBucket,
      "user-1",
      "req-body-log",
      expect.stringContaining('"model":"gpt-4o-mini"'),
    );
  });

  it("echoes X-NullSpend-Session header when session ID is present", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        id: "chatcmpl-test",
        model: "gpt-4o-mini",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        choices: [{ message: { role: "assistant", content: "hi" } }],
      }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-session" },
      }),
    );

    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { sessionId: "session-echo-test" }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("X-NullSpend-Session")).toBe("session-echo-test");
    await res.text();
  });

  it("does not include X-NullSpend-Session when session ID is null", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        id: "chatcmpl-test",
        model: "gpt-4o-mini",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        choices: [{ message: { role: "assistant", content: "hi" } }],
      }), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-no-session" },
      }),
    );

    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { sessionId: null }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("X-NullSpend-Session")).toBeNull();
    await res.text();
  });
});
