import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import type { RequestContext } from "../lib/context.js";

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

import { handleProviderRequest } from "../routes/provider-handler.js";
import { geminiChatAdapter } from "../providers/gemini.js";
import { matchProviderRoute } from "../providers/registry.js";

// ── Helpers ─────────────────────────────────────────────────────────

function makeRequest(
  path: string,
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Request {
  return new Request(`http://localhost${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer gemini-api-key-test",
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

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    bodyByteLength: new TextEncoder().encode(JSON.stringify(body)).byteLength,
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

function makeGeminiSSEStream(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(event));
      }
      controller.close();
    },
  });
}

const GEMINI_NON_STREAMING_RESPONSE = {
  candidates: [{
    content: {
      role: "model",
      parts: [{ text: "Hello from Gemini!" }],
    },
    finishReason: "STOP",
  }],
  usageMetadata: {
    promptTokenCount: 25,
    candidatesTokenCount: 10,
    totalTokenCount: 35,
  },
  modelVersion: "gemini-2.5-flash",
  responseId: "resp_abc123",
};

// ── Registry Tests ──────────────────────────────────────────────────

describe("matchProviderRoute — Gemini paths", () => {
  it("matches :generateContent to geminiChatAdapter", () => {
    const adapter = matchProviderRoute("/v1beta/models/gemini-2.5-flash:generateContent");
    expect(adapter).toBeDefined();
    expect(adapter?.provider).toBe("google");
  });

  it("matches :streamGenerateContent to geminiChatAdapter", () => {
    const adapter = matchProviderRoute("/v1beta/models/gemini-2.5-pro:streamGenerateContent");
    expect(adapter).toBeDefined();
    expect(adapter?.provider).toBe("google");
  });

  it("returns undefined for unknown Gemini methods", () => {
    expect(matchProviderRoute("/v1beta/models/gemini-2.5-flash:countTokens")).toBeUndefined();
  });

  it("returns undefined for prefix without method", () => {
    expect(matchProviderRoute("/v1beta/models/gemini-2.5-flash")).toBeUndefined();
  });

  it("returns undefined for bare prefix", () => {
    expect(matchProviderRoute("/v1beta/models/")).toBeUndefined();
  });

  it("still matches OpenAI exact route", () => {
    const adapter = matchProviderRoute("/v1/chat/completions");
    expect(adapter?.provider).toBe("openai");
  });

  it("still matches Anthropic exact route", () => {
    const adapter = matchProviderRoute("/v1/messages");
    expect(adapter?.provider).toBe("anthropic");
  });

  it("matches model names with dots (version numbers)", () => {
    const adapter = matchProviderRoute("/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent");
    expect(adapter).toBeDefined();
    expect(adapter?.provider).toBe("google");
  });
});

// ── Adapter Hook Tests ──────────────────────────────────────────────

describe("geminiChatAdapter hooks", () => {
  it("extractModel parses model from URL path", () => {
    const req = new Request("http://localhost/v1beta/models/gemini-2.5-flash:generateContent");
    expect(geminiChatAdapter.extractModel!(req, {})).toBe("gemini-2.5-flash");
  });

  it("extractModel returns 'unknown' for malformed path", () => {
    const req = new Request("http://localhost/v1beta/models/:generateContent");
    expect(geminiChatAdapter.extractModel!(req, {})).toBe("unknown");
  });

  it("extractModel handles dated model names", () => {
    const req = new Request("http://localhost/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent");
    expect(geminiChatAdapter.extractModel!(req, {})).toBe("gemini-2.5-pro-exp-03-25");
  });

  it("extractModel rejects percent-encoded path traversal", () => {
    const req = new Request("http://localhost/v1beta/models/gemini-2.5-flash%2F..%2F..%2Finternal:generateContent");
    expect(geminiChatAdapter.extractModel!(req, {})).toBe("unknown");
  });

  it("extractModel rejects model names with special characters", () => {
    const req = new Request("http://localhost/v1beta/models/evil%00model:generateContent");
    expect(geminiChatAdapter.extractModel!(req, {})).toBe("unknown");
  });

  it("isStreaming returns true for :streamGenerateContent", () => {
    const req = new Request("http://localhost/v1beta/models/gemini-2.5-flash:streamGenerateContent");
    expect(geminiChatAdapter.isStreaming!(req, {})).toBe(true);
  });

  it("isStreaming returns false for :generateContent", () => {
    const req = new Request("http://localhost/v1beta/models/gemini-2.5-flash:generateContent");
    expect(geminiChatAdapter.isStreaming!(req, {})).toBe(false);
  });
});

// ── Route Integration Tests ─────────────────────────────────────────

describe("Gemini route via handleProviderRequest", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("non-streaming: proxies generateContent and returns response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(200);
    const responseBody = await res.json();
    expect(responseBody.candidates[0].content.parts[0].text).toBe("Hello from Gemini!");
  });

  it("includes X-NullSpend-Trace-Id on responses", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("returns x-request-id header for body retrieval", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    // Gemini has no upstream x-request-id, so the handler generates a UUID and sets it
    const requestId = res.headers.get("x-request-id");
    expect(requestId).not.toBeNull();
    expect(requestId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  });

  it("includes X-NullSpend-Trace-Id on upstream errors", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: 400, message: "Invalid request", status: "INVALID_ARGUMENT" } }), {
        status: 400,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(400);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("streaming: proxies streamGenerateContent and returns SSE", async () => {
    const sseData = [
      `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: "Hello" }] } }], usageMetadata: { promptTokenCount: 10 }, modelVersion: "gemini-2.5-flash", responseId: "resp_1" })}\n\n`,
      `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: " world" }] }, finishReason: "STOP" }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 }, modelVersion: "gemini-2.5-flash", responseId: "resp_1" })}\n\n`,
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeGeminiSSEStream(sseData), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:streamGenerateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(200);
    // Streaming response: body must be a ReadableStream, not consumed JSON
    expect(res.body).toBeInstanceOf(ReadableStream);
    // Verify streaming headers are set
    expect(res.headers.get("cache-control")).toBe("no-cache, no-transform");
    expect(res.headers.get("x-accel-buffering")).toBe("no");
  });

  it("streaming normalization: :generateContent is never streaming regardless of body", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    // Body has stream: true but endpoint is :generateContent (non-streaming)
    const body = { contents: [{ parts: [{ text: "hi" }] }], stream: true };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    // Should be treated as non-streaming (200 JSON, not SSE)
    expect(res.status).toBe(200);
    const responseBody = await res.json();
    expect(responseBody.candidates).toBeDefined();
  });

  it("body passes through unmodified (native Gemini format)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = {
      contents: [{ parts: [{ text: "hi" }] }],
      generationConfig: { temperature: 0.7, maxOutputTokens: 1000 },
      tools: [{ functionDeclarations: [{ name: "test_fn", description: "test" }] }],
    };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    const fetchCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const sentBody = JSON.parse(fetchCall[1].body);
    // Body should be identical to what was sent — no stripping, no mutation
    expect(sentBody.contents).toEqual(body.contents);
    expect(sentBody.generationConfig).toEqual(body.generationConfig);
    expect(sentBody.tools).toEqual(body.tools);
  });

  it("upstream URL includes model from request path", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-pro:generateContent", body);
    await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    const fetchCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const upstreamUrl = fetchCall[0] as string;
    expect(upstreamUrl).toContain("models/gemini-2.5-pro:generateContent");
  });

  it("streaming upstream URL includes ?alt=sse", async () => {
    const sseData = [
      `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: "hi" }], role: "model" }, finishReason: "STOP" }], usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 1, totalTokenCount: 6 }, modelVersion: "gemini-2.5-flash", responseId: "r1" })}\n\n`,
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeGeminiSSEStream(sseData), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:streamGenerateContent", body);
    await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    const fetchCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const upstreamUrl = fetchCall[0] as string;
    expect(upstreamUrl).toContain("?alt=sse");
  });

  it("forwards x-goog-api-key from Bearer token", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    const fetchCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const headers = fetchCall[1].headers;
    expect(headers.get("x-goog-api-key")).toBe("gemini-api-key-test");
  });

  it("handles upstream 500 with sanitized error", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: 500, message: "Internal server error", status: "INTERNAL" } }), {
        status: 500,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(500);
    const errorBody = await res.json();
    // sanitizeUpstreamError replaces 5xx messages with generic text
    expect(errorBody.error.message).toContain("Upstream returned 500");
  });

  it("passes through unknown models with $0 cost", async () => {
    mockIsKnownModel.mockReturnValueOnce(false);
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-unknown:generateContent", body);
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(200);
    // For unknown models, the cost calculator returns 0 (no pricing data).
    // But the PXY-3 fallback in provider-handler.ts detects cost=0 + tokens>0
    // and replaces with the pre-request estimate. The estimate for unknown models
    // is $1 (1_000_000 microdollars). Verify the cost event is logged with the
    // estimate and tagged as _ns_unpriced + _ns_estimated.
    expect(mockLogCostEvent).toHaveBeenCalled();
    const costEventArgs = mockLogCostEvent.mock.calls[0];
    const costEvent = costEventArgs[2]; // third arg: logCostEventQueued(queue, conn, event)
    expect(costEvent.costMicrodollars).toBe(1_000_000); // $1 fallback estimate
    expect(costEvent.provider).toBe("google");
  });

  it("x-nullspend-upstream override with allowed domain", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body, {
      "x-nullspend-upstream": "https://generativelanguage.googleapis.com",
    });
    await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    const fetchCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const upstreamUrl = fetchCall[0] as string;
    expect(upstreamUrl).toContain("generativelanguage.googleapis.com");
  });

  it("x-nullspend-upstream rejects unlisted domain", async () => {
    const body = { contents: [{ parts: [{ text: "hi" }] }] };
    const req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body, {
      "x-nullspend-upstream": "https://evil.example.com",
    });
    const res = await handleProviderRequest(req, makeEnv(), makeCtx(body), geminiChatAdapter);

    expect(res.status).toBe(400);
    const errorBody = await res.json();
    expect(errorBody.error.code).toBe("invalid_upstream");
  });

  it("extractResponseTags captures generationConfig metadata", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GEMINI_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const body = {
      contents: [{ parts: [{ text: "hi" }] }],
      generationConfig: { temperature: 0.9, maxOutputTokens: 2048 },
    };
    const _req = makeRequest("/v1beta/models/gemini-2.5-flash:generateContent", body);

    // Verify extractResponseTags directly
    const tags = geminiChatAdapter.extractResponseTags(
      new Response("", { status: 200 }),
      body,
      false,
    );
    expect(tags._ns_temperature).toBe("0.9");
    expect(tags._ns_max_tokens).toBe("2048");
  });

  it("extractResponseTags counts functionDeclarations as tool_count", () => {
    const body = {
      contents: [{ parts: [{ text: "hi" }] }],
      tools: [
        { functionDeclarations: [{ name: "fn1" }, { name: "fn2" }] },
        { functionDeclarations: [{ name: "fn3" }] },
      ],
    };

    const tags = geminiChatAdapter.extractResponseTags(
      new Response("", { status: 200 }),
      body,
      false,
    );
    expect(tags._ns_tool_count).toBe("3");
  });
});
