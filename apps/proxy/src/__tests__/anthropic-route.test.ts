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

import { handleAnthropicMessages } from "../routes/anthropic.js";

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

function makeAnthropicSSEStream(chunks: string[]): ReadableStream<Uint8Array> {
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

const ANTHROPIC_NON_STREAMING_RESPONSE = {
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

describe("handleAnthropicMessages", () => {
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

  it("passes through unknown models with $0 cost (no hard reject)", async () => {
    mockIsKnownModel.mockReturnValueOnce(false);
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "msg_1", type: "message", role: "assistant", model: "claude-unknown", content: [{ type: "text", text: "hi" }], stop_reason: "end_turn", usage: { input_tokens: 5, output_tokens: 2 } }), {
        status: 200,
        headers: { "content-type": "application/json", "request-id": "req-unknown" },
      }),
    );
    const body = { model: "claude-unknown", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
  });

  it("includes X-NullSpend-Trace-Id on upstream error responses", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: "bad" } }), {
        status: 500,
        headers: { "content-type": "application/json" },
      }),
    );
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
  });

  it("includes X-NullSpend-Trace-Id on successful non-streaming response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_trace_test",
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
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
    await res.text();
  });

  it("handles valid non-streaming request", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_018EeWyXxfu5pfWkrYcMdjWG",
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
    const json = await res.json();
    expect(json.model).toBe("claude-sonnet-4-20250514");
    expect(json.usage.input_tokens).toBe(25);
    expect(json.usage.output_tokens).toBe(10);
  });

  it("handles valid streaming request with anti-buffering headers", async () => {
    const sseChunks = [
      'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","usage":{"input_tokens":25,"output_tokens":0}}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "req_stream_test",
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
    expect(res.headers.get("cache-control")).toBe("no-cache, no-transform");
    expect(res.headers.get("x-accel-buffering")).toBe("no");
    expect(res.headers.get("connection")).toBe("keep-alive");

    const text = await res.text();
    expect(text).toContain("message_start");
    expect(text).toContain("Hello");
  });

  it("forwards upstream 4xx errors to client", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          type: "error",
          error: { type: "invalid_request_error", message: "max_tokens is required" },
        }),
        {
          status: 400,
          headers: {
            "content-type": "application/json",
            "request-id": "req_err_400",
          },
        },
      ),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.error.type).toBe("invalid_request_error");
  });

  it("forwards upstream 5xx errors to client", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          type: "error",
          error: { type: "api_error", message: "Internal server error" },
        }),
        {
          status: 500,
          headers: {
            "content-type": "application/json",
            "request-id": "req_err_500",
          },
        },
      ),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.status).toBe(500);
  });

  it("returns 502 when streaming response has no body", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "req_no_body",
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

    expect(res.status).toBe(502);
    const text = await res.text();
    expect(text).toContain("No response body");
  });

  it("handles non-streaming response with no usage gracefully", async () => {
    const responseWithoutUsage = {
      id: "msg_no_usage",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [{ type: "text", text: "Hello!" }],
      stop_reason: "end_turn",
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responseWithoutUsage), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_no_usage",
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
    const json = await res.json();
    expect(json).not.toHaveProperty("usage");
  });

  it("handles non-streaming response with unparseable body gracefully", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("this is not json at all!", {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_bad_json",
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
    const text = await res.text();
    expect(text).toBe("this is not json at all!");
  });

  it("includes enrichment fields in non-streaming cost event with tool_use", async () => {
    mockLogCostEvent.mockClear();

    const responseWithToolUse = {
      id: "msg_enrich",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      content: [
        { type: "text", text: "I'll check the weather." },
        { type: "tool_use", id: "toolu_01X", name: "get_weather", input: { city: "SF" } },
      ],
      stop_reason: "tool_use",
      usage: {
        input_tokens: 60,
        output_tokens: 25,
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responseWithToolUse), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_enrich_ant",
        },
      }),
    );

    const tools = [{ name: "get_weather", description: "Get weather", input_schema: { type: "object" } }];
    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "weather?" }],
      tools,
    };
    const res = await handleAnthropicMessages(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { sessionId: "sess-ant-1" }),
    );

    expect(res.status).toBe(200);
    await res.text();

    await new Promise((r) => setTimeout(r, 10));

    expect(mockLogCostEvent).toHaveBeenCalledWith(
      undefined,
      expect.anything(),
      expect.objectContaining({
        sessionId: "sess-ant-1",
        source: "proxy",
        upstreamDurationMs: expect.any(Number),
        toolDefinitionTokens: expect.any(Number),
        toolCallsRequested: [{ name: "get_weather", id: "toolu_01X" }],
      }),
    );
    const callArgs = mockLogCostEvent.mock.calls[0][2];
    expect(callArgs.toolDefinitionTokens).toBeGreaterThan(0);
  });

  it("includes NullSpend-Version header on successful non-streaming response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_version_header_test",
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
    expect(res.headers.get("NullSpend-Version")).toBe("2026-04-01");
    await res.text();
  });

  describe("latency timing headers", () => {
    it("non-streaming response includes x-nullspend-overhead-ms header", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
          status: 200,
          headers: { "content-type": "application/json", "request-id": "req-timing" },
        }),
      );

      const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
      const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

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
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","usage":{"input_tokens":25,"output_tokens":0}}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
      ];

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(makeAnthropicSSEStream(sseChunks), {
          status: 200,
          headers: { "content-type": "text/event-stream", "request-id": "req-stream-timing" },
        }),
      );

      const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }], stream: true };
      const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

      expect(res.status).toBe(200);
      expect(res.headers.get("x-nullspend-overhead-ms")).toMatch(/^\d+$/);
      expect(res.headers.get("Server-Timing")).toContain("overhead;dur=");
      await res.text();
    });

    it("upstream error response includes timing headers", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ type: "error", error: { type: "api_error", message: "Server error" } }), {
          status: 500,
          headers: { "content-type": "application/json", "request-id": "req-err-timing" },
        }),
      );

      const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
      const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

      expect(res.status).toBe(500);
      expect(res.headers.get("x-nullspend-overhead-ms")).toMatch(/^\d+$/);
      expect(res.headers.get("Server-Timing")).toContain("upstream;dur=");
    });

    it("emits proxy_latency metric on non-streaming response", async () => {
      const logSpy = vi.spyOn(console, "log");

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
          status: 200,
          headers: { "content-type": "application/json", "request-id": "req-metric" },
        }),
      );

      const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
      const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

      expect(res.status).toBe(200);
      await res.text();

      const metricCall = logSpy.mock.calls.find(
        (call) => typeof call[0] === "string" && call[0].includes('"_metric":"proxy_latency"'),
      );
      expect(metricCall).toBeTruthy();
      const parsed = JSON.parse(metricCall![0] as string);
      expect(parsed.provider).toBe("anthropic");
      expect(parsed.model).toBe("claude-sonnet-4-20250514");
      expect(typeof parsed.overheadMs).toBe("number");
      expect(typeof parsed.upstreamMs).toBe("number");
      expect(typeof parsed.totalMs).toBe("number");
      expect(parsed.streaming).toBe(false);
    });
  });

  it("extracts request-id and forwards as x-request-id", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_018EeWyXxfu5pfWkrYcMdjWG",
        },
      }),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));

    expect(res.headers.get("x-request-id")).toBe("req_018EeWyXxfu5pfWkrYcMdjWG");
    await res.text();
  });

  it("stores streaming response body in R2 when requestLoggingEnabled", async () => {
    const sseChunks = [
      'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","usage":{"input_tokens":25,"output_tokens":0}}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ];

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(makeAnthropicSSEStream(sseChunks), {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "req-body-log",
        },
      }),
    );

    const body = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    };

    const mockBucket = { put: vi.fn().mockResolvedValue(undefined), get: vi.fn().mockResolvedValue(null) };
    const res = await handleAnthropicMessages(
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
      expect.stringContaining("message_start"),
    );
    expect(mockStoreRequestBody).toHaveBeenCalledWith(
      mockBucket,
      "user-1",
      "req-body-log",
      JSON.stringify(body),
    );
  });

  it("echoes X-NullSpend-Session header when session ID is present", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json", "request-id": "req-session" },
      }),
    );

    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const res = await handleAnthropicMessages(
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
      new Response(JSON.stringify(ANTHROPIC_NON_STREAMING_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json", "request-id": "req-no-session" },
      }),
    );

    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const res = await handleAnthropicMessages(
      makeRequest(body),
      makeEnv(),
      makeCtx(body, { sessionId: null }),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("X-NullSpend-Session")).toBeNull();
    await res.text();
  });
});
