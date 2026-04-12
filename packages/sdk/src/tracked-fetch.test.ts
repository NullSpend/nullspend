import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { buildTrackedFetch } from "./tracked-fetch.js";
import {
  BudgetExceededError,
  MandateViolationError,
  SessionLimitExceededError,
  VelocityExceededError,
  TagBudgetExceededError,
} from "./errors.js";
import type { CostEventInput, TrackedFetchOptions, DenialReason } from "./types.js";
import type { PolicyCache, PolicyResponse } from "./policy-cache.js";

vi.mock("@nullspend/cost-engine", () => ({
  getModelPricing: vi.fn(() => ({
    inputPerMTok: 2.5,
    outputPerMTok: 10,
    cachedInputPerMTok: 1.25,
  })),
  costComponent: vi.fn((tokens: number, rate: number) => {
    if (tokens <= 0 || rate <= 0) return 0;
    return tokens * rate;
  }),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";

function makeOpenAIBody(model = "gpt-4o", stream = false): string {
  return JSON.stringify({ model, stream, messages: [{ role: "user", content: "Hi" }] });
}

function makeAnthropicBody(model = "claude-sonnet-4-20250514", stream = false): string {
  return JSON.stringify({ model, stream, messages: [{ role: "user", content: "Hi" }] });
}

function mockFetchJsonResponse(
  body: unknown,
  status = 200,
  extraHeaders?: Record<string, string>,
): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 ? "OK" : `Status ${status}`,
    headers: { "content-type": "application/json", ...extraHeaders },
  });
}

/**
 * Header that gates SDK 429 interception. The proxy stamps this on every
 * NullSpend denial Response (5 paths in shared.ts, 4 in mcp.ts). Tests that
 * exercise the typed-error path must set this; tests that exercise upstream
 * fall-through must NOT set it.
 */
const DENIED_HEADERS = { "X-NullSpend-Denied": "1" };

function mockFetchStreamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function openaiJsonResponse(model = "gpt-4o") {
  return mockFetchJsonResponse({
    id: "chatcmpl-1",
    model,
    usage: { prompt_tokens: 100, completion_tokens: 50 },
    choices: [{ message: { role: "assistant", content: "Hello!" } }],
  });
}

function anthropicJsonResponse(model = "claude-sonnet-4-20250514") {
  return mockFetchJsonResponse({
    id: "msg-1",
    model,
    usage: { input_tokens: 80, output_tokens: 40 },
    content: [{ type: "text", text: "Hello!" }],
  });
}

function openaiStreamChunks(model = "gpt-4o"): string[] {
  return [
    `data: ${JSON.stringify({ id: "chatcmpl-1", model, choices: [{ delta: { content: "Hi" } }] })}\n\n`,
    `data: ${JSON.stringify({ id: "chatcmpl-1", model, choices: [{ delta: {} }], usage: { prompt_tokens: 100, completion_tokens: 50 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
}

function anthropicStreamChunks(model = "claude-sonnet-4-20250514"): string[] {
  return [
    `event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: { model, usage: { input_tokens: 80, output_tokens: 0 } } })}\n\n`,
    `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", delta: { text: "Hi" } })}\n\n`,
    `event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", usage: { output_tokens: 40 } })}\n\n`,
    `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
  ];
}

function createMockPolicyCache(overrides: Partial<PolicyCache> = {}): PolicyCache {
  return {
    getPolicy: vi.fn().mockResolvedValue({
      budget: null,
      allowed_models: null,
      allowed_providers: null,
      cheapest_per_provider: null,
      cheapest_overall: null,
      restrictions_active: false,
    } satisfies PolicyResponse),
    checkMandate: vi.fn().mockReturnValue({ allowed: true }),
    checkBudget: vi.fn().mockReturnValue({ allowed: true }),
    getSessionLimit: vi.fn().mockReturnValue(null),
    invalidate: vi.fn(),
    ...overrides,
  };
}

async function consumeStream(response: Response): Promise<string> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let result = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("buildTrackedFetch", () => {
  let originalFetch: typeof globalThis.fetch;
  let mockFetch: ReturnType<typeof vi.fn>;
  let queueCost: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch;
    queueCost = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // Non-streaming
  // -------------------------------------------------------------------------

  describe("non-streaming OpenAI", () => {
    it("calls fetch and queues a cost event with correct values", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).toHaveBeenCalledTimes(1);

      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.provider).toBe("openai");
      expect(event.model).toBe("gpt-4o");
      expect(event.inputTokens).toBe(100);
      expect(event.outputTokens).toBe(50);
      expect(event.costMicrodollars).toBeGreaterThan(0);
    });
  });

  describe("non-streaming Anthropic", () => {
    it("calls fetch and queues a cost event with correct values", async () => {
      mockFetch.mockResolvedValue(anthropicJsonResponse());
      const trackedFetch = buildTrackedFetch("anthropic", undefined, queueCost, null);

      const response = await trackedFetch(ANTHROPIC_URL, {
        method: "POST",
        body: makeAnthropicBody(),
      });

      expect(response.status).toBe(200);
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).toHaveBeenCalledTimes(1);

      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.provider).toBe("anthropic");
      expect(event.model).toBe("claude-sonnet-4-20250514");
      expect(event.inputTokens).toBe(80);
      expect(event.outputTokens).toBe(40);
    });
  });

  // -------------------------------------------------------------------------
  // Streaming
  // -------------------------------------------------------------------------

  describe("streaming OpenAI", () => {
    it("injects stream_options.include_usage, returns readable body, queues cost after stream", async () => {
      mockFetch.mockResolvedValue(mockFetchStreamResponse(openaiStreamChunks()));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody("gpt-4o", true),
      });

      expect(response.body).toBeTruthy();

      // Consume the stream to trigger cost event
      const text = await consumeStream(response);
      expect(text).toContain("chatcmpl-1");

      // The stream_options.include_usage should have been injected
      const calledBody = JSON.parse(mockFetch.mock.calls[0][1]?.body as string);
      expect(calledBody.stream_options?.include_usage).toBe(true);

      // Wait a tick for the fire-and-forget promise
      await new Promise((r) => setTimeout(r, 10));

      expect(queueCost).toHaveBeenCalledTimes(1);
      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.provider).toBe("openai");
      expect(event.inputTokens).toBe(100);
      expect(event.outputTokens).toBe(50);
    });
  });

  describe("streaming Anthropic", () => {
    it("returns readable body and queues cost after stream completes", async () => {
      mockFetch.mockResolvedValue(mockFetchStreamResponse(anthropicStreamChunks()));
      const trackedFetch = buildTrackedFetch("anthropic", undefined, queueCost, null);

      const response = await trackedFetch(ANTHROPIC_URL, {
        method: "POST",
        body: makeAnthropicBody("claude-sonnet-4-20250514", true),
      });

      expect(response.body).toBeTruthy();
      await consumeStream(response);

      await new Promise((r) => setTimeout(r, 10));

      expect(queueCost).toHaveBeenCalledTimes(1);
      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.provider).toBe("anthropic");
      expect(event.model).toBe("claude-sonnet-4-20250514");
    });
  });

  // -------------------------------------------------------------------------
  // Passthrough cases
  // -------------------------------------------------------------------------

  describe("passthrough", () => {
    it("passes through GET /models without cost tracking", async () => {
      mockFetch.mockResolvedValue(mockFetchJsonResponse({ data: [] }));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch("https://api.openai.com/v1/models", { method: "GET" });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through non-POST methods", async () => {
      mockFetch.mockResolvedValue(mockFetchJsonResponse({}));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      // DELETE request
      await trackedFetch(OPENAI_URL, { method: "DELETE" });
      expect(queueCost).not.toHaveBeenCalled();
    });

    it("does not track 4xx error responses", async () => {
      mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: "bad request" }, 400));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(400);
      expect(queueCost).not.toHaveBeenCalled();
    });

    it("does not track 5xx error responses", async () => {
      mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: "internal" }, 500));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(500);
      expect(queueCost).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Proxy detection guard
  // -------------------------------------------------------------------------

  describe("proxy detection guard", () => {
    it("passes through when URL starts with the configured proxyUrl", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        undefined,
        queueCost,
        null,
        "https://proxy.example.com",
      );

      await trackedFetch("https://proxy.example.com/v1/chat/completions", {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).not.toHaveBeenCalled();
    });

    // Regression: header-only detection WITHOUT proxyUrl used to skip tracking.
    // That was a bypass (Codex finding #1). Now the header fallback only works
    // when proxyUrl is configured. These tests use a dummy proxyUrl to verify
    // the header fallback still works in the correct context.
    it("passes through when x-nullspend-key header is present + proxyUrl configured (Headers object)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null, "https://proxy.example.com");

      const headers = new Headers({ "x-nullspend-key": "ns_live_sk_test" });
      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers,
      });

      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through when x-nullspend-key header is present + proxyUrl configured (plain object)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null, "https://proxy.example.com");

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: { "x-nullspend-key": "ns_live_sk_test" },
      });

      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through when x-nullspend-key header is present + proxyUrl configured (array tuples)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null, "https://proxy.example.com");

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: [["x-nullspend-key", "ns_live_sk_test"]],
      });

      expect(queueCost).not.toHaveBeenCalled();
    });

    // Regression: Codex finding #1 — x-nullspend-key header WITHOUT proxyUrl
    // must NOT bypass tracking. A caller adding this header to a direct OpenAI
    // request should still get tracked.
    it("does NOT bypass tracking when x-nullspend-key header is present but NO proxyUrl configured", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: { "x-nullspend-key": "ns_live_sk_test" },
      });

      // With no proxyUrl, the SDK should track the request even though
      // the header is present — the header alone is not sufficient.
      expect(queueCost).toHaveBeenCalled();
    });

    it("passes through when configurable proxyUrl matches the request URL", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        undefined,
        queueCost,
        null,
        "https://nullspend.cjones6489.workers.dev",
      );

      await trackedFetch(
        "https://nullspend.cjones6489.workers.dev/v1/chat/completions",
        { method: "POST", body: makeOpenAIBody() },
      );

      expect(mockFetch).toHaveBeenCalledTimes(1);
      // No client-side cost tracking when proxy is detected
      expect(queueCost).not.toHaveBeenCalled();
    });

    it("does NOT match a different host even when proxyUrl is configured", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        undefined,
        queueCost,
        null,
        "https://nullspend.cjones6489.workers.dev",
      );

      // Direct call to api.openai.com should still be tracked client-side
      await trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });

      expect(queueCost).toHaveBeenCalledTimes(1);
    });

    // ────────────────────────────────────────────────────────────────
    // Regression: proxyUrl origin comparison (no substring bypass)
    // ────────────────────────────────────────────────────────────────

    it("does NOT match a confusable hostname (origin comparison, not substring)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        undefined,
        queueCost,
        null,
        "https://proxy.example.com",
      );

      // Attacker-style URL that starts with the proxy URL but hits a different origin
      await trackedFetch(
        "https://proxy.example.com.evil.com/v1/chat/completions",
        { method: "POST", body: makeOpenAIBody() },
      );

      // Must NOT be treated as proxied — client-side tracking should still run
      expect(queueCost).toHaveBeenCalledTimes(1);
    });

    it("matches different ports on same host correctly", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        undefined,
        queueCost,
        null,
        "http://localhost:8787",
      );

      // Exact port match should be detected as proxied
      await trackedFetch(
        "http://localhost:8787/v1/chat/completions",
        { method: "POST", body: makeOpenAIBody() },
      );
      expect(queueCost).not.toHaveBeenCalled();

      // Different port — same hostname but different origin — should NOT match
      await trackedFetch(
        "http://localhost:3000/v1/chat/completions",
        { method: "POST", body: makeOpenAIBody() },
      );
      expect(queueCost).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // Customer header injection (X-NullSpend-Customer)
  // -------------------------------------------------------------------------

  describe("X-NullSpend-Customer header injection", () => {
    it("injects X-NullSpend-Customer when options.customer is set (direct mode)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
    });

    it("injects X-NullSpend-Customer BEFORE the proxy bailout (proxy mode)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
        "https://nullspend.cjones6489.workers.dev",
      );

      await trackedFetch(
        "https://nullspend.cjones6489.workers.dev/v1/chat/completions",
        { method: "POST", body: makeOpenAIBody() },
      );

      // SDK should bail out (no client-side tracking) BUT the header must be on the wire
      expect(queueCost).not.toHaveBeenCalled();
      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
    });

    it("does NOT inject the header when customer is undefined", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers ?? {});
      expect(headers.has("X-NullSpend-Customer")).toBe(false);
    });

    it("preserves other headers when injecting customer", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: { Authorization: "Bearer sk-test", "X-Custom": "keep-me" },
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
      expect(headers.get("Authorization")).toBe("Bearer sk-test");
      expect(headers.get("X-Custom")).toBe("keep-me");
    });

    it("preserves existing headers when they are an array of tuples", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: [
          ["Authorization", "Bearer sk-test"],
          ["X-Custom", "keep-me"],
        ],
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
      expect(headers.get("Authorization")).toBe("Bearer sk-test");
      expect(headers.get("X-Custom")).toBe("keep-me");
    });

    it("preserves existing headers when they are a Headers instance", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      const existing = new Headers({
        Authorization: "Bearer sk-test",
        "X-Custom": "keep-me",
      });
      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: existing,
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
      expect(headers.get("Authorization")).toBe("Bearer sk-test");
      expect(headers.get("X-Custom")).toBe("keep-me");
      // Original Headers instance should not be mutated
      expect(existing.has("X-NullSpend-Customer")).toBe(false);
    });

    // ────────────────────────────────────────────────────────────────
    // Regression: Request-object input must preserve Authorization
    // ────────────────────────────────────────────────────────────────

    it("preserves Authorization when input is a Request and init is absent", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      // Caller passes a Request object (not url+init). Without the fix, the
      // addHeader path would create a new init with only the customer header,
      // and fetch(request, init) would REPLACE the Request's headers entirely.
      const request = new Request(OPENAI_URL, {
        method: "POST",
        headers: { Authorization: "Bearer sk-test", "X-Custom": "keep-me" },
        body: makeOpenAIBody(),
      });

      await trackedFetch(request);

      // The fetch call should have received a Request (not url+init) with
      // the customer header injected but the original headers preserved.
      expect(mockFetch).toHaveBeenCalledTimes(1);
      const passedInput = mockFetch.mock.calls[0][0];
      expect(passedInput).toBeInstanceOf(Request);
      const passedHeaders = (passedInput as Request).headers;
      expect(passedHeaders.get("X-NullSpend-Customer")).toBe("acme-corp");
      expect(passedHeaders.get("Authorization")).toBe("Bearer sk-test");
      expect(passedHeaders.get("X-Custom")).toBe("keep-me");
      // Original Request should not be mutated
      expect(request.headers.has("X-NullSpend-Customer")).toBe(false);
    });

    it("preserves Authorization when input is a Request AND init has headers", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      // When init.headers is set, fetch uses init.headers and ignores the
      // Request's headers. The customer header must be added to init.
      const request = new Request(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });
      await trackedFetch(request, {
        headers: { Authorization: "Bearer sk-test", "X-Custom": "keep-me" },
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const headers = new Headers(callInit.headers);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
      expect(headers.get("Authorization")).toBe("Bearer sk-test");
      expect(headers.get("X-Custom")).toBe("keep-me");
    });

    // ────────────────────────────────────────────────────────────────
    // Regression: case-insensitive header dedup
    // ────────────────────────────────────────────────────────────────

    it("case-insensitively dedupes when caller sets x-nullspend-customer in lowercase (plain object)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: { "x-nullspend-customer": "stale-value" }, // lowercase, stale
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const rawHeaders = callInit.headers as Record<string, string>;
      // Should only have one customer header entry, with the new value
      const keys = Object.keys(rawHeaders).filter((k) => k.toLowerCase() === "x-nullspend-customer");
      expect(keys).toHaveLength(1);
      const headers = new Headers(rawHeaders);
      expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
    });

    it("case-insensitively dedupes when caller sets x-nullspend-customer in array tuples", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch(
        "openai",
        { customer: "acme-corp" },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: [
          ["x-nullspend-customer", "stale-value"],
          ["Authorization", "Bearer sk-test"],
        ],
      });

      const callInit = mockFetch.mock.calls[0][1] as RequestInit;
      const tuples = callInit.headers as [string, string][];
      // Should have exactly one customer entry (new), plus the Authorization
      const customerEntries = tuples.filter(([k]) => k.toLowerCase() === "x-nullspend-customer");
      expect(customerEntries).toHaveLength(1);
      expect(customerEntries[0][1]).toBe("acme-corp");
      expect(tuples.some(([k, v]) => k === "Authorization" && v === "Bearer sk-test")).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // Error resilience
  // -------------------------------------------------------------------------

  describe("error resilience", () => {
    it("returns the response even when cost tracking fails (JSON parse error)", async () => {
      // Return a response that will fail JSON parsing on the cloned body
      const resp = new Response("not json", {
        status: 200,
        headers: { "content-type": "application/json" },
      });
      mockFetch.mockResolvedValue(resp);

      const onCostError = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { onCostError },
        queueCost,
        null,
      );

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
      expect(queueCost).not.toHaveBeenCalled();
      expect(onCostError).toHaveBeenCalledTimes(1);
      expect(onCostError.mock.calls[0][0]).toBeInstanceOf(Error);
    });

    it("invokes onCostError callback on tracking failure", async () => {
      const resp = new Response("bad", {
        status: 200,
        headers: { "content-type": "application/json" },
      });
      mockFetch.mockResolvedValue(resp);

      const onCostError = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { onCostError },
        queueCost,
        null,
      );

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(onCostError).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // stream_options merging
  // -------------------------------------------------------------------------

  describe("stream_options merging", () => {
    it("merges include_usage with existing stream_options (does not overwrite)", async () => {
      mockFetch.mockResolvedValue(mockFetchStreamResponse(openaiStreamChunks()));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const body = JSON.stringify({
        model: "gpt-4o",
        stream: true,
        stream_options: { custom_field: "keep_me" },
        messages: [],
      });

      await trackedFetch(OPENAI_URL, { method: "POST", body });
      const calledBody = JSON.parse(mockFetch.mock.calls[0][1]?.body as string);

      expect(calledBody.stream_options.include_usage).toBe(true);
      expect(calledBody.stream_options.custom_field).toBe("keep_me");
    });
  });

  // -------------------------------------------------------------------------
  // AbortSignal
  // -------------------------------------------------------------------------

  describe("AbortSignal", () => {
    it("preserves AbortSignal through to real fetch", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const controller = new AbortController();
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        signal: controller.signal,
      });

      // The init passed to real fetch should have the signal
      const calledInit = mockFetch.mock.calls[0][1];
      expect(calledInit.signal).toBe(controller.signal);
    });
  });

  // -------------------------------------------------------------------------
  // Cancelled stream
  // -------------------------------------------------------------------------

  describe("cancelled stream", () => {
    it("does not queue cost event when usage is null from cancelled stream", async () => {
      // Create a stream that never sends usage
      const chunks = [
        `data: ${JSON.stringify({ id: "chatcmpl-1", model: "gpt-4o", choices: [{ delta: { content: "Hi" } }] })}\n\n`,
      ];
      const encoder = new TextEncoder();
      const body = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
          // Don't close
        },
      });
      const resp = new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
      mockFetch.mockResolvedValue(resp);

      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody("gpt-4o", true),
      });

      // Cancel the stream
      const reader = response.body!.getReader();
      await reader.read();
      await reader.cancel();

      await new Promise((r) => setTimeout(r, 10));

      // No usage in stream => no cost event
      expect(queueCost).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Metadata passthrough
  // -------------------------------------------------------------------------

  describe("metadata passthrough", () => {
    it("passes sessionId, traceId, and tags to cost events", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const options: TrackedFetchOptions = {
        sessionId: "sess-abc",
        traceId: "trace-123",
        tags: { env: "prod", team: "ai" },
      };
      const trackedFetch = buildTrackedFetch("openai", options, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.sessionId).toBe("sess-abc");
      expect(event.traceId).toBe("trace-123");
      expect(event.tags).toEqual({ env: "prod", team: "ai" });
    });
  });

  // -------------------------------------------------------------------------
  // Enforcement: mandates
  // -------------------------------------------------------------------------

  describe("enforcement — mandates", () => {
    it("throws MandateViolationError when model is denied", async () => {
      const policyCache = createMockPolicyCache({
        checkMandate: vi.fn().mockReturnValue({
          allowed: false,
          mandate: "allowed_models",
          requested: "gpt-4o",
          allowed_list: ["gpt-4o-mini"],
        }),
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(MandateViolationError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("invokes onDenied callback before throwing MandateViolationError", async () => {
      const policyCache = createMockPolicyCache({
        checkMandate: vi.fn().mockReturnValue({
          allowed: false,
          mandate: "allowed_models",
          requested: "gpt-4o",
          allowed_list: ["gpt-4o-mini"],
        }),
      });

      const onDenied = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(MandateViolationError);

      expect(onDenied).toHaveBeenCalledTimes(1);
      const reason: DenialReason = onDenied.mock.calls[0][0];
      expect(reason.type).toBe("mandate");
    });
  });

  // -------------------------------------------------------------------------
  // Enforcement: budget
  // -------------------------------------------------------------------------

  describe("enforcement — budget", () => {
    it("throws BudgetExceededError when budget is exceeded", async () => {
      const policyCache = createMockPolicyCache({
        checkBudget: vi.fn().mockReturnValue({ allowed: false, remaining: 50 }),
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(BudgetExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("invokes onDenied callback before throwing BudgetExceededError", async () => {
      const policyCache = createMockPolicyCache({
        checkBudget: vi.fn().mockReturnValue({ allowed: false, remaining: 25 }),
      });

      const onDenied = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(BudgetExceededError);

      expect(onDenied).toHaveBeenCalledTimes(1);
      const reason: DenialReason = onDenied.mock.calls[0][0];
      expect(reason.type).toBe("budget");
      if (reason.type === "budget") {
        expect(reason.remaining).toBe(25);
      }
    });
  });

  // -------------------------------------------------------------------------
  // Enforcement: fail-open
  // -------------------------------------------------------------------------

  describe("enforcement — fail-open on policy fetch failure", () => {
    it("proceeds with the request when getPolicy rejects", async () => {
      const policyCache = createMockPolicyCache({
        getPolicy: vi.fn().mockRejectedValue(new Error("network error")),
      });

      mockFetch.mockResolvedValue(openaiJsonResponse());
      const onCostError = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onCostError },
        queueCost,
        policyCache,
      );

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(onCostError).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // Enforcement disabled by default
  // -------------------------------------------------------------------------

  describe("enforcement off by default", () => {
    it("does not check policies when enforcement is not set", async () => {
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, policyCache);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(policyCache.getPolicy).not.toHaveBeenCalled();
      expect(policyCache.checkMandate).not.toHaveBeenCalled();
      expect(policyCache.checkBudget).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Request input types
  // -------------------------------------------------------------------------

  describe("Request input resolution", () => {
    it("handles URL object as input", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(new URL(OPENAI_URL), {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).toHaveBeenCalledTimes(1);
    });

    it("handles string URL as input", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(queueCost).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // Model resolution from response
  // -------------------------------------------------------------------------

  describe("model resolution", () => {
    it("uses model from response JSON over request body when available", async () => {
      // Request says gpt-4o but response says gpt-4o-2024-08-06
      mockFetch.mockResolvedValue(openaiJsonResponse("gpt-4o-2024-08-06"));
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody("gpt-4o"),
      });

      const event: CostEventInput = queueCost.mock.calls[0][0];
      expect(event.model).toBe("gpt-4o-2024-08-06");
    });
  });

  // -------------------------------------------------------------------------
  // No body edge case
  // -------------------------------------------------------------------------

  describe("edge cases", () => {
    it("defaults to 'unknown' model when body cannot be parsed", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      // POST with no body — method from init makes it tracked
      await trackedFetch(OPENAI_URL, { method: "POST" });

      // The model was unknown from the request, but resolved from response
      expect(queueCost).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // Edge-case: injectStreamUsage preserves init.signal
  // -------------------------------------------------------------------------

  it("preserves AbortSignal through stream_options injection", async () => {
    const controller = new AbortController();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetchStreamResponse(openaiStreamChunks()),
    );

    const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);
    const body = makeOpenAIBody("gpt-4o", true);
    await trackedFetch(OPENAI_URL, {
      method: "POST",
      body,
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
    });

    // The signal should be passed through to the real fetch call
    const calledInit = fetchSpy.mock.calls[0][1] as RequestInit;
    expect(calledInit.signal).toBe(controller.signal);
    fetchSpy.mockRestore();
  });

  // -------------------------------------------------------------------------
  // Edge-case: stream_options already has include_usage (no-op merge)
  // -------------------------------------------------------------------------

  it("does not break when stream_options already has include_usage", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetchStreamResponse(openaiStreamChunks()),
    );

    const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);
    const bodyObj = { model: "gpt-4o", stream: true, stream_options: { include_usage: true }, messages: [] };
    await trackedFetch(OPENAI_URL, {
      method: "POST",
      body: JSON.stringify(bodyObj),
      headers: { "Content-Type": "application/json" },
    });

    // Should still have include_usage: true (not duplicated or broken)
    const calledInit = fetchSpy.mock.calls[0][1] as RequestInit;
    const calledBody = JSON.parse(calledInit.body as string);
    expect(calledBody.stream_options.include_usage).toBe(true);
    fetchSpy.mockRestore();
  });

  // -------------------------------------------------------------------------
  // Edge-case: warns when body can't be extracted for a tracked route
  // -------------------------------------------------------------------------

  it("calls onCostError when body cannot be extracted for OpenAI route", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetchJsonResponse({
        model: "gpt-4o",
        usage: { prompt_tokens: 100, completion_tokens: 50 },
      }),
    );
    const errorHandler = vi.fn();

    const trackedFetch = buildTrackedFetch("openai", { onCostError: errorHandler }, queueCost, null);
    // Pass without body in init — body extraction returns null
    await trackedFetch(OPENAI_URL, { method: "POST" });

    expect(errorHandler).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Could not extract request body"),
      }),
    );
    fetchSpy.mockRestore();
  });

  // -------------------------------------------------------------------------
  // Enforcement: session limits
  // -------------------------------------------------------------------------

  describe("enforcement — session limits", () => {
    it("throws SessionLimitExceededError when spend + estimate > manual limit", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100 },
        queueCost,
        policyCache,
      );

      // The estimate for gpt-4o with default max_tokens is nonzero, exceeds 100 microdollars
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("calls onDenied with session_limit type before throwing", async () => {
      const policyCache = createMockPolicyCache();
      const onDenied = vi.fn();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(onDenied).toHaveBeenCalledTimes(1);
      const reason: DenialReason = onDenied.mock.calls[0][0];
      expect(reason.type).toBe("session_limit");
      if (reason.type === "session_limit") {
        expect(reason.sessionSpend).toBe(0);
        expect(reason.sessionLimit).toBe(100);
      }
    });

    it("does not enforce session limit when sessionId is absent", async () => {
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionLimitMicrodollars: 100 },
        queueCost,
        policyCache,
      );

      // Should succeed — no sessionId means no session enforcement
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
    });

    it("does not enforce session limit when enforcement is false", async () => {
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch(
        "openai",
        { sessionId: "sess-1", sessionLimitMicrodollars: 100 },
        queueCost,
        policyCache,
      );

      // Should succeed — enforcement is off
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
    });

    it("accumulates cost from non-streaming responses and denies Nth request", async () => {
      const policyCache = createMockPolicyCache();

      // Estimate: 1000 * 2.5 + 4096 * 10 = 43,460 microdollars (with mocked pricing)
      // Actual cost per request: 100 * 2.5 + 50 * 10 = 750 microdollars
      // Set limit so first request passes (0 + 43460 < 44000) but second fails
      // (750 + 43460 = 44210 > 44000)
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 44_000 },
        queueCost,
        policyCache,
      );

      // First request succeeds (0 + estimate < limit)
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const response1 = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });
      expect(response1.status).toBe(200);
      expect(queueCost).toHaveBeenCalledTimes(1);

      const cost1 = queueCost.mock.calls[0][0].costMicrodollars;
      expect(cost1).toBeGreaterThan(0);

      // Second request denied (cost1 + estimate > limit)
      mockFetch.mockResolvedValue(openaiJsonResponse());
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("accumulates cost from streaming responses and denies subsequent request", async () => {
      const policyCache = createMockPolicyCache();

      // Same math as non-streaming accumulation test:
      // Estimate ~43,460 microdollars, actual cost ~750 microdollars per request
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 44_000 },
        queueCost,
        policyCache,
      );

      // First streaming request succeeds (0 + estimate < 44000)
      mockFetch.mockResolvedValue(mockFetchStreamResponse(openaiStreamChunks()));
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody("gpt-4o", true),
      });
      await consumeStream(response);
      await new Promise((r) => setTimeout(r, 10));

      expect(queueCost).toHaveBeenCalledTimes(1);
      const cost = queueCost.mock.calls[0][0].costMicrodollars;
      expect(cost).toBeGreaterThan(0);

      // Second request denied (accumulated streaming cost + estimate > 44000)
      mockFetch.mockResolvedValue(openaiJsonResponse());
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("uses policy-fetched session limit when manual is not set", async () => {
      const policyCache = createMockPolicyCache({
        getSessionLimit: vi.fn().mockReturnValue(50), // 50 microdollars — very low
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1" },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("manual limit takes precedence over policy limit", async () => {
      const policyCache = createMockPolicyCache({
        getSessionLimit: vi.fn().mockReturnValue(100_000_000), // very high policy limit
      });
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 50 }, // very low manual
        queueCost,
        policyCache,
      );

      // Manual limit (50) takes precedence — should deny
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("falls open on policy failure but enforces manual session limit", async () => {
      const policyCache = createMockPolicyCache({
        getPolicy: vi.fn().mockRejectedValue(new Error("network error")),
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 50 },
        queueCost,
        policyCache,
      );

      // Policy failed but manual limit should still be enforced
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("session spend starts at 0 for new tracked fetch instance", async () => {
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      // High limit — should pass
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100_000_000 },
        queueCost,
        policyCache,
      );

      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(response.status).toBe(200);
    });

    it("SessionLimitExceededError is re-thrown (not swallowed by fall-open catch)", async () => {
      // Simulate a scenario where getPolicy succeeds but getSessionLimit returns a low value
      const policyCache = createMockPolicyCache({
        getSessionLimit: vi.fn().mockReturnValue(10),
      });

      const onCostError = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", onCostError },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      // Should NOT have called onCostError — the error is not swallowed
      expect(onCostError).not.toHaveBeenCalled();
    });

    it("failed response (4xx/5xx) does NOT accumulate session spend", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100_000 },
        queueCost,
        policyCache,
      );

      // First request returns 400
      mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: "bad request" }, 400));
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });
      expect(response.status).toBe(400);
      expect(queueCost).not.toHaveBeenCalled();

      // Second request should still have 0 session spend
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const response2 = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });
      expect(response2.status).toBe(200);
      expect(queueCost).toHaveBeenCalledTimes(1);
    });

    it("cost tracking error in response parsing does NOT accumulate session spend", async () => {
      const policyCache = createMockPolicyCache();
      const onCostError = vi.fn();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100_000, onCostError },
        queueCost,
        policyCache,
      );

      // Return a response that fails JSON parsing
      mockFetch.mockResolvedValue(new Response("not json", {
        status: 200,
        headers: { "content-type": "application/json" },
      }));
      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(onCostError).toHaveBeenCalledTimes(1);
      expect(queueCost).not.toHaveBeenCalled();

      // Next request should still succeed (spend is still 0)
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });
      expect(response.status).toBe(200);
    });

    it("enforces session limit for Anthropic provider", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "anthropic",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100 },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(ANTHROPIC_URL, { method: "POST", body: makeAnthropicBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("Anthropic streaming accumulates and denies subsequent request", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "anthropic",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 44_000 },
        queueCost,
        policyCache,
      );

      // First streaming request succeeds
      mockFetch.mockResolvedValue(mockFetchStreamResponse(anthropicStreamChunks()));
      const response = await trackedFetch(ANTHROPIC_URL, {
        method: "POST",
        body: makeAnthropicBody("claude-sonnet-4-20250514", true),
      });
      await consumeStream(response);
      await new Promise((r) => setTimeout(r, 10));

      expect(queueCost).toHaveBeenCalledTimes(1);

      // Second request denied (accumulated cost + estimate > limit)
      mockFetch.mockResolvedValue(anthropicJsonResponse());
      await expect(
        trackedFetch(ANTHROPIC_URL, { method: "POST", body: makeAnthropicBody() }),
      ).rejects.toThrow(SessionLimitExceededError);
    });

    it("sessionLimitMicrodollars: 0 blocks all requests (block-all)", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 0 },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("emits console.warn when session limit denies a request", async () => {
      const policyCache = createMockPolicyCache();
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100 },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("[nullspend] Session limit denied:"),
      );
      warnSpy.mockRestore();
    });

    it("emits console.warn when session limit denies in fallback path", async () => {
      const policyCache = createMockPolicyCache({
        getPolicy: vi.fn().mockRejectedValue(new Error("network error")),
      });
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 50 },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("[nullspend] Session limit denied (fallback):"),
      );
      warnSpy.mockRestore();
    });

    it("does not accumulate session spend when enforcement is off", async () => {
      // Verify that the costSink optimization works — queueCost is called directly
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch(
        "openai",
        { sessionId: "sess-1", sessionLimitMicrodollars: 100_000 },
        queueCost,
        policyCache,
      );

      await trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });
      expect(queueCost).toHaveBeenCalledTimes(1);
    });

    it("onDenied throwing does not bypass session limit enforcement", async () => {
      const policyCache = createMockPolicyCache();
      const onDenied = vi.fn().mockImplementation(() => {
        throw new Error("callback bug");
      });
      const onCostError = vi.fn();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 100, onDenied, onCostError },
        queueCost,
        policyCache,
      );

      // Should still throw SessionLimitExceededError even though onDenied throws
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      // onDenied was called (and threw)
      expect(onDenied).toHaveBeenCalledTimes(1);
      // onCostError received the callback's error
      expect(onCostError).toHaveBeenCalledTimes(1);
      expect(onCostError.mock.calls[0][0].message).toBe("callback bug");
      // fetch was never called
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("onDenied throwing in fallback path does not bypass session limit enforcement", async () => {
      const policyCache = createMockPolicyCache({
        getPolicy: vi.fn().mockRejectedValue(new Error("network error")),
      });
      const onDenied = vi.fn().mockImplementation(() => {
        throw new Error("callback bug");
      });
      const onCostError = vi.fn();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 50, onDenied, onCostError },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(onDenied).toHaveBeenCalledTimes(1);
    });

    it("onDenied throwing does not bypass budget enforcement", async () => {
      const policyCache = createMockPolicyCache({
        checkBudget: vi.fn().mockReturnValue({ allowed: false, remaining: 50 }),
      });
      const onDenied = vi.fn().mockImplementation(() => {
        throw new Error("callback bug");
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(BudgetExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("onDenied throwing does not bypass mandate enforcement", async () => {
      const policyCache = createMockPolicyCache({
        checkMandate: vi.fn().mockReturnValue({
          allowed: false,
          mandate: "allowed_models",
          requested: "gpt-4o",
          allowed_list: ["gpt-4o-mini"],
        }),
      });
      const onDenied = vi.fn().mockImplementation(() => {
        throw new Error("callback bug");
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(MandateViolationError);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("unknown model (estimate=0) still enforces session limit based on accumulated spend", async () => {
      // Mock getModelPricing to return null for unknown models
      const { getModelPricing } = await import("@nullspend/cost-engine");
      const mocked = vi.mocked(getModelPricing);
      const originalImpl = mocked.getMockImplementation();
      mocked.mockReturnValue(null); // unknown model → estimate=0

      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: 500 },
        queueCost,
        policyCache,
      );

      // First request passes (0 + 0 > 500 = false)
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody("unknown-model"),
      });
      expect(response.status).toBe(200);

      // Restore pricing so cost calculator returns real cost for the response
      // The response model is what matters for cost calculation, not the request model
      // But since getModelPricing is still mocked to null, costMicrodollars = 0
      // So session spend stays at 0. This means unknown models never trigger session limits
      // via accumulation — only if spend from OTHER requests exceeds the limit.
      expect(queueCost).toHaveBeenCalledTimes(1);
      // costMicrodollars is 0 because pricing is null
      expect(queueCost.mock.calls[0][0].costMicrodollars).toBe(0);

      // Restore original mock for other tests
      if (originalImpl) mocked.mockImplementation(originalImpl);
      else mocked.mockReturnValue({ inputPerMTok: 2.5, outputPerMTok: 10, cachedInputPerMTok: 1.25 } as any);
    });

    it("sessionLimitMicrodollars: Infinity acts as no-limit", async () => {
      const policyCache = createMockPolicyCache();
      mockFetch.mockResolvedValue(openaiJsonResponse());

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: Infinity },
        queueCost,
        policyCache,
      );

      // Should always pass — 0 + any_estimate > Infinity is always false
      const response = await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
      });
      expect(response.status).toBe(200);
    });

    it("sessionLimitMicrodollars: -1 blocks all requests", async () => {
      const policyCache = createMockPolicyCache();

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, sessionId: "sess-1", sessionLimitMicrodollars: -1 },
        queueCost,
        policyCache,
      );

      // 0 + any_positive_estimate > -1 is true → blocked
      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(SessionLimitExceededError);

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Enriched budget entity details
  // -------------------------------------------------------------------------

  describe("enriched budget entity details", () => {
    it("enriched BudgetExceededError includes entity details", async () => {
      const policyCache = createMockPolicyCache({
        checkBudget: vi.fn().mockReturnValue({
          allowed: false,
          remaining: 100,
          entityType: "api_key",
          entityId: "key-1",
          limit: 5_000_000,
          spend: 4_900_000,
        }),
      });

      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true },
        queueCost,
        policyCache,
      );

      try {
        await trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() });
        expect.unreachable("should have thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(BudgetExceededError);
        const budgetErr = err as InstanceType<typeof BudgetExceededError>;
        expect(budgetErr.entityType).toBe("api_key");
        expect(budgetErr.entityId).toBe("key-1");
        expect(budgetErr.limitMicrodollars).toBe(5_000_000);
        expect(budgetErr.spendMicrodollars).toBe(4_900_000);
      }

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("onDenied receives entity details in budget denial", async () => {
      const policyCache = createMockPolicyCache({
        checkBudget: vi.fn().mockReturnValue({
          allowed: false,
          remaining: 100,
          entityType: "api_key",
          entityId: "key-1",
          limit: 5_000_000,
          spend: 4_900_000,
        }),
      });

      const onDenied = vi.fn();
      const trackedFetch = buildTrackedFetch(
        "openai",
        { enforcement: true, onDenied },
        queueCost,
        policyCache,
      );

      await expect(
        trackedFetch(OPENAI_URL, { method: "POST", body: makeOpenAIBody() }),
      ).rejects.toThrow(BudgetExceededError);

      expect(onDenied).toHaveBeenCalledTimes(1);
      const reason: DenialReason = onDenied.mock.calls[0][0];
      expect(reason.type).toBe("budget");
      if (reason.type === "budget") {
        expect(reason.entityType).toBe("api_key");
        expect(reason.entityId).toBe("key-1");
        expect(reason.limit).toBe(5_000_000);
        expect(reason.spend).toBe(4_900_000);
      }
    });
  });

  // -------------------------------------------------------------------------
  // Proxy 429 interception
  // -------------------------------------------------------------------------

  describe("proxy 429 interception", () => {
    const PROXY_URL = "https://proxy.example.com";
    const PROXY_REQUEST_URL = "https://proxy.example.com/v1/chat/completions";

    // -----------------------------------------------------------------------
    // Sub-describe: via proxyUrl (the proxied path — exercises the actual fix
    // from §15c-1, where the SDK now intercepts proxy 429 denials in the
    // proxied path instead of bailing out before interception ran)
    // -----------------------------------------------------------------------

    describe("via proxyUrl (proxied path — the actual fix)", () => {
      it("proxy 429 with budget_exceeded code throws BudgetExceededError", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            message: "Budget exceeded",
            details: {
              entity_type: "api_key",
              entity_id: "key-1",
              budget_limit_microdollars: 5_000_000,
              budget_spend_microdollars: 4_900_000,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.remainingMicrodollars).toBe(100_000);
          expect(budgetErr.entityType).toBe("api_key");
          expect(budgetErr.entityId).toBe("key-1");
          expect(budgetErr.limitMicrodollars).toBe(5_000_000);
          expect(budgetErr.spendMicrodollars).toBe(4_900_000);
          // No upgrade_url in this denial body → undefined on the error
          expect(budgetErr.upgradeUrl).toBeUndefined();
        }
      });

      it("proxy 429 with budget_exceeded + upgrade_url surfaces upgradeUrl on the error", async () => {
        // Phase 0 finish, Item 2: parser reads error.upgrade_url at top level
        // (peer of code/message/details, NOT inside details).
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            message: "Budget exceeded",
            upgrade_url: "https://acme.com/upgrade",
            details: {
              entity_type: "api_key",
              entity_id: "key-1",
              budget_limit_microdollars: 5_000_000,
              budget_spend_microdollars: 4_900_000,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.upgradeUrl).toBe("https://acme.com/upgrade");
        }
      });

      it("proxy 429 with customer_budget_exceeded + upgrade_url surfaces upgradeUrl", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            message: "Request blocked: estimated cost exceeds customer budget limit.",
            upgrade_url: "https://acme.com/upgrade?customer=acme-corp",
            details: {
              customer_id: "acme-corp",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 999_500,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.upgradeUrl).toBe("https://acme.com/upgrade?customer=acme-corp");
        }
      });

      it("proxy 429 with customer_budget_exceeded code throws BudgetExceededError", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Proxy emits this code (not budget_exceeded) for customer-entity denials.
        // Details schema: { customer_id, budget_limit_microdollars, budget_spend_microdollars }
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            message: "Request blocked: estimated cost exceeds customer budget limit.",
            details: {
              customer_id: "acme-corp",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 999_500,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.entityType).toBe("customer");
          expect(budgetErr.entityId).toBe("acme-corp");
          expect(budgetErr.limitMicrodollars).toBe(1_000_000);
          expect(budgetErr.spendMicrodollars).toBe(999_500);
          expect(budgetErr.remainingMicrodollars).toBe(500);
        }
      });

      it("customer_budget_exceeded with null customer_id falls back to SDK-side customer", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Proxy's shared.ts:212 can emit customer_id as null, not just string
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            details: {
              customer_id: null, // proxy null case
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 999_500,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.entityType).toBe("customer");
          // Should fall back to the SDK-side customer, not remain null
          expect(budgetErr.entityId).toBe("acme-corp");
          expect(budgetErr.limitMicrodollars).toBe(1_000_000);
          expect(budgetErr.spendMicrodollars).toBe(999_500);
        }
      });

      it("customer_budget_exceeded with missing details produces undefined fields, not NaN", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Missing details object entirely
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: { code: "customer_budget_exceeded" },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.entityId).toBe("acme-corp"); // fallback
          expect(budgetErr.limitMicrodollars).toBeUndefined();
          expect(budgetErr.spendMicrodollars).toBeUndefined();
          expect(budgetErr.remainingMicrodollars).toBe(0); // max(0, 0-0)
          expect(Number.isNaN(budgetErr.remainingMicrodollars)).toBe(false);
        }
      });

      it("customer_budget_exceeded fires onDenied with type=budget and entityType=customer", async () => {
        const policyCache = createMockPolicyCache();
        const onDenied = vi.fn();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onDenied, customer: "acme-corp" },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            details: {
              customer_id: "acme-corp",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 1_000_000,
            },
          },
        }, 429, DENIED_HEADERS));

        await expect(
          trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() }),
        ).rejects.toThrow(BudgetExceededError);

        expect(onDenied).toHaveBeenCalledTimes(1);
        const reason = onDenied.mock.calls[0][0] as DenialReason;
        expect(reason.type).toBe("budget");
        if (reason.type === "budget") {
          expect(reason.entityType).toBe("customer");
          expect(reason.entityId).toBe("acme-corp");
          expect(reason.remaining).toBe(0);
        }
      });

      it("proxy 429 BudgetExceededError has computed remaining from limit - spend", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            message: "Request blocked",
            details: {
              entity_type: "api_key",
              entity_id: "key-1",
              budget_limit_microdollars: 5_000_000,
              budget_spend_microdollars: 4_800_000,
              estimated_cost_microdollars: 500_000,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          // remaining = limit - spend = 5_000_000 - 4_800_000 = 200_000, NOT 0
          expect(budgetErr.remainingMicrodollars).toBe(200_000);
          expect(budgetErr.entityType).toBe("api_key");
          expect(budgetErr.entityId).toBe("key-1");
          expect(budgetErr.limitMicrodollars).toBe(5_000_000);
          expect(budgetErr.spendMicrodollars).toBe(4_800_000);
        }
      });

      // Velocity — uses actual proxy response shape:
      // details: { limitMicrodollars, windowSeconds, currentMicrodollars }
      // Retry-After: HTTP header (not in JSON body)

      it("proxy 429 with velocity_exceeded reads Retry-After header and details", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Actual proxy response shape from shared.ts:101-117
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              code: "velocity_exceeded",
              message: "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.",
              details: {
                limitMicrodollars: 500_000,
                windowSeconds: 60,
                currentMicrodollars: 750_000,
              },
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "30",
              ...DENIED_HEADERS,
            },
          },
        ));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(VelocityExceededError);
          const velErr = err as InstanceType<typeof VelocityExceededError>;
          expect(velErr.retryAfterSeconds).toBe(30);
          expect(velErr.limitMicrodollars).toBe(500_000);
          expect(velErr.windowSeconds).toBe(60);
          expect(velErr.currentMicrodollars).toBe(750_000);
        }
      });

      it("proxy 429 velocity_exceeded fires onDenied with full velocity context", async () => {
        const policyCache = createMockPolicyCache();
        const denied: DenialReason[] = [];

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onDenied: (r) => denied.push(r) },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              code: "velocity_exceeded",
              message: "Spending too fast",
              details: {
                limitMicrodollars: 200_000,
                windowSeconds: 120,
                currentMicrodollars: 300_000,
              },
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "15",
              ...DENIED_HEADERS,
            },
          },
        ));

        await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() }).catch(() => {});

        expect(denied).toHaveLength(1);
        expect(denied[0]).toEqual({
          type: "velocity",
          retryAfterSeconds: 15,
          limit: 200_000,
          window: 120,
          current: 300_000,
        });
      });

      it("proxy 429 velocity_exceeded with null details still reads Retry-After", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Proxy sends details: null when velocityDetails is undefined
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              code: "velocity_exceeded",
              message: "Velocity limit exceeded",
              details: null,
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "60",
              ...DENIED_HEADERS,
            },
          },
        ));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(VelocityExceededError);
          const velErr = err as InstanceType<typeof VelocityExceededError>;
          expect(velErr.retryAfterSeconds).toBe(60);
          expect(velErr.limitMicrodollars).toBeUndefined();
          expect(velErr.windowSeconds).toBeUndefined();
          expect(velErr.currentMicrodollars).toBeUndefined();
        }
      });

      it("proxy 429 with session_limit_exceeded throws SessionLimitExceededError", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Actual proxy response shape from shared.ts:133-152
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "session_limit_exceeded",
            message: "Request blocked: session spend exceeds session limit. Start a new session.",
            details: {
              session_id: "task-042",
              session_spend_microdollars: 950_000,
              session_limit_microdollars: 1_000_000,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(SessionLimitExceededError);
          const sessErr = err as InstanceType<typeof SessionLimitExceededError>;
          expect(sessErr.sessionSpendMicrodollars).toBe(950_000);
          expect(sessErr.sessionLimitMicrodollars).toBe(1_000_000);
        }
      });

      // Tag budget — uses actual proxy response shape:
      // details: { tag_key, tag_value, budget_limit_microdollars, budget_spend_microdollars }
      // remaining is computed (limit - spend), NOT sent by proxy

      it("proxy 429 with tag_budget_exceeded computes remaining from limit - spend", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Actual proxy response shape from shared.ts:169-189
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "tag_budget_exceeded",
            message: "Request blocked: estimated cost exceeds tag budget limit.",
            details: {
              tag_key: "env",
              tag_value: "prod",
              budget_limit_microdollars: 5_000_000,
              budget_spend_microdollars: 4_800_000,
            },
          },
        }, 429, DENIED_HEADERS));

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(TagBudgetExceededError);
          const tagErr = err as InstanceType<typeof TagBudgetExceededError>;
          expect(tagErr.tagKey).toBe("env");
          expect(tagErr.tagValue).toBe("prod");
          expect(tagErr.limitMicrodollars).toBe(5_000_000);
          expect(tagErr.spendMicrodollars).toBe(4_800_000);
          // remaining = limit - spend = 5M - 4.8M = 200K
          expect(tagErr.remainingMicrodollars).toBe(200_000);
        }
      });

      it("proxy 429 tag_budget_exceeded fires onDenied with computed remaining and spend", async () => {
        const policyCache = createMockPolicyCache();
        const denied: DenialReason[] = [];

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onDenied: (r) => denied.push(r) },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Fully exhausted tag budget
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "tag_budget_exceeded",
            message: "Tag budget exceeded",
            details: {
              tag_key: "customer",
              tag_value: "acme",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 1_200_000,
            },
          },
        }, 429, DENIED_HEADERS));

        await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() }).catch(() => {});

        expect(denied).toHaveLength(1);
        expect(denied[0]).toEqual({
          type: "tag_budget",
          tagKey: "customer",
          tagValue: "acme",
          remaining: 0, // clamped: max(0, 1M - 1.2M) = 0
          limit: 1_000_000,
          spend: 1_200_000,
        });
      });

      it("proxy 429 with rate_limited code passes through silently (not a NullSpend denial)", async () => {
        const policyCache = createMockPolicyCache();
        const onCostError = vi.fn();
        const onDenied = vi.fn();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          policyCache,
          PROXY_URL,
        );

        // Proxy IP/key rate limit — different from budget denials
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              code: "rate_limited",
              message: "Too many requests",
              details: null,
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "60",
            },
          },
        ));

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // rate_limited is not a NullSpend denial — passes through silently.
        // Critical: must NOT fire onCostError (would pollute logs with every
        // upstream rate limit). The drift signal would be reintroduced once
        // the proxy adds an X-NullSpend-Denied header.
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });

      // -----------------------------------------------------------------------
      // New tests added with §15c-1 fix:
      //   - Test 1: proxied 200 → no cost tracking (no double-count guarantee)
      //   - Test 2: proxied 429 + enforcement: false → raw response (gate lock)
      //   - Test 3: customer header injection ordering under proxied path
      //   - Test 4: malformed JSON body → onCostError + raw response (Finding 2)
      //   - Test 5: error: null → silent fall-through (upstream 429 routed via proxy)
      //   - Test 6: Retry-After: 0 → retryAfterSeconds === 0 (Number.isFinite lock)
      //   - Test 7: unknown denial code → onCostError + raw response (drift signal)
      // -----------------------------------------------------------------------

      it("proxied 200 response does NOT track cost client-side (no double-count)", async () => {
        // Regression for the no-double-count guarantee. Without this, a future
        // refactor could move cost tracking inside the proxied branch and
        // double-count silently in the cost_event table.
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(200);
        expect(queueCost).not.toHaveBeenCalled();
      });

      it("proxied 429 with enforcement disabled returns raw response (no interception)", async () => {
        // Locks the enforcement gate. Without this, a future refactor could
        // flip the conditional and the test suite wouldn't catch it. Asserts
        // ALL three observable side effects of interception are absent: no
        // throw (response is returned), no onDenied fire, no cost track.
        const onDenied = vi.fn();
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            details: { entity_type: "api_key", budget_limit_microdollars: 100, budget_spend_microdollars: 100 },
          },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: false, onDenied, onCostError }, // ← gate is OFF
          queueCost,
          null,
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Raw 429 returned, no exception thrown, no callbacks fired
        expect(response.status).toBe(429);
        expect(queueCost).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
        expect(onCostError).not.toHaveBeenCalled();
      });

      it("proxied path injects X-NullSpend-Customer header before fetch (string URL + init)", async () => {
        // Regression for the customer header injection at tracked-fetch.ts:84-92.
        // Cheap insurance against a future refactor that moves the injection
        // below the isProxied check. Exercises the addHeader(init) branch
        // (line 90) — input is a string, init has no Request semantics.
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          { customer: "acme-corp" },
          queueCost,
          null,
          PROXY_URL,
        );

        await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Inspect the captured init from mockFetch
        const capturedInit = mockFetch.mock.calls[0][1] as RequestInit | undefined;
        const headers = new Headers(capturedInit?.headers as HeadersInit);
        expect(headers.get("X-NullSpend-Customer")).toBe("acme-corp");
      });

      it("proxied path injects X-NullSpend-Customer header when input is a Request without init.headers (WHATWG branch)", async () => {
        // Regression for the OTHER customer-injection branch at
        // tracked-fetch.ts:85-88 — when input is a Request and the caller
        // didn't provide init.headers, we MUST clone the Request and inject
        // into its headers (not synthesize an init.headers, because that
        // would replace the entire Request headers per WHATWG fetch spec).
        // The previous test only exercised the addHeader(init) branch.
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          { customer: "acme-corp" },
          queueCost,
          null,
          PROXY_URL,
        );

        const request = new Request(PROXY_REQUEST_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json", "Authorization": "Bearer sk-test" },
          body: makeOpenAIBody(),
        });

        await trackedFetch(request);

        // mockFetch should receive a Request (the cloned one) with the
        // customer header injected AND the original Authorization preserved.
        const capturedInput = mockFetch.mock.calls[0][0];
        expect(capturedInput).toBeInstanceOf(Request);
        const capturedRequest = capturedInput as Request;
        expect(capturedRequest.headers.get("X-NullSpend-Customer")).toBe("acme-corp");
        expect(capturedRequest.headers.get("Authorization")).toBe("Bearer sk-test");
        expect(capturedRequest.headers.get("Content-Type")).toBe("application/json");
      });

      it("proxied 429 with malformed JSON body passes through silently", async () => {
        // Parse failure → fall through to raw response WITHOUT firing onCostError.
        // Surfacing parse failures was reverted (audit Bug 1) — would fire on
        // every upstream provider 429 with a non-JSON or truncated body.
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(new Response("not valid json{", {
          status: 429,
          headers: { "Content-Type": "application/json" },
        }));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(429); // raw response
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });

      it("proxied 429 with error: null falls through silently (upstream 429 routed via proxy)", async () => {
        // null error is the "upstream provider 429 routed through proxy" case
        // and should NOT fire onCostError (silent fall-through).
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: null }, 429));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled(); // null error is silent
      });

      it("velocity_exceeded with Retry-After: 0 sets retryAfterSeconds to 0 (not undefined)", async () => {
        // Locks the Number.isFinite change vs the original `|| undefined` pattern.
        // The old code treated 0 as falsy and returned undefined; the new helper
        // distinguishes 0 from undefined.
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              code: "velocity_exceeded",
              details: { limitMicrodollars: 1000, windowSeconds: 60, currentMicrodollars: 1000 },
            },
          }),
          { status: 429, headers: { "Content-Type": "application/json", "Retry-After": "0", ...DENIED_HEADERS } },
        ));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(VelocityExceededError);
          expect((err as InstanceType<typeof VelocityExceededError>).retryAfterSeconds).toBe(0);
        }
      });

      it("proxied 429 with X-NullSpend-Denied + unknown code surfaces drift signal via onCostError", async () => {
        // With the X-NullSpend-Denied header gate in place, an unknown denial
        // code is no longer ambiguous — it can ONLY be a real proxy/SDK contract
        // drift (e.g., proxy added a new code the SDK hasn't shipped support for
        // yet). Surface it via onCostError as a drift signal, but still return
        // the raw response so the caller's existing 429 handling runs.
        // Critical: do NOT fire onDenied or throw — we don't know what type
        // of denial it is, so we have nothing meaningful to dispatch.
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "future_denial_code_we_dont_handle",
            details: { foo: "bar" },
          },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(429); // raw response, not thrown
        expect(onDenied).not.toHaveBeenCalled(); // can't classify, don't dispatch
        expect(onCostError).toHaveBeenCalledTimes(1);
        const errArg = onCostError.mock.calls[0][0] as Error;
        expect(errArg).toBeInstanceOf(Error);
        expect(errArg.message).toContain("future_denial_code_we_dont_handle");
      });

      it("proxied 429 with valid NullSpend code body but MISSING X-NullSpend-Denied header falls through silently", async () => {
        // The header is the gate. Without it, even a perfectly-shaped
        // NullSpend denial body must be treated as an upstream rate limit
        // and fall through to raw response. This locks the contract:
        // body shape alone is NEVER sufficient.
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            details: {
              entity_type: "api_key",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 1_000_000,
            },
          },
        }, 429)); // ← no DENIED_HEADERS

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Raw 429 returned, no exception, no callbacks fired
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });

      // -----------------------------------------------------------------------
      // Regression tests for audit Bug 1: upstream provider 429s routed through
      // the proxy must pass through SILENTLY. The proxy preserves OpenAI's
      // `error.code` and Anthropic's body shape (no `error.code`), so both
      // hit the parse → unknown-or-no-code paths in the helper. Without these
      // tests in place, future "drift signal" additions could regress and
      // pollute production logs with every upstream rate limit.
      // -----------------------------------------------------------------------

      it("OpenAI-shape upstream rate limit (error.code = rate_limit_exceeded) passes through silently", async () => {
        // Real OpenAI 429 body shape after passing through the proxy's
        // sanitizeUpstreamError (which preserves error.code if string).
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              type: "tokens",
              message: "Rate limit reached for gpt-4o in organization org-xxx",
              code: "rate_limit_exceeded",
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "60",
            },
          },
        ));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Caller gets the raw 429 to handle as they would for any rate limit.
        // Critical: zero noise — no onCostError, no onDenied, no exception.
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });

      it("Anthropic-shape upstream rate limit (error.type only, no error.code) passes through silently", async () => {
        // Real Anthropic 429 body shape after passing through the proxy's
        // sanitizeUpstreamError. Anthropic uses error.type, never error.code.
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({
            error: {
              type: "rate_limit_error",
              message: "Number of request tokens has exceeded your per-minute rate limit",
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "30",
            },
          },
        ));

        const trackedFetch = buildTrackedFetch(
          "anthropic",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Caller gets the raw 429. Critical: zero noise.
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });
    });

    // -----------------------------------------------------------------------
    // Sub-describe: via x-nullspend-key header (proxied path — header-based
    // detection). Proves that interception fires for both proxy detection
    // mechanisms (URL match and header match), not just one.
    // -----------------------------------------------------------------------

    describe("via x-nullspend-key header (proxied path — header-based detection)", () => {
      it("header-based proxy detection (x-nullspend-key) triggers 429 interception", async () => {
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            details: {
              customer_id: "acme-corp",
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 1_000_000,
            },
          },
        }, 429, DENIED_HEADERS));

        // proxyUrl configured — header fallback detects as proxied
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          createMockPolicyCache(),
          "https://proxy.example.com",
        );

        await expect(
          trackedFetch(OPENAI_URL, {
            method: "POST",
            body: makeOpenAIBody(),
            headers: { "x-nullspend-key": "ns_live_sk_test" }, // ← header-based detection
          }),
        ).rejects.toBeInstanceOf(BudgetExceededError);
      });
    });

    // -----------------------------------------------------------------------
    // Sub-describe: edge cases from the post-implementation audit. These
    // cover Risks 1, 2, 4, 5, 6, 8 from the edge-case audit + remaining test
    // gaps (falsy error fields, non-string code variants, streaming proxied
    // responses, Retry-After parsing edges, response.bodyUsed guarantees,
    // numeric customer_id fallback, defensive callback handling).
    // -----------------------------------------------------------------------

    describe("edge cases (audit follow-ups)", () => {
      // ── Risk 1: Request with x-nullspend-key in Request.headers ──
      // Updated: header-only detection requires proxyUrl to be configured.
      // Without proxyUrl, x-nullspend-key alone does NOT bypass tracking (Codex finding #1).
      it("Request with x-nullspend-key in Request.headers + proxyUrl is detected as proxied", async () => {
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          {},
          queueCost,
          null,
          "https://proxy.example.com", // proxyUrl required for header fallback
        );

        const request = new Request(OPENAI_URL, {
          method: "POST",
          headers: { "x-nullspend-key": "ns_live_sk_test", "Content-Type": "application/json" },
          body: makeOpenAIBody(),
        });

        await trackedFetch(request);

        // With proxyUrl configured, the header fallback detects the request as proxied
        expect(queueCost).not.toHaveBeenCalled();
      });

      it("Request with x-nullspend-key + customer header injection wraps but still detects as proxied", async () => {
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          { customer: "acme-corp" },
          queueCost,
          null,
          "https://proxy.example.com", // proxyUrl required for header fallback
        );

        const request = new Request(OPENAI_URL, {
          method: "POST",
          headers: { "x-nullspend-key": "ns_live_sk_test", "Content-Type": "application/json" },
          body: makeOpenAIBody(),
        });

        await trackedFetch(request);

        // Captured by mockFetch should be the WRAPPED Request with both headers.
        const capturedInput = mockFetch.mock.calls[0][0] as Request;
        expect(capturedInput.headers.get("x-nullspend-key")).toBe("ns_live_sk_test");
        expect(capturedInput.headers.get("X-NullSpend-Customer")).toBe("acme-corp");
        // Proxied path → no cost tracking
        expect(queueCost).not.toHaveBeenCalled();
      });

      // ── Risk 2: port mismatch in proxyUrl ──
      it("does NOT match a different port even when host matches (origin includes port)", async () => {
        // Risk 2: origin comparison is strict on port. proxyUrl without explicit
        // port doesn't match a request URL with an explicit port. This test
        // LOCKS the current strict behavior — port-normalization would mask
        // real misconfigurations. The remediation is documentation (see types.ts
        // proxyUrl JSDoc), not code change.
        mockFetch.mockResolvedValue(openaiJsonResponse());
        const trackedFetch = buildTrackedFetch(
          "openai",
          {},
          queueCost,
          null,
          "https://proxy.example.com", // ← no explicit port
        );

        await trackedFetch("https://proxy.example.com:8443/v1/chat/completions", {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Origins differ → SDK takes direct path → cost tracking runs.
        // This is the expected (and documented) failure mode for misconfigured
        // proxyUrl. Users with custom-port proxies must include the port.
        expect(queueCost).toHaveBeenCalledTimes(1);
      });

      // ── Falsy error field variants (Risk 8 narrowing) ──
      it("proxied 429 with error: 0 (falsy non-null) passes through silently", async () => {
        const onCostError = vi.fn();
        const onDenied = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: 0 }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
        expect(onDenied).not.toHaveBeenCalled();
      });

      it("proxied 429 with error: false passes through silently", async () => {
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: false }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
      });

      it("proxied 429 with error: \"\" (empty string) passes through silently", async () => {
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: "" }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
      });

      it("proxied 429 with error: [] (array) passes through silently (Risk 8 runtime narrowing)", async () => {
        // The original cast `errObj as Record<string, unknown>` would have
        // accepted an array; the runtime narrowing rejects it.
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: [] }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
      });

      // ── Non-string error.code variants ──
      it("proxied 429 with error.code: 0 (numeric) passes through silently", async () => {
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: { code: 0 } }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
      });

      it("proxied 429 with error.code: false passes through silently", async () => {
        const onCostError = vi.fn();
        mockFetch.mockResolvedValue(mockFetchJsonResponse({ error: { code: false } }, 429));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(onCostError).not.toHaveBeenCalled();
      });

      // ── Streaming 200 in proxied path ──
      it("streaming 200 response in proxied path passes through untouched (no cost tracking)", async () => {
        // The proxied branch returns the response untouched for any non-429.
        // Streaming responses should NOT be parsed by the SDK.
        const sseBody = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(sseBody));
            controller.close();
          },
        });
        mockFetch.mockResolvedValue(new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        const response = await trackedFetch(PROXY_REQUEST_URL, {
          method: "POST",
          body: JSON.stringify({ model: "gpt-4o-mini", stream: true, messages: [] }),
        });

        expect(response.status).toBe(200);
        // Body should still be readable (proxied path doesn't consume it)
        expect(response.bodyUsed).toBe(false);
        // No cost tracking — proxy handles it
        expect(queueCost).not.toHaveBeenCalled();
      });

      // ── Retry-After parsing edge cases ──
      it("velocity_exceeded with negative Retry-After: -5 produces undefined retryAfterSeconds", async () => {
        // RFC 7231: Retry-After is a non-negative integer. The new helper
        // rejects negatives defensively (audit Bug 2 fix).
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({ error: { code: "velocity_exceeded", details: { limitMicrodollars: 1000 } } }),
          { status: 429, headers: { "Content-Type": "application/json", "Retry-After": "-5", ...DENIED_HEADERS } },
        ));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(VelocityExceededError);
          expect((err as InstanceType<typeof VelocityExceededError>).retryAfterSeconds).toBeUndefined();
        }
      });

      it("velocity_exceeded with non-numeric Retry-After: \"abc\" produces undefined retryAfterSeconds", async () => {
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({ error: { code: "velocity_exceeded", details: { limitMicrodollars: 1000 } } }),
          { status: 429, headers: { "Content-Type": "application/json", "Retry-After": "abc", ...DENIED_HEADERS } },
        ));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(VelocityExceededError);
          expect((err as InstanceType<typeof VelocityExceededError>).retryAfterSeconds).toBeUndefined();
        }
      });

      // ── response.bodyUsed guarantee on proxied 429 fall-through ──
      it("proxied 429 with non-NullSpend body returns response with bodyUsed === false", async () => {
        // The clone-and-parse pattern in parseDenialPayload must NOT consume
        // the original response body. Callers expect to be able to read it.
        mockFetch.mockResolvedValue(new Response(
          JSON.stringify({ error: { type: "rate_limit_error", message: "..." } }),
          { status: 429, headers: { "Content-Type": "application/json" } },
        ));
        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );
        const response = await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
        expect(response.status).toBe(429);
        expect(response.bodyUsed).toBe(false);
        // User can still read the body
        const body = await response.json() as { error?: { type?: string } };
        expect(body.error?.type).toBe("rate_limit_error");
      });

      // ── Numeric customer_id falls back to closure ──
      it("customer_budget_exceeded with numeric customer_id (42) falls back to SDK-side customer", async () => {
        // typeof 42 === "number", not "string", so the fallback fires.
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "customer_budget_exceeded",
            details: {
              customer_id: 42, // ← numeric, not string
              budget_limit_microdollars: 1_000_000,
              budget_spend_microdollars: 999_500,
            },
          },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          // Should fall back to the SDK-side customer, not coerce 42 to "42"
          expect(budgetErr.entityId).toBe("acme-corp");
        }
      });

      // ── customer_budget_exceeded with details: null (Risk 8 narrowing) ──
      it("customer_budget_exceeded with details: null narrows safely (no NaN, no crash)", async () => {
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: { code: "customer_budget_exceeded", details: null },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, customer: "acme-corp" },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        try {
          await trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() });
          expect.unreachable("should have thrown");
        } catch (err) {
          expect(err).toBeInstanceOf(BudgetExceededError);
          const budgetErr = err as InstanceType<typeof BudgetExceededError>;
          expect(budgetErr.entityId).toBe("acme-corp"); // closure fallback
          expect(budgetErr.limitMicrodollars).toBeUndefined();
          expect(budgetErr.spendMicrodollars).toBeUndefined();
          expect(budgetErr.remainingMicrodollars).toBe(0); // max(0, 0-0)
          expect(Number.isNaN(budgetErr.remainingMicrodollars)).toBe(false);
        }
      });

      // ── Risk 4: safeDenied swallows onCostError throws ──
      it("buggy onCostError that throws does NOT prevent typed error from being thrown", async () => {
        // Risk 4: if user's onDenied throws AND their onCostError throws too,
        // safeDenied must swallow both so the typed error still propagates.
        const onCostError = vi.fn(() => { throw new Error("logger broken"); });
        const onDenied = vi.fn(() => { throw new Error("denied handler broken"); });
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            details: { entity_type: "api_key", budget_limit_microdollars: 100, budget_spend_microdollars: 100 },
          },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        await expect(
          trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() }),
        ).rejects.toBeInstanceOf(BudgetExceededError);

        // Both callbacks were invoked even though both throw
        expect(onDenied).toHaveBeenCalledTimes(1);
        expect(onCostError).toHaveBeenCalledTimes(1);
      });

      // ── Risk 5: async onDenied that rejects doesn't surface as unhandled rejection ──
      it("async onDenied that rejects is caught by safeDenied (no unhandled rejection)", async () => {
        // Risk 5: TS allows async onDenied even though the type is `void`.
        // safeDenied attaches a no-op catch so the rejection becomes a
        // routed onCostError call instead of an unhandled promise rejection.
        const onCostError = vi.fn();
        const onDenied = vi.fn(async () => {
          await Promise.resolve(); // ensure async boundary
          throw new Error("async denied handler failed");
        });
        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            details: { entity_type: "api_key", budget_limit_microdollars: 100, budget_spend_microdollars: 100 },
          },
        }, 429, DENIED_HEADERS));

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true, onCostError, onDenied },
          queueCost,
          createMockPolicyCache(),
          PROXY_URL,
        );

        await expect(
          trackedFetch(PROXY_REQUEST_URL, { method: "POST", body: makeOpenAIBody() }),
        ).rejects.toBeInstanceOf(BudgetExceededError);

        // Wait a microtask for the async onDenied to settle so the catch fires.
        await new Promise(resolve => setTimeout(resolve, 0));

        // The async rejection should be routed to onCostError, not unhandled.
        expect(onCostError).toHaveBeenCalledTimes(1);
        expect(onCostError.mock.calls[0][0].message).toBe("async denied handler failed");
      });
    });

    // -----------------------------------------------------------------------
    // Sub-describe: direct mode — NO 429 interception. In direct mode there
    // is no NullSpend proxy to send denial codes. Trusting X-NullSpend-Denied
    // from an arbitrary LLM provider would be a trust-boundary violation
    // (Codex finding #6). All 429 interception is now proxy-only.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Finding #2: custom fetch passthrough
    // -----------------------------------------------------------------------

    describe("custom fetch passthrough (Codex finding #2)", () => {
      it("uses fetchFn instead of globalThis.fetch for tracked requests", async () => {
        const customFetch = vi.fn().mockResolvedValue(openaiJsonResponse());

        const trackedFetch = buildTrackedFetch(
          "openai",
          undefined,
          queueCost,
          null,
          undefined,
          customFetch,
        );

        await trackedFetch(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        // Custom fetch should have been called, not globalThis.fetch
        expect(customFetch).toHaveBeenCalledTimes(1);
        expect(mockFetch).not.toHaveBeenCalled();
      });

      it("uses fetchFn for non-tracked routes", async () => {
        const customFetch = vi.fn().mockResolvedValue(new Response("ok"));

        const trackedFetch = buildTrackedFetch(
          "openai",
          undefined,
          queueCost,
          null,
          undefined,
          customFetch,
        );

        await trackedFetch("https://api.openai.com/v1/models", { method: "GET" });

        expect(customFetch).toHaveBeenCalledTimes(1);
        expect(mockFetch).not.toHaveBeenCalled();
      });

      it("uses fetchFn for proxied requests", async () => {
        const customFetch = vi.fn().mockResolvedValue(openaiJsonResponse());
        const PROXY = "https://proxy.example.com";

        const trackedFetch = buildTrackedFetch(
          "openai",
          undefined,
          queueCost,
          null,
          PROXY,
          customFetch,
        );

        await trackedFetch(`${PROXY}/v1/chat/completions`, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(customFetch).toHaveBeenCalledTimes(1);
        expect(mockFetch).not.toHaveBeenCalled();
      });

      it("falls back to globalThis.fetch when fetchFn is not provided", async () => {
        mockFetch.mockResolvedValue(openaiJsonResponse());

        const trackedFetch = buildTrackedFetch(
          "openai",
          undefined,
          queueCost,
          null,
        );

        await trackedFetch(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(mockFetch).toHaveBeenCalledTimes(1);
      });
    });

    // -----------------------------------------------------------------------
    // Finding #3: Request object body parsing
    // -----------------------------------------------------------------------

    describe("Request object body parsing (Codex finding #3)", () => {
      it("extracts model from Request object body", async () => {
        mockFetch.mockResolvedValue(openaiJsonResponse("gpt-4o-mini"));

        const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

        const request = new Request(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody("gpt-4o-mini"),
          headers: { "Content-Type": "application/json" },
        });

        await trackedFetch(request);

        expect(queueCost).toHaveBeenCalledTimes(1);
        const event: CostEventInput = queueCost.mock.calls[0][0];
        // Model should be extracted from the request body, not "unknown"
        expect(event.model).toBe("gpt-4o-mini");
      });

      it("detects streaming from Request object body", async () => {
        mockFetch.mockResolvedValue(
          mockFetchStreamResponse(openaiStreamChunks()),
        );

        const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

        const request = new Request(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody("gpt-4o", true),
          headers: { "Content-Type": "application/json" },
        });

        const response = await trackedFetch(request);
        await consumeStream(response);

        expect(queueCost).toHaveBeenCalledTimes(1);
        const event: CostEventInput = queueCost.mock.calls[0][0];
        expect(event.model).toBe("gpt-4o");
      });

      it("prefers init.body over Request body when both provided", async () => {
        mockFetch.mockResolvedValue(openaiJsonResponse("gpt-4o-mini"));

        const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

        const request = new Request(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody("gpt-4o"),
          headers: { "Content-Type": "application/json" },
        });

        // init.body should take precedence
        await trackedFetch(request, {
          body: makeOpenAIBody("gpt-4o-mini"),
        });

        expect(queueCost).toHaveBeenCalledTimes(1);
        const event: CostEventInput = queueCost.mock.calls[0][0];
        expect(event.model).toBe("gpt-4o-mini");
      });
    });

    describe("direct mode (no interception — Codex finding #6)", () => {
      it("upstream provider 429 passes through without throwing", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            message: "Rate limit exceeded",
            type: "rate_limit_error",
          },
        }, 429));

        const response = await trackedFetch(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(429);
      });

      it("NullSpend-shaped 429 with X-NullSpend-Denied header does NOT throw in direct mode", async () => {
        // Regression: a malicious origin could stamp X-NullSpend-Denied: 1 and
        // budget_exceeded code. In direct mode the SDK must NOT trust it.
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
          // NO proxyUrl — direct mode
        );

        mockFetch.mockResolvedValue(mockFetchJsonResponse({
          error: {
            code: "budget_exceeded",
            message: "Budget exceeded",
            details: {
              entity_type: "api_key",
              entity_id: "key-1",
              budget_limit_microdollars: 5_000_000,
              budget_spend_microdollars: 4_900_000,
            },
          },
        }, 429, DENIED_HEADERS));

        // Must pass through (not throw) — the SDK has no proxy to trust
        const response = await trackedFetch(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });
        expect(response.status).toBe(429);
      });

      it("non-JSON 429 passes through", async () => {
        const policyCache = createMockPolicyCache();

        const trackedFetch = buildTrackedFetch(
          "openai",
          { enforcement: true },
          queueCost,
          policyCache,
        );

        mockFetch.mockResolvedValue(new Response("Too Many Requests", {
          status: 429,
          statusText: "Too Many Requests",
          headers: { "content-type": "text/plain" },
        }));

        const response = await trackedFetch(OPENAI_URL, {
          method: "POST",
          body: makeOpenAIBody(),
        });

        expect(response.status).toBe(429);
      });
    });
  });
});
