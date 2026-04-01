import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { buildTrackedFetch } from "./tracked-fetch.js";
import { BudgetExceededError, MandateViolationError } from "./errors.js";
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

function mockFetchJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 ? "OK" : `Status ${status}`,
    headers: { "content-type": "application/json" },
  });
}

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
    it("passes through when URL contains proxy.nullspend.com", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch("https://proxy.nullspend.com/v1/chat/completions", {
        method: "POST",
        body: makeOpenAIBody(),
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through when x-nullspend-key header is present (Headers object)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      const headers = new Headers({ "x-nullspend-key": "ns_live_sk_test" });
      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers,
      });

      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through when x-nullspend-key header is present (plain object)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: { "x-nullspend-key": "ns_live_sk_test" },
      });

      expect(queueCost).not.toHaveBeenCalled();
    });

    it("passes through when x-nullspend-key header is present (array tuples)", async () => {
      mockFetch.mockResolvedValue(openaiJsonResponse());
      const trackedFetch = buildTrackedFetch("openai", undefined, queueCost, null);

      await trackedFetch(OPENAI_URL, {
        method: "POST",
        body: makeOpenAIBody(),
        headers: [["x-nullspend-key", "ns_live_sk_test"]],
      });

      expect(queueCost).not.toHaveBeenCalled();
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
});
