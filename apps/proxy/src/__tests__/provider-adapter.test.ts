/**
 * Unit tests for the provider adapter pattern (P2-1).
 *
 * Tests the adapter objects and generic handler behavior that was
 * NOT covered by the existing route-level regression tests.
 * The 1,434 existing tests cover end-to-end behavior; these tests
 * verify the adapter contract and new error paths.
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll } from "vitest";
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

// Mock cost-engine to avoid loading pricing data
vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn().mockReturnValue(null),
  costComponent: vi.fn().mockReturnValue(0),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/budget-orchestrator.js", () => ({
  checkBudget: vi.fn().mockResolvedValue({ status: "skipped", reservationId: null, budgetEntities: [] }),
  reconcileBudget: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: vi.fn(),
}));

vi.mock("../lib/write-metric.js", () => ({
  writeLatencyDataPoint: vi.fn(),
}));

vi.mock("../lib/body-storage.js", () => ({
  storeRequestBody: vi.fn().mockResolvedValue(undefined),
  storeResponseBody: vi.fn().mockResolvedValue(undefined),
  storeStreamingResponseBody: vi.fn().mockResolvedValue(undefined),
  createStreamBodyAccumulator: vi.fn(),
}));

vi.mock("../routes/shared.js", () => ({
  handleBudgetDenials: vi.fn().mockResolvedValue(null),
  dispatchVelocityRecoveryWebhooks: vi.fn(),
  dispatchCostEventWebhooks: vi.fn().mockResolvedValue(undefined),
  buildBudgetHeaders: vi.fn().mockReturnValue({}),
}));

vi.mock("../lib/sanitize-upstream-error.js", () => ({
  sanitizeUpstreamError: vi.fn().mockResolvedValue('{"error":"upstream error"}'),
}));

vi.mock("../lib/env.js", () => ({
  optionalBinding: vi.fn().mockReturnValue(undefined),
}));

import { handleProviderRequest } from "../routes/provider-handler.js";
import { openaiChatAdapter } from "../providers/openai.js";
import { anthropicChatAdapter } from "../providers/anthropic.js";
import type { ProviderAdapter, StreamResult, ParsedResponse } from "../lib/provider-types.js";
import { emitMetric } from "../lib/metrics.js";

function makeCtx(overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    body: { model: "gpt-4o", messages: [{ role: "user", content: "hi" }] },
    bodyText: '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}',
    bodyByteLength: 60,
    auth: {
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: "org-1",
      apiVersion: "2026-04-01",
      defaultTags: {},
      requestLoggingEnabled: false,
      allowedModels: null,
      allowedProviders: null,
      allowedCustomers: null,
      requireCustomerId: false,
      orgUpgradeUrl: null,
    },
    ownerId: "org-1",
    connectionString: "postgres://test",
    skipDbWrites: false,
    sessionId: null,
    traceId: "trace-1",
    tags: {},
    customerId: null,
    customerWarning: null,
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    requestLoggingEnabled: false,
    finalize: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Adapter-throws-502 test
// ---------------------------------------------------------------------------

describe("handleProviderRequest — adapter exception guard", () => {
  it("returns structured 502 when adapter.estimateMaxCost throws", async () => {
    const brokenAdapter: ProviderAdapter = {
      ...openaiChatAdapter,
      estimateMaxCost: () => { throw new Error("adapter bug"); },
    };

    const request = new Request("https://proxy.nullspend.dev/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
    });

    const response = await handleProviderRequest(
      request,
      {} as Env,
      makeCtx(),
      brokenAdapter,
    );

    expect(response.status).toBe(502);
    const body = await response.json() as { error: { code: string } };
    expect(body.error.code).toBe("internal_error");
    expect(vi.mocked(emitMetric)).toHaveBeenCalledWith("adapter_error", expect.objectContaining({ provider: "openai" }));
  });

  it("returns structured 502 when adapter.prepareBody throws", async () => {
    const brokenAdapter: ProviderAdapter = {
      ...openaiChatAdapter,
      prepareBody: () => { throw new Error("prepareBody bug"); },
    };

    const request = new Request("https://proxy.nullspend.dev/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
    });

    const response = await handleProviderRequest(
      request,
      {} as Env,
      makeCtx({ body: { model: "gpt-4o", stream: true, messages: [] } }),
      brokenAdapter,
    );

    expect(response.status).toBe(502);
  });
});

// ---------------------------------------------------------------------------
// OpenAI adapter calculateCost wrapper
// ---------------------------------------------------------------------------

describe("openaiChatAdapter", () => {
  it("calculateCost passes usage to calculateOpenAICost and returns CostEventInsert", () => {
    const mockUsage = { prompt_tokens: 100, completion_tokens: 50 };
    const parsed: ParsedResponse = {
      hasUsage: true,
      usage: mockUsage,
      model: "gpt-4o",
      stopReason: "stop",
      toolCalls: null,
    };

    const result = openaiChatAdapter.calculateCost(
      "gpt-4o", "gpt-4o", parsed, "req-1", 200,
      { userId: "u1", apiKeyId: "k1", actionId: null },
      0,
    );

    // calculateOpenAICost returns a CostEventInsert with the provider set
    expect(result.provider).toBe("openai");
    expect(result.requestId).toBe("req-1");
    expect(result.eventType).toBe("llm");
  });

  it("parseNonStreamingResponse maps finish_reason to stopReason", () => {
    const parsed = {
      model: "gpt-4o",
      usage: { prompt_tokens: 10, completion_tokens: 5 },
      choices: [{ finish_reason: "stop", message: { tool_calls: [{ id: "tc-1", function: { name: "search" } }] } }],
    };

    const result = openaiChatAdapter.parseNonStreamingResponse(parsed);

    expect(result.stopReason).toBe("stop");
    expect(result.model).toBe("gpt-4o");
    expect(result.hasUsage).toBe(true);
    expect(result.toolCalls).toEqual([{ name: "search", id: "tc-1" }]);
  });

  it("extractResponseTags uses max_completion_tokens over max_tokens", () => {
    const response = new Response("ok", { headers: {} });
    const body = { max_completion_tokens: 4096, max_tokens: 2048 };
    const tags = openaiChatAdapter.extractResponseTags(response, body, false);
    expect(tags._ns_max_tokens).toBe("4096");
  });
});

// ---------------------------------------------------------------------------
// Anthropic adapter calculateCost wrapper
// ---------------------------------------------------------------------------

describe("anthropicChatAdapter", () => {
  it("calculateCost unpacks cacheCreationDetail from providerExtra", () => {
    const mockUsage = { input_tokens: 100, output_tokens: 50 };
    const mockCacheDetail = { tokens: 500, ttl_seconds: 300 };
    const parsed: ParsedResponse = {
      hasUsage: true,
      usage: mockUsage,
      model: "claude-sonnet-4-20250514",
      stopReason: "end_turn",
      toolCalls: null,
      providerExtra: { cacheCreationDetail: mockCacheDetail },
    };

    const result = anthropicChatAdapter.calculateCost(
      "claude-sonnet-4-20250514", "claude-sonnet-4-20250514", parsed, "req-1", 200,
      { userId: "u1", apiKeyId: "k1", actionId: null },
      0,
    );

    expect(result.provider).toBe("anthropic");
    expect(result.requestId).toBe("req-1");
    expect(result.eventType).toBe("llm");
  });

  it("calculateCost handles null cacheCreationDetail gracefully", () => {
    const parsed: ParsedResponse = {
      hasUsage: true,
      usage: { input_tokens: 100, output_tokens: 50 },
      model: "claude-sonnet-4-20250514",
      stopReason: "end_turn",
      toolCalls: null,
      providerExtra: { cacheCreationDetail: null },
    };

    // Should not throw
    const result = anthropicChatAdapter.calculateCost(
      "claude-sonnet-4-20250514", "claude-sonnet-4-20250514", parsed, "req-1", 200,
      { userId: "u1", apiKeyId: "k1", actionId: null },
      0,
    );

    expect(result.provider).toBe("anthropic");
  });

  it("extractPostResponseTags captures cache write/read tokens and long context", () => {
    const parsed: StreamResult = {
      hasUsage: true,
      usage: {
        input_tokens: 150000,
        output_tokens: 1000,
        cache_creation_input_tokens: 60000,
        cache_read_input_tokens: 50000,
      },
      model: "claude-sonnet-4-20250514",
      stopReason: "end_turn",
      toolCalls: null,
      cancelled: false,
      firstChunkMs: null,
    };

    const tags = anthropicChatAdapter.extractPostResponseTags!(parsed);

    expect(tags._ns_cache_write_tokens).toBe("60000");
    expect(tags._ns_cache_read_tokens).toBe("50000");
    // 150000 + 60000 + 50000 = 260000 > 200000
    expect(tags._ns_long_context).toBe("true");
  });

  it("extractPostResponseTags returns empty for no usage", () => {
    const parsed: StreamResult = {
      hasUsage: false,
      usage: null,
      model: null,
      stopReason: null,
      toolCalls: null,
      cancelled: true,
      firstChunkMs: null,
    };

    const tags = anthropicChatAdapter.extractPostResponseTags!(parsed);
    expect(Object.keys(tags)).toHaveLength(0);
  });

  it("emitPreFetchMetrics tracks anthropic-beta header", () => {
    const request = new Request("https://proxy.nullspend.dev/v1/messages", {
      method: "POST",
      headers: { "anthropic-beta": "prompt-caching-2024-07-31" },
    });

    anthropicChatAdapter.emitPreFetchMetrics!(request, "claude-sonnet-4-20250514");

    expect(vi.mocked(emitMetric)).toHaveBeenCalledWith(
      "anthropic_beta_used",
      expect.objectContaining({ beta: "prompt-caching-2024-07-31", model: "claude-sonnet-4-20250514" }),
    );
  });
});
