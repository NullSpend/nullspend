import { cloudflareWorkersMock, makeEnv } from "./test-helpers.js";
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
    inputPerMTok: 3.0,
    outputPerMTok: 15.0,
    cachedInputPerMTok: 0.30,
  }),
  costComponent: vi.fn().mockReturnValue(100),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: vi.fn().mockResolvedValue({ status: "skipped", hasBudgets: false, checkedEntities: [] }),
  doBudgetReconcile: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/budget-spend.js", () => ({
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/anthropic-cost-estimator.js", () => ({
  estimateAnthropicMaxCost: vi.fn().mockReturnValue(500_000),
}));

import { handleAnthropicMessages } from "../routes/anthropic.js";
import type { RequestContext } from "../lib/context.js";

const ANTHROPIC_RESPONSE = {
  id: "msg-test",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-20250514",
  content: [{ type: "text", text: "Hello!" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 25, output_tokens: 10 },
};

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "sk-ant-test-key",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(body),
  });
}

function makeCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  return {
    body,
    bodyText: JSON.stringify(body),
    bodyByteLength: JSON.stringify(body).length,
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
    },
    ownerId: "org-1",
    connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    skipDbWrites: false,
    sessionId: null,
    traceId: "trace-456",
    tags: {},
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    requestLoggingEnabled: false,
    ...overrides,
  };
}

describe("Anthropic mandate enforcement", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ANTHROPIC_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json", "request-id": "req-test" },
      }),
    );
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("allows request when allowedProviders is null (unrestricted)", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
  });

  it("allows request when anthropic is in the allowlist", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["anthropic"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(200);
  });

  it("denies request when anthropic is not in the allowlist", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["openai"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.message).toContain("Provider anthropic is not allowed");
    expect(json.error.details.mandate).toBe("allowed_providers");
    expect(json.error.details.requested).toBe("anthropic");
    expect(json.error.details.allowed).toEqual(["openai"]);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-456");
  });

  it("denies request when allowedProviders is empty array (deny all)", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: [],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
  });

  it("allows request when model is in the allowlist", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-sonnet-4-20250514", "claude-haiku-3-20240307"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(200);
  });

  it("denies request when model is not in the allowlist", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-haiku-3-20240307"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.message).toContain("Model claude-sonnet-4-20250514 is not allowed");
    expect(json.error.details.mandate).toBe("allowed_models");
    expect(json.error.details.requested).toBe("claude-sonnet-4-20250514");
    expect(json.error.details.allowed).toEqual(["claude-haiku-3-20240307"]);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-456");
  });

  it("denies request when allowedModels is empty array (deny all)", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: [],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
  });

  it("checks provider before model", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["openai"],
        allowedModels: ["claude-sonnet-4-20250514"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.details.mandate).toBe("allowed_providers");
  });

  it("does not forward to upstream when mandate denies", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-haiku-3-20240307"],
      },
    });
    await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("emits mandate_denied metric with correct tags on provider denial", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["openai"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);

    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls).toHaveLength(1);
    expect(metricCalls[0]).toMatchObject({
      _metric: "mandate_denied",
      reason: "provider_not_allowed",
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    });
  });

  it("emits mandate_denied metric with correct tags on model denial", async () => {
    const body = { model: "claude-sonnet-4-20250514", max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-haiku-3-20240307"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);

    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls).toHaveLength(1);
    expect(metricCalls[0]).toMatchObject({
      _metric: "mandate_denied",
      reason: "model_not_allowed",
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    });
  });

  it("truncates oversized model name in error response and metric", async () => {
    const longModel = "y".repeat(500);
    const body = { model: longModel, max_tokens: 100, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-haiku-3-20240307"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);

    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.details.requested).toHaveLength(200);
    expect(json.error.message).toContain("y".repeat(200));
    expect(json.error.message).not.toContain("y".repeat(201));

    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls[0].model).toHaveLength(200);
  });

  it("denies when model field is missing (extractModelFromBody returns 'unknown')", async () => {
    const body = { max_tokens: 100, messages: [{ role: "user", content: "hi" }] }; // no model field
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["claude-haiku-3-20240307"],
      },
    });
    const res = await handleAnthropicMessages(makeRequest(body), makeEnv(), ctx);

    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.details.requested).toBe("unknown");
  });
});
