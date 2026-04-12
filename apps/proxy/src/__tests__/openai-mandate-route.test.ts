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
    inputPerMTok: 0.15,
    outputPerMTok: 0.60,
    cachedInputPerMTok: 0.075,
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

vi.mock("../lib/cost-estimator.js", () => ({
  estimateMaxCost: vi.fn().mockReturnValue(500_000),
}));

import { handleChatCompletions } from "../routes/openai.js";
import type { RequestContext } from "../lib/context.js";

const OPENAI_RESPONSE = {
  id: "chatcmpl-test",
  model: "gpt-4o-mini-2024-07-18",
  choices: [{ index: 0, message: { role: "assistant", content: "Hello!" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 25, completion_tokens: 10 },
};

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test-key" },
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
    traceId: "trace-123",
    tags: {},
    customerId: null,
    customerWarning: null,
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    requestLoggingEnabled: false,
    ...overrides,
  };
}

describe("OpenAI mandate enforcement", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(OPENAI_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req-test" },
      }),
    );
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("allows request when allowedProviders is null (unrestricted)", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
  });

  it("allows request when provider is in the allowlist", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["openai"],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(200);
  });

  it("denies request when provider is not in the allowlist", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["anthropic"],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.message).toContain("Provider openai is not allowed");
    expect(json.error.details.mandate).toBe("allowed_providers");
    expect(json.error.details.requested).toBe("openai");
    expect(json.error.details.allowed).toEqual(["anthropic"]);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-123");
  });

  it("denies request when allowedProviders is empty array (deny all)", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: [],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
  });

  it("allows request when allowedModels is null (unrestricted)", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), makeCtx(body));
    expect(res.status).toBe(200);
  });

  it("allows request when model is in the allowlist", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o-mini", "gpt-4o"],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(200);
  });

  it("denies request when model is not in the allowlist", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o"],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.message).toContain("Model gpt-4o-mini is not allowed");
    expect(json.error.message).toContain("gpt-4o");
    expect(json.error.details.mandate).toBe("allowed_models");
    expect(json.error.details.requested).toBe("gpt-4o-mini");
    expect(json.error.details.allowed).toEqual(["gpt-4o"]);
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-123");
  });

  it("denies request when allowedModels is empty array (deny all)", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: [],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
  });

  it("checks provider before model (provider denied = no model check needed)", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["anthropic"],
        allowedModels: ["gpt-4o-mini"], // model would pass, but provider fails first
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.details.mandate).toBe("allowed_providers");
  });

  it("does not forward to upstream when mandate denies", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o"],
      },
    });
    await handleChatCompletions(makeRequest(body), makeEnv(), ctx);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("emits mandate_denied metric with correct tags on provider denial", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedProviders: ["anthropic"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    await handleChatCompletions(makeRequest(body), makeEnv(), ctx);

    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls).toHaveLength(1);
    expect(metricCalls[0]).toMatchObject({
      _metric: "mandate_denied",
      reason: "provider_not_allowed",
      provider: "openai",
      model: "gpt-4o-mini",
    });
  });

  it("emits mandate_denied metric with correct tags on model denial", async () => {
    const body = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    await handleChatCompletions(makeRequest(body), makeEnv(), ctx);

    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls).toHaveLength(1);
    expect(metricCalls[0]).toMatchObject({
      _metric: "mandate_denied",
      reason: "model_not_allowed",
      provider: "openai",
      model: "gpt-4o-mini",
    });
  });

  it("truncates oversized model name in error response and metric", async () => {
    const longModel = "x".repeat(500);
    const body = { model: longModel, messages: [{ role: "user", content: "hi" }] };
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o"],
      },
    });
    const logSpy = vi.spyOn(console, "log");
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);

    expect(res.status).toBe(403);
    const json = await res.json();
    // Error message and details.requested should be truncated to 200 chars
    expect(json.error.details.requested).toHaveLength(200);
    expect(json.error.message).toContain("x".repeat(200));
    expect(json.error.message).not.toContain("x".repeat(201));

    // Metric model should also be truncated
    const metricCalls = logSpy.mock.calls
      .map(([arg]) => { try { return JSON.parse(arg as string); } catch { return null; } })
      .filter((m) => m?._metric === "mandate_denied");
    expect(metricCalls[0].model).toHaveLength(200);
  });

  it("denies when model field is missing (extractModelFromBody returns 'unknown')", async () => {
    const body = { messages: [{ role: "user", content: "hi" }] }; // no model field
    const ctx = makeCtx(body, {
      auth: {
        ...makeCtx(body).auth,
        allowedModels: ["gpt-4o"],
      },
    });
    const res = await handleChatCompletions(makeRequest(body), makeEnv(), ctx);

    expect(res.status).toBe(403);
    const json = await res.json();
    expect(json.error.code).toBe("mandate_violation");
    expect(json.error.details.requested).toBe("unknown");
  });
});
