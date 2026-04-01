/**
 * Velocity Limits Tests
 *
 * Tests velocity limit functionality across the proxy stack:
 * 1. BudgetCheckOutcome velocity fields via orchestrator
 * 2. Route handler velocity denial (OpenAI + Anthropic)
 * 3. Webhook event builder (buildVelocityExceededPayload)
 * 4. populateIfEmpty velocity config pass-through (doBudgetUpsertEntities)
 * 5. CheckResult interface velocity fields
 * 6. Velocity recovery pass-through
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
  mockPopulateIfEmpty,
  mockGetWebhookEndpoints,
  mockGetWebhookEndpointsWithSecrets,
  mockDispatchToEndpoints,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((promise: Promise<unknown>) => { promise.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockEstimateMaxCost: vi.fn(),
  mockUpdateBudgetSpend: vi.fn(),
  mockCalculateOpenAICost: vi.fn(),
  mockPopulateIfEmpty: vi.fn(),
  mockGetWebhookEndpoints: vi.fn(),
  mockGetWebhookEndpointsWithSecrets: vi.fn(),
  mockDispatchToEndpoints: vi.fn(),
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
  doBudgetUpsertEntities: vi.fn(async (_env: unknown, _userId: string, entities: unknown[]) => {
    for (const e of entities) {
      await mockPopulateIfEmpty(e);
    }
  }),
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
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/cost-calculator.js", () => ({
  calculateOpenAICost: (...args: unknown[]) => mockCalculateOpenAICost(...args),
}));

vi.mock("../lib/anthropic-cost-calculator.js", () => ({
  calculateAnthropicCost: vi.fn().mockReturnValue({ costMicrodollars: 42_000 }),
}));
vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpoints: (...args: unknown[]) => mockGetWebhookEndpoints(...args),
  getWebhookEndpointsWithSecrets: (...args: unknown[]) => mockGetWebhookEndpointsWithSecrets(...args),
  invalidateWebhookCache: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/webhook-thresholds.js", () => ({
  detectThresholdCrossings: vi.fn().mockReturnValue([]),
}));

vi.mock("../lib/webhook-dispatch.js", () => ({
  dispatchToEndpoints: (...args: unknown[]) => mockDispatchToEndpoints(...args),
  createWebhookDispatcher: vi.fn().mockReturnValue({ dispatch: vi.fn() }),
}));

vi.mock("../lib/webhook-expiry.js", () => ({
  expireRotatedSecrets: vi.fn().mockResolvedValue(undefined),
}));

// ── Imports ────────────────────────────────────────────────────────

import { handleChatCompletions } from "../routes/openai.js";
import { handleAnthropicMessages } from "../routes/anthropic.js";
import { checkBudget } from "../lib/budget-orchestrator.js";
import { doBudgetUpsertEntities } from "../lib/budget-do-client.js";
import {
  buildVelocityExceededPayload,
  CURRENT_API_VERSION,
} from "../lib/webhook-events.js";
import type { RequestContext } from "../lib/context.js";
import type { CheckResult } from "../durable-objects/user-budget.js";

// ── Helpers ────────────────────────────────────────────────────────

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

function makeAnthropicRequest(
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Request {
  return new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "sk-ant-test-key",
      "anthropic-version": "2023-06-01",
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

function makeEnv(overrides: Partial<Record<string, unknown>> = {}): Env {
  return {
    ANTHROPIC_API_KEY: "sk-ant-test-key",
    HYPERDRIVE: {
      connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    },
    CACHE_KV: { get: vi.fn(), put: vi.fn(), delete: vi.fn() },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue({ toString: () => "do-id" }),
      get: vi.fn().mockReturnValue({
        checkAndReserve: vi.fn(),
        reconcile: vi.fn(),
        populateIfEmpty: mockPopulateIfEmpty,
        removeBudget: vi.fn(),
        resetSpend: vi.fn(),
      }),
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
    auth: { userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
    ownerId: "user-1",
    connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
    sessionId: null,
    traceId: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    tags: {},
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    ...overrides,
  };
}

async function drainWaitUntil() {
  await Promise.all(
    mockWaitUntil.mock.calls.map(([p]: [Promise<unknown>]) => p.catch(() => {})),
  );
}

const defaultBody = {
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "hi" }],
};

const anthropicBody = {
  model: "claude-sonnet-4-20250514",
  messages: [{ role: "user", content: "hi" }],
  max_tokens: 1024,
};

const checkedEntity = {
  entityType: "api_key",
  entityId: "key-1",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  policy: "strict_block",
};

const velocityDeniedCheckResult: CheckResult = {
  status: "denied",
  hasBudgets: true,
  deniedEntity: "api_key:key-1",
  velocityDenied: true,
  retryAfterSeconds: 45,
  velocityDetails: {
    limitMicrodollars: 10_000_000,
    windowSeconds: 300,
    currentMicrodollars: 12_500_000,
  },
  checkedEntities: [checkedEntity],
};

// ════════════════════════════════════════════════════════════════════
// Group 1: BudgetCheckOutcome velocity fields via orchestrator
// ════════════════════════════════════════════════════════════════════

describe("BudgetCheckOutcome — velocity fields", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEstimateMaxCost.mockReturnValue(500_000);
  });

  it("carries velocityDenied=true through from DO CheckResult", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("denied");
    expect(result.velocityDenied).toBe(true);
  });

  it("passes retryAfterSeconds from DO CheckResult", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.retryAfterSeconds).toBe(45);
  });

  it("passes velocityDetails from DO CheckResult", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.velocityDetails).toEqual({
      limitMicrodollars: 10_000_000,
      windowSeconds: 300,
      currentMicrodollars: 12_500_000,
    });
  });

  it("parses deniedEntityType and deniedEntityId from deniedEntity string", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.deniedEntityType).toBe("api_key");
    expect(result.deniedEntityId).toBe("key-1");
  });

  it("reservationId is null on velocity denial (no reservation made)", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.reservationId).toBeNull();
  });

  it("budgetEntities are still populated on velocity denial", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.budgetEntities).toHaveLength(1);
    expect(result.budgetEntities[0].entityType).toBe("api_key");
    expect(result.budgetEntities[0].entityId).toBe("key-1");
  });

  it("regular budget denial does not set velocityDenied", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      remaining: 500,
      maxBudget: 100_000_000,
      spend: 99_999_500,
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("denied");
    expect(result.velocityDenied).toBeUndefined();
    expect(result.retryAfterSeconds).toBeUndefined();
    expect(result.velocityDetails).toBeUndefined();
  });

  it("approved result does not carry velocity fields", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("approved");
    expect(result.velocityDenied).toBeUndefined();
    expect(result.retryAfterSeconds).toBeUndefined();
  });

  it("velocity denial with user entity type parses correctly", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      ...velocityDeniedCheckResult,
      deniedEntity: "user:user-1",
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.deniedEntityType).toBe("user");
    expect(result.deniedEntityId).toBe("user-1");
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 2: Route handler velocity denial (OpenAI)
// ════════════════════════════════════════════════════════════════════

describe("OpenAI route — velocity denial", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Optimistic execution: fetch starts in parallel with budget check.
    // Default mock returns a pending promise (aborted on denial).
    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset();
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
    mockCalculateOpenAICost.mockReset();
    mockGetWebhookEndpoints.mockReset();
    mockGetWebhookEndpointsWithSecrets.mockReset();
    mockDispatchToEndpoints.mockReset();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("returns 429 with velocity_exceeded code", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("velocity_exceeded");
  });

  it("returns Retry-After header with retryAfterSeconds value", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));

    expect(res.headers.get("Retry-After")).toBe("45");
  });

  it("defaults Retry-After to 60 when retryAfterSeconds is undefined", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      ...velocityDeniedCheckResult,
      retryAfterSeconds: undefined,
    });

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));

    expect(res.headers.get("Retry-After")).toBe("60");
  });

  it("response body has correct error shape with velocity details", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const body = await res.json();

    expect(body.error).toEqual({
      code: "velocity_exceeded",
      message: "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.",
      details: {
        limitMicrodollars: 10_000_000,
        windowSeconds: 300,
        currentMicrodollars: 12_500_000,
      },
    });
  });

  it("response body has null details when velocityDetails is undefined", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      ...velocityDeniedCheckResult,
      velocityDetails: undefined,
    });

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const body = await res.json();

    expect(body.error.details).toBeNull();
  });

  it("does not call doBudgetReconcile on velocity denial (no reservation to clean up)", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    await drainWaitUntil();

    expect(mockDoBudgetReconcile).not.toHaveBeenCalled();
  });

  it("aborts upstream fetch on velocity denial", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    const fetchSpy = vi.fn().mockReturnValue(new Promise(() => {}));
    globalThis.fetch = fetchSpy;

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));

    expect(res.status).toBe(429);
    // Optimistic execution: fetch starts in parallel, aborted when budget denies
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const signal = fetchSpy.mock.calls[0][1]?.signal as AbortSignal;
    expect(signal?.aborted).toBe(true);
  });

  it("velocity denial is distinct from budget_exceeded", async () => {
    // First call: velocity denial
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    const velRes = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const velBody = await velRes.json();

    // Second call: budget denial
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-1",
      remaining: 100,
      maxBudget: 50_000_000,
      spend: 49_999_900,
      checkedEntities: [checkedEntity],
    });
    const budRes = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const budBody = await budRes.json();

    expect(velBody.error.code).toBe("velocity_exceeded");
    expect(budBody.error.code).toBe("budget_exceeded");
    expect(velRes.headers.get("Retry-After")).toBe("45");
    expect(budRes.headers.get("Retry-After")).toBeNull();
  });

  it("does not contain sensitive data in velocity 429 response", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    const text = await res.text();

    expect(text).not.toContain("x-nullspend-key");
    expect(text).not.toContain("ns_live_sk_");
    expect(text).not.toContain("postgresql://");
    expect(text).not.toContain("upstash.io");
    expect(text).not.toContain("fake-token");
    expect(text).not.toContain("sk-test-key");
  });

  it("dispatches velocity.exceeded webhook when hasWebhooks is true", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([
      { id: "ep-1", url: "https://hook.example.com", apiVersion: "2026-04-01", defaultTags: {}, currentSecret: "sec-1", rotatedSecret: null, rotatedAt: null },
    ]);
    mockDispatchToEndpoints.mockResolvedValue(undefined);

    const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
    const ctx = makeCtx(defaultBody, {
      auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      webhookDispatcher: mockDispatcher as any,
    });

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), ctx);
    await drainWaitUntil();

    expect(mockDispatchToEndpoints).toHaveBeenCalledWith(
      mockDispatcher,
      expect.arrayContaining([
        expect.objectContaining({ id: "ep-1" }),
      ]),
      expect.objectContaining({
        type: "velocity.exceeded",
        data: expect.objectContaining({
          object: expect.objectContaining({
            velocity_limit_microdollars: 10_000_000,
            velocity_window_seconds: 300,
            velocity_current_microdollars: 12_500_000,
            cooldown_seconds: 45,
            model: "gpt-4o-mini",
            provider: "openai",
          }),
        }),
      }),
    );
  });

  it("does not dispatch webhook when hasWebhooks is false", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), makeCtx(defaultBody));
    await drainWaitUntil();

    expect(mockGetWebhookEndpoints).not.toHaveBeenCalled();
    expect(mockDispatchToEndpoints).not.toHaveBeenCalled();
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 3: Anthropic route — velocity denial
// ════════════════════════════════════════════════════════════════════

describe("Anthropic route — velocity denial", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Optimistic execution: fetch starts in parallel with budget check.
    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset();
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
    mockGetWebhookEndpoints.mockReset();
    mockGetWebhookEndpointsWithSecrets.mockReset();
    mockDispatchToEndpoints.mockReset();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("returns 429 with velocity_exceeded code", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleAnthropicMessages(
      makeAnthropicRequest(anthropicBody), makeEnv(), makeCtx(anthropicBody),
    );

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("velocity_exceeded");
  });

  it("returns Retry-After header", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const res = await handleAnthropicMessages(
      makeAnthropicRequest(anthropicBody), makeEnv(), makeCtx(anthropicBody),
    );

    expect(res.headers.get("Retry-After")).toBe("45");
  });

  it("aborts upstream fetch on velocity denial", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    const fetchSpy = vi.fn().mockReturnValue(new Promise(() => {}));
    globalThis.fetch = fetchSpy;

    const res = await handleAnthropicMessages(
      makeAnthropicRequest(anthropicBody), makeEnv(), makeCtx(anthropicBody),
    );

    expect(res.status).toBe(429);
    // Optimistic execution: fetch starts in parallel, aborted when budget denies
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const signal = fetchSpy.mock.calls[0][1]?.signal as AbortSignal;
    expect(signal?.aborted).toBe(true);
  });

  it("webhook dispatch uses provider='anthropic'", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([
      { id: "ep-1", url: "https://hook.example.com", apiVersion: "2026-04-01", defaultTags: {}, currentSecret: "sec-1", rotatedSecret: null, rotatedAt: null },
    ]);
    mockDispatchToEndpoints.mockResolvedValue(undefined);

    const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
    const ctx = makeCtx(anthropicBody, {
      auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      webhookDispatcher: mockDispatcher as any,
    });

    await handleAnthropicMessages(makeAnthropicRequest(anthropicBody), makeEnv(), ctx);
    await drainWaitUntil();

    expect(mockDispatchToEndpoints).toHaveBeenCalledWith(
      mockDispatcher,
      expect.any(Array),
      expect.objectContaining({
        type: "velocity.exceeded",
        data: expect.objectContaining({
          object: expect.objectContaining({
            provider: "anthropic",
          }),
        }),
      }),
    );
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 4: Webhook event builder — buildVelocityExceededPayload
// ════════════════════════════════════════════════════════════════════

describe("buildVelocityExceededPayload", () => {
  it("produces correct event shape", () => {
    const event = buildVelocityExceededPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key-abc",
      velocityLimitMicrodollars: 5_000_000,
      velocityWindowSeconds: 300,
      velocityCurrentMicrodollars: 7_500_000,
      cooldownSeconds: 60,
      model: "gpt-4o-mini",
      provider: "openai",
    });

    expect(event.type).toBe("velocity.exceeded");
    expect(event.api_version).toBe(CURRENT_API_VERSION);
    expect(event.id).toMatch(/^evt_/);
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object).toEqual(expect.objectContaining({
      budget_entity_type: "api_key",
      budget_entity_id: "key-abc",
      velocity_limit_microdollars: 5_000_000,
      velocity_window_seconds: 300,
      velocity_current_microdollars: 7_500_000,
      cooldown_seconds: 60,
      model: "gpt-4o-mini",
      provider: "openai",
    }));
  });

  it("includes blocked_at ISO timestamp", () => {
    const event = buildVelocityExceededPayload({
      budgetEntityType: "user",
      budgetEntityId: "user-1",
      velocityLimitMicrodollars: 1_000_000,
      velocityWindowSeconds: 60,
      velocityCurrentMicrodollars: 1_200_000,
      cooldownSeconds: 120,
      model: "claude-sonnet-4-20250514",
      provider: "anthropic",
    });

    expect(typeof event.data.object.blocked_at).toBe("string");
    // Verify it is a valid ISO date string
    expect(new Date(event.data.object.blocked_at as string).toISOString()).toBe(
      event.data.object.blocked_at,
    );
  });

  it("uses custom apiVersion when provided", () => {
    const event = buildVelocityExceededPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key-1",
      velocityLimitMicrodollars: 10_000_000,
      velocityWindowSeconds: 300,
      velocityCurrentMicrodollars: 10_000_001,
      cooldownSeconds: 60,
      model: "gpt-4o-mini",
      provider: "openai",
    }, "2025-01-01");

    expect(event.api_version).toBe("2025-01-01");
  });

  it("generates unique event IDs", () => {
    const data = {
      budgetEntityType: "api_key",
      budgetEntityId: "key-1",
      velocityLimitMicrodollars: 1_000_000,
      velocityWindowSeconds: 60,
      velocityCurrentMicrodollars: 1_500_000,
      cooldownSeconds: 30,
      model: "gpt-4o-mini",
      provider: "openai",
    };

    const event1 = buildVelocityExceededPayload(data);
    const event2 = buildVelocityExceededPayload(data);

    expect(event1.id).not.toBe(event2.id);
  });

  it("all required fields are present in data.object", () => {
    const event = buildVelocityExceededPayload({
      budgetEntityType: "user",
      budgetEntityId: "user-1",
      velocityLimitMicrodollars: 10_000_000,
      velocityWindowSeconds: 300,
      velocityCurrentMicrodollars: 12_000_000,
      cooldownSeconds: 90,
      model: "o3-mini",
      provider: "openai",
    });

    const obj = event.data.object;
    expect(obj).toHaveProperty("budget_entity_type");
    expect(obj).toHaveProperty("budget_entity_id");
    expect(obj).toHaveProperty("velocity_limit_microdollars");
    expect(obj).toHaveProperty("velocity_window_seconds");
    expect(obj).toHaveProperty("velocity_current_microdollars");
    expect(obj).toHaveProperty("cooldown_seconds");
    expect(obj).toHaveProperty("model");
    expect(obj).toHaveProperty("provider");
    expect(obj).toHaveProperty("blocked_at");
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 5: populateIfEmpty velocity config pass-through
// ════════════════════════════════════════════════════════════════════

describe("doBudgetUpsertEntities — velocity config pass-through", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPopulateIfEmpty.mockResolvedValue(undefined);
  });

  it("passes velocity fields to populateIfEmpty for each entity", async () => {
    const entities = [
      {
        entityType: "user",
        entityId: "user-1",
        maxBudget: 100_000_000,
        spend: 0,
        policy: "strict_block",
        resetInterval: "monthly" as const,
        periodStart: 1_700_000_000_000,
        velocityLimit: 10_000_000,
        velocityWindow: 300_000,
        velocityCooldown: 60_000,
      },
    ];

    await doBudgetUpsertEntities(makeEnv(), "user-1", entities);

    expect(mockPopulateIfEmpty).toHaveBeenCalledWith(
      expect.objectContaining({
        velocityLimit: 10_000_000,
        velocityWindow: 300_000,
        velocityCooldown: 60_000,
      }),
    );
  });

  it("passes null velocityLimit when not configured", async () => {
    const entities = [
      {
        entityType: "api_key",
        entityId: "key-1",
        maxBudget: 50_000_000,
        spend: 0,
        policy: "strict_block",
        resetInterval: null,
        periodStart: 1_700_000_000_000,
        velocityLimit: null,
        velocityWindow: 60_000,
        velocityCooldown: 60_000,
      },
    ];

    await doBudgetUpsertEntities(makeEnv(), "user-1", entities);

    expect(mockPopulateIfEmpty).toHaveBeenCalledWith(
      expect.objectContaining({
        velocityLimit: null,
      }),
    );
  });

  it("calls populateIfEmpty for each entity in array", async () => {
    const entities = [
      {
        entityType: "user",
        entityId: "user-1",
        maxBudget: 100_000_000,
        spend: 0,
        policy: "strict_block",
        resetInterval: "monthly" as const,
        periodStart: 1_700_000_000_000,
        velocityLimit: 10_000_000,
        velocityWindow: 300_000,
        velocityCooldown: 60_000,
      },
      {
        entityType: "api_key",
        entityId: "key-1",
        maxBudget: 50_000_000,
        spend: 5_000_000,
        policy: "strict_block",
        resetInterval: null,
        periodStart: 1_700_000_000_000,
        velocityLimit: 5_000_000,
        velocityWindow: 60_000,
        velocityCooldown: 120_000,
      },
    ];

    await doBudgetUpsertEntities(makeEnv(), "user-1", entities);

    expect(mockPopulateIfEmpty).toHaveBeenCalledTimes(2);
    expect(mockPopulateIfEmpty).toHaveBeenNthCalledWith(1,
      expect.objectContaining({ velocityLimit: 10_000_000, velocityWindow: 300_000 }),
    );
    expect(mockPopulateIfEmpty).toHaveBeenNthCalledWith(2,
      expect.objectContaining({ velocityLimit: 5_000_000, velocityWindow: 60_000 }),
    );
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 6: CheckResult interface — velocity fields structure
// ════════════════════════════════════════════════════════════════════

describe("CheckResult interface — velocity fields", () => {
  it("velocityDenied field is boolean", () => {
    const result: CheckResult = {
      status: "denied",
      hasBudgets: true,
      velocityDenied: true,
    };
    expect(typeof result.velocityDenied).toBe("boolean");
  });

  it("retryAfterSeconds field is number", () => {
    const result: CheckResult = {
      status: "denied",
      hasBudgets: true,
      velocityDenied: true,
      retryAfterSeconds: 120,
    };
    expect(typeof result.retryAfterSeconds).toBe("number");
    expect(result.retryAfterSeconds).toBe(120);
  });

  it("velocityDetails has all required sub-fields", () => {
    const result: CheckResult = {
      status: "denied",
      hasBudgets: true,
      velocityDenied: true,
      velocityDetails: {
        limitMicrodollars: 10_000_000,
        windowSeconds: 300,
        currentMicrodollars: 15_000_000,
      },
    };
    expect(result.velocityDetails!.limitMicrodollars).toBe(10_000_000);
    expect(result.velocityDetails!.windowSeconds).toBe(300);
    expect(result.velocityDetails!.currentMicrodollars).toBe(15_000_000);
  });

  it("velocityRecovered is an array of entity descriptors with velocity config", () => {
    const result: CheckResult = {
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      velocityRecovered: [
        { entityType: "api_key", entityId: "key-1", velocityLimitMicrodollars: 10_000_000, velocityWindowSeconds: 300, velocityCooldownSeconds: 60 },
        { entityType: "user", entityId: "user-1", velocityLimitMicrodollars: 5_000_000, velocityWindowSeconds: 60, velocityCooldownSeconds: 120 },
      ],
    };
    expect(result.velocityRecovered).toHaveLength(2);
    expect(result.velocityRecovered![0]).toEqual({
      entityType: "api_key", entityId: "key-1",
      velocityLimitMicrodollars: 10_000_000, velocityWindowSeconds: 300, velocityCooldownSeconds: 60,
    });
  });

  it("all velocity fields are optional (backward compatible)", () => {
    const result: CheckResult = {
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
    };
    expect(result.velocityDenied).toBeUndefined();
    expect(result.retryAfterSeconds).toBeUndefined();
    expect(result.velocityDetails).toBeUndefined();
    expect(result.velocityRecovered).toBeUndefined();
  });

  it("velocity denial can coexist with deniedEntity and standard fields", () => {
    const result: CheckResult = {
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-1",
      velocityDenied: true,
      retryAfterSeconds: 30,
      velocityDetails: {
        limitMicrodollars: 1_000_000,
        windowSeconds: 60,
        currentMicrodollars: 1_200_000,
      },
      checkedEntities: [
        { entityType: "api_key", entityId: "key-1", maxBudget: 50_000_000, spend: 10_000_000, policy: "strict_block" },
      ],
    };

    expect(result.status).toBe("denied");
    expect(result.deniedEntity).toBe("api_key:key-1");
    expect(result.velocityDenied).toBe(true);
    expect(result.checkedEntities).toHaveLength(1);
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 7: Sliding window counter behavior via orchestrator
// ════════════════════════════════════════════════════════════════════

describe("Sliding window — requests within/over limit", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEstimateMaxCost.mockReturnValue(500_000);
  });

  it("request within velocity limit is approved", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-within",
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("approved");
    expect(result.velocityDenied).toBeUndefined();
  });

  it("request over velocity limit is denied with details", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-1",
      velocityDenied: true,
      retryAfterSeconds: 120,
      velocityDetails: {
        limitMicrodollars: 5_000_000,
        windowSeconds: 60,
        currentMicrodollars: 5_200_000,
      },
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("denied");
    expect(result.velocityDenied).toBe(true);
    expect(result.velocityDetails!.currentMicrodollars).toBeGreaterThan(
      result.velocityDetails!.limitMicrodollars,
    );
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 8: Circuit breaker behavior
// ════════════════════════════════════════════════════════════════════

describe("Circuit breaker — tripped/cooldown/recovery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEstimateMaxCost.mockReturnValue(500_000);
  });

  it("tripped circuit breaker returns velocity denial with cooldown", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      velocityDenied: true,
      retryAfterSeconds: 90,
      velocityDetails: {
        limitMicrodollars: 10_000_000,
        windowSeconds: 300,
        currentMicrodollars: 10_500_000,
      },
      checkedEntities: [
        { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 50_000_000, policy: "strict_block" },
      ],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("denied");
    expect(result.velocityDenied).toBe(true);
    expect(result.retryAfterSeconds).toBe(90);
  });

  it("recovered circuit breaker returns approved with velocityRecovered", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-recovered",
      velocityRecovered: [
        { entityType: "user", entityId: "user-1", velocityLimitMicrodollars: 10_000_000, velocityWindowSeconds: 300, velocityCooldownSeconds: 60 },
      ],
      checkedEntities: [
        { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 50_000_000, policy: "strict_block" },
      ],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.status).toBe("approved");
    expect(result.velocityDenied).toBeUndefined();
    // velocityRecovered is passed through in the CheckResult but not
    // directly mapped into BudgetCheckOutcome — the DO handles recovery internally
    expect(result.reservationId).toBe("rsv-recovered");
  });

  it("retryAfterSeconds=0 means immediate retry is allowed", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      ...velocityDeniedCheckResult,
      retryAfterSeconds: 0,
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.retryAfterSeconds).toBe(0);
  });

  it("large retryAfterSeconds (long cooldown) is passed through", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      ...velocityDeniedCheckResult,
      retryAfterSeconds: 3600,
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.retryAfterSeconds).toBe(3600);
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 9: Edge cases
// ════════════════════════════════════════════════════════════════════

describe("Velocity limits — edge cases", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.clearAllMocks();
    mockEstimateMaxCost.mockReturnValue(500_000);
    mockDoBudgetReconcile.mockResolvedValue(undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("velocity denial with zero currentMicrodollars (fresh window)", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "api_key:key-1",
      velocityDenied: true,
      retryAfterSeconds: 60,
      velocityDetails: {
        limitMicrodollars: 0,
        windowSeconds: 60,
        currentMicrodollars: 0,
      },
      checkedEntities: [checkedEntity],
    });

    const result = await checkBudget(makeEnv(), makeCtx(defaultBody), 500_000);

    expect(result.velocityDenied).toBe(true);
    expect(result.velocityDetails!.currentMicrodollars).toBe(0);
  });

  it("streaming request with velocity denial returns 429 without streaming", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);

    const streamBody = { ...defaultBody, stream: true };
    const res = await handleChatCompletions(
      makeRequest(streamBody), makeEnv(), makeCtx(streamBody),
    );

    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("velocity_exceeded");
    // Content-Type should be JSON, not text/event-stream
    expect(res.headers.get("Content-Type")).toBe("application/json");
  });

  it("velocity denial webhook failure does not affect 429 response", async () => {
    mockDoBudgetCheck.mockResolvedValue(velocityDeniedCheckResult);
    mockGetWebhookEndpoints.mockRejectedValue(new Error("Redis down"));

    const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
    const ctx = makeCtx(defaultBody, {
      auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      webhookDispatcher: mockDispatcher as any,
    });

    const res = await handleChatCompletions(makeRequest(defaultBody), makeEnv(), ctx);
    await drainWaitUntil();

    // Response should still be 429 even though webhook dispatch failed
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error.code).toBe("velocity_exceeded");
  });

  it("velocity denial with missing deniedEntity defaults to unknown in webhook", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      // No deniedEntity field
      velocityDenied: true,
      retryAfterSeconds: 60,
      velocityDetails: {
        limitMicrodollars: 1_000_000,
        windowSeconds: 60,
        currentMicrodollars: 1_500_000,
      },
      checkedEntities: [checkedEntity],
    });

    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([
      { id: "ep-1", url: "https://hook.example.com", apiVersion: "2026-04-01", defaultTags: {}, currentSecret: "sec-1", rotatedSecret: null, rotatedAt: null },
    ]);
    mockDispatchToEndpoints.mockResolvedValue(undefined);

    const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
    const ctx = makeCtx(defaultBody, {
      auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
      webhookDispatcher: mockDispatcher as any,
    });

    await handleChatCompletions(makeRequest(defaultBody), makeEnv(), ctx);
    await drainWaitUntil();

    expect(mockDispatchToEndpoints).toHaveBeenCalledWith(
      mockDispatcher,
      expect.any(Array),
      expect.objectContaining({
        data: expect.objectContaining({
          object: expect.objectContaining({
            budget_entity_type: "unknown",
            budget_entity_id: "unknown",
          }),
        }),
      }),
    );
  });

  it("multiple velocity fields are numeric and within expected ranges", () => {
    const result: CheckResult = {
      status: "denied",
      hasBudgets: true,
      velocityDenied: true,
      retryAfterSeconds: 300,
      velocityDetails: {
        limitMicrodollars: 100_000_000,
        windowSeconds: 3600,
        currentMicrodollars: 150_000_000,
      },
    };

    expect(Number.isFinite(result.retryAfterSeconds)).toBe(true);
    expect(Number.isFinite(result.velocityDetails!.limitMicrodollars)).toBe(true);
    expect(Number.isFinite(result.velocityDetails!.windowSeconds)).toBe(true);
    expect(Number.isFinite(result.velocityDetails!.currentMicrodollars)).toBe(true);
    expect(result.retryAfterSeconds).toBeGreaterThanOrEqual(0);
    expect(result.velocityDetails!.windowSeconds).toBeGreaterThan(0);
  });
});
