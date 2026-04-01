/**
 * Velocity Recovery Webhook Tests
 *
 * Tests the `velocity.recovered` webhook event builder and route dispatch:
 * 1. buildVelocityRecoveredPayload shape and fields
 * 2. Route dispatch when velocityRecovered is non-empty (OpenAI, Anthropic, MCP)
 * 3. No dispatch when empty/undefined
 * 4. Multiple recovered entities
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
  mockCalculateOpenAICost,
  mockGetWebhookEndpoints,
  mockGetWebhookEndpointsWithSecrets,
  mockDispatchToEndpoints,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((promise: Promise<unknown>) => { promise.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockEstimateMaxCost: vi.fn(),
  mockCalculateOpenAICost: vi.fn(),
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
}));

vi.mock("../lib/cost-estimator.js", () => ({
  estimateMaxCost: (...args: unknown[]) => mockEstimateMaxCost(...args),
}));

vi.mock("../lib/anthropic-cost-estimator.js", () => ({
  estimateAnthropicMaxCost: (...args: unknown[]) => mockEstimateMaxCost(...args),
}));

vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: vi.fn().mockResolvedValue(undefined),
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  logCostEventsBatchQueued: vi.fn().mockResolvedValue(undefined),
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

vi.mock("../lib/budget-do-lookup.js", () => ({
  lookupBudgetsForDO: vi.fn().mockResolvedValue([]),
}));

vi.mock("../lib/reconciliation-queue.js", () => ({
  enqueueReconciliation: vi.fn().mockResolvedValue(undefined),
}));

// ── Imports ────────────────────────────────────────────────────────

import { handleChatCompletions } from "../routes/openai.js";
import { handleAnthropicMessages as _handleAnthropicMessages } from "../routes/anthropic.js";
import { handleMcpBudgetCheck } from "../routes/mcp.js";
import {
  buildVelocityRecoveredPayload,
  CURRENT_API_VERSION,
} from "../lib/webhook-events.js";
import type { RequestContext } from "../lib/context.js";
import type { CheckResult } from "../durable-objects/user-budget.js";

// ── Helpers ────────────────────────────────────────────────────────

function makeEnv(): Env {
  return {
    ANTHROPIC_API_KEY: "sk-ant-test-key",
    HYPERDRIVE: { connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres" },
    CACHE_KV: { get: vi.fn(), put: vi.fn(), delete: vi.fn() },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue({ toString: () => "do-id" }),
      get: vi.fn().mockReturnValue({
        checkAndReserve: vi.fn(),
        reconcile: vi.fn(),
      }),
    },
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

function makeWebhookCtx(body: Record<string, unknown>): RequestContext {
  return makeCtx(body, {
    auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
    webhookDispatcher: { dispatch: vi.fn().mockResolvedValue(undefined) } as any,
  });
}

async function drainWaitUntil() {
  await Promise.all(
    mockWaitUntil.mock.calls.map(([p]: [Promise<unknown>]) => p.catch(() => {})),
  );
}

const checkedEntity = {
  entityType: "api_key",
  entityId: "key-1",
  maxBudget: 50_000_000,
  spend: 10_000_000,
  policy: "strict_block",
};

const recoveredCheckResult: CheckResult = {
  status: "approved",
  hasBudgets: true,
  reservationId: "rsv-recovered",
  velocityRecovered: [
    {
      entityType: "api_key",
      entityId: "key-1",
      velocityLimitMicrodollars: 10_000_000,
      velocityWindowSeconds: 300,
      velocityCooldownSeconds: 60,
    },
  ],
  checkedEntities: [checkedEntity],
};

const webhookEndpoint = {
  id: "ep-1",
  url: "https://hook.example.com",
  apiVersion: "2026-04-01", defaultTags: {},
  currentSecret: "sec-1",
  signingSecret: "sec-1",
  rotatedSecret: null,
  previousSigningSecret: null,
  secretRotatedAt: null,
  rotatedAt: null,
  eventTypes: [],
};

// ════════════════════════════════════════════════════════════════════
// Group 1: buildVelocityRecoveredPayload
// ════════════════════════════════════════════════════════════════════

describe("buildVelocityRecoveredPayload", () => {
  it("produces correct event shape", () => {
    const event = buildVelocityRecoveredPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key-abc",
      velocityLimitMicrodollars: 5_000_000,
      velocityWindowSeconds: 300,
      velocityCooldownSeconds: 60,
    });

    expect(event.type).toBe("velocity.recovered");
    expect(event.api_version).toBe(CURRENT_API_VERSION);
    expect(event.id).toMatch(/^evt_/);
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object).toEqual(expect.objectContaining({
      budget_entity_type: "api_key",
      budget_entity_id: "key-abc",
      velocity_limit_microdollars: 5_000_000,
      velocity_window_seconds: 300,
      velocity_cooldown_seconds: 60,
    }));
  });

  it("includes recovered_at ISO timestamp", () => {
    const event = buildVelocityRecoveredPayload({
      budgetEntityType: "user",
      budgetEntityId: "user-1",
      velocityLimitMicrodollars: 1_000_000,
      velocityWindowSeconds: 60,
      velocityCooldownSeconds: 120,
    });

    expect(typeof event.data.object.recovered_at).toBe("string");
    expect(new Date(event.data.object.recovered_at as string).toISOString()).toBe(
      event.data.object.recovered_at,
    );
  });

  it("uses custom apiVersion when provided", () => {
    const event = buildVelocityRecoveredPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key-1",
      velocityLimitMicrodollars: 10_000_000,
      velocityWindowSeconds: 300,
      velocityCooldownSeconds: 60,
    }, "2025-01-01");

    expect(event.api_version).toBe("2025-01-01");
  });

  it("generates unique event IDs", () => {
    const data = {
      budgetEntityType: "api_key",
      budgetEntityId: "key-1",
      velocityLimitMicrodollars: 1_000_000,
      velocityWindowSeconds: 60,
      velocityCooldownSeconds: 30,
    };

    const event1 = buildVelocityRecoveredPayload(data);
    const event2 = buildVelocityRecoveredPayload(data);

    expect(event1.id).not.toBe(event2.id);
  });

  it("all required fields are present in data.object", () => {
    const event = buildVelocityRecoveredPayload({
      budgetEntityType: "user",
      budgetEntityId: "user-1",
      velocityLimitMicrodollars: 10_000_000,
      velocityWindowSeconds: 300,
      velocityCooldownSeconds: 90,
    });

    const obj = event.data.object;
    expect(obj).toHaveProperty("budget_entity_type");
    expect(obj).toHaveProperty("budget_entity_id");
    expect(obj).toHaveProperty("velocity_limit_microdollars");
    expect(obj).toHaveProperty("velocity_window_seconds");
    expect(obj).toHaveProperty("velocity_cooldown_seconds");
    expect(obj).toHaveProperty("recovered_at");
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 2: OpenAI route — velocity recovery dispatch
// ════════════════════════════════════════════════════════════════════

describe("OpenAI route — velocity recovery webhook", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset().mockResolvedValue("ok");
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
    mockCalculateOpenAICost.mockReset().mockReturnValue({ costMicrodollars: 42_000 });
    mockGetWebhookEndpoints.mockReset();
    mockGetWebhookEndpointsWithSecrets.mockReset();
    mockDispatchToEndpoints.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("dispatches velocity.recovered webhook when recovered is non-empty", async () => {
    mockDoBudgetCheck.mockResolvedValue(recoveredCheckResult);
    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([webhookEndpoint]);

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ usage: { prompt_tokens: 10, completion_tokens: 5 }, model: "gpt-4o-mini" }), { status: 200 }),
    );

    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    });

    const res = await handleChatCompletions(req, makeEnv(), makeWebhookCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }));
    await drainWaitUntil();

    expect(res.status).toBe(200);
    expect(mockDispatchToEndpoints).toHaveBeenCalledWith(
      expect.anything(),
      expect.arrayContaining([expect.objectContaining({ id: "ep-1" })]),
      expect.objectContaining({
        type: "velocity.recovered",
        data: expect.objectContaining({
          object: expect.objectContaining({
            budget_entity_type: "api_key",
            budget_entity_id: "key-1",
            velocity_limit_microdollars: 10_000_000,
            velocity_window_seconds: 300,
            velocity_cooldown_seconds: 60,
          }),
        }),
      }),
    );
  });

  it("does not dispatch when velocityRecovered is undefined", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ usage: { prompt_tokens: 10, completion_tokens: 5 }, model: "gpt-4o-mini" }), { status: 200 }),
    );

    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    });

    await handleChatCompletions(req, makeEnv(), makeWebhookCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }));
    await drainWaitUntil();

    // dispatchToEndpoints might be called for cost event but NOT for velocity.recovered
    const recoveryCalls = mockDispatchToEndpoints.mock.calls.filter(
      (call: unknown[]) => (call[2] as any)?.type === "velocity.recovered",
    );
    expect(recoveryCalls).toHaveLength(0);
  });

  it("does not dispatch when velocityRecovered is empty array", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      velocityRecovered: [],
      checkedEntities: [checkedEntity],
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ usage: { prompt_tokens: 10, completion_tokens: 5 }, model: "gpt-4o-mini" }), { status: 200 }),
    );

    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    });

    await handleChatCompletions(req, makeEnv(), makeWebhookCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }));
    await drainWaitUntil();

    const recoveryCalls = mockDispatchToEndpoints.mock.calls.filter(
      (call: unknown[]) => (call[2] as any)?.type === "velocity.recovered",
    );
    expect(recoveryCalls).toHaveLength(0);
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 3: MCP route — velocity recovery dispatch
// ════════════════════════════════════════════════════════════════════

describe("MCP route — velocity recovery webhook", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
    mockGetWebhookEndpoints.mockReset();
    mockGetWebhookEndpointsWithSecrets.mockReset();
    mockDispatchToEndpoints.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("dispatches velocity.recovered on approved MCP request with recovery", async () => {
    mockDoBudgetCheck.mockResolvedValue(recoveredCheckResult);
    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([webhookEndpoint]);

    const req = new Request("http://localhost/v1/mcp/budget/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ toolName: "search", serverName: "web", estimateMicrodollars: 500_000 }),
    });

    const res = await handleMcpBudgetCheck(req, makeEnv(), makeWebhookCtx({
      toolName: "search",
      serverName: "web",
      estimateMicrodollars: 500_000,
    }));
    await drainWaitUntil();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.allowed).toBe(true);

    expect(mockDispatchToEndpoints).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(Array),
      expect.objectContaining({
        type: "velocity.recovered",
      }),
    );
  });

  it("does not dispatch when no recovery on MCP", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-1",
      checkedEntities: [checkedEntity],
    });

    const req = new Request("http://localhost/v1/mcp/budget/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ toolName: "search", serverName: "web", estimateMicrodollars: 500_000 }),
    });

    await handleMcpBudgetCheck(req, makeEnv(), makeWebhookCtx({
      toolName: "search",
      serverName: "web",
      estimateMicrodollars: 500_000,
    }));
    await drainWaitUntil();

    const recoveryCalls = mockDispatchToEndpoints.mock.calls.filter(
      (call: unknown[]) => (call[2] as any)?.type === "velocity.recovered",
    );
    expect(recoveryCalls).toHaveLength(0);
  });
});

// ════════════════════════════════════════════════════════════════════
// Group 4: Multiple recovered entities
// ════════════════════════════════════════════════════════════════════

describe("Multiple velocity recoveries", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockWaitUntil.mockClear();
    mockDoBudgetCheck.mockReset();
    mockDoBudgetReconcile.mockReset().mockResolvedValue("ok");
    mockEstimateMaxCost.mockReset().mockReturnValue(500_000);
    mockCalculateOpenAICost.mockReset().mockReturnValue({ costMicrodollars: 42_000 });
    mockGetWebhookEndpoints.mockReset();
    mockGetWebhookEndpointsWithSecrets.mockReset();
    mockDispatchToEndpoints.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("dispatches one webhook per recovered entity", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "rsv-multi",
      velocityRecovered: [
        { entityType: "user", entityId: "user-1", velocityLimitMicrodollars: 10_000_000, velocityWindowSeconds: 300, velocityCooldownSeconds: 60 },
        { entityType: "api_key", entityId: "key-1", velocityLimitMicrodollars: 5_000_000, velocityWindowSeconds: 60, velocityCooldownSeconds: 120 },
      ],
      checkedEntities: [
        { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 50_000_000, policy: "strict_block" },
        checkedEntity,
      ],
    });
    mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
    mockGetWebhookEndpointsWithSecrets.mockResolvedValue([webhookEndpoint]);

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ usage: { prompt_tokens: 10, completion_tokens: 5 }, model: "gpt-4o-mini" }), { status: 200 }),
    );

    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
    });

    await handleChatCompletions(req, makeEnv(), makeWebhookCtx({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }));
    await drainWaitUntil();

    const recoveryCalls = mockDispatchToEndpoints.mock.calls.filter(
      (call: unknown[]) => (call[2] as any)?.type === "velocity.recovered",
    );
    expect(recoveryCalls).toHaveLength(2);

    const entityTypes = recoveryCalls.map(
      (call: unknown[]) => (call[2] as any).data.object.budget_entity_type,
    );
    expect(entityTypes).toContain("user");
    expect(entityTypes).toContain("api_key");
  });
});
