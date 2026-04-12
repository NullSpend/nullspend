/**
 * Session Limits Tests
 *
 * Tests session limit functionality across the proxy stack:
 * 1. Session limit set + sessionId + under limit → approved
 * 2. Session limit set + sessionId + over limit → denied with sessionLimitDenied
 * 3. Session limit set + no sessionId → no enforcement (approved)
 * 4. No session limit on budget → no enforcement regardless of sessionId
 * 5. Session denial response: 429 with session_limit_exceeded code, no Retry-After
 * 6. Session denial webhook: session.limit_exceeded event dispatched
 * 7. Orchestrator session denial pass-through
 * 8. MCP route session denial format
 * 9. Anthropic route session denial
 * 10. buildSessionLimitExceededPayload builder
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

vi.mock("../lib/reconciliation-queue.js", () => ({
  enqueueReconciliation: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/sanitize-upstream-error.js", () => ({
  sanitizeUpstreamError: vi.fn().mockResolvedValue('{"error":"upstream"}'),
}));

// ── Imports ────────────────────────────────────────────────────────

import { handleChatCompletions } from "../routes/openai.js";
import { handleAnthropicMessages } from "../routes/anthropic.js";
import { handleMcpBudgetCheck } from "../routes/mcp.js";
import { checkBudget } from "../lib/budget-orchestrator.js";
import {
  buildSessionLimitExceededPayload,
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

function makeMcpRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/mcp/budget/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
    customerId: null,
    customerWarning: null,
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
  reserved: 0,
  policy: "strict_block",
  thresholdPercentages: [50, 80, 90, 95],
  sessionLimit: 5_000_000,
};

const sessionDeniedCheckResult: CheckResult = {
  status: "denied",
  hasBudgets: true,
  deniedEntity: "api_key:key-1",
  sessionLimitDenied: true,
  sessionId: "sess-abc-123",
  sessionSpend: 4_800_000,
  sessionLimit: 5_000_000,
  checkedEntities: [checkedEntity],
};

const approvedCheckResult: CheckResult = {
  status: "approved",
  hasBudgets: true,
  reservationId: "rsv-test-123",
  checkedEntities: [checkedEntity],
};

// ── Tests ──────────────────────────────────────────────────────────

describe("Session Limits", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Optimistic execution: fetch starts in parallel with budget check.
    // Default mock returns a pending promise (aborted on denial).
    globalThis.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.clearAllMocks();
    mockEstimateMaxCost.mockReturnValue(500_000);
    mockCalculateOpenAICost.mockReturnValue({ costMicrodollars: 42_000 });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // ── 1. Orchestrator: session check passes through DO result ──────

  describe("checkBudget (orchestrator)", () => {
    it("passes sessionId from ctx to doBudgetCheck", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);
      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: "sess-abc-123" });

      await checkBudget(env, ctx, 500_000);

      expect(mockDoBudgetCheck).toHaveBeenCalledWith(
        env, "user-1", "key-1", 500_000, "sess-abc-123", [], null,
      );
    });

    it("passes null sessionId when ctx.sessionId is null", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);
      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: null });

      await checkBudget(env, ctx, 500_000);

      expect(mockDoBudgetCheck).toHaveBeenCalledWith(
        env, "user-1", "key-1", 500_000, null, [], null,
      );
    });

    it("returns sessionLimitDenied outcome on session limit denial", async () => {
      mockDoBudgetCheck.mockResolvedValue(sessionDeniedCheckResult);
      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: "sess-abc-123" });

      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("denied");
      expect(outcome.sessionLimitDenied).toBe(true);
      expect(outcome.sessionId).toBe("sess-abc-123");
      expect(outcome.sessionSpend).toBe(4_800_000);
      expect(outcome.sessionLimit).toBe(5_000_000);
      expect(outcome.reservationId).toBeNull();
    });

    it("returns approved when session check passes", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);
      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: "sess-abc-123" });

      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("approved");
      expect(outcome.sessionLimitDenied).toBeUndefined();
    });
  });

  // ── 2. OpenAI route: session denial response ─────────────────────

  describe("OpenAI route session denial", () => {
    it("returns 429 with session_limit_exceeded code, no Retry-After", async () => {
      mockDoBudgetCheck.mockResolvedValue(sessionDeniedCheckResult);
      mockEstimateMaxCost.mockReturnValue(500_000);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: "sess-abc-123" });
      const request = makeRequest(defaultBody);

      const response = await handleChatCompletions(request, env, ctx);

      expect(response.status).toBe(429);
      // No Retry-After header for session limits (session is done)
      expect(response.headers.get("Retry-After")).toBeNull();
      expect(response.headers.get("X-NullSpend-Trace-Id")).toBe(ctx.traceId);

      const body = await response.json() as { error: { code: string; message: string; details: Record<string, unknown> } };
      expect(body.error.code).toBe("session_limit_exceeded");
      expect(body.error.details.session_id).toBe("sess-abc-123");
      expect(body.error.details.session_spend_microdollars).toBe(4_800_000);
      expect(body.error.details.session_limit_microdollars).toBe(5_000_000);
    });

    it("dispatches session.limit_exceeded webhook on denial", async () => {
      mockDoBudgetCheck.mockResolvedValue(sessionDeniedCheckResult);
      mockEstimateMaxCost.mockReturnValue(500_000);
      mockGetWebhookEndpoints.mockResolvedValue([{ url: "https://example.com/wh" }]);
      mockGetWebhookEndpointsWithSecrets.mockResolvedValue([{
        url: "https://example.com/wh",
        signingSecret: "secret",
        apiVersion: "2026-04-01", defaultTags: {},
      }]);

      const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
      const env = makeEnv();
      const ctx = makeCtx(defaultBody, {
        sessionId: "sess-abc-123",
        auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
        webhookDispatcher: mockDispatcher as any,
      });

      await handleChatCompletions(makeRequest(defaultBody), env, ctx);
      await drainWaitUntil();

      expect(mockDispatchToEndpoints).toHaveBeenCalledTimes(1);
      const event = mockDispatchToEndpoints.mock.calls[0][2];
      expect(event.type).toBe("session.limit_exceeded");
      expect(event.data.object.session_id).toBe("sess-abc-123");
    });
  });

  // ── 3. Anthropic route: session denial response ──────────────────

  describe("Anthropic route session denial", () => {
    it("returns 429 with session_limit_exceeded code, no Retry-After", async () => {
      mockDoBudgetCheck.mockResolvedValue(sessionDeniedCheckResult);
      mockEstimateMaxCost.mockReturnValue(500_000);

      const env = makeEnv();
      const ctx = makeCtx(anthropicBody, { sessionId: "sess-abc-123" });

      const response = await handleAnthropicMessages(
        makeAnthropicRequest(anthropicBody), env, ctx,
      );

      expect(response.status).toBe(429);
      expect(response.headers.get("Retry-After")).toBeNull();

      const body = await response.json() as { error: { code: string; details: Record<string, unknown> } };
      expect(body.error.code).toBe("session_limit_exceeded");
      expect(body.error.details.session_id).toBe("sess-abc-123");
    });
  });

  // ── 4. MCP route: session denial response ────────────────────────

  describe("MCP route session denial", () => {
    it("returns 429 with session_limit_exceeded code in error envelope", async () => {
      // Post-2026-04-08 MCP migration: denial body uses { error: { code, message, details } }
      // shape identical to OpenAI/Anthropic. The flat shape was hard-cut.
      mockDoBudgetCheck.mockResolvedValue(sessionDeniedCheckResult);

      const env = makeEnv();
      const mcpBody = {
        toolName: "test-tool",
        serverName: "test-server",
        estimateMicrodollars: 500_000,
      };
      const ctx = makeCtx(mcpBody, { sessionId: "sess-abc-123" });

      const response = await handleMcpBudgetCheck(
        makeMcpRequest(mcpBody), env, ctx,
      );

      expect(response.status).toBe(429);

      const body = await response.json() as { error: { code: string; details: Record<string, unknown> } };
      expect(body.error.code).toBe("session_limit_exceeded");
      expect(body.error.details.session_id).toBe("sess-abc-123");
      expect(body.error.details.session_spend_microdollars).toBe(4_800_000);
      expect(body.error.details.session_limit_microdollars).toBe(5_000_000);
      expect(response.headers.get("X-NullSpend-Denied")).toBe("1");
    });
  });

  // ── 5. Webhook event builder ─────────────────────────────────────

  describe("buildSessionLimitExceededPayload", () => {
    it("builds a valid session.limit_exceeded payload", () => {
      const event = buildSessionLimitExceededPayload({
        budgetEntityType: "api_key",
        budgetEntityId: "key-1",
        sessionId: "sess-xyz",
        sessionSpendMicrodollars: 4_500_000,
        sessionLimitMicrodollars: 5_000_000,
        model: "gpt-4o-mini",
        provider: "openai",
      });

      expect(event.type).toBe("session.limit_exceeded");
      expect(event.api_version).toBe(CURRENT_API_VERSION);
      expect(event.id).toMatch(/^evt_/);
      expect(event.data.object.session_id).toBe("sess-xyz");
      expect(event.data.object.session_spend_microdollars).toBe(4_500_000);
      expect(event.data.object.session_limit_microdollars).toBe(5_000_000);
      expect(event.data.object.budget_entity_type).toBe("api_key");
      expect(event.data.object.budget_entity_id).toBe("key-1");
      expect(event.data.object.model).toBe("gpt-4o-mini");
      expect(event.data.object.provider).toBe("openai");
      expect(event.data.object.blocked_at).toBeDefined();
    });

    it("uses custom API version when provided", () => {
      const event = buildSessionLimitExceededPayload({
        budgetEntityType: "user",
        budgetEntityId: "user-1",
        sessionId: "sess-1",
        sessionSpendMicrodollars: 1_000,
        sessionLimitMicrodollars: 2_000,
        model: "gpt-4o",
        provider: "openai",
      }, "2025-01-01");

      expect(event.api_version).toBe("2025-01-01");
    });
  });

  // ── 6. No session limit + sessionId → no enforcement ─────────────

  describe("No enforcement when session limit not set", () => {
    it("approves when budget has no session limit regardless of sessionId", async () => {
      const noSessionLimitResult: CheckResult = {
        status: "approved",
        hasBudgets: true,
        reservationId: "rsv-1",
        checkedEntities: [{
          entityType: "user",
          entityId: "user-1",
          maxBudget: 100_000_000,
          spend: 0,
          reserved: 0,
          policy: "strict_block",
          thresholdPercentages: [50, 80, 90, 95],
          sessionLimit: null,
        }],
      };
      mockDoBudgetCheck.mockResolvedValue(noSessionLimitResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: "sess-123" });

      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("approved");
      expect(outcome.sessionLimitDenied).toBeUndefined();
    });
  });

  // ── 7. Session limit + no sessionId → no enforcement ─────────────

  describe("No enforcement when sessionId not provided", () => {
    it("approves when sessionId is null even if budget has session limit", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { sessionId: null });

      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("approved");
      // sessionId=null is passed to DO — DO skips session check
      expect(mockDoBudgetCheck).toHaveBeenCalledWith(
        env, "user-1", "key-1", 500_000, null, [], null,
      );
    });
  });

  // ── 8. CheckedEntity includes sessionLimit ───────────────────────

  describe("CheckedEntity includes sessionLimit", () => {
    it("propagates sessionLimit in checkedEntities from DO", async () => {
      mockDoBudgetCheck.mockResolvedValue({
        ...approvedCheckResult,
        checkedEntities: [{
          ...checkedEntity,
          sessionLimit: 10_000_000,
        }],
      });

      const env = makeEnv();
      const ctx = makeCtx(defaultBody);

      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("approved");
      // budgetEntities are mapped from checkedEntities
      expect(outcome.budgetEntities.length).toBe(1);
    });
  });
});
