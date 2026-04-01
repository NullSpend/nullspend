/**
 * Tag Budget Enforcement Tests
 *
 * Tests tag budget functionality across the proxy stack:
 * 1. OpenAI route: 429 with tag_budget_exceeded code, details include tag_key/tag_value
 * 2. Anthropic route: same
 * 3. MCP route: MCP-format response
 * 4. Webhook dispatched on denial
 * 5. No enforcement when no tags on request
 * 6. No enforcement when tags present but no matching tag budgets (approved)
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
import type { RequestContext } from "../lib/context.js";
import type { CheckResult } from "../durable-objects/user-budget.js";

// ── Helpers ────────────────────────────────────────────────────────

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer sk-test-key" },
    body: JSON.stringify(body),
  });
}

function makeAnthropicRequest(body: Record<string, unknown>): Request {
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

function makeMcpRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/mcp/budget/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function makeEnv(): Env {
  return {
    ANTHROPIC_API_KEY: "sk-ant-test-key",
    HYPERDRIVE: { connectionString: "postgresql://test" },
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
    connectionString: "postgresql://test",
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

const defaultBody = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };
const anthropicBody = { model: "claude-sonnet-4-20250514", messages: [{ role: "user", content: "hi" }], max_tokens: 1024 };

const tagDeniedCheckResult: CheckResult = {
  status: "denied",
  hasBudgets: true,
  deniedEntity: "tag:project=openclaw",
  remaining: 0,
  maxBudget: 50_000_000,
  spend: 49_500_000,
  checkedEntities: [
    { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 10_000_000, policy: "strict_block", thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
    { entityType: "tag", entityId: "project=openclaw", maxBudget: 50_000_000, spend: 49_500_000, policy: "strict_block", thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
  ],
};

const approvedCheckResult: CheckResult = {
  status: "approved",
  hasBudgets: true,
  reservationId: "rsv-test-123",
  checkedEntities: [
    { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 10_000_000, policy: "strict_block", thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
  ],
};

const approvedWithTagCheckResult: CheckResult = {
  status: "approved",
  hasBudgets: true,
  reservationId: "rsv-tag-456",
  checkedEntities: [
    { entityType: "user", entityId: "user-1", maxBudget: 100_000_000, spend: 10_000_000, policy: "strict_block", thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
    { entityType: "tag", entityId: "project=openclaw", maxBudget: 50_000_000, spend: 5_000_000, policy: "strict_block", thresholdPercentages: [50, 80, 90, 95], sessionLimit: null },
  ],
};

// ── Tests ──────────────────────────────────────────────────────────

describe("Tag Budget Enforcement", () => {
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

  // ── OpenAI route ───────────────────────────────────────────────────

  describe("OpenAI route tag budget denial", () => {
    it("returns 429 with tag_budget_exceeded code", async () => {
      mockDoBudgetCheck.mockResolvedValue(tagDeniedCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { tags: { project: "openclaw" } });
      const response = await handleChatCompletions(makeRequest(defaultBody), env, ctx);

      expect(response.status).toBe(429);
      expect(response.headers.get("X-NullSpend-Trace-Id")).toBe(ctx.traceId);

      const body = await response.json() as { error: { code: string; details: Record<string, unknown> } };
      expect(body.error.code).toBe("tag_budget_exceeded");
      expect(body.error.details.tag_key).toBe("project");
      expect(body.error.details.tag_value).toBe("openclaw");
      expect(body.error.details.budget_limit_microdollars).toBe(50_000_000);
    });

    it("dispatches tag_budget.exceeded webhook on denial", async () => {
      mockDoBudgetCheck.mockResolvedValue(tagDeniedCheckResult);
      const mockDispatcher = { dispatch: vi.fn().mockResolvedValue(undefined) };
      mockGetWebhookEndpoints.mockResolvedValue([{ id: "ep-1" }]);
      mockGetWebhookEndpointsWithSecrets.mockResolvedValue([{
        id: "ep-1",
        url: "https://example.com/hook",
        signingSecret: "test-secret",
        eventTypes: [],
        enabled: true,
        apiVersion: "2026-04-01", defaultTags: {},
        payloadMode: "full",
      }]);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, {
        tags: { project: "openclaw" },
        auth: { userId: "user-1", keyId: "key-1", hasWebhooks: true, hasBudgets: true, orgId: null, apiVersion: "2026-04-01", defaultTags: {} },
        webhookDispatcher: mockDispatcher as any,
      });
      await handleChatCompletions(makeRequest(defaultBody), env, ctx);
      await drainWaitUntil();

      expect(mockDispatchToEndpoints).toHaveBeenCalled();
      const dispatchedEvent = mockDispatchToEndpoints.mock.calls[0][2];
      expect(dispatchedEvent.type).toBe("tag_budget.exceeded");
      expect(dispatchedEvent.data.object.tag_key).toBe("project");
      expect(dispatchedEvent.data.object.tag_value).toBe("openclaw");
    });
  });

  // ── Anthropic route ────────────────────────────────────────────────

  describe("Anthropic route tag budget denial", () => {
    it("returns 429 with tag_budget_exceeded code", async () => {
      mockDoBudgetCheck.mockResolvedValue(tagDeniedCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(anthropicBody, { tags: { project: "openclaw" } });
      const response = await handleAnthropicMessages(makeAnthropicRequest(anthropicBody), env, ctx);

      expect(response.status).toBe(429);
      const body = await response.json() as { error: { code: string; details: Record<string, unknown> } };
      expect(body.error.code).toBe("tag_budget_exceeded");
      expect(body.error.details.tag_key).toBe("project");
      expect(body.error.details.tag_value).toBe("openclaw");
    });
  });

  // ── MCP route ──────────────────────────────────────────────────────

  describe("MCP route tag budget denial", () => {
    it("returns 429 with MCP-format tag_budget_exceeded response", async () => {
      mockDoBudgetCheck.mockResolvedValue(tagDeniedCheckResult);

      const env = makeEnv();
      const mcpBody = { toolName: "search", serverName: "brave", estimateMicrodollars: 500_000 };
      const ctx = makeCtx(mcpBody, { tags: { project: "openclaw" } });
      const response = await handleMcpBudgetCheck(makeMcpRequest(mcpBody), env, ctx);

      expect(response.status).toBe(429);
      const body = await response.json() as Record<string, unknown>;
      expect(body.allowed).toBe(false);
      expect(body.denied).toBe(true);
      expect(body.reason).toBe("tag_budget_exceeded");
      expect(body.tagKey).toBe("project");
      expect(body.tagValue).toBe("openclaw");
    });
  });

  // ── No enforcement cases ───────────────────────────────────────────

  describe("no enforcement", () => {
    it("no tags on request → approved (no tag budget check)", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { tags: {} });

      // Mock fetch for upstream response
      const originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }), { status: 200 }),
      );

      try {
        const response = await handleChatCompletions(makeRequest(defaultBody), env, ctx);
        expect(response.status).toBe(200);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    it("tags present but no matching tag budgets → approved", async () => {
      // The DO has no tag budgets populated, so it approves
      mockDoBudgetCheck.mockResolvedValue(approvedCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { tags: { project: "openclaw" } });

      const originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } }), { status: 200 }),
      );

      try {
        const response = await handleChatCompletions(makeRequest(defaultBody), env, ctx);
        expect(response.status).toBe(200);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  // ── Approved + reconciled path ─────────────────────────────────────

  describe("approved path with tag budgets", () => {
    it("approved request includes tag entity in budgetEntities for reconciliation", async () => {
      mockDoBudgetCheck.mockResolvedValue(approvedWithTagCheckResult);
      mockCalculateOpenAICost.mockReturnValue({
        requestId: "req-1",
        provider: "openai",
        model: "gpt-4o-mini",
        inputTokens: 10,
        outputTokens: 5,
        costMicrodollars: 42_000,
      });
      mockDoBudgetReconcile.mockResolvedValue("ok");

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { tags: { project: "openclaw" } });

      const originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          id: "chatcmpl-1",
          model: "gpt-4o-mini",
          choices: [{ message: { content: "hi" } }],
          usage: { prompt_tokens: 10, completion_tokens: 5 },
        }), { status: 200, headers: { "content-type": "application/json" } }),
      );

      try {
        const response = await handleChatCompletions(makeRequest(defaultBody), env, ctx);
        expect(response.status).toBe(200);
        await drainWaitUntil();

        // Verify reconciliation was called
        expect(mockWaitUntil).toHaveBeenCalled();

        // The budgetEntities passed to reconcile should include the tag entity
        // checkBudget builds budgetEntities from checkedEntities, which includes the tag
        // We verify the orchestrator produced the correct entities by checking doBudgetCheck was called with tags
        expect(mockDoBudgetCheck).toHaveBeenCalledWith(
          expect.anything(), "user-1", "key-1", expect.any(Number), null,
          ["project=openclaw"],
        );
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    it("budgetEntities from approved response includes tag entity with correct entityKey", async () => {
      // Directly test the orchestrator output via checkBudget
      const { checkBudget } = await import("../lib/budget-orchestrator.js");

      mockDoBudgetCheck.mockResolvedValue(approvedWithTagCheckResult);

      const env = makeEnv();
      const ctx = makeCtx(defaultBody, { tags: { project: "openclaw" } });
      const outcome = await checkBudget(env, ctx, 500_000);

      expect(outcome.status).toBe("approved");
      expect(outcome.budgetEntities).toHaveLength(2);

      const tagEntity = outcome.budgetEntities.find(e => e.entityType === "tag");
      expect(tagEntity).toBeDefined();
      expect(tagEntity!.entityId).toBe("project=openclaw");
      expect(tagEntity!.entityKey).toBe("{budget}:tag:project=openclaw");
      expect(tagEntity!.maxBudget).toBe(50_000_000);
      expect(tagEntity!.spend).toBe(5_000_000);
    });
  });
});
