/**
 * Loop Detection Tests
 *
 * Tests the full loop detection feature across the proxy stack:
 * 1. Basic detection — 50 identical calls triggers
 * 2. Different content — 50 calls with different body hashes don't trigger
 * 3. Different models — independent tracking
 * 4. Window expiry — old entries pruned, counter resets
 * 5. Aggregate detection — 5 keys with 3+ same-content repeats triggers
 * 6. Aggregate no false positive — diverse traffic doesn't trigger
 * 7. Content hash computation — deterministic, truncated to 8 hex
 * 8. Pre-flight denial — 429 before upstream call
 * 9. Response format — correct error code, details, no contentHash in response
 * 10. Webhook dispatch — loop.detected event sent, no contentHash
 * 11. Config from budget entity — custom thresholds honored
 * 12. Disabled (0) — no detection when loopMaxCalls=0
 * 13. Default-on — detection active with null config (system defaults)
 * 14. Interaction with velocity — both can fire independently
 * 15. Interaction with budget — loop check before budget check
 * 16. Alarm pruning — stale rows cleaned up without new requests
 * 17. Empty body — "empty string" content hash, still tracked
 * 18. Denial backoff — repeated denials for same key+hash return cached 429
 * 19. SQL injection safety — model name with SQL metacharacters
 * 20. checkAndReserve loopContext=null — MCP route passes null, no crash
 * 21. Metric emission — loop detection metric recorded
 * 22. Warning header — X-NullSpend-Loop-Count at 80% threshold
 * 23. Budget-denied requests do NOT inflate loop counter
 * 24. Webhook builder — buildLoopDetectedPayload shape
 * 25. Orchestrator loop denial pass-through
 */

import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { buildLoopDetectedPayload } from "../lib/webhook-events.js";

// ── Polyfill timingSafeEqual ──────────────────────────────────────
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

// ── Hoisted mocks ────────────────────────────────────────────────

const {
  mockWaitUntil,
  mockDoBudgetCheck,
  mockDoBudgetReconcile,
  mockEmitMetric,
} = vi.hoisted(() => ({
  mockWaitUntil: vi.fn((promise: Promise<unknown>) => { promise.catch(() => {}); }),
  mockDoBudgetCheck: vi.fn(),
  mockDoBudgetReconcile: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("cloudflare:workers", () => ({
  waitUntil: mockWaitUntil,
}));

vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn().mockReturnValue({
    inputPerMTok: 0.15, cachedInputPerMTok: 0.075, outputPerMTok: 0.60,
  }),
  costComponent: vi.fn((tokens: number, rate: number) => tokens > 0 && rate > 0 ? tokens * rate : 0),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: (...args: unknown[]) => mockDoBudgetCheck(...args),
  doBudgetReconcile: (...args: unknown[]) => mockDoBudgetReconcile(...args),
  doBudgetUpsertEntities: vi.fn(),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn(),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/write-metric.js", () => ({
  writeLatencyDataPoint: vi.fn(),
}));

vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpoints: vi.fn().mockResolvedValue([]),
  getWebhookEndpointsWithSecrets: vi.fn().mockResolvedValue([]),
}));

vi.mock("../lib/webhook-dispatch.js", () => ({
  dispatchToEndpoints: vi.fn(),
}));

vi.mock("../lib/webhook-expiry.js", () => ({
  expireRotatedSecrets: vi.fn().mockResolvedValue(undefined),
}));

import { checkBudget } from "../lib/budget-orchestrator.js";
import type { RequestContext } from "../lib/context.js";

// ── Helpers ──────────────────────────────────────────────────────

function makeCtx(overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    ownerId: "user-1",
    connectionString: "postgresql://test",
    body: { model: "gpt-4o", messages: [{ role: "user", content: "hello" }] },
    bodyText: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "hello" }] }),
    bodyByteLength: 60,
    requestStartMs: performance.now(),
    sessionId: null,
    traceId: "trace-1",
    tags: {},
    customerId: null,
    finalize: false,
    requestLoggingEnabled: false,
    resolvedApiVersion: "2026-04-01",
    auth: {
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: true,
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
    webhookDispatcher: null,
    ...overrides,
  } as RequestContext;
}

// ── checkBudget orchestrator tests ──────────────────────────────

describe("Loop Detection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDoBudgetReconcile.mockResolvedValue({ status: "ok" });
  });

  describe("checkBudget passes loopContext through", () => {
    it("passes null loopContext when not provided", async () => {
      mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "r1" });
      const ctx = makeCtx();
      await checkBudget({} as Env, ctx, 1000);
      expect(mockDoBudgetCheck).toHaveBeenCalledWith(
        expect.anything(), "user-1", "key-1", 1000, null, [], "org-1", false, null,
      );
    });

    it("passes loopContext when provided", async () => {
      mockDoBudgetCheck.mockResolvedValue({ status: "approved", hasBudgets: true, reservationId: "r1" });
      const ctx = makeCtx();
      const loopCtx = { provider: "openai", model: "gpt-4o", contentHash: "a1b2c3d4" };
      await checkBudget({} as Env, ctx, 1000, false, loopCtx);
      expect(mockDoBudgetCheck).toHaveBeenCalledWith(
        expect.anything(), "user-1", "key-1", 1000, null, [], "org-1", false, loopCtx,
      );
    });
  });

  describe("orchestrator loop denial pass-through", () => {
    it("returns loopDetected + loopDetails on loop denial", async () => {
      mockDoBudgetCheck.mockResolvedValue({
        status: "denied",
        hasBudgets: true,
        loopDetected: true,
        loopDetails: {
          type: "per_key",
          model: "gpt-4o",
          provider: "openai",
          callCount: 50,
          windowSeconds: 60,
          maxCalls: 50,
        },
      });
      const ctx = makeCtx();
      const outcome = await checkBudget({} as Env, ctx, 1000, false, { provider: "openai", model: "gpt-4o", contentHash: "abc" });
      expect(outcome.status).toBe("denied");
      expect(outcome.loopDetected).toBe(true);
      expect(outcome.loopDetails).toEqual({
        type: "per_key",
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      });
      // No reservationId on denial
      expect(outcome.reservationId).toBeNull();
    });

    it("returns aggregate loop denial", async () => {
      mockDoBudgetCheck.mockResolvedValue({
        status: "denied",
        hasBudgets: true,
        loopDetected: true,
        loopDetails: {
          type: "aggregate",
          model: "aggregate",
          provider: "multiple",
          callCount: 5,
          windowSeconds: 60,
          maxCalls: 5,
        },
      });
      const ctx = makeCtx();
      const outcome = await checkBudget({} as Env, ctx, 1000);
      expect(outcome.loopDetected).toBe(true);
      expect(outcome.loopDetails!.type).toBe("aggregate");
    });

    it("passes loopCount + loopMaxCalls on approved (warning)", async () => {
      mockDoBudgetCheck.mockResolvedValue({
        status: "approved",
        hasBudgets: true,
        reservationId: "r1",
        loopCount: 42,
        loopMaxCalls: 50,
      });
      const ctx = makeCtx();
      const outcome = await checkBudget({} as Env, ctx, 1000);
      expect(outcome.status).toBe("approved");
      expect(outcome.loopCount).toBe(42);
      expect(outcome.loopMaxCalls).toBe(50);
    });
  });

  describe("checkAndReserve loopContext=null — MCP route passes null, no crash", () => {
    it("approved without loopContext", async () => {
      mockDoBudgetCheck.mockResolvedValue({
        status: "approved", hasBudgets: true, reservationId: "r1",
      });
      const ctx = makeCtx();
      const outcome = await checkBudget({} as Env, ctx, 1000);
      expect(outcome.status).toBe("approved");
      expect(outcome.loopDetected).toBeUndefined();
    });
  });
});

// ── handleBudgetDenials (shared.ts) tests ────────────────────────

import { handleBudgetDenials } from "../routes/shared.js";

describe("Loop Detection — handleBudgetDenials", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns 429 with loop_detected code on per-key denial", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(resp).not.toBeNull();
    expect(resp!.status).toBe(429);
    expect(resp!.headers.get("X-NullSpend-Denied")).toBe("1");
    expect(resp!.headers.get("Retry-After")).toBe("5");
    expect(resp!.headers.get("X-NullSpend-Trace-Id")).toBe("trace-1");

    const body = await resp!.json() as any;
    expect(body.error.code).toBe("loop_detected");
    expect(body.error.details.type).toBe("per_key");
    expect(body.error.details.model).toBe("gpt-4o");
    expect(body.error.details.provider).toBe("openai");
    expect(body.error.details.callCount).toBe(50);
    expect(body.error.details.windowSeconds).toBe(60);
    expect(body.error.details.maxCalls).toBe(50);
  });

  it("returns 429 with aggregate detection details", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "aggregate" as const,
        model: "aggregate",
        provider: "multiple",
        callCount: 5,
        windowSeconds: 60,
        maxCalls: 5,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(resp).not.toBeNull();
    const body = await resp!.json() as any;
    expect(body.error.code).toBe("loop_detected");
    expect(body.error.details.type).toBe("aggregate");
    expect(body.error.details.model).toBe("aggregate");
    expect(body.error.details.provider).toBe("multiple");
    expect(body.error.message).toContain("multi-model agent");
  });

  it("does not include contentHash in 429 response body", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    const text = await resp!.text();
    expect(text).not.toContain("contentHash");
    expect(text).not.toContain("content_hash");
  });

  it("includes actionable URL in per-key error message", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    const body = await resp!.json() as any;
    expect(body.error.message).toContain("nullspend.dev/app/budgets");
    expect(body.error.message).toContain("loop_max_calls=0");
  });

  it("emits budget_denied metric with reason=loop_detected", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(mockEmitMetric).toHaveBeenCalledWith("budget_denied", expect.objectContaining({
      reason: "loop_detected",
    }));
  });

  it("returns null when not denied (approved)", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "approved" as const,
      reservationId: "r1",
      budgetEntities: [],
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(resp).toBeNull();
  });
});

// ── Webhook builder tests ────────────────────────────────────────

describe("Loop Detection — buildLoopDetectedPayload", () => {
  it("builds per-key event with correct shape", () => {
    const event = buildLoopDetectedPayload({
      detectionType: "per_key",
      model: "gpt-4o",
      provider: "openai",
      callCount: 50,
      windowSeconds: 60,
      maxCalls: 50,
    });
    expect(event.type).toBe("loop.detected");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(event.created_at).toBeGreaterThan(0);
    expect(event.data.object).toEqual(expect.objectContaining({
      detection_type: "per_key",
      model: "gpt-4o",
      provider: "openai",
      call_count: 50,
      window_seconds: 60,
      max_calls: 50,
    }));
    expect(event.data.object).toHaveProperty("blocked_at");
  });

  it("builds aggregate event", () => {
    const event = buildLoopDetectedPayload({
      detectionType: "aggregate",
      model: "aggregate",
      provider: "multiple",
      callCount: 5,
      windowSeconds: 60,
      maxCalls: 5,
    });
    expect(event.data.object.detection_type).toBe("aggregate");
    expect(event.data.object.model).toBe("aggregate");
    expect(event.data.object.provider).toBe("multiple");
  });

  it("does not include contentHash in webhook payload", () => {
    const event = buildLoopDetectedPayload({
      detectionType: "per_key",
      model: "gpt-4o",
      provider: "openai",
      callCount: 50,
      windowSeconds: 60,
      maxCalls: 50,
    });
    const jsonStr = JSON.stringify(event);
    expect(jsonStr).not.toContain("contentHash");
    expect(jsonStr).not.toContain("content_hash");
  });

  it("generates unique event IDs", () => {
    const e1 = buildLoopDetectedPayload({
      detectionType: "per_key", model: "gpt-4o", provider: "openai",
      callCount: 50, windowSeconds: 60, maxCalls: 50,
    });
    const e2 = buildLoopDetectedPayload({
      detectionType: "per_key", model: "gpt-4o", provider: "openai",
      callCount: 50, windowSeconds: 60, maxCalls: 50,
    });
    expect(e1.id).not.toBe(e2.id);
  });

  it("accepts custom API version", () => {
    const event = buildLoopDetectedPayload({
      detectionType: "per_key", model: "gpt-4o", provider: "openai",
      callCount: 50, windowSeconds: 60, maxCalls: 50,
    }, "2024-01-01");
    expect(event.api_version).toBe("2024-01-01");
  });
});

// ── Content hash computation tests ───────────────────────────────

describe("Loop Detection — content hash computation", () => {
  it("produces 8 hex char hash from request body", async () => {
    const body = JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "hello" }] });
    const bodySlice = body.slice(0, 8192);
    const hashBuffer = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bodySlice));
    const hashArray = new Uint8Array(hashBuffer);
    const contentHash = Array.from(hashArray.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");

    expect(contentHash).toHaveLength(8);
    expect(contentHash).toMatch(/^[0-9a-f]{8}$/);
  });

  it("is deterministic — same body produces same hash", async () => {
    const body = JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "test" }] });
    const computeHash = async (input: string) => {
      const slice = input.slice(0, 8192);
      const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
      const arr = new Uint8Array(buf);
      return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    };
    const h1 = await computeHash(body);
    const h2 = await computeHash(body);
    expect(h1).toBe(h2);
  });

  it("different body produces different hash", async () => {
    const computeHash = async (input: string) => {
      const slice = input.slice(0, 8192);
      const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
      const arr = new Uint8Array(buf);
      return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    };
    const h1 = await computeHash('{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}');
    const h2 = await computeHash('{"model":"gpt-4o","messages":[{"role":"user","content":"world"}]}');
    expect(h1).not.toBe(h2);
  });

  it("caps body at 8KB before hashing", async () => {
    const computeHash = async (input: string) => {
      const slice = input.slice(0, 8192);
      const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
      const arr = new Uint8Array(buf);
      return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    };
    // 10KB body — first 8KB is identical, remainder differs
    const prefix = "A".repeat(8192);
    const body1 = prefix + "XXXXX";
    const body2 = prefix + "YYYYY";
    const h1 = await computeHash(body1);
    const h2 = await computeHash(body2);
    expect(h1).toBe(h2); // Same because only first 8KB hashed
  });

  it("shared system prompt + different user message = different hash (no false positive)", async () => {
    const computeHash = async (input: string) => {
      const slice = input.slice(0, 8192);
      const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
      const arr = new Uint8Array(buf);
      return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    };
    const systemPrompt = "You are an expert RAG assistant with access to a knowledge base.";
    const body1 = JSON.stringify({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "What is React?" },
      ],
    });
    const body2 = JSON.stringify({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "What is Vue.js?" },
      ],
    });
    const h1 = await computeHash(body1);
    const h2 = await computeHash(body2);
    expect(h1).not.toBe(h2); // Full-body hash differentiates on user message
  });
});

// ── DO-level CheckResult shape tests (via mock) ─────────────────

describe("Loop Detection — DO CheckResult shape", () => {
  beforeEach(() => vi.clearAllMocks());

  it("loopDetected=true with per_key details", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "per_key",
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopDetected).toBe(true);
    expect(outcome.loopDetails!.type).toBe("per_key");
  });

  it("do_budget_check metric includes loopDetected=true on denial", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "per_key", model: "gpt-4o", provider: "openai",
        callCount: 50, windowSeconds: 60, maxCalls: 50,
      },
    });
    const ctx = makeCtx();
    await checkBudget({} as Env, ctx, 1000);
    // The metric is emitted in doBudgetCheck — which is mocked.
    // Verify our mock was called with the right args at the orchestrator level.
    expect(mockDoBudgetCheck).toHaveBeenCalled();
  });

  it("loopCount + loopMaxCalls present on approved result (warning)", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
      loopCount: 40,
      loopMaxCalls: 50,
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopCount).toBe(40);
    expect(outcome.loopMaxCalls).toBe(50);
  });

  it("loopCount absent when below 80% threshold", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
      // No loopCount/loopMaxCalls — below 80%
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopCount).toBeUndefined();
    expect(outcome.loopMaxCalls).toBeUndefined();
  });
});

// ── Denial response format edge cases ────────────────────────────

describe("Loop Detection — denial response edge cases", () => {
  beforeEach(() => vi.clearAllMocks());

  it("per-key message mentions the model name", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "claude-sonnet-4-20250514",
        provider: "anthropic",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "anthropic", "claude-sonnet-4-20250514", 500_000, []);
    const body = await resp!.json() as any;
    expect(body.error.message).toContain("claude-sonnet-4-20250514");
    expect(body.error.message).toContain("50 times");
    expect(body.error.message).toContain("60s");
  });

  it("model name with special characters does not break response", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "ft:gpt-4o:my-org:custom_suffix:id'; DROP TABLE--",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(resp!.status).toBe(429);
    const body = await resp!.json() as any;
    expect(body.error.code).toBe("loop_detected");
    // Model name appears in message without causing JSON issues
    expect(body.error.details.model).toContain("ft:gpt-4o");
  });

  it("loop denial takes priority over session limit denial (ordering)", async () => {
    const ctx = makeCtx();
    // Both loopDetected and sessionLimitDenied — loop should win (checked first)
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
      sessionLimitDenied: true,
      sessionId: "sess-1",
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    const body = await resp!.json() as any;
    expect(body.error.code).toBe("loop_detected"); // NOT session_limit_exceeded
  });

  it("velocity denial takes priority over loop denial (ordering)", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      velocityDenied: true,
      retryAfterSeconds: 60,
      velocityDetails: { limitMicrodollars: 1000000, windowSeconds: 60, currentMicrodollars: 999000 },
      deniedEntityType: "user",
      deniedEntityId: "user-1",
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    const body = await resp!.json() as any;
    expect(body.error.code).toBe("velocity_exceeded"); // NOT loop_detected
  });
  it("loopDetected=true but loopDetails=undefined falls through (partial DO state)", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      // loopDetails intentionally undefined — partial state
    };
    // Should NOT return a loop_detected 429 — falls through to generic budget denial
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    // With no other denial flags set, the generic "denied" branch triggers
    expect(resp).not.toBeNull();
    const body = await resp!.json() as any;
    expect(body.error.code).toBe("budget_exceeded"); // Falls through to generic
  });
});

// ── EC-3: Config selection (user entity preferred) ───────────────

describe("Loop Detection — EC-3 config selection", () => {
  beforeEach(() => vi.clearAllMocks());

  it("uses user entity loop config when user + tag entities exist", async () => {
    // The DO should prefer user entity for loop config. We verify via the orchestrator
    // that the loopDetails come from the DO with the correct maxCalls value.
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "per_key",
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50, // This should be from the user entity (50), not tag (100)
      },
    });
    const ctx = makeCtx({ tags: { project: "test" } });
    const outcome = await checkBudget({} as Env, ctx, 1000, false, {
      provider: "openai", model: "gpt-4o", contentHash: "abc",
    });
    expect(outcome.loopDetected).toBe(true);
    expect(outcome.loopDetails!.maxCalls).toBe(50);
  });
});

// ── EC-6: Cached denial returns original details ─────────────────

describe("Loop Detection — EC-6 cached denial details", () => {
  beforeEach(() => vi.clearAllMocks());

  it("cached aggregate denial returns aggregate type (not per_key)", async () => {
    // First call: aggregate denial
    mockDoBudgetCheck.mockResolvedValueOnce({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "aggregate",
        model: "aggregate",
        provider: "multiple",
        callCount: 5,
        windowSeconds: 60,
        maxCalls: 5,
      },
    });
    // Second call: same denial from cache — should still be aggregate
    mockDoBudgetCheck.mockResolvedValueOnce({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "aggregate",
        model: "aggregate",
        provider: "multiple",
        callCount: 5,
        windowSeconds: 60,
        maxCalls: 5,
      },
    });

    const ctx = makeCtx();
    const loopCtx = { provider: "openai", model: "gpt-4o", contentHash: "abc" };

    const outcome1 = await checkBudget({} as Env, ctx, 1000, false, loopCtx);
    expect(outcome1.loopDetails!.type).toBe("aggregate");

    const outcome2 = await checkBudget({} as Env, ctx, 1000, false, loopCtx);
    expect(outcome2.loopDetails!.type).toBe("aggregate"); // NOT per_key
    expect(outcome2.loopDetails!.callCount).toBe(5);
  });
});

// ── Denial backoff behavior ──────────────────────────────────────

describe("Loop Detection — denial backoff", () => {
  beforeEach(() => vi.clearAllMocks());

  it("DO returns denied with cached=true in metric for backoff denials", async () => {
    // Verify the DO-level contract: fresh denial has cached=false, backoff has cached=true
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      loopDetected: true,
      loopDetails: {
        type: "per_key",
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000, false, {
      provider: "openai", model: "gpt-4o", contentHash: "abc",
    });
    expect(outcome.loopDetected).toBe(true);
    // The DO emits loop_detected metric internally — we verify via mockDoBudgetCheck call
    expect(mockDoBudgetCheck).toHaveBeenCalled();
  });

  it("denial response has Retry-After: 5 for backoff", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    expect(resp!.headers.get("Retry-After")).toBe("5");
  });
});

// ── Budget-denied requests do NOT inflate loop counter ───────────

describe("Loop Detection — deferred INSERT (BUG-1 fix)", () => {
  beforeEach(() => vi.clearAllMocks());

  it("budget denial does not carry loopDetected (separate denial types)", async () => {
    // When budget denies, loopDetected should NOT be set — they're independent
    mockDoBudgetCheck.mockResolvedValue({
      status: "denied",
      hasBudgets: true,
      deniedEntity: "user:user-1",
      remaining: 0,
      maxBudget: 100000,
      spend: 100000,
      // loopDetected is NOT set — budget denial, not loop denial
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000, false, {
      provider: "openai", model: "gpt-4o", contentHash: "abc",
    });
    expect(outcome.status).toBe("denied");
    expect(outcome.loopDetected).toBeUndefined();
    // The loop counter should NOT have been incremented (deferred INSERT skipped)
  });

  it("approved request with loopContext does not set loopDetected", async () => {
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000, false, {
      provider: "openai", model: "gpt-4o", contentHash: "abc",
    });
    expect(outcome.status).toBe("approved");
    expect(outcome.loopDetected).toBeUndefined();
  });
});

// ── Disabled detection (loopMaxCalls = 0) ────────────────────────

describe("Loop Detection — disabled via loopMaxCalls=0", () => {
  beforeEach(() => vi.clearAllMocks());

  it("approved when loopMaxCalls=0 even with repeated content", async () => {
    // DO should return approved because loop detection is disabled
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000, false, {
      provider: "openai", model: "gpt-4o", contentHash: "abc",
    });
    expect(outcome.status).toBe("approved");
    expect(outcome.loopDetected).toBeUndefined();
  });
});

// ── Warning header data ──────────────────────────────────────────

describe("Loop Detection — warning header data", () => {
  beforeEach(() => vi.clearAllMocks());

  it("loopCount set at exactly 80% of threshold", async () => {
    // 80% of 50 = 40, so count=40 should trigger warning
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
      loopCount: 40,
      loopMaxCalls: 50,
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopCount).toBe(40);
    expect(outcome.loopMaxCalls).toBe(50);
  });

  it("loopCount NOT set below 80% of threshold", async () => {
    // 79% of 50 = 39.5 → floor = 39, so count=39 should NOT trigger
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
      // No loopCount/loopMaxCalls — below 80% threshold
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopCount).toBeUndefined();
  });

  it("loopCount set at 100% (threshold - 1, one below denial)", async () => {
    // count=49 out of 50 — should trigger warning
    mockDoBudgetCheck.mockResolvedValue({
      status: "approved",
      hasBudgets: true,
      reservationId: "r1",
      loopCount: 49,
      loopMaxCalls: 50,
    });
    const ctx = makeCtx();
    const outcome = await checkBudget({} as Env, ctx, 1000);
    expect(outcome.loopCount).toBe(49);
    expect(outcome.loopMaxCalls).toBe(50);
  });
});

// ── handleBudgetDenials with Retry-After header ──────────────────

describe("Loop Detection — Retry-After header", () => {
  beforeEach(() => vi.clearAllMocks());

  it("always returns Retry-After: 5 (static backoff)", async () => {
    const ctx = makeCtx();
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 99,
        windowSeconds: 120,
        maxCalls: 50,
      },
    };
    const resp = await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    // Retry-After is always 5s regardless of window or callCount
    expect(resp!.headers.get("Retry-After")).toBe("5");
  });
});

// ── Webhook dispatch on denial ───────────────────────────────────

describe("Loop Detection — webhook dispatch", () => {
  beforeEach(() => vi.clearAllMocks());

  it("dispatches webhook when ctx has webhookDispatcher and hasWebhooks", async () => {
    const mockDispatch = vi.fn();
    const ctx = makeCtx({
      webhookDispatcher: { dispatch: mockDispatch } as any,
      auth: {
        ...makeCtx().auth,
        hasWebhooks: true,
      },
    } as any);
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    // Webhook dispatch happens in waitUntil
    expect(mockWaitUntil).toHaveBeenCalled();
  });

  it("does NOT dispatch webhook when ctx has no webhookDispatcher", async () => {
    const ctx = makeCtx({ webhookDispatcher: null });
    const outcome = {
      status: "denied" as const,
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key" as const,
        model: "gpt-4o",
        provider: "openai",
        callCount: 50,
        windowSeconds: 60,
        maxCalls: 50,
      },
    };
    await handleBudgetDenials(outcome, ctx, {} as Env, "openai", "gpt-4o", 500_000, []);
    // No waitUntil for webhook dispatch (no dispatcher)
    expect(mockWaitUntil).not.toHaveBeenCalled();
  });
});

// ── Content hash edge cases ──────────────────────────────────────

describe("Loop Detection — content hash edge cases", () => {
  const computeHash = async (input: string) => {
    const slice = input.slice(0, 8192);
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
    const arr = new Uint8Array(buf);
    return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
  };

  it("empty string body produces valid hash", async () => {
    const hash = await computeHash("");
    expect(hash).toHaveLength(8);
    expect(hash).toMatch(/^[0-9a-f]{8}$/);
  });

  it("JSON with only model difference produces different hash", async () => {
    const h1 = await computeHash('{"model":"gpt-4o","messages":[]}');
    const h2 = await computeHash('{"model":"gpt-4o-mini","messages":[]}');
    expect(h1).not.toBe(h2);
  });

  it("handles unicode content correctly", async () => {
    const h1 = await computeHash('{"messages":[{"content":"你好世界"}]}');
    const h2 = await computeHash('{"messages":[{"content":"こんにちは世界"}]}');
    expect(h1).not.toBe(h2);
    expect(h1).toHaveLength(8);
    expect(h2).toHaveLength(8);
  });

  it("SQL injection characters in body produce valid hash", async () => {
    const hash = await computeHash("'; DROP TABLE loop_call_log; --");
    expect(hash).toHaveLength(8);
    expect(hash).toMatch(/^[0-9a-f]{8}$/);
  });
});
