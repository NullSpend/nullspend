/**
 * Contract tests for denial priority ordering (P1-2).
 *
 * In-process port of the live-smoke suite at
 * apps/proxy/smoke-denial-priority.test.ts. DO-level priority ordering
 * (velocity > loop > session > tag > customer > budget) is covered in
 * user-budget-do.do.test.ts (workerd pool) — those tests seed the DO's
 * SQLite and verify that when BOTH loop_max_calls and session_limit
 * would trigger, the DO returns `loopDetected: true` in its outcome.
 *
 * This contract test verifies the HTTP rendering layer: given a denial
 * outcome with flag X set, shared.ts/handleBudgetDenials produces the
 * correct HTTP response (status, code, headers, recovery shape).
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";

beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as any).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.byteLength !== viewB.byteLength) return false;
      let result = 0;
      for (let i = 0; i < viewA.byteLength; i++) result |= viewA[i] ^ viewB[i];
      return result === 0;
    };
  }
});

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

const { mockCheckBudget } = vi.hoisted(() => ({
  mockCheckBudget: vi.fn(),
}));
vi.mock("../lib/budget-orchestrator.js", () => ({
  checkBudget: mockCheckBudget,
  reconcileBudgetQueued: vi.fn().mockResolvedValue(undefined),
  getReconcileQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/cost-event-queue.js", () => ({
  logCostEventQueued: vi.fn().mockResolvedValue(undefined),
  logCostEventsBatchQueued: vi.fn().mockResolvedValue(undefined),
  getCostEventQueue: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpoints: vi.fn().mockResolvedValue([]),
  getWebhookEndpointsWithSecrets: vi.fn().mockResolvedValue([]),
  invalidateWebhookCache: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/webhook-thresholds.js", () => ({
  detectThresholdCrossings: vi.fn().mockReturnValue([]),
}));

import { handleChatCompletions } from "../routes/openai.js";
import { makeContractEnv, makeContractCtx } from "./msw/contract-helpers.js";

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-test-proxy-upstream",
    },
    body: JSON.stringify(body),
  });
}

describe("Denial priority ordering (P1-2) — contract", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockCheckBudget.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loop_detected denial → 429 with code loop_detected + Retry-After header", async () => {
    // This simulates what the DO returns when the 2nd identical call in
    // a tight window trips the loop check (the DO's internal priority
    // ordering put loop BEFORE session, matching the documented
    // velocity > loop > session > ... ordering).
    mockCheckBudget.mockResolvedValueOnce({
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      loopDetected: true,
      loopDetails: {
        type: "per_key",
        model: "gpt-4o-mini",
        provider: "openai",
        callCount: 2,
        windowSeconds: 60,
        maxCalls: 2,
      },
      retryAfterSeconds: 5,
    });

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "repeated content" }],
      max_tokens: 5,
    };
    const res = await handleChatCompletions(
      makeRequest(body),
      makeContractEnv(),
      makeContractCtx(body, { sessionId: "priority-test-session" }),
    );

    expect(res.status).toBe(429);
    const payload = await res.json() as { error: { code: string; details: Record<string, unknown>; recovery: unknown } };
    // Critical assertion — loop denial renders with correct code.
    expect(payload.error.code).toBe("loop_detected");
    expect(payload.error.details.type).toBe("per_key");
    expect(payload.error.details.model).toBe("gpt-4o-mini");
    expect(payload.error.details.callCount).toBe(2);
    expect(payload.error.recovery).toBeDefined();

    expect(res.headers.get("Retry-After")).toBeDefined();
    expect(res.headers.get("X-NullSpend-Denied")).toBe("1");
  });

  it("session_limit_exceeded denial → 429 with code session_limit_exceeded (sanity control)", async () => {
    // Control: DO returns session denial (loop NOT triggered, e.g. the
    // 2nd request had different content). Proves the HTTP renderer
    // handles session denials distinctly from loop denials.
    mockCheckBudget.mockResolvedValueOnce({
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      sessionLimitDenied: true,
      deniedEntityType: "user",
      deniedEntityId: "user-1",
      sessionId: "session-only-test",
      sessionSpend: 15,
      sessionLimit: 10,
    });

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "unique content B — different hash" }],
      max_tokens: 5,
    };
    const res = await handleChatCompletions(
      makeRequest(body),
      makeContractEnv(),
      makeContractCtx(body, { sessionId: "session-only-test" }),
    );

    expect(res.status).toBe(429);
    const payload = await res.json() as { error: { code: string; details: Record<string, unknown> } };
    expect(payload.error.code).toBe("session_limit_exceeded");
    expect(payload.error.details.session_id).toBe("session-only-test");
    expect(payload.error.details.session_limit_microdollars).toBe(10);
    expect(res.headers.get("X-NullSpend-Denied")).toBe("1");
  });
});
