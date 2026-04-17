/**
 * Contract tests for estimate input validation (P0-4 + NF-2).
 *
 * In-process port of the live-smoke suite at
 * apps/proxy/smoke-estimate-validation.test.ts. Happy-path tests use
 * MSW handlers populated from recorded cassettes (OpenAI gpt-4o-mini,
 * Anthropic claude-3-haiku). The 422 invalid_estimate test (previously
 * tautological — asserted a constant against itself) is now a real
 * test: mock `checkBudget` to return the NF-2 denial shape, call the
 * route handler, assert HTTP 422 + code "invalid_estimate".
 *
 * DB round-trip regression guards ("cost event lands in DB after
 * sanitization fallback") are preserved separately in
 * apps/proxy/smoke-estimate-validation-db.test.ts (nightly SMOKE_LIVE=1).
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

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
import { handleAnthropicMessages } from "../routes/anthropic.js";
import {
  makeContractEnv,
  makeContractCtx,
  TEST_TRACE_ID,
} from "./msw/contract-helpers.js";
import { server } from "./msw/server.js";
import { openaiChatCompletionHandler } from "./msw/openai-handlers.js";
import { anthropicMessagesHandler } from "./msw/anthropic-handlers.js";

const openaiCassette = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("./fixtures/cassettes/openai-chat-completion.json", import.meta.url)),
    "utf-8",
  ),
);
const anthropicCassette = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("./fixtures/cassettes/anthropic-messages.json", import.meta.url)),
    "utf-8",
  ),
);

function makeOpenAIRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-test-proxy-upstream",
    },
    body: JSON.stringify(body),
  });
}

function makeAnthropicRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "sk-ant-test-proxy-upstream",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(body),
  });
}

describe("Estimate input validation (P0-4 + NF-2) — contract", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockCheckBudget.mockReset();
    // Default: no budgets configured → skipped path, upstream reached.
    mockCheckBudget.mockResolvedValue({
      status: "skipped",
      reservationId: null,
      budgetEntities: [],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    server.resetHandlers();
  });

  // ── OpenAI path: 8 invalid shapes × does-not-crash ───────────────────

  it.each([
    ["negative", -1],
    ["string 'NaN'", "NaN"],
    ["string 'Infinity'", "Infinity"],
    ["boolean true", true],
    ["boolean false", false],
    ["array [100]", [100]],
    ["empty object", {}],
    ["null", null],
  ])(
    "OpenAI: max_tokens = %s does not crash — proxy returns 200 via sanitization fallback",
    async (_label, value) => {
      server.use(openaiChatCompletionHandler(openaiCassette));
      const body = {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "say ok" }],
        max_tokens: value,
      };
      const res = await handleChatCompletions(
        makeOpenAIRequest(body),
        makeContractEnv(),
        makeContractCtx(body),
      );
      expect(res.status).not.toBe(500);
      expect(res.status).not.toBe(502);
      expect(res.status).not.toBe(422);
    },
  );

  it("OpenAI: max_completion_tokens: -1 takes the same fall-through path", async () => {
    server.use(openaiChatCompletionHandler(openaiCassette));
    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "say ok" }],
      max_completion_tokens: -1,
    };
    const res = await handleChatCompletions(
      makeOpenAIRequest(body),
      makeContractEnv(),
      makeContractCtx(body),
    );
    expect(res.status).not.toBe(500);
    expect(res.status).not.toBe(422);
  });

  it("OpenAI: valid max_tokens → proxy enqueues cost event (queue-mock proof; DB round-trip lives in nightly smoke)", async () => {
    const cachedBatch = await import("../lib/cost-event-queue.js");
    const enqueueSpy = vi.mocked(cachedBatch.logCostEventQueued);
    enqueueSpy.mockClear();

    server.use(openaiChatCompletionHandler(openaiCassette));
    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "ok" }],
      max_tokens: 5,
    };
    const res = await handleChatCompletions(
      makeOpenAIRequest(body),
      makeContractEnv(),
      makeContractCtx(body),
    );
    expect(res.status).toBe(200);
    await res.text();
    expect(enqueueSpy).toHaveBeenCalled();
  });

  // ── Anthropic path: 5 invalid shapes × does-not-crash ────────────────

  it.each([
    ["negative", -1],
    ["string 'NaN'", "NaN"],
    ["boolean true", true],
    ["array [100]", [100]],
    ["empty object", {}],
  ])(
    "Anthropic: max_tokens = %s does not crash",
    async (_label, value) => {
      server.use(anthropicMessagesHandler(anthropicCassette));
      const body = {
        model: "claude-3-haiku-20240307",
        messages: [{ role: "user", content: "say ok" }],
        max_tokens: value,
      };
      const res = await handleAnthropicMessages(
        makeAnthropicRequest(body),
        makeContractEnv(),
        makeContractCtx(body),
      );
      expect(res.status).not.toBe(500);
      expect(res.status).not.toBe(502);
      expect(res.status).not.toBe(422);
    },
  );

  it("Anthropic: valid max_tokens → proxy enqueues cost event (queue-mock proof; DB round-trip lives in nightly smoke)", async () => {
    const cachedBatch = await import("../lib/cost-event-queue.js");
    const enqueueSpy = vi.mocked(cachedBatch.logCostEventQueued);
    enqueueSpy.mockClear();

    server.use(anthropicMessagesHandler(anthropicCassette));
    const body = {
      model: "claude-3-haiku-20240307",
      messages: [{ role: "user", content: "ok" }],
      max_tokens: 10,
    };
    const res = await handleAnthropicMessages(
      makeAnthropicRequest(body),
      makeContractEnv(),
      makeContractCtx(body),
    );
    expect(res.status).toBe(200);
    await res.text();
    expect(enqueueSpy).toHaveBeenCalled();
  });

  // ── NF-2: 422 invalid_estimate path (REAL test, was tautological in smoke) ──

  it("OpenAI: 422 invalid_estimate fires when orchestrator flags invalid estimate", async () => {
    // Override checkBudget for this test: simulate NF-2 detection.
    mockCheckBudget.mockResolvedValueOnce({
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      invalidEstimate: true,
    });

    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "trigger nf-2" }],
      max_tokens: 5,
    };
    const res = await handleChatCompletions(
      makeOpenAIRequest(body),
      makeContractEnv(),
      makeContractCtx(body),
    );

    expect(res.status).toBe(422);
    const payload = await res.json() as {
      error: { code: string; message: string; details: null };
    };
    expect(payload.error.code).toBe("invalid_estimate");
    expect(payload.error.details).toBeNull();
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe(TEST_TRACE_ID);
    expect(res.headers.get("X-NullSpend-Denied")).toBe("1");
  });

  it("Anthropic: 422 invalid_estimate fires via same orchestrator path", async () => {
    mockCheckBudget.mockResolvedValueOnce({
      status: "denied",
      reservationId: null,
      budgetEntities: [],
      invalidEstimate: true,
    });

    const body = {
      model: "claude-3-haiku-20240307",
      messages: [{ role: "user", content: "trigger nf-2" }],
      max_tokens: 5,
    };
    const res = await handleAnthropicMessages(
      makeAnthropicRequest(body),
      makeContractEnv(),
      makeContractCtx(body),
    );

    expect(res.status).toBe(422);
    const payload = await res.json() as {
      error: { code: string; details: null };
    };
    expect(payload.error.code).toBe("invalid_estimate");
    expect(payload.error.details).toBeNull();
  });
});
