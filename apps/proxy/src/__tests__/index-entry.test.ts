/**
 * Integration tests for the main Worker entry point (index.ts).
 * Tests the full request→route→response flow without hitting OpenAI,
 * covering body parsing, JSON validation, fail-closed behavior, body size limits, and routing logic.
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
      for (let i = 0; i < viewA.byteLength; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

const mockIpLimit = vi.fn().mockResolvedValue({ success: true });
const mockKeyLimit = vi.fn().mockResolvedValue({ success: true });

vi.mock("@nullspend/cost-engine", () => ({
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn().mockReturnValue(null),
  costComponent: vi.fn().mockReturnValue(0),
}));

const mockAuthenticateRequest = vi.fn();
vi.mock("../lib/auth.js", () => ({
  authenticateRequest: (...args: unknown[]) => mockAuthenticateRequest(...args),
}));

vi.mock("../lib/webhook-dispatch.js", () => ({
  createWebhookDispatcher: vi.fn().mockReturnValue(null),
}));
vi.mock("../lib/budget-orchestrator.js", () => ({
  checkBudget: vi.fn().mockResolvedValue({ status: "skipped", reservationId: null, budgetEntities: [] }),
  reconcileBudgetQueued: vi.fn().mockResolvedValue(undefined),
  getReconcileQueue: vi.fn().mockReturnValue(undefined),
}));

const mockHandleReconciliationQueue = vi.fn().mockResolvedValue(undefined);
const mockHandleDlqQueue = vi.fn().mockResolvedValue(undefined);
const mockHandleCostEventQueue = vi.fn().mockResolvedValue(undefined);
const mockHandleCostEventDlq = vi.fn().mockResolvedValue(undefined);

vi.mock("../queue-handler.js", () => ({
  handleReconciliationQueue: (...args: unknown[]) => mockHandleReconciliationQueue(...args),
}));
vi.mock("../dlq-handler.js", () => ({
  handleDlqQueue: (...args: unknown[]) => mockHandleDlqQueue(...args),
  DLQ_QUEUE_NAME: "nullspend-reconcile-dlq",
}));
vi.mock("../cost-event-queue-handler.js", () => ({
  handleCostEventQueue: (...args: unknown[]) => mockHandleCostEventQueue(...args),
  COST_EVENT_QUEUE_NAME: "nullspend-cost-events",
}));
vi.mock("../cost-event-dlq-handler.js", () => ({
  handleCostEventDlq: (...args: unknown[]) => mockHandleCostEventDlq(...args),
  COST_EVENT_DLQ_NAME: "nullspend-cost-events-dlq",
}));

import entrypoint from "../index.js";

function makeEnv(): Env {
  return {
    HYPERDRIVE: { connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres" },
    IP_RATE_LIMITER: { limit: mockIpLimit },
    KEY_RATE_LIMITER: { limit: mockKeyLimit },
    CACHE_KV: { get: vi.fn().mockResolvedValue(null), put: vi.fn() },
  } as unknown as Env;
}

function makeCtx(): ExecutionContext {
  return {
    waitUntil: vi.fn(),
    passThroughOnException: vi.fn(),
  } as unknown as ExecutionContext;
}

describe("Worker entry point routing", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockAuthenticateRequest.mockReset();
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: null,
      apiVersion: "2026-04-01", defaultTags: {},
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  describe("health endpoints", () => {
    it("GET /health returns 200 with service info", async () => {
      const req = new Request("http://localhost/health");
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");
      expect(body.service).toBe("nullspend-proxy");
    });

    it("GET /health/metrics returns 200 with metrics shape", async () => {
      const req = new Request("http://localhost/health/metrics");
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("overhead_ms");
      expect(body).toHaveProperty("request_count");
      expect(body).toHaveProperty("window_seconds");
      expect(body).toHaveProperty("measured_at");
    });

  });

  describe("body parsing", () => {
    it("empty body returns 400 with bad_request", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
    });

    it("non-JSON body returns 400", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not json {{{",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("bad_request");
      expect(body.error.message).toContain("Invalid JSON");
    });

    it("JSON array returns 400 (must be object)", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "[1, 2, 3]",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.message).toContain("JSON object");
    });

    it("JSON null returns 400", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "null",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
    });

    it("JSON string returns 400", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: '"hello"',
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
    });

    it("JSON number returns 400", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "42",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(400);
    });

    it("deeply nested valid JSON object passes body validation", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "hi" }],
          nested: { deep: { value: true } },
        }),
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
    });
  });

  describe("routing", () => {
    it("GET /v1/chat/completions returns 404 (POST only)", async () => {
      const req = new Request("http://localhost/v1/chat/completions", { method: "GET" });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
    });

    it("POST /v1/embeddings returns 404 (not supported)", async () => {
      const req = new Request("http://localhost/v1/embeddings", {
        method: "POST",
        body: "{}",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
      const body = await res.json();
      expect(body.error.code).toBe("not_found");
      expect(body.error.message).toContain("not yet supported");
    });

    it("POST /v1/audio/transcriptions returns 404", async () => {
      const req = new Request("http://localhost/v1/audio/transcriptions", {
        method: "POST",
        body: "{}",
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
    });

    it("unknown root path returns 404", async () => {
      const req = new Request("http://localhost/unknown");
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
      const body = await res.json();
      expect(body.error.code).toBe("not_found");
    });

    it("root path returns 404", async () => {
      const req = new Request("http://localhost/");
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
    });

    it("POST /v1/messages reaches Anthropic handler", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            id: "msg_test",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-20250514",
            content: [{ type: "text", text: "Hello" }],
            usage: { input_tokens: 10, output_tokens: 5 },
          }),
          { status: 200, headers: { "content-type": "application/json", "request-id": "req_test123" } },
        ),
      );

      const req = new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-ant-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 100,
          messages: [{ role: "user", content: "hi" }],
        }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.model).toBe("claude-sonnet-4-20250514");
    });

    it("GET /v1/messages returns 404 (POST only)", async () => {
      const req = new Request("http://localhost/v1/messages", { method: "GET" });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(404);
    });
  });

  describe("fail-closed behavior", () => {
    it("returns 500 when route handler throws (never forwards to origin)", async () => {
      globalThis.fetch = vi.fn().mockImplementation(async () => {
        throw new Error("Simulated internal error");
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "hi" }],
        }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(500);
      const body = await res.json();
      expect(body.error.code).toBe("internal_error");
      expect(res.headers.get("X-NullSpend-Trace-Id")).toMatch(/^[0-9a-f]{32}$/);
    });

    it("does not call passThroughOnException", async () => {
      const ctx = makeCtx();
      const req = new Request("http://localhost/health");
      await entrypoint.fetch(req, makeEnv(), ctx);
      expect(ctx.passThroughOnException).not.toHaveBeenCalled();
    });

    it("returns 401 when authentication fails", async () => {
      mockAuthenticateRequest.mockResolvedValueOnce(null);
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(401);
      const body = await res.json();
      expect(body.error.code).toBe("unauthorized");
    });
  });

  describe("effective tags header", () => {
    it("sets X-NullSpend-Effective-Tags when tags are present", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      mockAuthenticateRequest.mockResolvedValueOnce({
        userId: "user-1",
        keyId: "key-1",
        hasWebhooks: false,
        hasBudgets: false,
        orgId: null,
        apiVersion: "2026-04-01",
        defaultTags: { project: "openclaw" },
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const effectiveTags = res.headers.get("X-NullSpend-Effective-Tags");
      expect(effectiveTags).toBeTruthy();
      expect(JSON.parse(effectiveTags!)).toEqual({ project: "openclaw" });
    });

    it("does not set X-NullSpend-Effective-Tags when no tags", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      expect(res.headers.get("X-NullSpend-Effective-Tags")).toBeNull();
    });

    it("merges default tags with request tags (request wins)", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      mockAuthenticateRequest.mockResolvedValueOnce({
        userId: "user-1",
        keyId: "key-1",
        hasWebhooks: false,
        hasBudgets: false,
        orgId: null,
        apiVersion: "2026-04-01",
        defaultTags: { project: "openclaw", team: "backend" },
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
          "X-NullSpend-Tags": '{"project":"other","env":"prod"}',
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const effectiveTags = JSON.parse(res.headers.get("X-NullSpend-Effective-Tags")!);
      expect(effectiveTags).toEqual({
        project: "other",  // request wins
        team: "backend",   // from defaults
        env: "prod",       // from request
      });
    });
  });

  describe("customer header", () => {
    it("sets X-NullSpend-Warning when customer header is invalid", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      mockAuthenticateRequest.mockResolvedValueOnce({
        userId: "user-1",
        keyId: "key-1",
        hasWebhooks: false,
        hasBudgets: false,
        orgId: null,
        apiVersion: "2026-04-01",
        defaultTags: {},
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
          "X-NullSpend-Customer": "acme corp invalid!",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      expect(res.headers.get("X-NullSpend-Warning")).toBe("invalid_customer");
    });

    it("does not set X-NullSpend-Warning for valid customer header", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      mockAuthenticateRequest.mockResolvedValueOnce({
        userId: "user-1",
        keyId: "key-1",
        hasWebhooks: false,
        hasBudgets: false,
        orgId: null,
        apiVersion: "2026-04-01",
        defaultTags: {},
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
          "X-NullSpend-Customer": "acme-corp",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      expect(res.headers.get("X-NullSpend-Warning")).toBeNull();
    });

    it("auto-injects customer into effective tags", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      mockAuthenticateRequest.mockResolvedValueOnce({
        userId: "user-1",
        keyId: "key-1",
        hasWebhooks: false,
        hasBudgets: false,
        orgId: null,
        apiVersion: "2026-04-01",
        defaultTags: {},
      });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
          "X-NullSpend-Customer": "acme-corp",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
      const effectiveTags = JSON.parse(res.headers.get("X-NullSpend-Effective-Tags")!);
      expect(effectiveTags.customer).toBe("acme-corp");
    });
  });

  describe("body size limits", () => {
    it("rejects requests with Content-Length exceeding 1MB", async () => {
      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": "2000000",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(413);
      const body = await res.json();
      expect(body.error.code).toBe("payload_too_large");
    });

    it("allows requests with Content-Length under 1MB", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).not.toBe(413);
    });
  });

  describe("rate limiting", () => {
    it("returns 429 when rate limit is exceeded", async () => {
      mockIpLimit.mockResolvedValueOnce({ success: false });

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(429);
      const body = await res.json();
      expect(body.error.code).toBe("rate_limited");
      expect(res.headers.get("Retry-After")).toBe("60");
    });

    it("continues processing when rate limit is not exceeded", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
    });

    it("continues processing when rate limiter throws (fail-open for availability)", async () => {
      mockIpLimit.mockRejectedValueOnce(new Error("Rate limiter binding error"));

      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ choices: [], model: "gpt-4o-mini" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );

      const req = new Request("http://localhost/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer sk-test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), makeCtx());
      expect(res.status).toBe(200);
    });
  });
});

describe("Queue routing", () => {
  function makeBatch(queueName: string): MessageBatch<any> {
    return {
      queue: queueName,
      messages: [],
      ackAll: vi.fn(),
      retryAll: vi.fn(),
    };
  }

  beforeEach(() => {
    mockHandleReconciliationQueue.mockClear();
    mockHandleDlqQueue.mockClear();
    mockHandleCostEventQueue.mockClear();
    mockHandleCostEventDlq.mockClear();
  });

  it("routes cost event queue to handleCostEventQueue", async () => {
    const batch = makeBatch("nullspend-cost-events");
    await entrypoint.queue(batch, makeEnv());

    expect(mockHandleCostEventQueue).toHaveBeenCalledTimes(1);
    expect(mockHandleCostEventQueue).toHaveBeenCalledWith(batch, expect.anything());
    expect(mockHandleReconciliationQueue).not.toHaveBeenCalled();
    expect(mockHandleDlqQueue).not.toHaveBeenCalled();
    expect(mockHandleCostEventDlq).not.toHaveBeenCalled();
  });

  it("routes cost event DLQ to handleCostEventDlq", async () => {
    const batch = makeBatch("nullspend-cost-events-dlq");
    await entrypoint.queue(batch, makeEnv());

    expect(mockHandleCostEventDlq).toHaveBeenCalledTimes(1);
    expect(mockHandleCostEventDlq).toHaveBeenCalledWith(batch, expect.anything());
    expect(mockHandleCostEventQueue).not.toHaveBeenCalled();
    expect(mockHandleReconciliationQueue).not.toHaveBeenCalled();
    expect(mockHandleDlqQueue).not.toHaveBeenCalled();
  });

  it("routes reconciliation DLQ to handleDlqQueue", async () => {
    const batch = makeBatch("nullspend-reconcile-dlq");
    await entrypoint.queue(batch, makeEnv());

    expect(mockHandleDlqQueue).toHaveBeenCalledTimes(1);
    expect(mockHandleReconciliationQueue).not.toHaveBeenCalled();
    expect(mockHandleCostEventQueue).not.toHaveBeenCalled();
    expect(mockHandleCostEventDlq).not.toHaveBeenCalled();
  });

  it("routes reconciliation queue to handleReconciliationQueue", async () => {
    const batch = makeBatch("nullspend-reconcile");
    await entrypoint.queue(batch, makeEnv());

    expect(mockHandleReconciliationQueue).toHaveBeenCalledTimes(1);
    expect(mockHandleDlqQueue).not.toHaveBeenCalled();
    expect(mockHandleCostEventQueue).not.toHaveBeenCalled();
    expect(mockHandleCostEventDlq).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Regression: Codex audit P1 #7 — auth DB error returns 503, not 401
  // -------------------------------------------------------------------------

  describe("auth DB error handling", () => {
    it("returns 503 service_unavailable when authenticateRequest throws (DB down)", async () => {
      mockAuthenticateRequest.mockRejectedValue(new Error("ECONNREFUSED"));

      const req = new Request("https://proxy.example.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "x-nullspend-key": "ns_live_sk_test",
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "test" }] }),
      });

      const res = await entrypoint.fetch(req, makeEnv(), { waitUntil: vi.fn(), passThroughOnException: vi.fn() } as unknown as ExecutionContext);
      expect(res.status).toBe(503);
      const body = await res.json() as { error: { code: string } };
      expect(body.error.code).toBe("service_unavailable");
    });
  });

  // -------------------------------------------------------------------------
  // Regression: Codex audit P1 #8 — Unicode body byte length
  // -------------------------------------------------------------------------

  describe("body byte length accuracy", () => {
    it("computes UTF-8 byte length, not JS string length, for CJK content", async () => {
      // CJK characters are 3 bytes each in UTF-8 but 1 char in JS
      const cjkBody = JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "你好世界" }] });
      const jsLength = cjkBody.length;
      const utf8Length = new TextEncoder().encode(cjkBody).byteLength;
      // Sanity: UTF-8 byte count MUST be larger for CJK content
      expect(utf8Length).toBeGreaterThan(jsLength);

      mockAuthenticateRequest.mockResolvedValue({
        userId: "user-1", keyId: "key-1", hasWebhooks: false, hasBudgets: false,
        orgId: "org-1", apiVersion: "2026-04-01", defaultTags: {},
      });
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          model: "gpt-4o",
          usage: { prompt_tokens: 10, completion_tokens: 5 },
          choices: [{ message: { content: "hi" } }],
        }), { headers: { "content-type": "application/json" } }),
      );

      const req = new Request("https://proxy.example.com/v1/chat/completions", {
        method: "POST",
        headers: { "x-nullspend-key": "ns_live_sk_test", "content-type": "application/json" },
        body: cjkBody,
      });

      const res = await entrypoint.fetch(req, makeEnv(), { waitUntil: vi.fn(), passThroughOnException: vi.fn() } as unknown as ExecutionContext);
      // The request should succeed (not fail on body parsing)
      expect(res.status).toBe(200);
      // The upstream fetch was called, meaning body was parsed correctly
      expect(globalThis.fetch).toHaveBeenCalled();
    });
  });
});
