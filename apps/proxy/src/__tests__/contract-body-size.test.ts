/**
 * Contract tests for body-size 413 enforcement (P0-3).
 *
 * In-process port of the live-smoke suite previously at
 * apps/proxy/smoke-body-size.test.ts. Invokes the top-level
 * worker fetch handler directly — body-size check lives in
 * parseRequestBody() BEFORE routing, so route-handler invocation
 * won't reach it.
 *
 * What this proves:
 * - 1.2MB body with accurate Content-Length → 413 (pre-read path)
 * - 1.2MB body via ReadableStream (no CL) → 413 (streaming path, REGRESSION GUARD P0-3)
 * - Valid body under cap → passes body-size check
 * - Oversize body → `globalThis.fetch` was NEVER called (no upstream attempt)
 * - Body EXACTLY at MAX_BODY_SIZE (1048576 bytes) → passes (boundary)
 * - Zero-byte body → passes body-size check (may fail downstream)
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

// Auth must succeed so body-size check is reached.
vi.mock("../lib/auth.js", () => ({
  authenticateRequest: vi.fn().mockResolvedValue({
    userId: "user-1",
    keyId: "key-1",
    hasWebhooks: false,
    hasBudgets: false,
    orgId: "org-test",
    apiVersion: "2026-04-01",
    defaultTags: {},
  }),
}));

// Short-circuit budget orchestration — we never reach it.
vi.mock("../lib/budget-orchestrator.js", () => ({
  checkBudget: vi.fn().mockResolvedValue({ status: "skipped", reservationId: null, budgetEntities: [] }),
  reconcileBudget: vi.fn().mockResolvedValue(undefined),
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

import {
  MAX_BODY_SIZE,
  makeContractEnv,
  makeExecutionContext,
  invokeWorker,
} from "./msw/contract-helpers.js";

function buildOversizeBody(targetBytes: number): string {
  const overhead = 200;
  const filler = "x".repeat(Math.max(0, targetBytes - overhead));
  return JSON.stringify({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: filler }],
    max_tokens: 1,
  });
}

function makePostRequest(body: BodyInit, headers: Record<string, string> = {}): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-nullspend-key": "ns_key_test_contract",
      ...headers,
    },
    body,
  });
}

describe("Body-size enforcement (P0-3) — contract", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects 1.2MB body with 413 when Content-Length is accurate (pre-read path)", async () => {
    const body = buildOversizeBody(1.2 * MAX_BODY_SIZE);
    expect(body.length).toBeGreaterThan(MAX_BODY_SIZE);

    const request = makePostRequest(body, { "Content-Length": String(body.length) });
    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    expect(res.status).toBe(413);
    const payload = await res.json() as { error: { code: string } };
    expect(payload.error.code).toBe("payload_too_large");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects 1.2MB body via ReadableStream with 413 (streaming path, REGRESSION GUARD)", async () => {
    const body = buildOversizeBody(1.2 * MAX_BODY_SIZE);
    const encoder = new TextEncoder();
    const chunk = encoder.encode(body);
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    });

    const request = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": "ns_key_test_contract",
      },
      body: stream,
      // @ts-expect-error duplex is Node 18+ fetch, not in lib.dom
      duplex: "half",
    });

    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    expect(res.status).toBe(413);
    const payload = await res.json() as { error: { code: string } };
    expect(payload.error.code).toBe("payload_too_large");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("allows 500KB body — passes body-size check", async () => {
    const body = buildOversizeBody(0.5 * MAX_BODY_SIZE);
    expect(body.length).toBeLessThan(MAX_BODY_SIZE);

    const request = makePostRequest(body);
    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    // Under-cap body must pass body-size check. Downstream may still
    // error (no MSW handler for OpenAI — hits default bypass and fails
    // network), but must NOT be 413.
    expect(res.status).not.toBe(413);
  });

  it("oversized body — fetch is NEVER called (replaces wall-time assertion)", async () => {
    const body = buildOversizeBody(1.2 * MAX_BODY_SIZE);
    const request = makePostRequest(body, { "Content-Length": String(body.length) });

    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    expect(res.status).toBe(413);
    // The invariant the old `elapsed < 2000ms` check really tested.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  // ── Boundary cases added in Phase 2 (D6) ─────────────────────────────

  it("body at EXACTLY MAX_BODY_SIZE bytes — passes body-size check (boundary >=  vs >)", async () => {
    // Craft a JSON body whose serialized byte length is exactly MAX_BODY_SIZE.
    // The envelope adds overhead; adjust the filler to land on the boundary.
    const envelope = JSON.stringify({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "" }],
      max_tokens: 1,
    });
    const overhead = envelope.length;
    const filler = "x".repeat(MAX_BODY_SIZE - overhead);
    const body = JSON.stringify({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: filler }],
      max_tokens: 1,
    });
    expect(body.length).toBe(MAX_BODY_SIZE);

    const request = makePostRequest(body, { "Content-Length": String(body.length) });
    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    // The check uses `> MAX_BODY_SIZE` not `>=`, so a body exactly at the cap must pass.
    expect(res.status).not.toBe(413);
  });

  it("zero-byte body — passes body-size check (empty ReadableStream path)", async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.close();
      },
    });

    const request = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": "ns_key_test_contract",
      },
      body: stream,
      // @ts-expect-error duplex is Node 18+ fetch
      duplex: "half",
    });

    const res = await invokeWorker(request, makeContractEnv(), makeExecutionContext());

    // Zero-byte body passes body-size check (no bytes consumed is under any cap).
    // Downstream JSON parse will reject with 400 bad_request — that's fine.
    expect(res.status).not.toBe(413);
  });
});
