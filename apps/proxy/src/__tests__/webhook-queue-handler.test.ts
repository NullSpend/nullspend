/**
 * Tests for the webhook queue consumer.
 * Covers endpoint lookup, HMAC signing at delivery time, delivery with
 * retry/backoff, permanent failure detection, batch caching, and
 * rotation window handling.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockGetEndpointsWithSecrets, mockDualSign, mockEmitMetric } = vi.hoisted(() => ({
  mockGetEndpointsWithSecrets: vi.fn(),
  mockDualSign: vi.fn().mockResolvedValue("t=1000,v1=abc123"),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/webhook-cache.js", () => ({
  getWebhookEndpointsWithSecrets: (...args: unknown[]) => mockGetEndpointsWithSecrets(...args),
}));

vi.mock("../lib/webhook-signer.js", () => ({
  dualSignWebhookPayload: (...args: unknown[]) => mockDualSign(...args),
  SECRET_ROTATION_WINDOW_SECONDS: 86_400,
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { handleWebhookQueue } from "../webhook-queue-handler.js";
import type { WebhookQueueMessage } from "../lib/webhook-queue.js";

function makeEndpoint(overrides: Record<string, unknown> = {}) {
  return {
    id: "ep-1",
    url: "https://hooks.example.com/webhook",
    signingSecret: "whsec_current",
    previousSigningSecret: null,
    secretRotatedAt: null,
    eventTypes: [],
    apiVersion: "2026-04-01",
    payloadMode: "full",
    ...overrides,
  };
}

function makeEvent(overrides: Record<string, unknown> = {}) {
  return {
    id: "evt_test123",
    type: "cost_event.created",
    api_version: "2026-04-01",
    created_at: 1710547200,
    data: { object: { request_id: "req_1" } },
    ...overrides,
  };
}

function makeMessage(overrides: Partial<WebhookQueueMessage> = {}) {
  const body: WebhookQueueMessage = {
    ownerId: "user-1",
    endpointId: "ep-1",
    event: makeEvent() as any,
    ...overrides,
  };
  return {
    body,
    id: "msg-1",
    timestamp: new Date(),
    attempts: 1,
    ack: vi.fn(),
    retry: vi.fn(),
  };
}

function makeBatch(messages: ReturnType<typeof makeMessage>[]) {
  return {
    queue: "nullspend-webhooks",
    messages,
  } as unknown as MessageBatch<WebhookQueueMessage>;
}

function makeEnv() {
  return {
    HYPERDRIVE: { connectionString: "postgresql://test:test@localhost/test" },
  } as unknown as Env;
}

describe("handleWebhookQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockGetEndpointsWithSecrets.mockResolvedValue([makeEndpoint()]);
    // Mock global fetch
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("ok", { status: 200 })));
  });

  // ── Delivery tests ──

  it("successful delivery — acks message and emits webhook_delivered metric", async () => {
    const msg = makeMessage();
    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.ack).toHaveBeenCalledOnce();
    expect(msg.retry).not.toHaveBeenCalled();
    expect(mockEmitMetric).toHaveBeenCalledWith("webhook_delivered", expect.objectContaining({
      eventType: "cost_event.created",
      endpointId: "ep-1",
      attempts: 1,
    }));
  });

  it("server error (500) — retries with exponential backoff", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("error", { status: 500 })));
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.retry).toHaveBeenCalledWith({ delaySeconds: 10 }); // 10 * 2^0 = 10
    expect(msg.ack).not.toHaveBeenCalled();
    expect(mockEmitMetric).toHaveBeenCalledWith("webhook_retry", expect.objectContaining({
      statusCode: 500,
      attempt: 1,
    }));
  });

  it("rate limited (429) — retries with backoff (treated as transient)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("rate limited", { status: 429 })));
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.retry).toHaveBeenCalledWith({ delaySeconds: 10 });
    expect(msg.ack).not.toHaveBeenCalled();
    expect(mockEmitMetric).toHaveBeenCalledWith("webhook_retry", expect.objectContaining({
      statusCode: 429,
    }));
  });

  it("permanent failure (400) — acks without retry", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("bad request", { status: 400 })));
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.ack).toHaveBeenCalledOnce();
    expect(msg.retry).not.toHaveBeenCalled();
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining("Permanent failure 400"));
  });

  it("network error (fetch throws) — retries with backoff", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.retry).toHaveBeenCalledWith({ delaySeconds: 10 });
    expect(msg.ack).not.toHaveBeenCalled();
    expect(mockEmitMetric).toHaveBeenCalledWith("webhook_retry", expect.objectContaining({
      statusCode: 0,
    }));
  });

  it("exponential backoff increases with attempts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("error", { status: 502 })));

    const msg1 = makeMessage();
    msg1.attempts = 1;
    await handleWebhookQueue(makeBatch([msg1]), makeEnv());
    expect(msg1.retry).toHaveBeenCalledWith({ delaySeconds: 10 }); // 10 * 2^0

    const msg2 = makeMessage();
    msg2.attempts = 3;
    await handleWebhookQueue(makeBatch([msg2]), makeEnv());
    expect(msg2.retry).toHaveBeenCalledWith({ delaySeconds: 40 }); // 10 * 2^2

    const msg3 = makeMessage();
    msg3.attempts = 10;
    await handleWebhookQueue(makeBatch([msg3]), makeEnv());
    expect(msg3.retry).toHaveBeenCalledWith({ delaySeconds: 3600 }); // capped at 3600
  });

  // ── Endpoint lookup tests ──

  it("endpoint deleted between enqueue and delivery — acks and skips", async () => {
    mockGetEndpointsWithSecrets.mockResolvedValue([]); // no endpoints found
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(msg.ack).toHaveBeenCalledOnce();
    expect(fetch).not.toHaveBeenCalled();
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("not found"));
  });

  // PXY-6 regression: DB error during dequeue must retry, not ack.
  // Before this fix, getWebhookEndpointsWithSecrets caught DB errors
  // and returned [], which the consumer treated as "endpoint deleted"
  // and acked — permanently losing the webhook message.
  it("DB error during endpoint lookup retries instead of acking (PXY-6)", async () => {
    mockGetEndpointsWithSecrets.mockRejectedValue(new Error("ECONNREFUSED"));
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    // Must retry (not ack) — message stays in queue for re-delivery
    expect(msg.retry).toHaveBeenCalledOnce();
    expect(msg.ack).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("batch cache — 2 messages for same ownerId result in 1 DB call", async () => {
    const msg1 = makeMessage({ endpointId: "ep-1" });
    const msg2 = makeMessage({ endpointId: "ep-1" });
    mockGetEndpointsWithSecrets.mockResolvedValue([makeEndpoint()]);

    await handleWebhookQueue(makeBatch([msg1, msg2]), makeEnv());

    expect(mockGetEndpointsWithSecrets).toHaveBeenCalledTimes(1);
    expect(msg1.ack).toHaveBeenCalled();
    expect(msg2.ack).toHaveBeenCalled();
  });

  // ── Signing tests ──

  it("signs with fresh timestamp at delivery time", async () => {
    const msg = makeMessage();
    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(mockDualSign).toHaveBeenCalledWith(
      expect.any(String), // payload
      "whsec_current",    // signing secret
      null,               // no previous secret (not in rotation)
      expect.any(Number), // fresh timestamp
    );

    // Timestamp should be close to now
    const calledTimestamp = mockDualSign.mock.calls[0][3] as number;
    const now = Math.floor(Date.now() / 1000);
    expect(Math.abs(now - calledTimestamp)).toBeLessThan(5);
  });

  it("dual-signs within rotation window (previousSecret passed)", async () => {
    mockGetEndpointsWithSecrets.mockResolvedValue([makeEndpoint({
      previousSigningSecret: "whsec_old",
      secretRotatedAt: new Date().toISOString(), // just rotated
    })]);
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(mockDualSign).toHaveBeenCalledWith(
      expect.any(String),
      "whsec_current",
      "whsec_old", // previous secret passed for dual-signing
      expect.any(Number),
    );
  });

  it("no dual-signing outside rotation window (previousSecret = null)", async () => {
    const expired = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString(); // 25h ago
    mockGetEndpointsWithSecrets.mockResolvedValue([makeEndpoint({
      previousSigningSecret: "whsec_old",
      secretRotatedAt: expired,
    })]);
    const msg = makeMessage();

    await handleWebhookQueue(makeBatch([msg]), makeEnv());

    expect(mockDualSign).toHaveBeenCalledWith(
      expect.any(String),
      "whsec_current",
      null, // expired, so null
      expect.any(Number),
    );
  });

  // ── Empty batch ──

  it("empty batch does nothing", async () => {
    await handleWebhookQueue(makeBatch([]), makeEnv());
    expect(mockGetEndpointsWithSecrets).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });
});
