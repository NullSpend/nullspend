/**
 * Tests for the webhook dead letter queue consumer.
 * Verifies always-ack behavior, metric emission, and logging.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockEmitMetric } = vi.hoisted(() => ({
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { handleWebhookDlq } from "../webhook-dlq-handler.js";
import type { WebhookQueueMessage } from "../lib/webhook-queue.js";

function makeMessage(overrides: Partial<WebhookQueueMessage> = {}) {
  const body: WebhookQueueMessage = {
    ownerId: "user-1",
    endpointId: "ep-1",
    event: {
      id: "evt_test123",
      type: "cost_event.created",
      api_version: "2026-04-01",
      created_at: 1710547200,
      data: { object: { request_id: "req_1" } },
    } as any,
    ...overrides,
  };
  return {
    body,
    id: "msg-1",
    timestamp: new Date(),
    attempts: 6, // exhausted retries
    ack: vi.fn(),
    retry: vi.fn(),
  };
}

function makeBatch(messages: ReturnType<typeof makeMessage>[]) {
  return {
    queue: "nullspend-webhooks-dlq",
    messages,
  } as unknown as MessageBatch<WebhookQueueMessage>;
}

describe("handleWebhookDlq", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("always acks every message", async () => {
    const msg1 = makeMessage();
    const msg2 = makeMessage({ endpointId: "ep-2" });

    await handleWebhookDlq(makeBatch([msg1, msg2]));

    expect(msg1.ack).toHaveBeenCalledOnce();
    expect(msg2.ack).toHaveBeenCalledOnce();
    expect(msg1.retry).not.toHaveBeenCalled();
    expect(msg2.retry).not.toHaveBeenCalled();
  });

  it("emits webhook_delivery_failed metric for each message", async () => {
    const msg = makeMessage();
    await handleWebhookDlq(makeBatch([msg]));

    expect(mockEmitMetric).toHaveBeenCalledWith("webhook_delivery_failed", {
      endpointId: "ep-1",
      eventType: "cost_event.created",
    });
  });

  it("logs endpointId, eventType, eventId, and attempts", async () => {
    const msg = makeMessage();
    await handleWebhookDlq(makeBatch([msg]));

    expect(console.error).toHaveBeenCalledWith(
      expect.stringContaining("[webhook-dlq]"),
      expect.objectContaining({
        ownerId: "user-1",
        endpointId: "ep-1",
        eventType: "cost_event.created",
        eventId: "evt_test123",
        attempts: 6,
      }),
    );
  });
});
