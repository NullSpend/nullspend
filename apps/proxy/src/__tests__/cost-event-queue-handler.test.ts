/**
 * Tests for the cost event queue consumer.
 * Covers batch INSERT, per-message fallback on failure, and empty batch handling.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockLogCostEventsBatch, mockLogCostEvent, mockEmitMetric } = vi.hoisted(() => ({
  mockLogCostEventsBatch: vi.fn(),
  mockLogCostEvent: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/cost-logger.js", () => ({
  logCostEventsBatch: (...args: unknown[]) => mockLogCostEventsBatch(...args),
  logCostEvent: (...args: unknown[]) => mockLogCostEvent(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { handleCostEventQueue } from "../cost-event-queue-handler.js";

function makeCostEventMessage(overrides: Record<string, unknown> = {}) {
  return {
    type: "cost_event" as const,
    event: {
      requestId: "req-test-123",
      provider: "openai",
      model: "gpt-4o-mini",
      inputTokens: 50,
      outputTokens: 10,
      cachedInputTokens: 0,
      reasoningTokens: 0,
      costMicrodollars: 150,
      durationMs: 250,
      userId: "user-test",
      apiKeyId: null,
      actionId: null,
      ...overrides,
    },
    enqueuedAt: Date.now(),
  };
}

function makeMessage(body: ReturnType<typeof makeCostEventMessage>) {
  return {
    body,
    ack: vi.fn(),
    retry: vi.fn(),
    id: crypto.randomUUID(),
    timestamp: new Date(),
    attempts: 1,
  };
}

function makeBatch(
  messages: ReturnType<typeof makeMessage>[],
  queue = "nullspend-cost-events",
): MessageBatch<any> {
  return {
    messages,
    queue,
    ackAll: vi.fn(),
    retryAll: vi.fn(),
  };
}

function makeEnv(): Env {
  return {
    HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
  } as unknown as Env;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("handleCostEventQueue", () => {
  it("batch INSERTs all events and acks all messages on success", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const msg1 = makeMessage(makeCostEventMessage({ requestId: "r1" }));
    const msg2 = makeMessage(makeCostEventMessage({ requestId: "r2" }));
    const batch = makeBatch([msg1, msg2]);

    await handleCostEventQueue(batch, makeEnv());

    expect(mockLogCostEventsBatch).toHaveBeenCalledTimes(1);
    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      "postgres://test:test@localhost:5432/test",
      expect.arrayContaining([
        expect.objectContaining({ requestId: "r1" }),
        expect.objectContaining({ requestId: "r2" }),
      ]),
      { throwOnError: true },
    );
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
    expect(msg1.retry).not.toHaveBeenCalled();
    expect(msg2.retry).not.toHaveBeenCalled();
  });

  it("falls back to per-message processing on batch failure", async () => {
    mockLogCostEventsBatch.mockRejectedValue(new Error("batch INSERT failed"));
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg1 = makeMessage(makeCostEventMessage({ requestId: "r1" }));
    const msg2 = makeMessage(makeCostEventMessage({ requestId: "r2" }));
    const batch = makeBatch([msg1, msg2]);

    await handleCostEventQueue(batch, makeEnv());

    // Batch was attempted first
    expect(mockLogCostEventsBatch).toHaveBeenCalledTimes(1);

    // Per-message fallback
    expect(mockLogCostEvent).toHaveBeenCalledTimes(2);
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
  });

  it("retries individual messages that fail in per-message fallback", async () => {
    mockLogCostEventsBatch.mockRejectedValue(new Error("batch INSERT failed"));
    mockLogCostEvent
      .mockResolvedValueOnce(undefined)    // msg1 succeeds
      .mockRejectedValueOnce(new Error("constraint violation")); // msg2 fails

    const msg1 = makeMessage(makeCostEventMessage({ requestId: "r1" }));
    const msg2 = makeMessage(makeCostEventMessage({ requestId: "r2" }));
    const batch = makeBatch([msg1, msg2]);

    await handleCostEventQueue(batch, makeEnv());

    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg1.retry).not.toHaveBeenCalled();
    expect(msg2.ack).not.toHaveBeenCalled();
    expect(msg2.retry).toHaveBeenCalledTimes(1);
  });

  it("handles empty batch without DB call", async () => {
    const batch = makeBatch([]);

    await handleCostEventQueue(batch, makeEnv());

    expect(mockLogCostEventsBatch).not.toHaveBeenCalled();
    expect(mockLogCostEvent).not.toHaveBeenCalled();
  });

  it("emits cost_event_queue_dwell_ms metric with max dwell across the batch", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const now = Date.now();
    // msg1 enqueued 8s ago, msg2 enqueued 20s ago (above 15s slow threshold)
    const msg1 = makeMessage({ ...makeCostEventMessage({ requestId: "r1" }), enqueuedAt: now - 8_000 });
    const msg2 = makeMessage({ ...makeCostEventMessage({ requestId: "r2" }), enqueuedAt: now - 20_000 });
    const batch = makeBatch([msg1, msg2]);

    await handleCostEventQueue(batch, makeEnv());

    const dwellCall = mockEmitMetric.mock.calls.find(
      ([name]) => name === "cost_event_queue_dwell_ms",
    );
    expect(dwellCall).toBeDefined();
    const payload = dwellCall![1] as { max: number; slowCount: number; batchSize: number };
    expect(payload.batchSize).toBe(2);
    expect(payload.slowCount).toBe(1);
    // Max dwell is ~20_000ms; allow small jitter from Date.now() drift during test
    expect(payload.max).toBeGreaterThanOrEqual(20_000);
    expect(payload.max).toBeLessThan(21_000);
  });

  it("skips dwell tracking for messages with missing or invalid enqueuedAt", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const msgMissing = makeMessage({ ...makeCostEventMessage({ requestId: "r1" }), enqueuedAt: 0 });
    const batch = makeBatch([msgMissing]);

    await handleCostEventQueue(batch, makeEnv());

    const dwellCall = mockEmitMetric.mock.calls.find(
      ([name]) => name === "cost_event_queue_dwell_ms",
    );
    expect(dwellCall).toBeDefined();
    const payload = dwellCall![1] as { max: number; slowCount: number };
    expect(payload.max).toBe(0);
    expect(payload.slowCount).toBe(0);
  });

  it("reads connectionString from env.HYPERDRIVE", async () => {
    mockLogCostEventsBatch.mockResolvedValue(undefined);

    const msg = makeMessage(makeCostEventMessage());
    const batch = makeBatch([msg]);
    const env = {
      HYPERDRIVE: { connectionString: "postgres://custom:5432/db" },
    } as unknown as Env;

    await handleCostEventQueue(batch, env);

    expect(mockLogCostEventsBatch).toHaveBeenCalledWith(
      "postgres://custom:5432/db",
      expect.any(Array),
      { throwOnError: true },
    );
  });
});
