/**
 * Tests for the cost event queue module.
 * Covers enqueue, getCostEventQueue, queue-first-with-fallback helpers,
 * timeout behavior, and fallback metric emission.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../lib/cost-logger.js", () => ({
  logCostEvent: vi.fn().mockResolvedValue(undefined),
  logCostEventsBatch: vi.fn().mockResolvedValue(undefined),
}));

const mockEmitMetric = vi.fn();
vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import {
  enqueueCostEvent,
  enqueueCostEventsBatch,
  getCostEventQueue,
  logCostEventQueued,
  logCostEventsBatchQueued,
} from "../lib/cost-event-queue.js";
import { logCostEvent, logCostEventsBatch } from "../lib/cost-logger.js";

function makeCostEvent(overrides: Record<string, unknown> = {}) {
  return {
    requestId: "req-test-123",
    provider: "openai" as const,
    model: "gpt-4o-mini",
    inputTokens: 50,
    outputTokens: 10,
    cachedInputTokens: 0,
    reasoningTokens: 0,
    costMicrodollars: 150,
    durationMs: 250,
    userId: null,
    apiKeyId: null,
    actionId: null,
    ...overrides,
  };
}

function makeQueue(overrides: Partial<Queue> = {}): Queue {
  return {
    send: vi.fn().mockResolvedValue(undefined),
    sendBatch: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as unknown as Queue;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("enqueueCostEvent", () => {
  it("sends a single message with type, event, and enqueuedAt", async () => {
    const queue = makeQueue();
    const event = makeCostEvent();

    await enqueueCostEvent(queue, event);

    expect(queue.send).toHaveBeenCalledTimes(1);
    const msg = (queue.send as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(msg.type).toBe("cost_event");
    expect(msg.event).toEqual(event);
    expect(msg.enqueuedAt).toBeGreaterThan(0);
  });

  it("rejects when queue.send times out", async () => {
    vi.useFakeTimers();

    const queue = makeQueue({
      send: vi.fn().mockImplementation(() => new Promise(() => {})), // never resolves
    });
    const event = makeCostEvent();

    const promise = enqueueCostEvent(queue, event);
    // Attach rejection handler before advancing timers to avoid unhandled rejection
    const assertion = expect(promise).rejects.toThrow(/timed out/);
    await vi.advanceTimersByTimeAsync(5_000);
    await assertion;

    vi.useRealTimers();
  });
});

describe("enqueueCostEventsBatch", () => {
  it("uses queue.sendBatch for multiple events", async () => {
    const queue = makeQueue();
    const events = [makeCostEvent({ requestId: "a" }), makeCostEvent({ requestId: "b" })];

    await enqueueCostEventsBatch(queue, events);

    expect(queue.sendBatch).toHaveBeenCalledTimes(1);
    const batch = (queue.sendBatch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(batch).toHaveLength(2);
    expect(batch[0].body.type).toBe("cost_event");
    expect(batch[0].body.event.requestId).toBe("a");
    expect(batch[1].body.event.requestId).toBe("b");
  });

  it("is a no-op for empty events array", async () => {
    const queue = makeQueue();

    await enqueueCostEventsBatch(queue, []);

    expect(queue.sendBatch).not.toHaveBeenCalled();
  });
});

describe("getCostEventQueue", () => {
  it("returns the COST_EVENT_QUEUE binding when present", () => {
    const mockQueue = makeQueue();
    const env = { COST_EVENT_QUEUE: mockQueue } as unknown as Env;

    expect(getCostEventQueue(env)).toBe(mockQueue);
  });

  it("returns undefined when binding is absent", () => {
    const env = {} as unknown as Env;

    expect(getCostEventQueue(env)).toBeUndefined();
  });
});

describe("logCostEventQueued", () => {
  it("enqueues to queue when available", async () => {
    const queue = makeQueue();
    const event = makeCostEvent();

    await logCostEventQueued(queue, "postgres://test", event);

    expect(queue.send).toHaveBeenCalledTimes(1);
    expect(logCostEvent).not.toHaveBeenCalled();
  });

  it("falls back to direct logCostEvent when queue is undefined", async () => {
    const event = makeCostEvent();

    await logCostEventQueued(undefined, "postgres://test", event);

    expect(logCostEvent).toHaveBeenCalledWith("postgres://test", event);
  });

  it("falls back to direct logCostEvent when queue.send fails", async () => {
    const queue = makeQueue({ send: vi.fn().mockRejectedValue(new Error("queue down")) });
    const event = makeCostEvent();

    await logCostEventQueued(queue, "postgres://test", event);

    expect(queue.send).toHaveBeenCalledTimes(1);
    expect(logCostEvent).toHaveBeenCalledWith("postgres://test", event);
  });

  it("emits cost_event_queue_fallback metric on queue send failure", async () => {
    const queue = makeQueue({ send: vi.fn().mockRejectedValue(new Error("queue down")) });
    const event = makeCostEvent();

    await logCostEventQueued(queue, "postgres://test", event);

    expect(mockEmitMetric).toHaveBeenCalledWith("cost_event_queue_fallback", { reason: "send_failed" });
  });

  it("does not emit fallback metric when queue is undefined (expected local dev path)", async () => {
    const event = makeCostEvent();

    await logCostEventQueued(undefined, "postgres://test", event);

    expect(mockEmitMetric).not.toHaveBeenCalled();
  });
});

describe("logCostEventsBatchQueued", () => {
  it("enqueues batch via queue.sendBatch when available", async () => {
    const queue = makeQueue();
    const events = [makeCostEvent({ requestId: "a" }), makeCostEvent({ requestId: "b" })];

    await logCostEventsBatchQueued(queue, "postgres://test", events);

    expect(queue.sendBatch).toHaveBeenCalledTimes(1);
    expect(logCostEventsBatch).not.toHaveBeenCalled();
  });

  it("falls back to direct logCostEventsBatch when queue is undefined", async () => {
    const events = [makeCostEvent()];

    await logCostEventsBatchQueued(undefined, "postgres://test", events);

    expect(logCostEventsBatch).toHaveBeenCalledWith("postgres://test", events);
  });

  it("falls back to direct logCostEventsBatch when queue.sendBatch fails", async () => {
    const queue = makeQueue({ sendBatch: vi.fn().mockRejectedValue(new Error("queue down")) });
    const events = [makeCostEvent()];

    await logCostEventsBatchQueued(queue, "postgres://test", events);

    expect(queue.sendBatch).toHaveBeenCalledTimes(1);
    expect(logCostEventsBatch).toHaveBeenCalledWith("postgres://test", events);
  });

  it("emits cost_event_queue_fallback metric on queue sendBatch failure", async () => {
    const queue = makeQueue({ sendBatch: vi.fn().mockRejectedValue(new Error("queue down")) });
    const events = [makeCostEvent(), makeCostEvent()];

    await logCostEventsBatchQueued(queue, "postgres://test", events);

    expect(mockEmitMetric).toHaveBeenCalledWith("cost_event_queue_fallback", { reason: "send_batch_failed", count: 2 });
  });

  it("is a no-op for empty events array", async () => {
    const queue = makeQueue();

    await logCostEventsBatchQueued(queue, "postgres://test", []);

    expect(queue.sendBatch).not.toHaveBeenCalled();
    expect(logCostEventsBatch).not.toHaveBeenCalled();
  });
});
