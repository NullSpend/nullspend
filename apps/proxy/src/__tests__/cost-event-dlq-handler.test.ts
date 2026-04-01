/**
 * Tests for the cost event DLQ consumer.
 * Covers always-ack behavior, metric emission, individual best-effort writes,
 * and handling of null userId.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockLogCostEvent, mockEmitMetric } = vi.hoisted(() => ({
  mockLogCostEvent: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/cost-logger.js", () => ({
  logCostEvent: (...args: unknown[]) => mockLogCostEvent(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { handleCostEventDlq, COST_EVENT_DLQ_NAME } from "../cost-event-dlq-handler.js";

function makeCostEventMessage(overrides: Record<string, unknown> = {}) {
  return {
    type: "cost_event" as const,
    event: {
      requestId: "req-dlq-123",
      provider: "openai",
      model: "gpt-4o-mini",
      inputTokens: 50,
      outputTokens: 10,
      cachedInputTokens: 0,
      reasoningTokens: 0,
      costMicrodollars: 150,
      durationMs: 250,
      userId: "user-1",
      apiKeyId: null,
      actionId: null,
      ...overrides,
    },
    enqueuedAt: Date.now() - 5000,
  };
}

function makeMessage(body: ReturnType<typeof makeCostEventMessage>) {
  return {
    body,
    ack: vi.fn(),
    retry: vi.fn(),
    id: crypto.randomUUID(),
    timestamp: new Date(),
    attempts: 4,
  };
}

function makeBatch(
  messages: ReturnType<typeof makeMessage>[],
): MessageBatch<any> {
  return {
    messages,
    queue: COST_EVENT_DLQ_NAME,
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

describe("handleCostEventDlq", () => {
  it("always acks every message", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg1 = makeMessage(makeCostEventMessage({ requestId: "r1" }));
    const msg2 = makeMessage(makeCostEventMessage({ requestId: "r2" }));
    const batch = makeBatch([msg1, msg2]);

    await handleCostEventDlq(batch, makeEnv());

    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
    expect(msg1.retry).not.toHaveBeenCalled();
    expect(msg2.retry).not.toHaveBeenCalled();
  });

  it("emits cost_event_dlq metric for each message", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg = makeMessage(makeCostEventMessage({
      requestId: "req-metric-test",
      costMicrodollars: 42000,
      userId: "user-metrics",
    }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith("cost_event_dlq", expect.objectContaining({
      requestId: "req-metric-test",
      costMicrodollars: 42000,
      userId: "user-metrics",
    }));
  });

  it("attempts best-effort individual write for each message", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg = makeMessage(makeCostEventMessage({ requestId: "req-write-test" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    expect(mockLogCostEvent).toHaveBeenCalledWith(
      "postgres://test:test@localhost:5432/test",
      expect.objectContaining({ requestId: "req-write-test" }),
    );
  });

  it("acks even when logCostEvent throws", async () => {
    mockLogCostEvent.mockRejectedValue(new Error("DB down"));

    const msg = makeMessage(makeCostEventMessage());
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    expect(msg.ack).toHaveBeenCalledTimes(1);
  });

  it("handles userId in metric", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg = makeMessage(makeCostEventMessage({ userId: "user-dlq" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith("cost_event_dlq", expect.objectContaining({
      userId: "user-dlq",
    }));
  });

  it("calculates ageMs from enqueuedAt", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const msgBody = makeCostEventMessage();
    msgBody.enqueuedAt = Date.now() - 10_000;
    const msg = makeMessage(msgBody);
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    const metricCall = mockEmitMetric.mock.calls[0];
    expect(metricCall[1].ageMs).toBeGreaterThanOrEqual(9000);
    expect(metricCall[1].ageMs).toBeLessThan(20000);
  });

  it("exports the correct DLQ queue name", () => {
    expect(COST_EVENT_DLQ_NAME).toBe("nullspend-cost-events-dlq");
  });

  it("acks all messages and emits metrics when HYPERDRIVE binding is unavailable", async () => {
    const msg1 = makeMessage(makeCostEventMessage({ requestId: "r1" }));
    const msg2 = makeMessage(makeCostEventMessage({ requestId: "r2" }));
    const batch = makeBatch([msg1, msg2]);

    const brokenEnv = {} as unknown as Env; // no HYPERDRIVE

    await handleCostEventDlq(batch, brokenEnv);

    // All messages acked despite binding failure
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
    // Metric emitted for each
    expect(mockEmitMetric).toHaveBeenCalledTimes(2);
    // No DB write attempted
    expect(mockLogCostEvent).not.toHaveBeenCalled();
  });
});
