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
      expect.objectContaining({ throwOnError: true }),
    );
  });

  it("passes throwOnError:true to logCostEvent (regression: P0-2)", async () => {
    // REGRESSION GUARD: without throwOnError:true, logCostEvent swallows its
    // own pg errors and returns normally. That makes dbWriteOk always true
    // and renders the R2 fallback below unreachable in production.
    // This test ensures the flag is present; the "persists to R2 when DB
    // write fails" test validates the outer catch + R2 persistence path.
    mockLogCostEvent.mockResolvedValue(undefined);

    const msg = makeMessage(makeCostEventMessage({ requestId: "req-flag-test" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, makeEnv());

    const callArgs = mockLogCostEvent.mock.calls[0];
    expect(callArgs).toHaveLength(3);
    expect(callArgs[2]).toEqual({ throwOnError: true });
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

    const brokenEnv = {} as unknown as Env; // no HYPERDRIVE, no R2

    await handleCostEventDlq(batch, brokenEnv);

    // All messages acked despite binding failure
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
    // P2 fix: each message emits cost_event_dlq + cost_event_dlq_lost (no R2 in broken env)
    expect(mockEmitMetric).toHaveBeenCalledTimes(4);
    // No DB write attempted
    expect(mockLogCostEvent).not.toHaveBeenCalled();
  });

  it("persists to R2 when DB write fails and R2 is available", async () => {
    mockLogCostEvent.mockRejectedValue(new Error("DB down"));

    const mockR2Put = vi.fn().mockResolvedValue(undefined);
    const envWithR2 = {
      HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
      BODY_STORAGE: { put: mockR2Put },
    } as unknown as Env;

    const msg = makeMessage(makeCostEventMessage({ requestId: "req-r2-test" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, envWithR2);

    expect(msg.ack).toHaveBeenCalledTimes(1);
    // DB write attempted and failed
    expect(mockLogCostEvent).toHaveBeenCalledTimes(1);
    // R2 persistence attempted
    expect(mockR2Put).toHaveBeenCalledTimes(1);
    const r2Key = mockR2Put.mock.calls[0][0] as string;
    expect(r2Key).toMatch(/^_dlq\/req-r2-test-\d+\.json$/);
    // Persisted metric emitted (not lost)
    const persistedCalls = mockEmitMetric.mock.calls.filter(
      (c: unknown[]) => c[0] === "cost_event_dlq_persisted_r2",
    );
    expect(persistedCalls).toHaveLength(1);
    expect(persistedCalls[0][1].requestId).toBe("req-r2-test");
  });

  it("emits cost_event_dlq_lost when both DB and R2 fail", async () => {
    mockLogCostEvent.mockRejectedValue(new Error("DB down"));

    const mockR2Put = vi.fn().mockRejectedValue(new Error("R2 down"));
    const envWithBrokenR2 = {
      HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
      BODY_STORAGE: { put: mockR2Put },
    } as unknown as Env;

    const msg = makeMessage(makeCostEventMessage({ requestId: "req-lost" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, envWithBrokenR2);

    expect(msg.ack).toHaveBeenCalledTimes(1);
    expect(mockLogCostEvent).toHaveBeenCalledTimes(1);
    expect(mockR2Put).toHaveBeenCalledTimes(1);
    // Lost metric emitted
    const lostCalls = mockEmitMetric.mock.calls.filter(
      (c: unknown[]) => c[0] === "cost_event_dlq_lost",
    );
    expect(lostCalls).toHaveLength(1);
    expect(lostCalls[0][1].requestId).toBe("req-lost");
  });

  it("skips R2 persistence when DB write succeeds", async () => {
    mockLogCostEvent.mockResolvedValue(undefined);

    const mockR2Put = vi.fn();
    const envWithR2 = {
      HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
      BODY_STORAGE: { put: mockR2Put },
    } as unknown as Env;

    const msg = makeMessage(makeCostEventMessage({ requestId: "req-ok" }));
    const batch = makeBatch([msg]);

    await handleCostEventDlq(batch, envWithR2);

    expect(msg.ack).toHaveBeenCalledTimes(1);
    expect(mockLogCostEvent).toHaveBeenCalledTimes(1);
    // R2 NOT called when DB write succeeds
    expect(mockR2Put).not.toHaveBeenCalled();
    // No lost or persisted metrics
    const r2Calls = mockEmitMetric.mock.calls.filter(
      (c: unknown[]) => c[0] === "cost_event_dlq_persisted_r2" || c[0] === "cost_event_dlq_lost",
    );
    expect(r2Calls).toHaveLength(0);
  });
});
