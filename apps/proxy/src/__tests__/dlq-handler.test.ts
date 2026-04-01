import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockReconcileBudget, mockEmitMetric } = vi.hoisted(() => ({
  mockReconcileBudget: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

vi.mock("../lib/budget-orchestrator.js", () => ({
  reconcileBudget: (...args: unknown[]) => mockReconcileBudget(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { handleDlqQueue, DLQ_QUEUE_NAME } from "../dlq-handler.js";

function makeEnv(): any {
  return {
    HYPERDRIVE: { connectionString: "postgresql://test:test@db:5432/test" },
  };
}

function makeMessage(body: any): any {
  return {
    body,
    ack: vi.fn(),
    retry: vi.fn(),
  };
}

function makeBatch(messages: any[], queue = DLQ_QUEUE_NAME): any {
  return { messages, queue };
}

describe("handleDlqQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockReconcileBudget.mockResolvedValue(undefined);
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("always acks, never retries", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-dlq-1",
      actualCostMicrodollars: 50_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now() - 60_000,
    });

    await handleDlqQueue(makeBatch([msg]), makeEnv());

    expect(msg.ack).toHaveBeenCalledTimes(1);
    expect(msg.retry).not.toHaveBeenCalled();
  });

  it("emits reconciliation_dlq metric with correct tags", async () => {
    const enqueuedAt = Date.now() - 120_000;
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-metric",
      actualCostMicrodollars: 75_000,
      budgetEntities: [
        { entityKey: "{budget}:user:u1", entityType: "user", entityId: "u1" },
        { entityKey: "{budget}:api_key:k1", entityType: "api_key", entityId: "k1" },
      ],
      ownerId: "user-abc",
      enqueuedAt,
    });

    await handleDlqQueue(makeBatch([msg]), makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith("reconciliation_dlq", {
      reservationId: "res-metric",
      costMicrodollars: 75_000,
      ownerId: "user-abc",
      ageMs: expect.any(Number),
      entityCount: 2,
    });
    // ageMs should be roughly 120s
    const tags = mockEmitMetric.mock.calls[0][1];
    expect(tags.ageMs).toBeGreaterThanOrEqual(119_000);
  });

  it("logs structured error with [dlq] prefix", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-log",
      actualCostMicrodollars: 10_000,
      budgetEntities: [],
      ownerId: "user-log",
      enqueuedAt: Date.now(),
    });

    await handleDlqQueue(makeBatch([msg]), makeEnv());

    expect(console.error).toHaveBeenCalledWith(
      "[dlq] Dead-lettered reconciliation message:",
      expect.stringContaining("res-log"),
    );
  });

  it("calls reconcileBudget without throwOnError (6 args)", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-retry",
      actualCostMicrodollars: 30_000,
      budgetEntities: [
        { entityKey: "{budget}:user:u1", entityType: "user", entityId: "u1" },
      ],
      ownerId: "user-retry",
      enqueuedAt: Date.now(),
    });

    const env = makeEnv();
    await handleDlqQueue(makeBatch([msg]), env);

    expect(mockReconcileBudget).toHaveBeenCalledWith(
      env,
      "user-retry",
      "res-retry",
      30_000,
      expect.arrayContaining([
        expect.objectContaining({ entityType: "user", entityId: "u1" }),
      ]),
      env.HYPERDRIVE.connectionString,
    );
    // Verify only 6 args (no options object)
    expect(mockReconcileBudget.mock.calls[0]).toHaveLength(6);
  });

  it("acks even when reconcileBudget throws", async () => {
    mockReconcileBudget.mockRejectedValueOnce(new Error("DB exploded"));

    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-throw",
      actualCostMicrodollars: 50_000,
      budgetEntities: [],
      ownerId: "user-throw",
      enqueuedAt: Date.now(),
    });

    await handleDlqQueue(makeBatch([msg]), makeEnv());

    expect(msg.ack).toHaveBeenCalledTimes(1);
    expect(msg.retry).not.toHaveBeenCalled();
  });

  it("processes all messages in a multi-message batch", async () => {
    const msg1 = makeMessage({
      type: "reconcile",
      reservationId: "res-batch-1",
      actualCostMicrodollars: 10_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now(),
    });
    const msg2 = makeMessage({
      type: "reconcile",
      reservationId: "res-batch-2",
      actualCostMicrodollars: 20_000,
      budgetEntities: [],
      ownerId: "user-2",
      enqueuedAt: Date.now(),
    });

    await handleDlqQueue(makeBatch([msg1, msg2]), makeEnv());

    expect(mockReconcileBudget).toHaveBeenCalledTimes(2);
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
  });

  it("uses 'unknown' for null ownerId in metric", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-null-user",
      actualCostMicrodollars: 5_000,
      budgetEntities: [],
      ownerId: null,
      enqueuedAt: Date.now(),
    });

    await handleDlqQueue(makeBatch([msg]), makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "reconciliation_dlq",
      expect.objectContaining({ ownerId: "unknown" }),
    );
  });

  it("exports correct DLQ_QUEUE_NAME constant", () => {
    expect(DLQ_QUEUE_NAME).toBe("nullspend-reconcile-dlq");
  });

  it("second message processes after first fails", async () => {
    mockReconcileBudget
      .mockRejectedValueOnce(new Error("first fails"))
      .mockResolvedValueOnce(undefined);

    const msg1 = makeMessage({
      type: "reconcile",
      reservationId: "res-fail-1",
      actualCostMicrodollars: 10_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now(),
    });
    const msg2 = makeMessage({
      type: "reconcile",
      reservationId: "res-ok-2",
      actualCostMicrodollars: 20_000,
      budgetEntities: [],
      ownerId: "user-2",
      enqueuedAt: Date.now(),
    });

    await handleDlqQueue(makeBatch([msg1, msg2]), makeEnv());

    expect(mockReconcileBudget).toHaveBeenCalledTimes(2);
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
  });
});
