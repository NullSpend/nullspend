import { describe, it, expect, vi } from "vitest";
import { enqueueReconciliation, type ReconciliationMessage } from "../lib/reconciliation-queue.js";

describe("enqueueReconciliation", () => {
  it("sends message to queue", async () => {
    const mockQueue = { send: vi.fn().mockResolvedValue(undefined) };

    const msg: ReconciliationMessage = {
      type: "reconcile",
      reservationId: "res-123",
      actualCostMicrodollars: 50_000,
      budgetEntities: [
        { entityKey: "{budget}:api_key:key-1", entityType: "api_key", entityId: "key-1" },
      ],
      ownerId: "user-abc",
      enqueuedAt: 1710000000000,
    };

    await enqueueReconciliation(mockQueue as any, msg);

    expect(mockQueue.send).toHaveBeenCalledWith(msg);
    expect(mockQueue.send).toHaveBeenCalledTimes(1);
  });

  it("propagates queue send errors", async () => {
    const mockQueue = { send: vi.fn().mockRejectedValue(new Error("queue full")) };

    const msg: ReconciliationMessage = {
      type: "reconcile",
      reservationId: "res-456",
      actualCostMicrodollars: 10_000,
      budgetEntities: [],
      ownerId: null,
      enqueuedAt: Date.now(),
    };

    await expect(enqueueReconciliation(mockQueue as any, msg)).rejects.toThrow("queue full");
  });

  it("serializes all required fields", async () => {
    const mockQueue = { send: vi.fn().mockResolvedValue(undefined) };

    const msg: ReconciliationMessage = {
      type: "reconcile",
      reservationId: "res-789",
      actualCostMicrodollars: 100_000,
      budgetEntities: [
        { entityKey: "{budget}:user:u1", entityType: "user", entityId: "u1" },
        { entityKey: "{budget}:api_key:k1", entityType: "api_key", entityId: "k1" },
      ],
      ownerId: "user-xyz",
      enqueuedAt: 1710000000000,
    };

    await enqueueReconciliation(mockQueue as any, msg);

    const sent = mockQueue.send.mock.calls[0][0];
    expect(sent.type).toBe("reconcile");
    expect(sent.reservationId).toBe("res-789");
    expect(sent.actualCostMicrodollars).toBe(100_000);
    expect(sent.budgetEntities).toHaveLength(2);
    expect(sent.ownerId).toBe("user-xyz");
    expect(sent.enqueuedAt).toBe(1710000000000);
  });
});
