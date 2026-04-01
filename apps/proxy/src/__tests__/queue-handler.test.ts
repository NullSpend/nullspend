import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockReconcileBudget } = vi.hoisted(() => ({
  mockReconcileBudget: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

vi.mock("../lib/budget-orchestrator.js", () => ({
  reconcileBudget: (...args: unknown[]) => mockReconcileBudget(...args),
}));

import { handleReconciliationQueue } from "../queue-handler.js";

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

function makeBatch(messages: any[]): any {
  return { messages };
}

describe("handleReconciliationQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockReconcileBudget.mockResolvedValue(undefined);
  });

  it("acks message on successful reconciliation", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-123",
      actualCostMicrodollars: 50_000,
      budgetEntities: [
        { entityKey: "{budget}:api_key:key-1", entityType: "api_key", entityId: "key-1" },
      ],
      ownerId: "user-abc",
      enqueuedAt: Date.now(),
    });

    await handleReconciliationQueue(makeBatch([msg]), makeEnv());

    expect(mockReconcileBudget).toHaveBeenCalledTimes(1);
    expect(msg.ack).toHaveBeenCalledTimes(1);
    expect(msg.retry).not.toHaveBeenCalled();
  });

  it("retries message on reconciliation failure", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockReconcileBudget.mockRejectedValueOnce(new Error("DB unavailable"));

    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-fail",
      actualCostMicrodollars: 25_000,
      budgetEntities: [],
      ownerId: null,
      enqueuedAt: Date.now(),
    });

    await handleReconciliationQueue(makeBatch([msg]), makeEnv());

    expect(msg.retry).toHaveBeenCalledTimes(1);
    expect(msg.ack).not.toHaveBeenCalled();
    expect(console.error).toHaveBeenCalledWith(
      "[queue] Reconciliation failed, retrying:",
      expect.any(Error),
    );
  });

  it("processes multiple messages in a batch", async () => {
    const msg1 = makeMessage({
      type: "reconcile",
      reservationId: "res-1",
      actualCostMicrodollars: 10_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now(),
    });
    const msg2 = makeMessage({
      type: "reconcile",
      reservationId: "res-2",
      actualCostMicrodollars: 20_000,
      budgetEntities: [],
      ownerId: "user-2",
      enqueuedAt: Date.now(),
    });

    await handleReconciliationQueue(makeBatch([msg1, msg2]), makeEnv());

    expect(mockReconcileBudget).toHaveBeenCalledTimes(2);
    expect(msg1.ack).toHaveBeenCalledTimes(1);
    expect(msg2.ack).toHaveBeenCalledTimes(1);
  });

  it("passes correct arguments to reconcileBudget with throwOnError", async () => {
    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-args",
      actualCostMicrodollars: 75_000,
      budgetEntities: [
        { entityKey: "{budget}:user:u1", entityType: "user", entityId: "u1" },
      ],
      ownerId: "user-xyz",
      enqueuedAt: Date.now(),
    });

    const env = makeEnv();
    await handleReconciliationQueue(makeBatch([msg]), env);

    expect(mockReconcileBudget).toHaveBeenCalledWith(
      env,
      "user-xyz",
      "res-args",
      75_000,
      expect.arrayContaining([
        expect.objectContaining({ entityType: "user", entityId: "u1" }),
      ]),
      env.HYPERDRIVE.connectionString,
      { throwOnError: true },
    );
  });

  it("retries when reconcileBudget throws due to non-ok status (throwOnError)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockReconcileBudget.mockRejectedValueOnce(new Error("Reconciliation failed with status: pg_failed"));

    const msg = makeMessage({
      type: "reconcile",
      reservationId: "res-status-fail",
      actualCostMicrodollars: 50_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now(),
    });

    await handleReconciliationQueue(makeBatch([msg]), makeEnv());

    expect(msg.retry).toHaveBeenCalledTimes(1);
    expect(msg.ack).not.toHaveBeenCalled();
  });

  it("tolerates old messages with extra mode field", async () => {
    const msg = makeMessage({
      type: "reconcile",
      mode: "redis", // legacy field from old messages
      reservationId: "res-compat",
      actualCostMicrodollars: 50_000,
      budgetEntities: [],
      ownerId: "user-1",
      enqueuedAt: Date.now(),
    });

    await handleReconciliationQueue(makeBatch([msg]), makeEnv());

    expect(mockReconcileBudget).toHaveBeenCalledTimes(1);
    expect(msg.ack).toHaveBeenCalledTimes(1);
  });
});
