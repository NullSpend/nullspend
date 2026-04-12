import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockEnqueueReconciliation } = vi.hoisted(() => ({
  mockEnqueueReconciliation: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: vi.fn(),
  doBudgetReconcile: vi.fn(),
}));

vi.mock("../lib/budget-spend.js", () => ({
  resetBudgetPeriod: vi.fn(),
}));

vi.mock("../lib/reconciliation-queue.js", () => ({
  enqueueReconciliation: (...args: unknown[]) => mockEnqueueReconciliation(...args),
}));

// We need to mock reconcileBudget but import reconcileBudgetQueued
// Since reconcileBudgetQueued calls reconcileBudget internally, we need a different approach
// Let's use the actual module but mock the internal dependencies

import { reconcileBudgetQueued } from "../lib/budget-orchestrator.js";

function makeEnv(): any {
  return {
    HYPERDRIVE: { connectionString: "postgresql://test:test@db:5432/test" },
  };
}

const budgetEntities = [
  {
    entityKey: "{budget}:user:user-1",
    entityType: "user",
    entityId: "user-1",
    maxBudget: 100_000_000,
    spend: 20_000_000,
    reserved: 0,
    policy: "strict_block",
  },
];

describe("reconcileBudgetQueued", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEnqueueReconciliation.mockResolvedValue(undefined);
  });

  it("uses queue when available", async () => {
    const mockQueue = {} as any;

    await reconcileBudgetQueued(
      mockQueue, makeEnv(), "user-1", "org-test", "res-123", 50_000,
      budgetEntities, "postgresql://test",
    );

    expect(mockEnqueueReconciliation).toHaveBeenCalledWith(
      mockQueue,
      expect.objectContaining({
        type: "reconcile",
        reservationId: "res-123",
        actualCostMicrodollars: 50_000,
        ownerId: "user-1",
        orgId: "org-test",
      }),
    );
  });

  it("falls back to direct reconciliation when queue is undefined", async () => {
    // When queue is undefined, reconcileBudgetQueued should call reconcileBudget directly
    // (which calls doBudgetReconcile for DO mode)
    await reconcileBudgetQueued(
      undefined, makeEnv(), "user-1", "org-test", "res-456", 25_000,
      budgetEntities, "postgresql://test",
    );

    // Queue was not called
    expect(mockEnqueueReconciliation).not.toHaveBeenCalled();
  });

  it("falls back to direct when queue send fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const mockQueue = {} as any;
    mockEnqueueReconciliation.mockRejectedValueOnce(new Error("queue unavailable"));

    await reconcileBudgetQueued(
      mockQueue, makeEnv(), "user-1", "org-test", "res-789", 30_000,
      budgetEntities, "postgresql://test",
    );

    expect(mockEnqueueReconciliation).toHaveBeenCalledTimes(1);
    expect(console.error).toHaveBeenCalledWith(
      "[budget-orchestrator] Queue send failed, falling back to direct:",
      expect.any(Error),
    );
  });

  it("skips queue when reservationId is null", async () => {
    const mockQueue = {} as any;

    await reconcileBudgetQueued(
      mockQueue, makeEnv(), "user-1", "org-test", null, 0,
      [], "postgresql://test",
    );

    // No queue or direct reconciliation for null reservationId
    expect(mockEnqueueReconciliation).not.toHaveBeenCalled();
  });
});
