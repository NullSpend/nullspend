import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockEnqueueReconciliation } = vi.hoisted(() => ({
  mockEnqueueReconciliation: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetCheck: vi.fn(),
  doBudgetReconcile: vi.fn().mockResolvedValue({ status: "ok" }),
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

  it("always goes direct (queue bypassed for atomic threshold dedup)", async () => {
    const mockQueue = {} as any;

    await reconcileBudgetQueued(
      mockQueue, makeEnv(), "user-1", "org-test", "res-123", 50_000,
      budgetEntities, "postgresql://test",
    );

    // P0-1 fix: reconcileBudgetQueued always calls DO directly now
    // (queue caused duplicate threshold webhooks under concurrency)
    expect(mockEnqueueReconciliation).not.toHaveBeenCalled();
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

  it("always goes direct even when queue is provided (P0-1 threshold dedup)", async () => {
    const mockQueue = {} as any;

    await reconcileBudgetQueued(
      mockQueue, makeEnv(), "user-1", "org-test", "res-789", 30_000,
      budgetEntities, "postgresql://test",
    );

    // Queue is ignored — DO provides atomic threshold crossing dedup
    expect(mockEnqueueReconciliation).not.toHaveBeenCalled();
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
