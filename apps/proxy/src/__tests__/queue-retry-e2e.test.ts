/**
 * End-to-end test for the queue retry fix.
 *
 * Bug: `doBudgetReconcile` never throws (catches errors internally),
 * so the queue handler's try/catch never triggered `message.retry()`.
 *
 * Fix: `doBudgetReconcile` returns a status string ("ok" | "error")
 * and `reconcileBudget` throws when `throwOnError` is set.
 *
 * Strategy E removed the optimistic PG write — the DO's outbox + alarm
 * handler owns Postgres sync entirely. The only failure mode that triggers
 * queue retry is a DO error.
 *
 * These tests run the FULL chain:
 *   handleReconciliationQueue → reconcileBudget → doBudgetReconcile
 * with only the DO stub mocked.
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Hoisted mocks — must be declared before any import that touches them
// ---------------------------------------------------------------------------

const { mockReconcileStub } = vi.hoisted(() => ({
  mockReconcileStub: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

// Mock budget-spend to avoid loading real Postgres dependencies.
// Strategy E removed the optimistic PG write from doBudgetReconcile,
// but budget-orchestrator still imports resetBudgetPeriod for checkBudget.
vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: vi.fn().mockResolvedValue(undefined),
  resetBudgetPeriod: vi.fn().mockResolvedValue(undefined),
}));

// Mock emitMetric to suppress console noise
vi.mock("../lib/metrics.js", () => ({
  emitMetric: vi.fn(),
}));

// We do NOT mock budget-orchestrator or budget-do-client —
// the real code paths run end-to-end.

import { handleReconciliationQueue } from "../queue-handler.js";
import { reconcileBudget } from "../lib/budget-orchestrator.js";
import { doBudgetReconcile } from "../lib/budget-do-client.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEnv(): any {
  return {
    HYPERDRIVE: { connectionString: "postgresql://test:test@db:5432/test" },
    USER_BUDGET: {
      idFromName: (name: string) => ({ name }),
      get: (_id: any) => ({
        reconcile: mockReconcileStub,
      }),
    },
  };
}

function makeMessage(body: any): {
  body: any;
  ack: ReturnType<typeof vi.fn>;
  retry: ReturnType<typeof vi.fn>;
} {
  return {
    body,
    ack: vi.fn(),
    retry: vi.fn(),
  };
}

function makeBatch(messages: any[]): any {
  return { messages };
}

const ENTITIES = [
  { entityKey: "{budget}:api_key:key-1", entityType: "api_key", entityId: "key-1" },
];

function makeBody(overrides?: Record<string, unknown>) {
  return {
    type: "reconcile",
    reservationId: "res-123",
    actualCostMicrodollars: 50_000,
    budgetEntities: ENTITIES,
    ownerId: "user-abc",
    orgId: "org-test",
    enqueuedAt: Date.now(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Queue retry fix — end-to-end chain", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  // -----------------------------------------------------------------------
  // 1. DO failure → retry triggered
  // -----------------------------------------------------------------------
  describe("DO failure → retry triggered", () => {
    it("doBudgetReconcile returns 'error' when DO stub rejects", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const env = makeEnv();
      const status = await doBudgetReconcile(
        env,
        "user-abc",
        "org-test",
        "res-123",
        50_000,
        [{ entityType: "api_key", entityId: "key-1" }],
      );

      expect(status.status).toBe("error");
    });

    it("reconcileBudget with throwOnError throws on DO error", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
          "org-test",
          "res-123",
          50_000,
          [
            {
              entityKey: "{budget}:api_key:key-1",
              entityType: "api_key",
              entityId: "key-1",
              maxBudget: 0,
              spend: 0,
              reserved: 0,
              policy: "strict_block",
            },
          ],
          "postgresql://test:test@db:5432/test",
          { throwOnError: true },
        ),
      ).rejects.toThrow("Reconciliation failed with status: error");
    });

    it("queue handler calls message.retry() (not ack) when DO fails", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const msg = makeMessage(makeBody());
      await handleReconciliationQueue(makeBatch([msg]), makeEnv());

      expect(msg.retry).toHaveBeenCalledTimes(1);
      expect(msg.ack).not.toHaveBeenCalled();
    });
  });

  // -----------------------------------------------------------------------
  // 2. Success → ack (not retry)
  // -----------------------------------------------------------------------
  describe("Success → ack (not retry)", () => {
    it("doBudgetReconcile returns 'ok' when DO succeeds", async () => {
      mockReconcileStub.mockResolvedValue({ status: "reconciled", thresholdCrossings: [] });

      const env = makeEnv();
      const status = await doBudgetReconcile(
        env,
        "user-abc",
        "org-test",
        "res-123",
        50_000,
        [{ entityType: "api_key", entityId: "key-1" }],
      );

      expect(status.status).toBe("ok");
    });

    it("reconcileBudget does NOT throw on success", async () => {
      mockReconcileStub.mockResolvedValue({ status: "reconciled", thresholdCrossings: [] });

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
          "org-test",
          "res-123",
          50_000,
          [
            {
              entityKey: "{budget}:api_key:key-1",
              entityType: "api_key",
              entityId: "key-1",
              maxBudget: 0,
              spend: 0,
              reserved: 0,
              policy: "strict_block",
            },
          ],
          "postgresql://test:test@db:5432/test",
          { throwOnError: true },
        ),
      ).resolves.toEqual({ thresholdCrossings: [] });
    });

    it("queue handler calls message.ack() (not retry) on success", async () => {
      mockReconcileStub.mockResolvedValue({ status: "reconciled", thresholdCrossings: [] });

      const msg = makeMessage(makeBody());
      await handleReconciliationQueue(makeBatch([msg]), makeEnv());

      expect(msg.ack).toHaveBeenCalledTimes(1);
      expect(msg.retry).not.toHaveBeenCalled();
    });
  });

  // -----------------------------------------------------------------------
  // 3. Without throwOnError — reconcileBudget should NOT throw on failure
  // -----------------------------------------------------------------------
  describe("Without throwOnError (fallback/direct path)", () => {
    it("reconcileBudget does NOT throw on DO error without throwOnError", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
          "org-test",
          "res-123",
          50_000,
          [
            {
              entityKey: "{budget}:api_key:key-1",
              entityType: "api_key",
              entityId: "key-1",
              maxBudget: 0,
              spend: 0,
              reserved: 0,
              policy: "strict_block",
            },
          ],
          "postgresql://test:test@db:5432/test",
        ),
      ).resolves.toEqual({ thresholdCrossings: [] });
    });
  });
});
