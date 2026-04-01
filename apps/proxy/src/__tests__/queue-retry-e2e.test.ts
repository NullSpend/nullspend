/**
 * End-to-end test for the queue retry fix.
 *
 * Bug: `doBudgetReconcile` never throws (catches errors internally),
 * so the queue handler's try/catch never triggered `message.retry()`.
 *
 * Fix: `doBudgetReconcile` returns a status string ("ok" | "pg_failed" | "error")
 * and `reconcileBudget` throws when `throwOnError` is set.
 *
 * These tests run the FULL chain:
 *   handleReconciliationQueue → reconcileBudget → doBudgetReconcile
 * with only the DO stub and `updateBudgetSpend` mocked.
 */
import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Hoisted mocks — must be declared before any import that touches them
// ---------------------------------------------------------------------------

const { mockUpdateBudgetSpend, mockReconcileStub } = vi.hoisted(() => ({
  mockUpdateBudgetSpend: vi.fn(),
  mockReconcileStub: vi.fn(),
}));

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

// Mock only the PG write — let everything else run for real
vi.mock("../lib/budget-spend.js", () => ({
  updateBudgetSpend: (...args: unknown[]) => mockUpdateBudgetSpend(...args),
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
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  // -----------------------------------------------------------------------
  // 1. PG failure → retry triggered
  // -----------------------------------------------------------------------
  describe("PG failure → retry triggered", () => {
    it("doBudgetReconcile returns 'pg_failed' when updateBudgetSpend fails on all retries", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockRejectedValue(new Error("PG connection refused"));

      const env = makeEnv();
      const status = await doBudgetReconcile(
        env,
        "user-abc",
        "res-123",
        50_000,
        [{ entityType: "api_key", entityId: "key-1" }],
        "postgresql://test:test@db:5432/test",
      );

      expect(status).toBe("pg_failed");
      // PG_MAX_RETRIES = 2 → attempts 0, 1, 2 = 3 total calls
      expect(mockUpdateBudgetSpend).toHaveBeenCalledTimes(3);
    });

    it("reconcileBudget with throwOnError throws on pg_failed status", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockRejectedValue(new Error("PG connection refused"));

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
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
      ).rejects.toThrow("Reconciliation failed with status: pg_failed");
    });

    it("queue handler calls message.retry() (not ack) when PG fails", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockRejectedValue(new Error("PG connection refused"));

      const msg = makeMessage(makeBody());
      await handleReconciliationQueue(makeBatch([msg]), makeEnv());

      expect(msg.retry).toHaveBeenCalledTimes(1);
      expect(msg.ack).not.toHaveBeenCalled();
    });
  });

  // -----------------------------------------------------------------------
  // 2. DO failure → retry triggered
  // -----------------------------------------------------------------------
  describe("DO failure → retry triggered", () => {
    it("doBudgetReconcile returns 'error' when DO stub rejects", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));
      mockUpdateBudgetSpend.mockResolvedValue(undefined);

      const env = makeEnv();
      const status = await doBudgetReconcile(
        env,
        "user-abc",
        "res-123",
        50_000,
        [{ entityType: "api_key", entityId: "key-1" }],
        "postgresql://test:test@db:5432/test",
      );

      expect(status).toBe("error");
      // updateBudgetSpend should NOT be called — DO failed before we got there
      expect(mockUpdateBudgetSpend).not.toHaveBeenCalled();
    });

    it("reconcileBudget with throwOnError throws on DO error", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
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
  // 3. Success → ack (not retry)
  // -----------------------------------------------------------------------
  describe("Success → ack (not retry)", () => {
    it("doBudgetReconcile returns 'ok' when both DO and PG succeed", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockResolvedValue(undefined);

      const env = makeEnv();
      const status = await doBudgetReconcile(
        env,
        "user-abc",
        "res-123",
        50_000,
        [{ entityType: "api_key", entityId: "key-1" }],
        "postgresql://test:test@db:5432/test",
      );

      expect(status).toBe("ok");
      expect(mockUpdateBudgetSpend).toHaveBeenCalledTimes(1);
    });

    it("reconcileBudget does NOT throw on success", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockResolvedValue(undefined);

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
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
      ).resolves.toBeUndefined();
    });

    it("queue handler calls message.ack() (not retry) on success", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockResolvedValue(undefined);

      const msg = makeMessage(makeBody());
      await handleReconciliationQueue(makeBatch([msg]), makeEnv());

      expect(msg.ack).toHaveBeenCalledTimes(1);
      expect(msg.retry).not.toHaveBeenCalled();
    });
  });

  // -----------------------------------------------------------------------
  // 4. Verify the OLD behavior is gone — without throwOnError,
  //    reconcileBudget should NOT throw even on failure
  // -----------------------------------------------------------------------
  describe("Without throwOnError (fallback/direct path)", () => {
    it("reconcileBudget does NOT throw on pg_failed status", async () => {
      mockReconcileStub.mockResolvedValue({ status: "ok" });
      mockUpdateBudgetSpend.mockRejectedValue(new Error("PG connection refused"));

      const env = makeEnv();
      // No throwOnError option — this is the direct/fallback path
      await expect(
        reconcileBudget(
          env,
          "user-abc",
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
          // no options — throwOnError defaults to false
        ),
      ).resolves.toBeUndefined();
    });

    it("reconcileBudget does NOT throw on DO error without throwOnError", async () => {
      mockReconcileStub.mockRejectedValue(new Error("DO unavailable"));

      const env = makeEnv();
      await expect(
        reconcileBudget(
          env,
          "user-abc",
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
      ).resolves.toBeUndefined();
    });
  });
});
