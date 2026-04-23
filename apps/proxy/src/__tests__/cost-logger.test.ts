/**
 * Unit tests for the cost-logger module.
 * Covers skipDbWrites behavior (console fallback), error resilience,
 * and batch-to-per-event fallback on batch INSERT failure (P1-4).
 *
 * Important: The cost-logger module is designed to NEVER throw,
 * because it runs inside waitUntil() where unhandled rejections
 * could crash the Cloudflare Workers runtime.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// --- Mock setup ---
// The sql tagged template function returned by getSql.
// Also needs .json() for JSONB serialization used by logCostEvent.
const mockSql = Object.assign(vi.fn().mockResolvedValue([]), {
  json: vi.fn((v: unknown) => v),
});

const mockEmitMetric = vi.fn();

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import { logCostEvent, logCostEventsBatch } from "../lib/cost-logger.js";

function makeCostEvent(overrides: Record<string, unknown> = {}) {
  return {
    requestId: "req-test-123",
    provider: "openai",
    model: "gpt-4o-mini",
    inputTokens: 50,
    outputTokens: 10,
    cachedInputTokens: 0,
    reasoningTokens: 0,
    costMicrodollars: 150,
    durationMs: 250,
    userId: "user-test",
    apiKeyId: null,
    actionId: null,
    ...overrides,
  };
}

const CONN = "postgresql://localhost:5432/db";

beforeEach(() => {
  vi.clearAllMocks();
  // Default: all SQL calls resolve successfully
  mockSql.mockResolvedValue([]);
});

describe("logCostEvent", () => {
  it("does not throw when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await expect(
      logCostEvent("postgresql://postgres:postgres@127.0.0.1:54322/postgres", makeCostEvent(), { skipDbWrites: true }),
    ).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });

  it("logs cost event to console when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(CONN, makeCostEvent(), { skipDbWrites: true });

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    const logArgs = consoleSpy.mock.calls[0];
    expect(logArgs[0]).toContain("[cost-logger]");
    expect(logArgs[0]).toContain("Local dev");

    const loggedEvent = logArgs[1] as Record<string, unknown>;
    expect(loggedEvent.requestId).toBe("req-test-123");
    expect(loggedEvent.provider).toBe("openai");
    expect(loggedEvent.model).toBe("gpt-4o-mini");
    expect(loggedEvent.inputTokens).toBe(50);
    expect(loggedEvent.outputTokens).toBe(10);
    expect(loggedEvent.costMicrodollars).toBe(150);
    consoleSpy.mockRestore();
  });

  it("includes durationMs in console log when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(
      CONN,
      makeCostEvent({ durationMs: 1234 }),
      { skipDbWrites: true },
    );

    const loggedEvent = consoleSpy.mock.calls[0][1] as Record<string, unknown>;
    expect(loggedEvent.durationMs).toBe(1234);
    consoleSpy.mockRestore();
  });

  it("does not attempt DB connection when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await logCostEvent(CONN, makeCostEvent(), { skipDbWrites: true });

    expect(errorSpy).not.toHaveBeenCalled();
    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("handles missing durationMs gracefully", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(
      CONN,
      makeCostEvent({ durationMs: undefined }),
      { skipDbWrites: true },
    );

    const loggedEvent = consoleSpy.mock.calls[0][1] as Record<string, unknown>;
    expect(loggedEvent.durationMs).toBeUndefined();
    consoleSpy.mockRestore();
  });

  it("handles zero-value cost event", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(
      CONN,
      makeCostEvent({
        inputTokens: 0,
        outputTokens: 0,
        cachedInputTokens: 0,
        reasoningTokens: 0,
        costMicrodollars: 0,
      }),
      { skipDbWrites: true },
    );

    const loggedEvent = consoleSpy.mock.calls[0][1] as Record<string, unknown>;
    expect(loggedEvent.inputTokens).toBe(0);
    expect(loggedEvent.outputTokens).toBe(0);
    expect(loggedEvent.costMicrodollars).toBe(0);
    consoleSpy.mockRestore();
  });

  it("handles very large token counts without overflow", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(
      CONN,
      makeCostEvent({
        inputTokens: 128_000,
        outputTokens: 16_384,
        costMicrodollars: 9_999_999,
      }),
      { skipDbWrites: true },
    );

    const loggedEvent = consoleSpy.mock.calls[0][1] as Record<string, unknown>;
    expect(loggedEvent.inputTokens).toBe(128_000);
    expect(loggedEvent.outputTokens).toBe(16_384);
    expect(loggedEvent.costMicrodollars).toBe(9_999_999);
    consoleSpy.mockRestore();
  });

  it("does not throw when DB write succeeds", async () => {
    await expect(
      logCostEvent(CONN, makeCostEvent()),
    ).resolves.toBeUndefined();
  });

  it("does not throw when DB write fails (graceful error handling)", async () => {
    mockSql.mockRejectedValueOnce(new Error("connection refused"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      logCostEvent(CONN, makeCostEvent()),
    ).resolves.toBeUndefined();

    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[cost-logger]"),
      "connection refused",
    );
    errorSpy.mockRestore();
  });

  it("throws when DB write fails and throwOnError is true", async () => {
    mockSql.mockRejectedValueOnce(new Error("connection refused"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      logCostEvent(CONN, makeCostEvent(), { throwOnError: true }),
    ).rejects.toThrow("connection refused");

    errorSpy.mockRestore();
  });

  // PR-2d: period stamping (codex R1 P1#3 + R4#1)
  describe("period_start / period_end stamping (PR-2d)", () => {
    // C52 — single-event INSERT stamps periods from caller-provided Dates
    it("C52: single INSERT passes period_start/period_end to sql tagged template", async () => {
      const periodStart = new Date("2026-04-01T00:00:00Z");
      const periodEnd = new Date("2026-05-01T00:00:00Z");

      await logCostEvent(CONN, makeCostEvent({ periodStart, periodEnd }));

      expect(mockSql).toHaveBeenCalledTimes(1);
      // sql tagged template: args[0] = TemplateStringsArray, args[1..] = interpolated values
      const args = mockSql.mock.calls[0].slice(1);
      expect(args).toContain(periodStart);
      expect(args).toContain(periodEnd);
    });

    // C53 — batch INSERT preserves per-event periods
    it("C53: batch INSERT preserves each event's periodStart/periodEnd", async () => {
      const marchStart = new Date("2026-03-01T00:00:00Z");
      const marchEnd = new Date("2026-04-01T00:00:00Z");
      const aprilStart = new Date("2026-04-01T00:00:00Z");
      const aprilEnd = new Date("2026-05-01T00:00:00Z");

      await logCostEventsBatch(CONN, [
        makeCostEvent({ requestId: "march-event", periodStart: marchStart, periodEnd: marchEnd }),
        makeCostEvent({ requestId: "april-event", periodStart: aprilStart, periodEnd: aprilEnd }),
      ]);

      // Batch path: sql(array) helper is called first, then sql`INSERT...` tagged template.
      // The first mockSql call is the sql(array) helper with the mapped rows.
      expect(mockSql).toHaveBeenCalled();
      const firstCallArg = mockSql.mock.calls[0][0] as unknown;
      // sql(array) passes the array as the first positional arg.
      expect(Array.isArray(firstCallArg)).toBe(true);
      const rows = firstCallArg as Array<Record<string, unknown>>;
      expect(rows).toHaveLength(2);
      expect(rows[0].period_start).toBe(marchStart);
      expect(rows[0].period_end).toBe(marchEnd);
      expect(rows[1].period_start).toBe(aprilStart);
      expect(rows[1].period_end).toBe(aprilEnd);
    });

    // C54 — missing periods insert NULL (backward-compat, no epoch-0 coercion)
    it("C54: missing periods pass null to sql (no epoch-0 coercion)", async () => {
      await logCostEvent(CONN, makeCostEvent());  // no periodStart/End in overrides

      expect(mockSql).toHaveBeenCalledTimes(1);
      const args = mockSql.mock.calls[0].slice(1);
      // The last two interpolated values in the INSERT are periodStart ?? null, periodEnd ?? null
      // Count nulls at the end: should include two trailing nulls (periods).
      // This is a structural assertion — batch mode also tested below.
      const trailingNulls = args.slice(-2);
      expect(trailingNulls).toEqual([null, null]);
    });

    // C54 batch variant: missing periods in batch map to null per row
    it("C54b: batch with missing periods yields per-row nulls", async () => {
      await logCostEventsBatch(CONN, [
        makeCostEvent({ requestId: "no-period-1" }),
        makeCostEvent({ requestId: "no-period-2" }),
      ]);

      const firstCallArg = mockSql.mock.calls[0][0] as unknown;
      expect(Array.isArray(firstCallArg)).toBe(true);
      const rows = firstCallArg as Array<Record<string, unknown>>;
      expect(rows[0].period_start).toBeNull();
      expect(rows[0].period_end).toBeNull();
      expect(rows[1].period_start).toBeNull();
      expect(rows[1].period_end).toBeNull();
    });
  });
});

describe("logCostEventsBatch", () => {
  it("returns immediately for empty array", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await expect(
      logCostEventsBatch(CONN, []),
    ).resolves.toBeUndefined();
    expect(consoleSpy).not.toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it("does not throw when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await expect(
      logCostEventsBatch(CONN, [
        makeCostEvent(),
        makeCostEvent({ requestId: "req-2" }),
      ], { skipDbWrites: true }),
    ).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });

  it("logs each event to console when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEventsBatch(CONN, [
      makeCostEvent({ requestId: "req-1", costMicrodollars: 100 }),
      makeCostEvent({ requestId: "req-2", costMicrodollars: 200 }),
      makeCostEvent({ requestId: "req-3", costMicrodollars: 300 }),
    ], { skipDbWrites: true });

    expect(consoleSpy).toHaveBeenCalledTimes(3);
    expect((consoleSpy.mock.calls[0][1] as Record<string, unknown>).requestId).toBe("req-1");
    expect((consoleSpy.mock.calls[1][1] as Record<string, unknown>).requestId).toBe("req-2");
    expect((consoleSpy.mock.calls[2][1] as Record<string, unknown>).requestId).toBe("req-3");
    consoleSpy.mockRestore();
  });

  it("handles single-element array", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEventsBatch(CONN, [
      makeCostEvent(),
    ], { skipDbWrites: true });

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    consoleSpy.mockRestore();
  });

  it("does not throw when DB write succeeds", async () => {
    await expect(
      logCostEventsBatch(CONN, [makeCostEvent()]),
    ).resolves.toBeUndefined();
  });
});

describe("logCostEventsBatch — per-event fallback (P1-4)", () => {
  it("falls back to per-event writes when batch INSERT fails", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // Batch INSERT calls mockSql twice: sql(array) helper + sql`...` tagged template.
    // Both must reject so the outer await throws. Subsequent per-event calls resolve.
    mockSql
      .mockRejectedValueOnce(new Error("batch syntax error")) // sql(array)
      .mockRejectedValueOnce(new Error("batch syntax error")) // sql`...`
      .mockResolvedValue([]); // per-event fallback calls

    await logCostEventsBatch(CONN, [
      makeCostEvent({ requestId: "req-1" }),
      makeCostEvent({ requestId: "req-2" }),
      makeCostEvent({ requestId: "req-3" }),
    ]);

    // Batch failure metric emitted
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_drop",
      { reason: "batch_pg_error_attempting_fallback", count: 3 },
    );

    // No per_event_fallback_pg_error metric (all 3 succeeded)
    expect(mockEmitMetric).not.toHaveBeenCalledWith(
      "cost_event_drop",
      expect.objectContaining({ reason: "per_event_fallback_pg_error" }),
    );

    // Recovery metric emitted — all 3 recovered
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_batch_recovered",
      { count: 3, total: 3 },
    );

    // Fallback result logged
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("3 succeeded, 0 failed out of 3"),
    );

    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("drops events and emits metric when per-event fallback also fails", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // All calls fail: batch + every per-event
    mockSql.mockRejectedValue(new Error("connection pool exhausted"));

    await logCostEventsBatch(CONN, [
      makeCostEvent({ requestId: "req-1" }),
      makeCostEvent({ requestId: "req-2" }),
    ]);

    // Batch failure metric
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_drop",
      { reason: "batch_pg_error_attempting_fallback", count: 2 },
    );

    // Per-event fallback failure metric — both events failed
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_drop",
      { reason: "per_event_fallback_pg_error", count: 2 },
    );

    // No recovery metric — zero succeeded
    expect(mockEmitMetric).not.toHaveBeenCalledWith(
      "cost_event_batch_recovered",
      expect.anything(),
    );

    // Fallback result logged
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("0 succeeded, 2 failed out of 2"),
    );

    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("partial per-event success emits metric only for failures", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // Batch: sql(array) + sql`...` both reject (2 calls).
    // Per-event fallback: event 1 → resolve, event 2 → reject, event 3 → resolve.
    mockSql
      .mockRejectedValueOnce(new Error("batch fail"))  // batch sql(array)
      .mockRejectedValueOnce(new Error("batch fail"))  // batch sql`...`
      .mockResolvedValueOnce([])                        // event 1
      .mockRejectedValueOnce(new Error("single fail")) // event 2
      .mockResolvedValueOnce([]);                       // event 3

    await logCostEventsBatch(CONN, [
      makeCostEvent({ requestId: "req-1" }),
      makeCostEvent({ requestId: "req-2" }),
      makeCostEvent({ requestId: "req-3" }),
    ]);

    // Per-event fallback failure metric — only 1 event failed
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_drop",
      { reason: "per_event_fallback_pg_error", count: 1 },
    );

    // Recovery metric — 2 out of 3 recovered
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "cost_event_batch_recovered",
      { count: 2, total: 3 },
    );

    // Fallback result
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("2 succeeded, 1 failed out of 3"),
    );

    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("does not throw with throwOnError when some per-event writes succeed", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // Batch: 2 calls reject. Per-event: event 1 succeeds, event 2 fails.
    mockSql
      .mockRejectedValueOnce(new Error("batch fail"))  // batch sql(array)
      .mockRejectedValueOnce(new Error("batch fail"))  // batch sql`...`
      .mockResolvedValueOnce([])                        // event 1 succeeds
      .mockRejectedValueOnce(new Error("single fail")); // event 2 fails

    // Should NOT throw because event 1 succeeded (not ALL failed)
    await expect(
      logCostEventsBatch(CONN, [
        makeCostEvent({ requestId: "req-1" }),
        makeCostEvent({ requestId: "req-2" }),
      ], { throwOnError: true }),
    ).resolves.toBeUndefined();

    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("throws with throwOnError only when ALL per-event writes fail", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // All calls fail (batch + per-event)
    mockSql.mockRejectedValue(new Error("total failure"));

    await expect(
      logCostEventsBatch(CONN, [
        makeCostEvent({ requestId: "req-1" }),
        makeCostEvent({ requestId: "req-2" }),
      ], { throwOnError: true }),
    ).rejects.toThrow("total failure");

    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });
});
