/**
 * Unit tests for the cost-logger module.
 * Covers skipDbWrites behavior (console fallback) and error resilience.
 *
 * Important: The cost-logger module is designed to NEVER throw,
 * because it runs inside waitUntil() where unhandled rejections
 * could crash the Cloudflare Workers runtime.
 */
import { describe, it, expect, vi } from "vitest";

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
    await logCostEvent("postgresql://localhost:5432/db", makeCostEvent(), { skipDbWrites: true });

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
      "postgresql://localhost:5432/db",
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

    await logCostEvent("postgresql://localhost:5432/db", makeCostEvent(), { skipDbWrites: true });

    expect(errorSpy).not.toHaveBeenCalled();
    consoleSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it("handles missing durationMs gracefully", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEvent(
      "postgresql://localhost:5432/db",
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
      "postgresql://localhost:5432/db",
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
      "postgresql://localhost:5432/db",
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

  it("does not throw for unreachable remote connection (graceful error handling)", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      logCostEvent("postgresql://user:pass@192.0.2.1:5432/db", makeCostEvent()),
    ).resolves.toBeUndefined();

    errorSpy.mockRestore();
  }, 15_000);
});

describe("logCostEventsBatch", () => {
  it("returns immediately for empty array", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await expect(
      logCostEventsBatch("postgresql://localhost:5432/db", []),
    ).resolves.toBeUndefined();
    expect(consoleSpy).not.toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it("does not throw when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await expect(
      logCostEventsBatch("postgresql://localhost:5432/db", [
        makeCostEvent(),
        makeCostEvent({ requestId: "req-2" }),
      ], { skipDbWrites: true }),
    ).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });

  it("logs each event to console when skipDbWrites is true", async () => {
    const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    await logCostEventsBatch("postgresql://localhost:5432/db", [
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
    await logCostEventsBatch("postgresql://localhost:5432/db", [
      makeCostEvent(),
    ], { skipDbWrites: true });

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    consoleSpy.mockRestore();
  });

  it("does not throw for unreachable remote connection (graceful error handling)", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      logCostEventsBatch("postgresql://user:pass@192.0.2.1:5432/db", [makeCostEvent()]),
    ).resolves.toBeUndefined();

    errorSpy.mockRestore();
  }, 15_000);
});
