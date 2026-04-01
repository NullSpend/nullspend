import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { CostReporter } from "./cost-reporter.js";
import { NullSpendError } from "./errors.js";
import type { CostEventInput } from "./types.js";

// We use vi.advanceTimersByTimeAsync() (not the sync variant) because
// setTimeout callbacks invoke async flush(). The sync version fires the
// callback but cannot resolve the awaited promises inside it, leading to
// false negatives where sendBatch appears uncalled.

function makeEvent(id = 1): CostEventInput {
  return {
    provider: "openai",
    model: "gpt-4o",
    inputTokens: 100 * id,
    outputTokens: 50 * id,
    costMicrodollars: 500 * id,
  };
}

function makeEvents(count: number): CostEventInput[] {
  return Array.from({ length: count }, (_, i) => makeEvent(i + 1));
}

let sendBatch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.useFakeTimers();
  sendBatch = vi.fn().mockResolvedValue(undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Batch size flush
// ---------------------------------------------------------------------------

describe("batch size flush", () => {
  it("auto-flushes when batchSize events are enqueued", async () => {
    const reporter = new CostReporter({ batchSize: 3 }, sendBatch);
    const events = makeEvents(3);

    events.forEach((e) => reporter.enqueue(e));
    // Let the fire-and-forget flush() resolve
    await vi.advanceTimersByTimeAsync(0);

    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch).toHaveBeenCalledWith(events);

    await reporter.shutdown();
  });

  it("chunks into multiple batches (25 events, batchSize 10)", async () => {
    const reporter = new CostReporter({ batchSize: 10 }, sendBatch);
    const events = makeEvents(25);

    events.forEach((e) => reporter.enqueue(e));
    // Let the fire-and-forget auto-flush (triggered at event 10) resolve
    await vi.advanceTimersByTimeAsync(0);
    // Flush remaining events (11-25)
    await reporter.flush();

    expect(sendBatch).toHaveBeenCalledTimes(3);
    expect(sendBatch.mock.calls[0][0]).toHaveLength(10);
    expect(sendBatch.mock.calls[1][0]).toHaveLength(10);
    expect(sendBatch.mock.calls[2][0]).toHaveLength(5);

    await reporter.shutdown();
  });

  it("does not auto-flush when fewer than batchSize events enqueued", async () => {
    const reporter = new CostReporter({ batchSize: 5 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    await vi.advanceTimersByTimeAsync(0);

    expect(sendBatch).not.toHaveBeenCalled();

    await reporter.shutdown();
  });

  it("flushes on every enqueue when batchSize is 1", async () => {
    const reporter = new CostReporter({ batchSize: 1 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    await vi.advanceTimersByTimeAsync(0);
    reporter.enqueue(makeEvent(2));
    await vi.advanceTimersByTimeAsync(0);

    expect(sendBatch).toHaveBeenCalledTimes(2);

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// Timer flush
// ---------------------------------------------------------------------------

describe("timer flush", () => {
  it("flushes after flushIntervalMs", async () => {
    const reporter = new CostReporter(
      { flushIntervalMs: 2000, batchSize: 100 },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    expect(sendBatch).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2000);
    expect(sendBatch).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });

  it("manual flush resets timer (no double fire)", async () => {
    const reporter = new CostReporter(
      { flushIntervalMs: 1000, batchSize: 100 },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    // Advance part of the interval
    await vi.advanceTimersByTimeAsync(500);
    expect(sendBatch).not.toHaveBeenCalled();

    // Manual flush
    await reporter.flush();
    expect(sendBatch).toHaveBeenCalledOnce();

    // Advance rest of original interval — should NOT double fire
    await vi.advanceTimersByTimeAsync(500);
    expect(sendBatch).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });

  it("timer is no-op when queue is empty", async () => {
    const reporter = new CostReporter(
      { flushIntervalMs: 1000, batchSize: 100 },
      sendBatch,
    );

    await vi.advanceTimersByTimeAsync(3000);
    expect(sendBatch).not.toHaveBeenCalled();

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------

describe("flush()", () => {
  it("sends current queue immediately", async () => {
    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    await reporter.flush();

    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(2);

    await reporter.shutdown();
  });

  it("is a no-op when queue is empty", async () => {
    const reporter = new CostReporter({}, sendBatch);

    await reporter.flush();
    expect(sendBatch).not.toHaveBeenCalled();

    await reporter.shutdown();
  });

  it("concurrent calls do not double-send (mutex)", async () => {
    let resolveFlush!: () => void;
    sendBatch.mockImplementation(
      () => new Promise<void>((r) => { resolveFlush = r; }),
    );

    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);
    reporter.enqueue(makeEvent(1));

    const p1 = reporter.flush();
    const p2 = reporter.flush();

    resolveFlush();
    await p1;
    await p2;

    // Only one sendBatch call despite two concurrent flush() calls
    expect(sendBatch).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// shutdown()
// ---------------------------------------------------------------------------

describe("shutdown()", () => {
  it("flushes remaining events and clears timer", async () => {
    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    await reporter.shutdown();

    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(2);
    expect(reporter.isShutDown).toBe(true);
  });

  it("is idempotent (multiple calls do not double-flush)", async () => {
    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    await reporter.shutdown();
    await reporter.shutdown();

    expect(sendBatch).toHaveBeenCalledOnce();
  });

  it("enqueue() after shutdown throws NullSpendError", async () => {
    const reporter = new CostReporter({}, sendBatch);
    await reporter.shutdown();

    let thrown: unknown;
    try {
      reporter.enqueue(makeEvent(1));
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(NullSpendError);
    expect((thrown as Error).message).toBe("CostReporter is shut down");
  });
});

// ---------------------------------------------------------------------------
// Queue overflow
// ---------------------------------------------------------------------------

describe("queue overflow", () => {
  it("drops oldest events and calls onDropped", async () => {
    const onDropped = vi.fn();
    const reporter = new CostReporter(
      { maxQueueSize: 3, batchSize: 100, onDropped },
      sendBatch,
    );

    for (let i = 1; i <= 5; i++) {
      reporter.enqueue(makeEvent(i));
    }

    // Overflow happens per-enqueue: events 4 and 5 each trigger a drop of 1
    expect(onDropped).toHaveBeenCalledTimes(2);
    expect(onDropped).toHaveBeenCalledWith(1);

    await reporter.flush();
    // Should have events 3, 4, 5 (oldest 1, 2 dropped)
    expect(sendBatch.mock.calls[0][0]).toHaveLength(3);
    expect(sendBatch.mock.calls[0][0][0].inputTokens).toBe(300); // event(3)

    await reporter.shutdown();
  });

  it("uses console.warn when no onDropped callback", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const reporter = new CostReporter(
      { maxQueueSize: 2, batchSize: 100 },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    reporter.enqueue(makeEvent(3));

    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("dropped 1 event(s)"),
    );

    await reporter.shutdown();
  });

  it("retains newest events when overflow occurs", async () => {
    const onDropped = vi.fn();
    const reporter = new CostReporter(
      { maxQueueSize: 2, batchSize: 100, onDropped },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    reporter.enqueue(makeEvent(3));
    reporter.enqueue(makeEvent(4));

    await reporter.flush();

    const sent = sendBatch.mock.calls[0][0];
    expect(sent).toHaveLength(2);
    expect(sent[0].inputTokens).toBe(300); // event(3)
    expect(sent[1].inputTokens).toBe(400); // event(4)

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe("error handling", () => {
  it("failed sendBatch does not re-queue events, calls onFlushError", async () => {
    const batchError = new Error("network failure");
    sendBatch.mockRejectedValue(batchError);
    const onFlushError = vi.fn();

    const reporter = new CostReporter(
      { batchSize: 100, onFlushError },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    await reporter.flush();

    expect(onFlushError).toHaveBeenCalledWith(batchError, [makeEvent(1)]);

    // Queue should be empty — events are dropped, not re-queued
    sendBatch.mockResolvedValue(undefined);
    await reporter.flush();
    expect(sendBatch).toHaveBeenCalledOnce(); // only the failed call

    await reporter.shutdown();
  });

  it("failure on batch 2 of 3 still sends batch 3", async () => {
    const onFlushError = vi.fn();
    let callCount = 0;
    sendBatch.mockImplementation(() => {
      callCount++;
      if (callCount === 2) return Promise.reject(new Error("batch 2 failed"));
      return Promise.resolve();
    });

    const reporter = new CostReporter(
      { batchSize: 2, onFlushError },
      sendBatch,
    );

    for (let i = 1; i <= 6; i++) {
      reporter.enqueue(makeEvent(i));
    }
    // Let auto-flush (triggered at event 2) complete first
    await vi.advanceTimersByTimeAsync(0);
    // Flush remaining events (3-6)
    await reporter.flush();

    // 6 events / batchSize 2 = 3 batches total. All 3 attempted despite batch 2 failing.
    expect(sendBatch).toHaveBeenCalledTimes(3);
    expect(onFlushError).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });

  it("uses console.warn when no onFlushError callback", async () => {
    sendBatch.mockRejectedValue(new Error("boom"));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    await reporter.flush();

    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("flush failed: boom"),
    );

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

describe("config validation", () => {
  it("batchSize: NaN falls back to default 10", async () => {
    const reporter = new CostReporter({ batchSize: NaN }, sendBatch);

    // Enqueue 10 events — should trigger auto-flush at default batchSize 10
    makeEvents(10).forEach((e) => reporter.enqueue(e));
    await vi.advanceTimersByTimeAsync(0);

    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(10);

    await reporter.shutdown();
  });

  it("batchSize: 0 clamped to 1; batchSize: 200 clamped to 100", async () => {
    // batchSize 0 → clamped to 1 → flush on every enqueue
    const reporter1 = new CostReporter({ batchSize: 0 }, sendBatch);
    reporter1.enqueue(makeEvent(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(sendBatch).toHaveBeenCalledOnce();
    await reporter1.shutdown();

    sendBatch.mockClear();

    // batchSize 200 → clamped to 100
    const reporter2 = new CostReporter({ batchSize: 200 }, sendBatch);
    makeEvents(100).forEach((e) => reporter2.enqueue(e));
    await vi.advanceTimersByTimeAsync(0);
    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(100);
    await reporter2.shutdown();
  });

  it("flushIntervalMs: 50 clamped to 100", async () => {
    const reporter = new CostReporter(
      { flushIntervalMs: 50, batchSize: 100 },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));

    // At 50ms — should not have flushed yet (clamped to 100)
    await vi.advanceTimersByTimeAsync(50);
    expect(sendBatch).not.toHaveBeenCalled();

    // At 100ms — should flush
    await vi.advanceTimersByTimeAsync(50);
    expect(sendBatch).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });

  it("maxQueueSize: -1 clamped to 1", async () => {
    const onDropped = vi.fn();
    const reporter = new CostReporter(
      { maxQueueSize: -1, batchSize: 100, onDropped },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));

    expect(onDropped).toHaveBeenCalledWith(1);

    await reporter.shutdown();
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("edge cases", () => {
  it("flush() during active timer flush returns same promise (mutex)", async () => {
    let resolveFlush!: () => void;
    sendBatch.mockImplementation(
      () => new Promise<void>((r) => { resolveFlush = r; }),
    );

    const reporter = new CostReporter(
      { flushIntervalMs: 1000, batchSize: 100 },
      sendBatch,
    );

    reporter.enqueue(makeEvent(1));

    // Trigger timer-based flush
    const timerPromise = vi.advanceTimersByTimeAsync(1000);

    // Now call flush() manually — should get the same promise (mutex)
    const manualFlush = reporter.flush();

    resolveFlush();
    await timerPromise;
    await manualFlush;

    expect(sendBatch).toHaveBeenCalledOnce();

    await reporter.shutdown();
  });

  it("empty CostReportingConfig {} uses all defaults", async () => {
    const reporter = new CostReporter({}, sendBatch);

    // Default batchSize is 10 — enqueue 10 events to trigger
    makeEvents(10).forEach((e) => reporter.enqueue(e));
    await vi.advanceTimersByTimeAsync(0);

    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(10);

    await reporter.shutdown();
  });

  it("flush() after shutdown() is a no-op", async () => {
    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);
    reporter.enqueue(makeEvent(1));
    await reporter.shutdown();

    sendBatch.mockClear();

    await reporter.flush();
    expect(sendBatch).not.toHaveBeenCalled();
  });

  it("fire-and-forget flush from enqueue() does not produce unhandled rejection", async () => {
    sendBatch.mockRejectedValue(new Error("batch failed"));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const reporter = new CostReporter({ batchSize: 1 }, sendBatch);

    // This triggers a fire-and-forget flush().catch(() => {}) — no unhandled rejection
    reporter.enqueue(makeEvent(1));
    await vi.advanceTimersByTimeAsync(0);

    // The error was handled (no process crash). onFlushError defaults to console.warn.
    expect(warnSpy).toHaveBeenCalled();

    await reporter.shutdown();
  });

  it("onFlushError throwing does not abort remaining chunks", async () => {
    let callCount = 0;
    sendBatch.mockImplementation(() => {
      callCount++;
      // Every batch "fails"
      return Promise.reject(new Error(`fail-${callCount}`));
    });

    const onFlushError = vi.fn().mockImplementation(() => {
      throw new Error("callback exploded");
    });

    const reporter = new CostReporter(
      { batchSize: 2, onFlushError },
      sendBatch,
    );

    for (let i = 1; i <= 4; i++) {
      reporter.enqueue(makeEvent(i));
    }
    // Let auto-flush complete
    await vi.advanceTimersByTimeAsync(0);
    await reporter.flush();

    // Both batches attempted despite onFlushError throwing each time
    expect(sendBatch).toHaveBeenCalledTimes(2);
    expect(onFlushError).toHaveBeenCalledTimes(2);

    await reporter.shutdown();
  });

  it("onDropped throwing does not break enqueue", async () => {
    const onDropped = vi.fn().mockImplementation(() => {
      throw new Error("callback exploded");
    });

    const reporter = new CostReporter(
      { maxQueueSize: 2, batchSize: 100, onDropped },
      sendBatch,
    );

    // Should not throw despite onDropped throwing internally
    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    reporter.enqueue(makeEvent(3)); // triggers overflow + onDropped which throws

    // onDropped was called, and the enqueue still succeeded
    expect(onDropped).toHaveBeenCalledWith(1);

    await reporter.flush();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(2);

    await reporter.shutdown();
  });

  it("events enqueued during active flush are captured by next flush", async () => {
    let resolveFirst!: () => void;
    let callCount = 0;
    sendBatch.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        // First batch: slow, gives time to enqueue more
        return new Promise<void>((r) => { resolveFirst = r; });
      }
      return Promise.resolve();
    });

    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));
    reporter.enqueue(makeEvent(2));
    const firstFlush = reporter.flush();

    // While first flush is in progress, enqueue more events
    reporter.enqueue(makeEvent(3));
    reporter.enqueue(makeEvent(4));

    // Complete the first flush
    resolveFirst();
    await firstFlush;

    // Events 3 and 4 should be in the queue, not sent yet
    expect(sendBatch).toHaveBeenCalledOnce();
    expect(sendBatch.mock.calls[0][0]).toHaveLength(2);

    // Second flush picks up the new events
    await reporter.flush();
    expect(sendBatch).toHaveBeenCalledTimes(2);
    expect(sendBatch.mock.calls[1][0]).toHaveLength(2);
    expect(sendBatch.mock.calls[1][0][0].inputTokens).toBe(300); // event(3)

    await reporter.shutdown();
  });

  it("concurrent shutdown calls do not double-flush", async () => {
    const reporter = new CostReporter({ batchSize: 100 }, sendBatch);

    reporter.enqueue(makeEvent(1));

    // Call shutdown concurrently (not sequentially)
    const [r1, r2] = await Promise.all([
      reporter.shutdown(),
      reporter.shutdown(),
    ]);

    expect(r1).toBeUndefined();
    expect(r2).toBeUndefined();
    expect(sendBatch).toHaveBeenCalledOnce();
  });
});
