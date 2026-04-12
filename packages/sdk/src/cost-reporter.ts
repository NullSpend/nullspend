import { NullSpendError } from "./errors.js";
import type { CostEventInput, CostReportingConfig } from "./types.js";

const DEFAULT_BATCH_SIZE = 10;
const DEFAULT_FLUSH_INTERVAL_MS = 5_000;
const DEFAULT_MAX_QUEUE_SIZE = 1_000;

function toFiniteInt(value: number | undefined, fallback: number): number {
  const v = value ?? fallback;
  if (!Number.isFinite(v)) return fallback;
  return Math.floor(v);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export class CostReporter {
  private readonly batchSize: number;
  private readonly flushIntervalMs: number;
  private readonly maxQueueSize: number;
  private readonly onDropped: ((count: number) => void) | undefined;
  private readonly onFlushError:
    | ((error: Error, events: CostEventInput[]) => void)
    | undefined;
  private readonly sendBatch: (events: CostEventInput[]) => Promise<void>;

  private queue: CostEventInput[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private flushing: Promise<void> | null = null;
  private _isShutDown = false;

  constructor(
    config: CostReportingConfig,
    sendBatch: (events: CostEventInput[]) => Promise<void>,
  ) {
    this.batchSize = clamp(
      toFiniteInt(config.batchSize, DEFAULT_BATCH_SIZE),
      1,
      100,
    );
    this.flushIntervalMs = Math.max(
      100,
      toFiniteInt(config.flushIntervalMs, DEFAULT_FLUSH_INTERVAL_MS),
    );
    this.maxQueueSize = Math.max(
      1,
      toFiniteInt(config.maxQueueSize, DEFAULT_MAX_QUEUE_SIZE),
    );
    this.onDropped = config.onDropped;
    this.onFlushError = config.onFlushError;
    this.sendBatch = sendBatch;

    this.scheduleFlush();
  }

  get isShutDown(): boolean {
    return this._isShutDown;
  }

  enqueue(event: CostEventInput): void {
    if (this._isShutDown) {
      throw new NullSpendError("CostReporter is shut down");
    }

    this.queue.push(event);

    // Check overflow — drop oldest events from front
    if (this.queue.length > this.maxQueueSize) {
      const dropCount = this.queue.length - this.maxQueueSize;
      this.queue.splice(0, dropCount);
      try {
        if (this.onDropped) {
          this.onDropped(dropCount);
        } else {
          console.warn(
            `CostReporter: dropped ${dropCount} event(s) due to queue overflow`,
          );
        }
      } catch {
        // Never let a user callback break the enqueue flow
      }
    }

    // Auto-flush when batch size reached
    if (this.queue.length >= this.batchSize) {
      this.flush().catch((err) => {
        if (typeof console !== "undefined") console.error("[CostReporter] Auto-flush failed:", err);
      });
    }
  }

  async flush(): Promise<void> {
    // Mutex: if already flushing, return existing promise
    if (this.flushing) {
      return this.flushing;
    }

    if (this.queue.length === 0) {
      return;
    }

    this.flushing = this.doFlush();
    try {
      await this.flushing;
    } finally {
      this.flushing = null;
      if (!this._isShutDown) {
        this.scheduleFlush();
      }
    }
  }

  async shutdown(): Promise<void> {
    if (this._isShutDown) {
      // Already shut down — wait for any in-flight flush to complete.
      if (this.flushing) {
        await this.flushing;
      }
      return;
    }

    this._isShutDown = true;

    if (this.flushTimer !== null) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    // Drain loop: flush() short-circuits if a flush is already in flight,
    // returning the existing promise. After awaiting that, the queue may
    // still contain events that were enqueue()d during the in-flight flush.
    // Loop until queue is empty AND no flush is in flight.
    //
    // Bounded to MAX_DRAIN_ITERATIONS so a pathological producer can't
    // keep us looping forever. Each iteration flushes the entire current
    // queue snapshot, so non-pathological cases converge in 1-2 iterations.
    const MAX_DRAIN_ITERATIONS = 16;
    let iterations = 0;
    while (
      (this.queue.length > 0 || this.flushing !== null) &&
      iterations < MAX_DRAIN_ITERATIONS
    ) {
      iterations++;
      // Wait for any in-flight flush first.
      if (this.flushing) {
        await this.flushing;
      }
      // If the queue accumulated more events during the in-flight flush,
      // run another flush() to drain them. flush() will splice the queue
      // atomically inside doFlush().
      if (this.queue.length > 0) {
        await this.flush();
      }
    }

    if (this.queue.length > 0) {
      // Pathological case: producer kept enqueueing for MAX_DRAIN_ITERATIONS
      // batches. Drop the snapshot (we cannot loop forever) and signal
      // the loss via BOTH onFlushError AND console.warn so a no-op or
      // throwing callback cannot make the loss silent.
      const dropped = this.queue.splice(0);
      const errMessage = `CostReporter.shutdown drain exhausted ${MAX_DRAIN_ITERATIONS} iterations, ${dropped.length} events dropped`;
      try {
        if (this.onFlushError) {
          this.onFlushError(new Error(errMessage), dropped);
        }
      } catch { /* ignore callback error */ }
      // Always emit a stderr warning, even if the callback fired correctly.
      // This guarantees the loss is visible in logs / CI output regardless
      // of how the caller's callback behaves.
      console.warn(`[CostReporter] ${errMessage}`);
    }
  }

  private async doFlush(): Promise<void> {
    // Snapshot: splice entire queue atomically
    const snapshot = this.queue.splice(0);

    // Chunk and send
    for (let i = 0; i < snapshot.length; i += this.batchSize) {
      const chunk = snapshot.slice(i, i + this.batchSize);
      try {
        await this.sendBatch(chunk);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        try {
          if (this.onFlushError) {
            this.onFlushError(error, chunk);
          } else {
            console.warn(`CostReporter: flush failed: ${error.message}`);
          }
        } catch {
          // Never let a user callback abort remaining chunks
        }
        // Drop failed batch, continue with remaining chunks
      }
    }
  }

  private scheduleFlush(): void {
    if (this.flushTimer !== null) {
      clearTimeout(this.flushTimer);
    }

    this.flushTimer = setTimeout(() => {
      if (this.queue.length === 0) {
        // Nothing to flush — reschedule directly (flush() would skip scheduleFlush)
        if (!this._isShutDown) {
          this.scheduleFlush();
        }
        return;
      }
      // flush() reschedules in its finally block
      this.flush().catch((err) => {
        if (typeof console !== "undefined") console.error("[CostReporter] Auto-flush failed:", err);
      });
    }, this.flushIntervalMs);

    // Unref the timer so it doesn't keep the process alive.
    // Guard at runtime since ReturnType<typeof setTimeout> may be `number` in some envs.
    if (
      this.flushTimer &&
      typeof this.flushTimer === "object" &&
      "unref" in this.flushTimer
    ) {
      (this.flushTimer as { unref: () => void }).unref();
    }
  }
}
