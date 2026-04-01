import type { NewCostEventRow } from "@nullspend/db";
import { logCostEvent, logCostEventsBatch } from "./cost-logger.js";
import { emitMetric } from "./metrics.js";

export interface CostEventMessage {
  type: "cost_event";
  event: Omit<NewCostEventRow, "id" | "createdAt">;
  enqueuedAt: number;
}

const QUEUE_SEND_TIMEOUT_MS = 5_000;

/**
 * Race a promise against a timeout. Rejects if the timeout fires first.
 * Clears the timer when the original promise settles to avoid dangling timers.
 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Queue send timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Enqueue a single cost event to Cloudflare Queues.
 * Resolves in <1ms (message written to disk). Times out after 5s.
 */
export async function enqueueCostEvent(
  queue: Queue,
  event: Omit<NewCostEventRow, "id" | "createdAt">,
): Promise<void> {
  const msg: CostEventMessage = {
    type: "cost_event",
    event,
    enqueuedAt: Date.now(),
  };
  await withTimeout(queue.send(msg), QUEUE_SEND_TIMEOUT_MS);
}

/**
 * Enqueue multiple cost events in a single atomic `queue.sendBatch()` call.
 * Up to 100 messages / 256KB total per Cloudflare Queues limits.
 * Times out after 5s.
 */
export async function enqueueCostEventsBatch(
  queue: Queue,
  events: Omit<NewCostEventRow, "id" | "createdAt">[],
): Promise<void> {
  if (events.length === 0) return;

  const messages = events.map((event) => ({
    body: {
      type: "cost_event" as const,
      event,
      enqueuedAt: Date.now(),
    },
  }));

  await withTimeout(queue.sendBatch(messages), QUEUE_SEND_TIMEOUT_MS);
}

/**
 * Returns the COST_EVENT_QUEUE binding from env, or undefined if absent.
 */
export function getCostEventQueue(env: Env): Queue | undefined {
  return (env as unknown as Record<string, unknown>).COST_EVENT_QUEUE as Queue | undefined;
}

/**
 * Queue-first cost event logging with direct fallback.
 * If the queue binding is present, enqueues the event. If absent or send fails,
 * falls back to direct `logCostEvent`.
 */
export async function logCostEventQueued(
  queue: Queue | undefined,
  connectionString: string,
  event: Omit<NewCostEventRow, "id" | "createdAt">,
): Promise<void> {
  if (queue) {
    try {
      await enqueueCostEvent(queue, event);
      return;
    } catch (err) {
      emitMetric("cost_event_queue_fallback", { reason: "send_failed" });
      console.error("[cost-event-queue] Queue send failed, falling back to direct:", err);
    }
  }
  await logCostEvent(connectionString, event);
}

/**
 * Queue-first batch cost event logging with direct fallback.
 * Uses `queue.sendBatch()` for atomic batch enqueue.
 * Falls back to `logCostEventsBatch` if queue is absent or send fails.
 */
export async function logCostEventsBatchQueued(
  queue: Queue | undefined,
  connectionString: string,
  events: Omit<NewCostEventRow, "id" | "createdAt">[],
): Promise<void> {
  if (events.length === 0) return;

  if (queue) {
    try {
      await enqueueCostEventsBatch(queue, events);
      return;
    } catch (err) {
      emitMetric("cost_event_queue_fallback", { reason: "send_batch_failed", count: events.length });
      console.error("[cost-event-queue] Queue sendBatch failed, falling back to direct:", err);
    }
  }
  await logCostEventsBatch(connectionString, events);
}
