import type { CostEventMessage } from "./lib/cost-event-queue.js";
import { logCostEventsBatch, logCostEvent } from "./lib/cost-logger.js";
import { emitMetric } from "./lib/metrics.js";

export const COST_EVENT_QUEUE_NAME = "nullspend-cost-events";

/**
 * Cloudflare Queue consumer for cost event messages.
 *
 * Batch-first strategy: collects all events from the batch and attempts a
 * single multi-row INSERT. On batch failure, falls back to per-message
 * processing to isolate poisoned messages.
 */
export async function handleCostEventQueue(
  batch: MessageBatch<CostEventMessage>,
  env: Env,
): Promise<void> {
  if (batch.messages.length === 0) return;

  const connectionString = env.HYPERDRIVE.connectionString;
  const batchStartMs = Date.now();

  // Collect all event bodies
  const events = batch.messages.map((m) => m.body.event);

  emitMetric("cost_event_batch_size", { count: events.length });

  // Queue dwell time: how long did each message sit between enqueue and now?
  // Surfaces queue backpressure / consumer-idle regressions that live-smoke
  // `waitForCostEvent` polling would otherwise catch only as opaque timeouts.
  // Tracked as max + count>15s because p99 matters more than mean for polling budgets.
  let maxDwellMs = 0;
  let slowCount = 0;
  for (const msg of batch.messages) {
    const enqueuedAt = msg.body.enqueuedAt;
    if (typeof enqueuedAt === "number" && enqueuedAt > 0) {
      const dwell = batchStartMs - enqueuedAt;
      if (dwell > maxDwellMs) maxDwellMs = dwell;
      if (dwell > 15_000) slowCount++;
    }
  }
  emitMetric("cost_event_queue_dwell_ms", { max: maxDwellMs, slowCount, batchSize: events.length });
  if (slowCount > 0) {
    console.warn(
      `[cost-event-queue] ${slowCount}/${events.length} messages exceeded 15s dwell (max=${maxDwellMs}ms)`,
    );
  }

  try {
    // Attempt batch INSERT — throwOnError so we can fall back on failure
    await logCostEventsBatch(connectionString, events, { throwOnError: true });

    const ingestionMs = Date.now() - batchStartMs;
    emitMetric("cost_event_ingestion_latency", { ms: ingestionMs, count: events.length, mode: "batch" });

    // Batch succeeded — ack all messages
    for (const message of batch.messages) {
      message.ack();
    }
  } catch (err) {
    console.error("[cost-event-queue] Batch INSERT failed, falling back to per-message:", err);

    // Per-message fallback: try each individually to isolate poisoned messages
    let acked = 0;
    let retried = 0;
    for (const message of batch.messages) {
      try {
        await logCostEvent(connectionString, message.body.event, { throwOnError: true });
        message.ack();
        acked++;
      } catch (innerErr) {
        console.error("[cost-event-queue] Per-message write failed, retrying:", innerErr);
        message.retry();
        retried++;
      }
    }

    const ingestionMs = Date.now() - batchStartMs;
    emitMetric("cost_event_ingestion_latency", { ms: ingestionMs, count: events.length, mode: "per_message", acked, retried });
  }
}
