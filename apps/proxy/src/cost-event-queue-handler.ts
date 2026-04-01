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
