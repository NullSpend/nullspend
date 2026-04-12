import type { CostEventMessage } from "./lib/cost-event-queue.js";
import { logCostEvent } from "./lib/cost-logger.js";
import { emitMetric } from "./lib/metrics.js";

export const COST_EVENT_DLQ_NAME = "nullspend-cost-events-dlq";

/**
 * Cloudflare Queue consumer for dead-lettered cost event messages.
 *
 * For each message: emit a metric, log a structured error, then attempt one
 * final best-effort write (no throwOnError — never throws). Always acks in
 * a finally block so messages are never retried at the queue level.
 */
export async function handleCostEventDlq(
  batch: MessageBatch<CostEventMessage>,
  env: Env,
): Promise<void> {
  let connectionString: string;
  try {
    connectionString = env.HYPERDRIVE.connectionString;
  } catch (err) {
    // Hyperdrive binding unavailable — ack all messages to avoid silent discard
    // (DLQ has max_retries=0, so unacked messages are permanently lost)
    console.error("[cost-event-dlq] HYPERDRIVE binding unavailable, acking all messages:", err);
    for (const message of batch.messages) {
      emitMetric("cost_event_dlq", {
        requestId: "unknown",
        costMicrodollars: 0,
        userId: "unknown",
        ageMs: -1,
      });
      message.ack();
    }
    return;
  }

  for (const message of batch.messages) {
    try {
      const msg = message.body;
      const ageMs =
        typeof msg.enqueuedAt === "number" && msg.enqueuedAt > 0
          ? Date.now() - msg.enqueuedAt
          : -1;

      emitMetric("cost_event_dlq", {
        requestId: msg.event.requestId ?? "unknown",
        costMicrodollars: msg.event.costMicrodollars ?? 0,
        userId: msg.event.userId ?? "unknown",
        ageMs,
      });

      console.error(
        "[cost-event-dlq] Dead-lettered cost event:",
        safeStringify(msg),
      );

      // Best-effort individual write — never throws
      await logCostEvent(connectionString, msg.event);
    } catch (err) {
      console.error("[cost-event-dlq] Unexpected error processing message:", err);
    } finally {
      message.ack();
    }
  }
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
