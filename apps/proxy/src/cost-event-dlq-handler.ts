import type { CostEventMessage } from "./lib/cost-event-queue.js";
import { logCostEvent } from "./lib/cost-logger.js";
import { emitMetric } from "./lib/metrics.js";
import { optionalBinding } from "./lib/env.js";

export const COST_EVENT_DLQ_NAME = "nullspend-cost-events-dlq";

/**
 * Cloudflare Queue consumer for dead-lettered cost event messages.
 *
 * For each message: emit a metric, log a structured error, then attempt one
 * final best-effort write. If the DB write also fails, persist the event to
 * R2 as a durable operator-visible record that can be replayed later (P2 fix:
 * accounting loss must never become permanent). Always acks in a finally block
 * so messages are never retried at the queue level.
 */
export async function handleCostEventDlq(
  batch: MessageBatch<CostEventMessage>,
  env: Env,
): Promise<void> {
  let connectionString: string | null = null;
  try {
    connectionString = env.HYPERDRIVE.connectionString;
  } catch {
    // Hyperdrive binding unavailable — will fall through to R2 persistence
    console.error("[cost-event-dlq] HYPERDRIVE binding unavailable");
  }

  const r2 = optionalBinding(env, "BODY_STORAGE");

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

      // Attempt DB write. REGRESSION GUARD (P0-2): logCostEvent without
      // throwOnError swallows pg errors internally and returns normally, which
      // makes dbWriteOk always true and renders the R2 fallback below
      // unreachable. MUST pass throwOnError:true so pg failures propagate to
      // the outer catch and trigger R2 persistence.
      let dbWriteOk = false;
      if (connectionString) {
        try {
          await logCostEvent(connectionString, msg.event, { throwOnError: true });
          dbWriteOk = true;
        } catch (dbErr) {
          console.error("[cost-event-dlq] DB write failed:", dbErr);
        }
      }

      // P2 fix: if DB write failed, persist to R2 as durable record.
      // Key format: _dlq/{requestId}-{timestamp}.json — operator can list
      // and replay these with a recovery script.
      if (!dbWriteOk && r2) {
        try {
          const requestId = msg.event.requestId ?? "unknown";
          const key = `_dlq/${requestId}-${Date.now()}.json`;
          await r2.put(key, safeStringify(msg), {
            httpMetadata: { contentType: "application/json" },
          });
          emitMetric("cost_event_dlq_persisted_r2", {
            requestId,
            costMicrodollars: msg.event.costMicrodollars ?? 0,
          });
        } catch (r2Err) {
          // Both DB and R2 failed — event is lost. Metric is the only record.
          console.error("[cost-event-dlq] R2 persistence also failed:", r2Err);
          emitMetric("cost_event_dlq_lost", {
            requestId: msg.event.requestId ?? "unknown",
            costMicrodollars: msg.event.costMicrodollars ?? 0,
          });
        }
      } else if (!dbWriteOk && !r2) {
        emitMetric("cost_event_dlq_lost", {
          requestId: msg.event.requestId ?? "unknown",
          costMicrodollars: msg.event.costMicrodollars ?? 0,
        });
      }
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
