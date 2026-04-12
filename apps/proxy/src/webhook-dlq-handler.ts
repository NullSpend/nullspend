import type { WebhookQueueMessage } from "./lib/webhook-queue.js";
import { emitMetric } from "./lib/metrics.js";

export const WEBHOOK_DLQ_NAME = "nullspend-webhooks-dlq";

/**
 * Dead letter queue consumer for permanently failed webhook deliveries.
 * Always acks — messages in the DLQ have exhausted all retries.
 * Logs the failure and emits a metric for alerting.
 */
export async function handleWebhookDlq(
  batch: MessageBatch<WebhookQueueMessage>,
): Promise<void> {
  for (const msg of batch.messages) {
    console.error("[webhook-dlq] Permanently failed webhook delivery:", {
      ownerId: msg.body.ownerId,
      endpointId: msg.body.endpointId,
      eventType: msg.body.event.type,
      eventId: msg.body.event.id,
      attempts: msg.attempts,
    });

    emitMetric("webhook_delivery_failed", {
      endpointId: msg.body.endpointId,
      eventType: msg.body.event.type,
    });

    msg.ack();
  }
}
