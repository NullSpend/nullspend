import type { AnyWebhookEvent } from "./webhook-events.js";
import { emitMetric } from "./metrics.js";

/**
 * Thin webhook queue message. Consumer fetches endpoint details
 * (URL, signing secret) at delivery time for fresh signatures
 * and to handle endpoint changes between enqueue and delivery.
 */
export interface WebhookQueueMessage {
  ownerId: string;
  endpointId: string;
  event: AnyWebhookEvent;
}

/**
 * Enqueue a webhook event for delivery via Cloudflare Queue.
 * Fire-and-forget: errors are logged but never thrown.
 */
export async function enqueueWebhook(
  queue: Queue,
  message: WebhookQueueMessage,
): Promise<void> {
  try {
    await queue.send(message);
    emitMetric("webhook_enqueued", { eventType: message.event.type });
  } catch (err) {
    emitMetric("webhook_enqueue_failed", { eventType: message.event.type });
    console.error(
      `[webhook-queue] Failed to enqueue ${message.event.type} for endpoint ${message.endpointId}:`,
      err instanceof Error ? err.message : String(err),
    );
  }
}
