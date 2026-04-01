import { enqueueWebhook } from "./webhook-queue.js";
import type { WebhookEndpointWithSecret } from "./webhook-cache.js";
import type { AnyWebhookEvent } from "./webhook-events.js";

export interface WebhookDispatcher {
  dispatch(endpoint: WebhookEndpointWithSecret, event: AnyWebhookEvent): Promise<void>;
}

/**
 * Create a webhook dispatcher backed by Cloudflare Queue.
 * Returns null if the WEBHOOK_QUEUE binding is not configured.
 *
 * The dispatcher enqueues a thin message (userId, endpointId, event).
 * The queue consumer handles signing, delivery, and retries.
 */
export function createWebhookDispatcher(
  queue: Queue | undefined,
  ownerId: string,
): WebhookDispatcher | null {
  if (!queue) return null;

  return {
    async dispatch(
      endpoint: WebhookEndpointWithSecret,
      event: AnyWebhookEvent,
    ): Promise<void> {
      // Event type filter: empty array = all events
      if (
        endpoint.eventTypes.length > 0 &&
        !endpoint.eventTypes.includes(event.type)
      ) {
        return;
      }

      await enqueueWebhook(queue, {
        ownerId,
        endpointId: endpoint.id,
        event,
      });
    },
  };
}

/**
 * Dispatch a webhook event to all matching endpoints.
 * Fail-open: errors are logged but never thrown.
 */
export async function dispatchToEndpoints(
  dispatcher: WebhookDispatcher,
  endpoints: WebhookEndpointWithSecret[],
  event: AnyWebhookEvent,
): Promise<void> {
  await Promise.allSettled(
    endpoints.map((endpoint) =>
      dispatcher.dispatch(endpoint, event).catch((err) => {
        console.error(
          `[webhook-dispatch] Failed to dispatch ${event.type} to ${endpoint.id}:`,
          err,
        );
      }),
    ),
  );
}
