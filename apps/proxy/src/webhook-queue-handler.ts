import type { WebhookQueueMessage } from "./lib/webhook-queue.js";
import { getWebhookEndpointsWithSecrets, type WebhookEndpointWithSecret } from "./lib/webhook-cache.js";
import { dualSignWebhookPayload, SECRET_ROTATION_WINDOW_SECONDS } from "./lib/webhook-signer.js";
import { emitMetric } from "./lib/metrics.js";

export const WEBHOOK_QUEUE_NAME = "nullspend-webhooks";

function isWithinRotationWindow(rotatedAt: string | null): boolean {
  if (!rotatedAt) return false;
  const elapsed = Date.now() - new Date(rotatedAt).getTime();
  return elapsed < SECRET_ROTATION_WINDOW_SECONDS * 1000;
}

/**
 * Cloudflare Queue consumer for webhook delivery.
 *
 * For each message:
 * 1. Look up endpoint with signing secret (batch-cached per userId)
 * 2. Sign the payload with a fresh timestamp
 * 3. Deliver via fetch() to the endpoint URL
 * 4. Ack on success (2xx), retry on transient failure (5xx/429/network), ack on permanent failure (4xx)
 */
export async function handleWebhookQueue(
  batch: MessageBatch<WebhookQueueMessage>,
  env: Env,
): Promise<void> {
  if (batch.messages.length === 0) return;

  const connectionString = env.HYPERDRIVE.connectionString;

  // Lazy per-batch cache: one DB call per unique ownerId.
  // Only cache non-empty results — empty could be a fail-open DB error
  // from getWebhookEndpointsWithSecrets, and caching it would silently
  // skip all remaining messages for that owner in the batch.
  const endpointCache = new Map<string, WebhookEndpointWithSecret[]>();
  async function getEndpoints(ownerId: string): Promise<WebhookEndpointWithSecret[]> {
    if (endpointCache.has(ownerId)) return endpointCache.get(ownerId)!;
    const endpoints = await getWebhookEndpointsWithSecrets(connectionString, ownerId);
    if (endpoints.length > 0) endpointCache.set(ownerId, endpoints);
    return endpoints;
  }

  for (const msg of batch.messages) {
    const { ownerId, endpointId, event } = msg.body;

    try {
      // Look up endpoint with secrets
      const endpoints = await getEndpoints(ownerId);
      const endpoint = endpoints.find((ep) => ep.id === endpointId);

      if (!endpoint) {
        // Endpoint deleted between enqueue and delivery — skip
        console.warn(`[webhook-queue] Endpoint ${endpointId} not found for owner ${ownerId}, skipping`);
        msg.ack();
        continue;
      }

      // Sign with fresh timestamp at delivery time
      const payload = JSON.stringify(event);
      const timestamp = Math.floor(Date.now() / 1000);
      const signature = await dualSignWebhookPayload(
        payload,
        endpoint.signingSecret,
        isWithinRotationWindow(endpoint.secretRotatedAt) ? endpoint.previousSigningSecret : null,
        timestamp,
      );

      // Deliver
      const startMs = Date.now();
      const res = await fetch(endpoint.url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-NullSpend-Signature": signature,
          "X-NullSpend-Webhook-Id": event.id,
          "X-NullSpend-Webhook-Timestamp": String(timestamp),
          "User-Agent": "NullSpend-Webhooks/1.0",
        },
        body: payload,
        signal: AbortSignal.timeout(30_000),
      });
      const durationMs = Date.now() - startMs;

      if (res.ok) {
        msg.ack();
        emitMetric("webhook_delivered", {
          eventType: event.type,
          endpointId,
          attempts: msg.attempts,
          durationMs,
        });
      } else if (res.status === 429 || res.status >= 500) {
        // Transient failure — retry with exponential backoff
        const delay = Math.min(10 * (2 ** (msg.attempts - 1)), 3600);
        msg.retry({ delaySeconds: delay });
        emitMetric("webhook_retry", {
          eventType: event.type,
          endpointId,
          statusCode: res.status,
          attempt: msg.attempts,
        });
      } else {
        // 4xx (not 429) — permanent failure, don't retry
        msg.ack();
        console.error(`[webhook-queue] Permanent failure ${res.status} for ${endpoint.url} (endpoint ${endpointId})`);
      }
    } catch (err) {
      // Network error, timeout, or endpoint lookup failure — retry
      const delay = Math.min(10 * (2 ** (msg.attempts - 1)), 3600);
      msg.retry({ delaySeconds: delay });
      emitMetric("webhook_retry", {
        eventType: event.type,
        endpointId,
        statusCode: 0,
        attempt: msg.attempts,
      });
      console.error(
        `[webhook-queue] Delivery error for endpoint ${endpointId}:`,
        err instanceof Error ? err.message : String(err),
      );
    }
  }
}
