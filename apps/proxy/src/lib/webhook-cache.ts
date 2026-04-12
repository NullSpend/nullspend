import { getSql } from "./db.js";
import {
  getCachedWebhookEndpoints,
  setCachedWebhookEndpoints,
  invalidateWebhookEndpoints,
  type CachedWebhookEndpoint,
} from "./cache-kv.js";

export type { CachedWebhookEndpoint } from "./cache-kv.js";

/**
 * Full endpoint data including signing secret.
 * Returned from DB queries; never stored in cache.
 */
export interface WebhookEndpointWithSecret extends CachedWebhookEndpoint {
  signingSecret: string;
  previousSigningSecret: string | null;
  secretRotatedAt: string | null;
  apiVersion: string;
  payloadMode: "full" | "thin";
}

/**
 * Get active webhook endpoints for a user (metadata only, no secrets).
 * Uses Workers KV cache with DB fallback.
 * Fail-open: returns [] on error.
 */
export async function getWebhookEndpoints(
  connectionString: string,
  ownerId: string,
  kv: KVNamespace,
): Promise<CachedWebhookEndpoint[]> {
  try {
    const kvCached = await getCachedWebhookEndpoints(kv, ownerId);
    if (kvCached) return kvCached;
  } catch (err) {
    console.error("[webhook-cache:kv] read error:", err);
    // Fail-open: fall through to DB query
  }

  let endpoints: WebhookEndpointWithSecret[];
  try {
    endpoints = await queryActiveEndpoints(connectionString, ownerId);
  } catch (err) {
    console.error("[webhook-cache:kv] DB query error:", err);
    return []; // Fail-open
  }

  const metadata: CachedWebhookEndpoint[] = endpoints.map(({ id, url, eventTypes }) => ({
    id,
    url,
    eventTypes,
  }));

  try {
    await setCachedWebhookEndpoints(kv, ownerId, metadata);
  } catch (err) {
    console.error("[webhook-cache:kv] write error:", err);
  }

  return metadata;
}

/**
 * Get active webhook endpoints WITH signing secrets for dispatch.
 * Always queries DB — secrets are never cached.
 *
 * THROWS on DB error. This is intentional: the queue consumer's per-message
 * catch block calls msg.retry() on error, so a transient PG outage retries
 * instead of acking (which would permanently lose the webhook). Before this
 * fix (PXY-6), DB errors returned [] and the consumer treated "empty" as
 * "endpoint deleted" and acked — silently discarding messages.
 */
export async function getWebhookEndpointsWithSecrets(
  connectionString: string,
  ownerId: string,
): Promise<WebhookEndpointWithSecret[]> {
  return await queryActiveEndpoints(connectionString, ownerId);
}

/**
 * Invalidate the webhook endpoint cache for a user.
 * Called from dashboard API on create/update/delete.
 */
export async function invalidateWebhookCache(
  ownerId: string,
  kv: KVNamespace,
): Promise<void> {
  try {
    await invalidateWebhookEndpoints(kv, ownerId);
  } catch (err) {
    console.error("[webhook-cache:kv] invalidation error:", err);
  }
}

async function queryActiveEndpoints(
  connectionString: string,
  ownerId: string,
): Promise<WebhookEndpointWithSecret[]> {
  const sql = getSql(connectionString);

  const rows = await sql`
    SELECT id, url, signing_secret, previous_signing_secret, secret_rotated_at, event_types, api_version, payload_mode
    FROM webhook_endpoints
    WHERE org_id = ${ownerId} AND enabled = true
  `;

  return rows.map((row) => ({
    id: row.id as string,
    url: row.url as string,
    signingSecret: row.signing_secret as string,
    previousSigningSecret: (row.previous_signing_secret as string) ?? null,
    secretRotatedAt: row.secret_rotated_at ? new Date(row.secret_rotated_at as string).toISOString() : null,
    eventTypes: (row.event_types as string[]) ?? [],
    apiVersion: (row.api_version as string) ?? "2026-04-01",
    payloadMode: ((row.payload_mode as string) ?? "full") as "full" | "thin",
  }));
}
