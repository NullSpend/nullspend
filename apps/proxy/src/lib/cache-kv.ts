/**
 * Shared webhook endpoint metadata. Defined here (not in webhook-cache.ts)
 * to avoid a circular dependency since webhook-cache.ts imports from this file.
 */
export interface CachedWebhookEndpoint {
  id: string;
  url: string;
  eventTypes: string[];
}

// KV minimum expirationTtl is 60 seconds (CF enforced)
const WEBHOOK_TTL = 300; // 5 minutes

export async function getCachedWebhookEndpoints(
  kv: KVNamespace,
  ownerId: string,
): Promise<CachedWebhookEndpoint[] | null> {
  return kv.get<CachedWebhookEndpoint[]>(`webhook:${ownerId}`, "json");
}

export async function setCachedWebhookEndpoints(
  kv: KVNamespace,
  ownerId: string,
  endpoints: CachedWebhookEndpoint[],
): Promise<void> {
  await kv.put(`webhook:${ownerId}`, JSON.stringify(endpoints), {
    expirationTtl: WEBHOOK_TTL,
  });
}

export async function invalidateWebhookEndpoints(
  kv: KVNamespace,
  ownerId: string,
): Promise<void> {
  await kv.delete(`webhook:${ownerId}`);
}
