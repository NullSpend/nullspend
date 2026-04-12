import { getSql } from "./db.js";
import { SECRET_ROTATION_WINDOW_SECONDS } from "./webhook-signer.js";
import type { WebhookEndpointWithSecret } from "./webhook-cache.js";

/**
 * Lazy-expire rotated secrets that are past the 24h window.
 * Fire-and-forget: wrapped in try/catch, never blocks dispatch.
 */
export async function expireRotatedSecrets(
  connectionString: string,
  endpoints: WebhookEndpointWithSecret[],
): Promise<void> {
  const expiredIds = endpoints
    .filter((ep) => {
      if (!ep.secretRotatedAt) return false;
      const elapsed = Date.now() - new Date(ep.secretRotatedAt).getTime();
      return elapsed >= SECRET_ROTATION_WINDOW_SECONDS * 1000;
    })
    .map((ep) => ep.id);

  if (expiredIds.length === 0) return;

  const sql = getSql(connectionString);

  await sql`
    UPDATE webhook_endpoints
    SET previous_signing_secret = NULL, secret_rotated_at = NULL
    WHERE id = ANY(${expiredIds}) AND secret_rotated_at < NOW() - INTERVAL '24 hours'
  `;
}
