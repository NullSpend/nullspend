import { authenticateApiKey, type ApiKeyIdentity } from "./api-key-auth.js";

export type { ApiKeyIdentity };

export interface AuthResult {
  userId: string;
  orgId: string | null;
  keyId: string;
  hasWebhooks: boolean;
  hasBudgets: boolean;
  requestLoggingEnabled: boolean;
  apiVersion: string;
  defaultTags: Record<string, string>;
  allowedModels: string[] | null;
  allowedProviders: string[] | null;
  orgUpgradeUrl: string | null;
}

/**
 * API key authentication for the proxy.
 *
 * Reads `x-nullspend-key` header, looks up by SHA-256 hash in DB.
 * Returns null for invalid/missing credentials (caller should return 401).
 */
export async function authenticateRequest(
  request: Request,
  connectionString: string,
): Promise<AuthResult | null> {
  const apiKey = request.headers.get("x-nullspend-key");
  if (!apiKey) return null;
  const identity = await authenticateApiKey(apiKey, connectionString);
  if (!identity) return null;
  return {
    userId: identity.userId,
    orgId: identity.orgId,
    keyId: identity.keyId,
    hasWebhooks: identity.hasWebhooks,
    hasBudgets: identity.hasBudgets,
    requestLoggingEnabled: identity.requestLoggingEnabled,
    apiVersion: identity.apiVersion,
    defaultTags: identity.defaultTags,
    allowedModels: identity.allowedModels,
    allowedProviders: identity.allowedProviders,
    orgUpgradeUrl: identity.orgUpgradeUrl,
  };
}
