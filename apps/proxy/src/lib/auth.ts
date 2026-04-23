import {
  authenticateApiKey,
  type ApiKeyIdentity,
  type PlanLimitMode,
  type TierLabel,
} from "./api-key-auth.js";

export type { ApiKeyIdentity, PlanLimitMode, TierLabel };

/**
 * `AuthResult` is a type alias for `ApiKeyIdentity` — NOT a separate interface
 * (per build-audit B2). Prior draft defined an independent AuthResult shape
 * that had to be kept in sync field-by-field with ApiKeyIdentity, which was
 * exactly the class of bug we fixed in PR-2a (plan-audit F13 / codex PR-2a-N2).
 * Type alias ensures future `ApiKeyIdentity` field additions automatically
 * flow through — no mapping function to update, no drift possible.
 *
 * If a downstream use case ever needs `AuthResult` to diverge from
 * `ApiKeyIdentity` (e.g., adding a request-derived field), use
 * `ApiKeyIdentity & { extraField: T }` instead of redefining the whole shape.
 */
export type AuthResult = ApiKeyIdentity;

/**
 * API key authentication for the proxy.
 *
 * Reads `x-nullspend-key` header, looks up by SHA-256 hash in DB.
 * Returns null for invalid/missing credentials (caller should return 401).
 *
 * `env` is REQUIRED (not optional) per codex PR-2a-N2 — the self-hosted
 * bypass (`env.NULLSPEND_CLOUD !== "true"` → enterprise-equivalent shape)
 * only activates if env actually reaches the leaf `authenticateApiKey`.
 * Making env required at this wrapper layer forces every caller in index.ts
 * to pass it; optional would silently hide the bug.
 */
export async function authenticateRequest(
  request: Request,
  env: { NULLSPEND_CLOUD?: string },
  connectionString: string,
): Promise<AuthResult | null> {
  const apiKey = request.headers.get("x-nullspend-key");
  if (!apiKey) return null;
  // Pass the identity through — AuthResult is an alias, so no mapping needed.
  return authenticateApiKey(apiKey, connectionString, env);
}
