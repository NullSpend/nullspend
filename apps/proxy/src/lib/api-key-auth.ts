import { getSql } from "./db.js";
import { toHex } from "./hex.js";

/**
 * Parse a Postgres text[] value which may arrive as:
 * - A JavaScript array (when postgres.js parses the type correctly)
 * - A Postgres array literal string like "{gpt-4o-mini,claude-haiku}" (when fetch_types:false skips parsing)
 * - null/undefined (column is NULL)
 *
 * Returns a string[] or null.
 */
function parseTextArray(value: unknown): string[] | null {
  if (Array.isArray(value)) return value as string[];
  if (typeof value === "string" && value.startsWith("{") && value.endsWith("}")) {
    const inner = value.slice(1, -1);
    if (inner === "") return [];
    return inner.split(",").map(s => {
      // Handle quoted elements: "{\"quoted value\",simple}" → ["quoted value", "simple"]
      if (s.startsWith('"') && s.endsWith('"')) return s.slice(1, -1).replace(/\\"/g, '"');
      return s;
    });
  }
  return null;
}

export interface ApiKeyIdentity {
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
}

const CACHE_MAX_SIZE = 256;
const NEGATIVE_CACHE_MAX_SIZE = 2048;
const POSITIVE_TTL_MS = 120_000; // 120s — longer TTL reduces DB lookups; invalidated actively via /internal/budget/invalidate
const NEGATIVE_TTL_MS = 30_000; // 30s — keep short to avoid blocking new valid keys
const TTL_JITTER_MS = 20_000;   // ±10s jitter to prevent thundering herd on isolate recycle

interface CacheEntry {
  identity: ApiKeyIdentity;
  expiresAt: number;
}

interface NegativeCacheEntry {
  expiresAt: number;
}

// Module-level caches — persist within the Workers isolate across requests
const positiveCache = new Map<string, CacheEntry>();
const negativeCache = new Map<string, NegativeCacheEntry>();

/**
 * SHA-256 hash using Web Crypto API (Workers runtime).
 * Returns hex string matching Node.js crypto.createHash("sha256").digest("hex").
 */
export async function hashApiKey(rawKey: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(rawKey),
  );
  return toHex(buf);
}

/**
 * Evict the oldest entry when the cache exceeds its max size.
 * Map iteration order in JS is insertion order, so the first key is the oldest.
 */
function evictIfNeeded(cache: Map<string, unknown>, maxSize: number): void {
  if (cache.size > maxSize) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }
}

/**
 * Look up an API key by its SHA-256 hash in the database.
 * Returns { userId, keyId } for valid, non-revoked keys.
 * Returns null for keys not found or revoked.
 * THROWS on DB errors — caller must distinguish "not found" from "DB down".
 *
 * Uses the shared postgres.js pool via getSql() with Hyperdrive
 * connection pooling. Connection limits handled by postgres.js max setting.
 */
async function lookupKeyInDb(
  keyHash: string,
  connectionString: string,
): Promise<ApiKeyIdentity | null> {
  const sql = getSql(connectionString);

  const rows = await sql`
    SELECT k.id, k.user_id, k.org_id, k.api_version, k.default_tags, k.allowed_models, k.allowed_providers,
      EXISTS(
        SELECT 1 FROM webhook_endpoints w
        WHERE w.org_id = k.org_id AND w.enabled = true
      ) AS has_webhooks,
      EXISTS(
        SELECT 1 FROM budgets b
        WHERE b.org_id = k.org_id
      ) AS has_budgets,
      COALESCE(
        (SELECT s.tier IN ('pro', 'enterprise') FROM subscriptions s WHERE s.org_id = k.org_id AND s.status IN ('active', 'past_due') LIMIT 1),
        false
      ) AS request_logging_enabled
    FROM api_keys k
    WHERE k.key_hash = ${keyHash} AND k.revoked_at IS NULL
  `;

  if (rows.length === 0) {
    return null;
  }

  const row = rows[0];
  return {
    userId: row.user_id as string,
    orgId: (row.org_id as string) ?? null,
    keyId: row.id as string,
    hasWebhooks: row.has_webhooks === true,
    hasBudgets: row.has_budgets === true,
    requestLoggingEnabled: row.request_logging_enabled === true,
    apiVersion: row.api_version as string,
    defaultTags: (typeof row.default_tags === "object" && row.default_tags !== null && !Array.isArray(row.default_tags))
      ? row.default_tags as Record<string, string>
      : {},
    allowedModels: parseTextArray(row.allowed_models),
    allowedProviders: parseTextArray(row.allowed_providers),
  };
}

/**
 * Authenticate a raw API key.
 *
 * 1. Hash the key with SHA-256
 * 2. Check positive cache (valid keys, 120s TTL ±10s jitter)
 * 3. Check negative cache (invalid keys, 30s TTL)
 * 4. Query the database
 * 5. On success: populate positive or negative cache
 *    On DB error: return null WITHOUT negative-caching (next request retries)
 *
 * Never throws — returns null for invalid/revoked keys or DB errors.
 */
export async function authenticateApiKey(
  rawKey: string,
  connectionString: string,
): Promise<ApiKeyIdentity | null> {
  const keyHash = await hashApiKey(rawKey);
  const now = Date.now();

  // Check positive cache
  const cached = positiveCache.get(keyHash);
  if (cached) {
    if (cached.expiresAt > now) {
      return cached.identity;
    }
    positiveCache.delete(keyHash);
  }

  // Check negative cache
  const negativeCached = negativeCache.get(keyHash);
  if (negativeCached) {
    if (negativeCached.expiresAt > now) {
      return null;
    }
    negativeCache.delete(keyHash);
  }

  // DB lookup — throws on DB errors, returns null for "not found"
  let identity: ApiKeyIdentity | null;
  try {
    identity = await lookupKeyInDb(keyHash, connectionString);
  } catch (err) {
    // DB error (Hyperdrive down, connection timeout, etc.)
    // Return null (deny request) but do NOT negative-cache.
    // Next request will retry the DB lookup instead of being locked out for 30s.
    console.error(
      "[api-key-auth] DB lookup failed (not negative-cached):",
      err instanceof Error ? err.message : "Unknown error",
    );
    return null;
  }

  if (identity) {
    positiveCache.set(keyHash, {
      identity,
      expiresAt: now + POSITIVE_TTL_MS + (Math.floor(Math.random() * TTL_JITTER_MS) - TTL_JITTER_MS / 2),
    });
    evictIfNeeded(positiveCache, CACHE_MAX_SIZE);
  } else {
    // Key genuinely not found or revoked — safe to negative-cache
    negativeCache.set(keyHash, {
      expiresAt: now + NEGATIVE_TTL_MS,
    });
    evictIfNeeded(negativeCache, NEGATIVE_CACHE_MAX_SIZE);
  }

  return identity;
}

/**
 * Invalidate auth cache entries for an org (or user fallback).
 * Needed when a budget/webhook config changes.
 * Returns the number of evicted entries.
 */
export function invalidateAuthCacheForOwner(ownerId: string): number {
  let evicted = 0;
  for (const [hash, entry] of positiveCache) {
    if (entry.identity.orgId === ownerId || entry.identity.userId === ownerId) {
      positiveCache.delete(hash);
      evicted++;
    }
  }
  return evicted;
}

/**
 * Reset caches — exposed for testing only.
 */
export function _resetCaches(): void {
  positiveCache.clear();
  negativeCache.clear();
}
