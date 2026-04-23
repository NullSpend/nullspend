import { getSql } from "./db.js";
import { toHex } from "./hex.js";
import { toFiniteNumber } from "./number-coerce.js";

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

export type TierLabel = "free" | "pro" | "scale" | "enterprise";
export type PlanLimitMode = "hard" | "soft";

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
  allowedCustomers: string[] | null;
  requireCustomerId: boolean;
  /**
   * Org-level upgrade URL from `organizations.metadata.upgradeUrl`.
   * Null when unset. Surfaced in `budget_exceeded` and
   * `customer_budget_exceeded` denial responses. Per-customer overrides
   * live in `customer_mappings.upgrade_url` and are resolved at denial
   * time (not auth time) — this field is the fallback for non-customer
   * denials and the org-level default for customer denials without
   * a per-customer override.
   */
  orgUpgradeUrl: string | null;

  // ── PR-2a fields (plan-limit + period — per Decisions #15, #23, #33, #36) ──

  /**
   * Plan-limit blockAt threshold (governed-request count per period).
   * - 100_000 for Free, 500_000 for Pro, 2_000_000 for Scale.
   * - `null` means no enforcement: Enterprise (unlimited) AND self-hosted (`NULLSPEND_CLOUD !== "true"`).
   * - Free = 100000 when `s.tier IS NULL OR s.tier = 'free'`, NOT just `tier = 'free'`
   *   (Free users have NO subscription row; LEFT JOIN miss → s.tier IS NULL).
   */
  planLimitBlockAt: number | null;

  /**
   * Plan-limit enforcement mode.
   * - "hard" for Free (returns 429 at threshold).
   * - "soft" for Pro / Scale / Enterprise / self-hosted (count-only; overage billed downstream).
   */
  planLimitMode: PlanLimitMode;

  /** Metric label, COALESCE-backed — never null. Tag-aggregation friendly. */
  tierLabel: TierLabel;

  /**
   * Subscription billing period (ms epoch).
   * - Paid orgs: `subscriptions.current_period_start/end` for the Stripe cycle.
   * - Free / unpaid / self-hosted: both `null` → `resolvePeriodBounds` falls back to UTC calendar month.
   *
   * Converted from TIMESTAMPTZ to ms-epoch in SQL via `(EXTRACT(EPOCH FROM ts) * 1000)::BIGINT`,
   * then Number-coerced in JS via `toFiniteNumber` (postgres.js with `fetch_types:false`
   * returns BIGINT as string/BigInt, not number).
   */
  subscriptionPeriodStart: number | null;
  subscriptionPeriodEnd: number | null;
}

/**
 * CACHE_SCHEMA_VERSION (per Decision #21 / plan-audit A7 / codex PR-2a-R3 verified).
 *
 * Prefix for cache keys — allows atomic invalidation of all prior cache entries when
 * `ApiKeyIdentity` shape changes. Initial value: "v1" (first introduction).
 *
 * **Bump policy:** bump whenever `ApiKeyIdentity` shape changes in ANY future PR
 * (2b/2c/2d, or later work). Bumping invalidates all pre-deploy isolate-local cache
 * entries atomically — prevents the 120s window of stale-shape reads post-deploy.
 */
export const CACHE_SCHEMA_VERSION = "v1";

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

// Module-level caches — persist within the Workers isolate across requests.
// Keys are `${CACHE_SCHEMA_VERSION}:${keyHash}` so a version bump atomically
// invalidates everything (stale entries become unreachable under the new prefix).
const positiveCache = new Map<string, CacheEntry>();
const negativeCache = new Map<string, NegativeCacheEntry>();

function cacheKey(keyHash: string): string {
  return `${CACHE_SCHEMA_VERSION}:${keyHash}`;
}

/**
 * Testing-only: expose the cache-key function so tests can verify the
 * CACHE_SCHEMA_VERSION prefix is actually applied (per build-audit B4).
 * Not for production use.
 */
export const _cacheKeyForTesting = cacheKey;

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
 * Coerce a raw SQL `plan_limit_mode` value to a valid PlanLimitMode.
 * SQL always returns "hard" or "soft" per the CASE expression; this
 * defensive check handles test mocks without the new column (default "soft"
 * matches the SQL's ELSE branch for non-free tiers).
 */
function normalizePlanLimitMode(v: unknown): PlanLimitMode {
  return v === "hard" ? "hard" : "soft";
}

/**
 * Coerce a raw SQL `tier_label` value to a valid TierLabel.
 * SQL uses `COALESCE(s.tier, 'free')` so the value is always a non-null tier string.
 * Defensive fallback to "free" for unexpected values.
 */
function normalizeTierLabel(v: unknown): TierLabel {
  if (v === "pro" || v === "scale" || v === "enterprise" || v === "free") return v;
  return "free";
}

/**
 * Self-hosted identity shape (per Decision #13 / plan-audit A2 / codex PR-2a-N2).
 *
 * Returned when `env.NULLSPEND_CLOUD !== "true"` — maps cleanly onto
 * Enterprise-tier semantics: unlimited, all features, no metering gates.
 *
 * All 5 new fields are populated explicitly — NOT just `planLimitBlockAt: null`
 * (which would leave the other 4 fields undefined and cause downstream bugs).
 *
 * **Future-proofing note (per edge-audit E4):** when adding numeric fields to
 * `ApiKeyIdentity`, call `toFiniteNumber(baseRow.x)` here — `postgres.js` with
 * `fetch_types:false` returns BIGINT as string/BigInt, and self-hosted rows
 * come from the same postgres.js path as cloud rows. Missing coercion would
 * silently break self-hosted only, which has poor signal visibility (cloud
 * monitoring doesn't cover it). Current 5 fields are all string/boolean/null
 * so no coercion needed today — but remember when extending.
 */
function buildSelfHostedIdentity(baseRow: Record<string, unknown>): ApiKeyIdentity {
  return {
    userId: baseRow.user_id as string,
    orgId: (baseRow.org_id as string) ?? null,
    keyId: baseRow.id as string,
    hasWebhooks: baseRow.has_webhooks === true,
    hasBudgets: baseRow.has_budgets === true,
    requestLoggingEnabled: true,        // self-hosted gets all features — no tier gate
    apiVersion: baseRow.api_version as string,
    defaultTags:
      typeof baseRow.default_tags === "object" &&
      baseRow.default_tags !== null &&
      !Array.isArray(baseRow.default_tags)
        ? (baseRow.default_tags as Record<string, string>)
        : {},
    allowedModels: parseTextArray(baseRow.allowed_models),
    allowedProviders: parseTextArray(baseRow.allowed_providers),
    allowedCustomers: parseTextArray(baseRow.allowed_customers),
    requireCustomerId: baseRow.require_customer_id === true,
    orgUpgradeUrl:
      typeof baseRow.org_upgrade_url === "string" ? baseRow.org_upgrade_url : null,
    planLimitBlockAt: null,             // unlimited — no enforcement
    planLimitMode: "soft",              // counter runs (for local observability) but never blocks
    tierLabel: "enterprise",            // metric label matches unlimited semantics
    subscriptionPeriodStart: null,      // no subscription → period math falls back to calendar month
    subscriptionPeriodEnd: null,
  };
}

/**
 * Look up an API key by its SHA-256 hash in the database.
 * Returns the full ApiKeyIdentity for valid, non-revoked keys.
 * Returns null for keys not found or revoked.
 * THROWS on DB errors — caller must distinguish "not found" from "DB down".
 *
 * Uses the shared postgres.js pool via getSql() with Hyperdrive
 * connection pooling. Connection limits handled by postgres.js max setting.
 *
 * When `env.NULLSPEND_CLOUD !== "true"` (self-hosted), short-circuits to the
 * self-hosted identity shape (per Decision #13 / plan-audit A2) — fully
 * unlimited, all features enabled, no metering enforcement.
 */
async function lookupKeyInDb(
  keyHash: string,
  connectionString: string,
  env?: { NULLSPEND_CLOUD?: string },
): Promise<ApiKeyIdentity | null> {
  const sql = getSql(connectionString);

  // Decisions #15, #20, #23, #24, #33 applied across a SINGLE LEFT JOIN LATERAL
  // (per build-audit B1 — prior 5 correlated subqueries added ~5x DB work on cache miss).
  //
  // - `s.status IN ('active','past_due','trialing')` — trialing treated as Pro-equivalent (#24).
  // - `s.tier IN ('pro','scale','enterprise')` in request_logging_enabled — Scale included (#20).
  // - `CASE WHEN s.tier IS NULL OR s.tier = 'free'` — for Free users the LATERAL produces
  //   ZERO rows; the LEFT JOIN leaves s fields as NULL; the IS NULL branch fires → 100000.
  //   This IS the Free-user fallback path (per build-audit B5 — prior misleading comment).
  // - `(EXTRACT(EPOCH FROM ...) * 1000)::BIGINT` — ms-epoch in SQL (#33) + `toFiniteNumber`
  //   coercion in JS (#36) because postgres.js with `fetch_types:false` returns BIGINT as
  //   string/BigInt.
  // - `COALESCE(s.tier, 'free')` — tier_label never null, metric-aggregation friendly (#23).
  const rows = await sql`
    SELECT k.id, k.user_id, k.org_id, k.api_version, k.default_tags, k.allowed_models, k.allowed_providers, k.allowed_customers, k.require_customer_id,
      EXISTS(
        SELECT 1 FROM webhook_endpoints w
        WHERE w.org_id = k.org_id AND w.enabled = true
      ) AS has_webhooks,
      EXISTS(
        SELECT 1 FROM budgets b
        WHERE b.org_id = k.org_id
      ) AS has_budgets,
      (SELECT o.metadata->>'upgradeUrl' FROM organizations o WHERE o.id = k.org_id) AS org_upgrade_url,
      -- All 5 tier-derived columns read from the SAME lateral row.
      COALESCE(s.tier IN ('pro', 'scale', 'enterprise'), false) AS request_logging_enabled,
      CASE
        WHEN s.tier IS NULL OR s.tier = 'free' THEN 100000
        WHEN s.tier = 'pro'   THEN 500000
        WHEN s.tier = 'scale' THEN 2000000
        ELSE NULL
      END AS plan_limit_block_at,
      CASE
        WHEN s.tier IS NULL OR s.tier = 'free' THEN 'hard'
        ELSE 'soft'
      END AS plan_limit_mode,
      COALESCE(s.tier, 'free') AS tier_label,
      CASE
        WHEN s.current_period_start IS NOT NULL
        THEN (EXTRACT(EPOCH FROM s.current_period_start) * 1000)::BIGINT
        ELSE NULL
      END AS subscription_period_start,
      CASE
        WHEN s.current_period_end IS NOT NULL
        THEN (EXTRACT(EPOCH FROM s.current_period_end) * 1000)::BIGINT
        ELSE NULL
      END AS subscription_period_end
    FROM api_keys k
    LEFT JOIN LATERAL (
      SELECT tier, current_period_start, current_period_end
      FROM subscriptions
      WHERE org_id = k.org_id AND status IN ('active', 'past_due', 'trialing')
      LIMIT 1
    ) s ON TRUE
    WHERE k.key_hash = ${keyHash} AND k.revoked_at IS NULL
  `;

  if (rows.length === 0) {
    return null;
  }

  const row = rows[0];

  // Self-hosted short-circuit (per Decision #13 / plan-audit A2 / edge-audit E1).
  // When env is omitted OR NULLSPEND_CLOUD is "true" (case-insensitive, whitespace-trimmed),
  // behaves as cloud-mode. Backwards-compatible with existing 2-arg call sites + test harnesses.
  //
  // Normalization protects against env-var misconfig typos (" true", "TRUE", "True") that
  // would otherwise silently flip a cloud deployment into self-hosted mode (unlimited
  // enforcement-off) with no visible signal — per edge-audit E1.
  if (env !== undefined && env.NULLSPEND_CLOUD?.trim().toLowerCase() !== "true") {
    return buildSelfHostedIdentity(row);
  }

  // Cloud-mode path — use SQL-derived values.
  //
  // Free orgs (no subscription row matching status IN (...)): the LEFT JOIN LATERAL
  // produces no row → s.tier is NULL → CASE's `WHEN s.tier IS NULL` branch fires →
  // plan_limit_block_at = 100000, plan_limit_mode = 'hard', tier_label = 'free' (via
  // COALESCE), period fields = null. No JS fallback needed.
  //
  // Paid orgs: s.tier matches a WHEN branch; derived columns return the paid values.
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
    allowedCustomers: parseTextArray(row.allowed_customers),
    requireCustomerId: row.require_customer_id === true,
    orgUpgradeUrl: typeof row.org_upgrade_url === "string" ? row.org_upgrade_url : null,
    // BIGINT columns come back as string/BigInt under fetch_types:false (Decision #36).
    // Normalize helpers guard against unexpected mode/tier values (defensive — SQL CASE
    // already constrains to valid values).
    planLimitBlockAt: toFiniteNumber(row.plan_limit_block_at),
    planLimitMode: normalizePlanLimitMode(row.plan_limit_mode),
    tierLabel: normalizeTierLabel(row.tier_label),
    subscriptionPeriodStart: toFiniteNumber(row.subscription_period_start),
    subscriptionPeriodEnd: toFiniteNumber(row.subscription_period_end),
  };
}

/**
 * Authenticate a raw API key.
 *
 * 1. Hash the key with SHA-256
 * 2. Check positive cache (valid keys, 120s TTL ±10s jitter) — keyed under CACHE_SCHEMA_VERSION prefix
 * 3. Check negative cache (invalid keys, 30s TTL)
 * 4. Query the database (cloud-mode) OR short-circuit to self-hosted shape
 * 5. On success: populate positive or negative cache
 *    On DB error: throw (caller returns 503 — never negative-cached)
 *
 * THROWS on DB errors — caller must distinguish "not found" (null) from "DB down" (thrown).
 * Returns null only for genuinely invalid/revoked keys.
 *
 * @param rawKey Raw API key (unhashed)
 * @param connectionString Postgres connection string (via Hyperdrive)
 * @param env Optional env object; when `env.NULLSPEND_CLOUD !== "true"`, returns the self-hosted
 *            identity shape (all features unlocked, no enforcement). Omitting env defaults to
 *            cloud-mode behavior — production paths always pass env explicitly; the optional
 *            shape is a test-ergonomics affordance, not a compat shim.
 */
export async function authenticateApiKey(
  rawKey: string,
  connectionString: string,
  env?: { NULLSPEND_CLOUD?: string },
): Promise<ApiKeyIdentity | null> {
  const keyHash = await hashApiKey(rawKey);
  const ck = cacheKey(keyHash);
  const now = Date.now();

  // Check positive cache
  const cached = positiveCache.get(ck);
  if (cached) {
    if (cached.expiresAt > now) {
      return cached.identity;
    }
    positiveCache.delete(ck);
  }

  // Check negative cache
  const negativeCached = negativeCache.get(ck);
  if (negativeCached) {
    if (negativeCached.expiresAt > now) {
      return null;
    }
    negativeCache.delete(ck);
  }

  // DB lookup — throws on DB errors, returns null for "not found"
  // Re-throw DB errors so the caller can return 503 instead of 401.
  // Do NOT negative-cache — next request will retry the DB lookup.
  const identity = await lookupKeyInDb(keyHash, connectionString, env);

  if (identity) {
    positiveCache.set(ck, {
      identity,
      expiresAt: now + POSITIVE_TTL_MS + (Math.floor(Math.random() * TTL_JITTER_MS) - TTL_JITTER_MS / 2),
    });
    evictIfNeeded(positiveCache, CACHE_MAX_SIZE);
  } else {
    // Key genuinely not found or revoked — safe to negative-cache
    negativeCache.set(ck, {
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
  for (const [key, entry] of positiveCache) {
    if (entry.identity.orgId === ownerId || entry.identity.userId === ownerId) {
      positiveCache.delete(key);
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
