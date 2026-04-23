/**
 * PR-2a tests — plan-limit + subscription-period SQL-derived columns + self-hosted bypass.
 *
 * Covers C14, C14b, C14c, C14d, C15, C16, C16-full, C17, C17b, C17b-trial,
 * C17b-trial-scale, C17c (cache version), plus regression for the old-shape
 * positive cache under `v1:` prefix.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// --- Hoisted mocks ---
const { mockSql } = vi.hoisted(() => {
  const mockSql = vi.fn().mockResolvedValue([]);
  return { mockSql };
});

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

import {
  authenticateApiKey,
  _resetCaches,
  _cacheKeyForTesting,
  CACHE_SCHEMA_VERSION,
} from "../lib/api-key-auth.js";

const RAW_KEY = "ns_live_sk_pr2a_test_a1b2c3d4e5f6";
const KEY_ID = "550e8400-e29b-41d4-a716-pr2aa";
const USER_ID = "user-pr2a-1";
const ORG_ID = "org-pr2a-1";
const CONN = "postgresql://postgres@db:5432/postgres";

// Base row for a Free org (no subscription row via LEFT JOIN LATERAL miss).
// When the LATERAL produces no row, the LEFT JOIN gives s.* as NULL, so the
// CASE expressions in the SELECT list evaluate the `WHEN s.tier IS NULL` branch
// and return concrete Free-tier values: 100000 / 'hard' / 'free'. Period fields
// stay null (LEFT JOIN miss).
const freeRow = {
  id: KEY_ID,
  user_id: USER_ID,
  org_id: ORG_ID,
  api_version: "2026-04-01",
  default_tags: {},
  allowed_models: null,
  allowed_providers: null,
  allowed_customers: null,
  require_customer_id: false,
  has_webhooks: false,
  has_budgets: false,
  request_logging_enabled: false,
  org_upgrade_url: null,
  // LATERAL miss → s.tier is NULL → CASE's IS NULL branch fires in SQL
  plan_limit_block_at: 100000,
  plan_limit_mode: "hard",
  tier_label: "free",
  subscription_period_start: null,
  subscription_period_end: null,
};

function withSub(extra: Partial<typeof freeRow>) {
  return { ...freeRow, ...extra };
}

describe("api-key-auth PR-2a — plan-limit + period columns", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  // C14 — Free org (no subscription row) gets hard 100K cap
  it("C14 — Free org returns planLimitBlockAt=100000, mode=hard, tierLabel='free'", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 100_000,
      planLimitMode: "hard",
      tierLabel: "free",
      subscriptionPeriodStart: null,
      subscriptionPeriodEnd: null,
      requestLoggingEnabled: false, // Free doesn't get body capture
    });
  });

  // C14b — Canceled Pro org falls out of status IN clause; proxy sees Free shape.
  // Simulate: the subqueries return null because no row has status IN (active/past_due/trialing).
  it("C14b — canceled Pro subscription → Free treatment", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        // Same as freeRow — proxy can't distinguish "no sub" from "canceled sub"
        // because our status filter excludes canceled. That's intentional (Decision #23).
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 100_000,
      planLimitMode: "hard",
      tierLabel: "free",
    });
  });

  // C14c — Trialing Pro subscription is treated as Pro
  it("C14c — trialing Pro → planLimitBlockAt=500000, mode=soft, tierLabel='pro'", async () => {
    const periodStart = Date.UTC(2026, 3, 17);
    const periodEnd = Date.UTC(2026, 4, 17);
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true, // trialing Pro gets body capture (Decision #24)
        plan_limit_block_at: "500000", // postgres.js under fetch_types:false returns BIGINT as string
        plan_limit_mode: "soft",
        tier_label: "pro",
        subscription_period_start: String(periodStart),
        subscription_period_end: String(periodEnd),
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 500_000,
      planLimitMode: "soft",
      tierLabel: "pro",
      subscriptionPeriodStart: periodStart,
      subscriptionPeriodEnd: periodEnd,
      requestLoggingEnabled: true,
    });
  });

  // C14d — TIMESTAMPTZ → ms-epoch coercion works for string AND BigInt inputs
  it("C14d — BIGINT coercion: string input round-trips to number", async () => {
    const startMs = Date.UTC(2026, 3, 17); // Apr 17 2026
    const endMs = Date.UTC(2026, 4, 17); // May 17 2026
    mockSql.mockResolvedValueOnce([
      withSub({
        plan_limit_block_at: "500000",
        plan_limit_mode: "soft",
        tier_label: "pro",
        subscription_period_start: String(startMs), // Stringified number from postgres.js
        subscription_period_end: String(endMs),
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(typeof identity!.subscriptionPeriodStart).toBe("number");
    expect(identity!.subscriptionPeriodStart).toBe(startMs);
    expect(typeof identity!.subscriptionPeriodEnd).toBe("number");
    expect(identity!.subscriptionPeriodEnd).toBe(endMs);
  });

  it("C14d — BIGINT coercion: BigInt input round-trips to number", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        plan_limit_block_at: 500000n as unknown as string, // BigInt primitive
        plan_limit_mode: "soft",
        tier_label: "pro",
        subscription_period_start: 1744848000000n as unknown as string,
        subscription_period_end: null,
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(typeof identity!.planLimitBlockAt).toBe("number");
    expect(identity!.planLimitBlockAt).toBe(500000);
    expect(typeof identity!.subscriptionPeriodStart).toBe("number");
    expect(identity!.subscriptionPeriodStart).toBe(1744848000000);
  });

  // C15 — Active Pro subscription → soft 500K, tierLabel 'pro'
  it("C15 — active Pro → planLimitBlockAt=500000, mode=soft, tierLabel='pro'", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true,
        plan_limit_block_at: "500000",
        plan_limit_mode: "soft",
        tier_label: "pro",
        subscription_period_start: String(Date.UTC(2026, 3, 1)),
        subscription_period_end: String(Date.UTC(2026, 4, 1)),
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 500_000,
      planLimitMode: "soft",
      tierLabel: "pro",
    });
    expect(typeof identity!.subscriptionPeriodStart).toBe("number");
  });

  // C17 — Enterprise org → null blockAt (unlimited), tierLabel 'enterprise'
  it("C17 — Enterprise → planLimitBlockAt=null, tierLabel='enterprise'", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true,
        plan_limit_block_at: null, // Enterprise uses SQL's ELSE branch returning null
        plan_limit_mode: "soft",
        tier_label: "enterprise",
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: null,
      planLimitMode: "soft",
      tierLabel: "enterprise",
    });
  });

  // C17b — Scale tier gets soft 2M cap + request_logging_enabled (per Decision #20 money-bug fix)
  it("C17b — Scale → planLimitBlockAt=2000000, mode=soft, tierLabel='scale', requestLoggingEnabled=true", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true, // Scale must have body capture ($199/mo feature)
        plan_limit_block_at: "2000000",
        plan_limit_mode: "soft",
        tier_label: "scale",
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 2_000_000,
      planLimitMode: "soft",
      tierLabel: "scale",
      requestLoggingEnabled: true,
    });
  });

  // C17b-trial — Trialing Pro must get request_logging_enabled=true per Decision #24
  it("C17b-trial — trialing Pro → requestLoggingEnabled=true", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true, // SQL includes 'trialing' in status filter now
        plan_limit_block_at: "500000",
        plan_limit_mode: "soft",
        tier_label: "pro",
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);
    expect(identity!.requestLoggingEnabled).toBe(true);
    expect(identity!.tierLabel).toBe("pro");
  });

  // C17b-trial-scale — Trialing Scale must also get request_logging_enabled
  it("C17b-trial-scale — trialing Scale → requestLoggingEnabled=true", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        request_logging_enabled: true,
        plan_limit_block_at: "2000000",
        plan_limit_mode: "soft",
        tier_label: "scale",
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN);
    expect(identity!.requestLoggingEnabled).toBe(true);
    expect(identity!.tierLabel).toBe("scale");
  });

  // SQL contract check — ensure required SQL keywords + derived columns are in the query.
  // Partial shield against SQL-shape regressions that mocked tests otherwise can't catch
  // (per build-audit B6 — full SQL not exercised against real Postgres).
  it("SQL SELECT contains plan-limit + period computed columns + LATERAL join", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);
    await authenticateApiKey(RAW_KEY, CONN);

    const call = mockSql.mock.calls[0];
    const templateStrings = (call[0] as string[]).join(" ");
    expect(templateStrings).toContain("plan_limit_block_at");
    expect(templateStrings).toContain("plan_limit_mode");
    expect(templateStrings).toContain("tier_label");
    expect(templateStrings).toContain("subscription_period_start");
    expect(templateStrings).toContain("subscription_period_end");
    // Verify status filter includes trialing (Decision #24 + plan-audit A3)
    expect(templateStrings).toContain("'trialing'");
    // Verify Scale in request_logging_enabled filter (Decision #20 money-bug fix)
    expect(templateStrings).toContain("'scale'");
    // Verify the LATERAL join (per build-audit B1 — prevents accidental regression
    // back to the 5-subquery pattern).
    expect(templateStrings).toContain("LEFT JOIN LATERAL");
  });
});

describe("api-key-auth PR-2a — self-hosted bypass (NULLSPEND_CLOUD)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  // C16 + C16-full — Self-hosted returns the FULL enterprise-equivalent shape, not just null blockAt
  it("C16-full — NULLSPEND_CLOUD=false returns full self-hosted shape (all 5 new fields)", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        // Even if the SQL returns Free-tier values, self-hosted bypass overrides all 5 fields
        plan_limit_block_at: null,
        plan_limit_mode: null,
        tier_label: null,
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: "false" });

    expect(identity).toMatchObject({
      planLimitBlockAt: null,         // unlimited
      planLimitMode: "soft",          // never blocks
      tierLabel: "enterprise",        // metric tag matches unlimited semantics
      subscriptionPeriodStart: null,
      subscriptionPeriodEnd: null,
      requestLoggingEnabled: true,    // self-hosted gets all features
    });
  });

  it("C16 — NULLSPEND_CLOUD unset (env omitted entirely) → cloud-mode default (BACKWARDS COMPAT)", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);
    // No env arg at all — matches all existing 2-arg call sites.
    const identity = await authenticateApiKey(RAW_KEY, CONN);

    expect(identity).toMatchObject({
      planLimitBlockAt: 100_000, // Free user, not self-hosted
      tierLabel: "free",
      requestLoggingEnabled: false,
    });
  });

  it("C16 — NULLSPEND_CLOUD='true' (explicit cloud) → cloud-mode behavior", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);
    const identity = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: "true" });

    expect(identity).toMatchObject({
      planLimitBlockAt: 100_000,
      tierLabel: "free",
    });
  });

  // Edge-audit E1 — NULLSPEND_CLOUD comparison is case-insensitive + whitespace-trimmed.
  // Prior strict `!== "true"` would silently flip a cloud deployment to self-hosted on typo.
  it.each([
    ["TRUE", "cloud"],
    ["True", "cloud"],
    [" true", "cloud"],
    ["true ", "cloud"],
    ["  TRUE  ", "cloud"],
    ["false", "self-hosted"],
    ["FALSE", "self-hosted"],
    ["", "self-hosted"],
    ["1", "self-hosted"],
    ["yes", "self-hosted"],
  ])("C16-E1 — NULLSPEND_CLOUD=%j resolves to %s mode", async (value, expected) => {
    mockSql.mockResolvedValueOnce([freeRow]);
    const identity = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: value });

    if (expected === "cloud") {
      // Cloud mode: Free-user semantics come from SQL (100K hard cap)
      expect(identity).toMatchObject({
        planLimitBlockAt: 100_000,
        tierLabel: "free",
      });
    } else {
      // Self-hosted: enterprise-equivalent, no enforcement
      expect(identity).toMatchObject({
        planLimitBlockAt: null,
        tierLabel: "enterprise",
        requestLoggingEnabled: true,
      });
    }
  });

  it("C16-full — self-hosted bypass preserves base identity fields (userId, orgId, keyId, etc.)", async () => {
    mockSql.mockResolvedValueOnce([
      withSub({
        default_tags: { env: "local" },
        allowed_models: ["gpt-4o"],
        require_customer_id: true,
        has_budgets: true,
      }),
    ]);
    const identity = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: "" });

    expect(identity).toMatchObject({
      userId: USER_ID,
      orgId: ORG_ID,
      keyId: KEY_ID,
      hasBudgets: true,
      defaultTags: { env: "local" },
      allowedModels: ["gpt-4o"],
      requireCustomerId: true,
      // And the self-hosted override:
      planLimitBlockAt: null,
      tierLabel: "enterprise",
    });
  });
});

describe("api-key-auth PR-2a — CACHE_SCHEMA_VERSION", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  // C17c — Cache key is version-prefixed so a future bump invalidates prior entries
  it("C17c — CACHE_SCHEMA_VERSION is a defined string (initial 'v1')", () => {
    expect(typeof CACHE_SCHEMA_VERSION).toBe("string");
    expect(CACHE_SCHEMA_VERSION).toBe("v1");
  });

  // C17c-mechanism (NEW per build-audit B4) — the cache-key wrapper actually applies the prefix.
  // If someone removes the wrapper, the bump contract breaks silently. Prior tests only
  // verified the constant existed + caching worked, not the key mechanism.
  it("C17c-mechanism — cache keys are prefixed with ${CACHE_SCHEMA_VERSION}:${keyHash}", () => {
    const hash = "abc123";
    expect(_cacheKeyForTesting(hash)).toBe(`${CACHE_SCHEMA_VERSION}:${hash}`);
    // Concrete value matches current constant — fails loudly if the constant drifts
    // without a corresponding cache wipe.
    expect(_cacheKeyForTesting(hash)).toBe("v1:abc123");
  });

  it("C17c-mechanism — different keyHashes get distinct cache keys under the same version", () => {
    expect(_cacheKeyForTesting("hashA")).not.toBe(_cacheKeyForTesting("hashB"));
    expect(_cacheKeyForTesting("hashA")).toBe("v1:hashA");
    expect(_cacheKeyForTesting("hashB")).toBe("v1:hashB");
  });

  // Cache key prefix contract — we can't inspect the internal Map directly,
  // but we can verify the cache effect by checking cache hit/miss behavior
  // and that cache entries respect the current version.
  it("positive cache is still effective under the current CACHE_SCHEMA_VERSION", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);

    const r1 = await authenticateApiKey(RAW_KEY, CONN);
    expect(r1).not.toBeNull();
    expect(mockSql).toHaveBeenCalledTimes(1);

    // Second call — cache hit under same version prefix
    const r2 = await authenticateApiKey(RAW_KEY, CONN);
    expect(r2).toEqual(r1);
    expect(mockSql).toHaveBeenCalledTimes(1); // no new DB lookup
  });

  // Regression: changing env mode (cloud vs self-hosted) should produce different
  // identities — but they share the same cache key (keyHash), so the cache will
  // return whatever was set first. This is a known limitation — tests document it.
  // In production, an isolate runs in a single env mode; this mismatch can't happen.
  it("documented: cache keys are not env-scoped (single-mode isolate assumption)", async () => {
    mockSql.mockResolvedValueOnce([freeRow]);

    // First call: self-hosted mode → returns enterprise-shape identity
    const r1 = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: "false" });
    expect(r1?.tierLabel).toBe("enterprise");

    // Second call: cloud mode → cache hit → returns the SAME enterprise identity
    // (not the Free identity the SQL would produce). This is OK because isolates
    // don't dynamically switch env mode. Test documents the assumption.
    const r2 = await authenticateApiKey(RAW_KEY, CONN, { NULLSPEND_CLOUD: "true" });
    expect(r2?.tierLabel).toBe("enterprise"); // cache hit, same as r1
    expect(mockSql).toHaveBeenCalledTimes(1);
  });
});
