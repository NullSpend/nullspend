/**
 * Unit tests for the api-key-auth module.
 *
 * Tests SHA-256 hashing, DB lookup, LRU caching (positive & negative),
 * cache eviction, and graceful error handling.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createHash } from "node:crypto";

// --- Hoisted mocks (must be defined before vi.mock calls) ---
const { mockSql } = vi.hoisted(() => {
  // Mock postgres.js tagged template function — returns rows array
  const mockSql = vi.fn().mockResolvedValue([]);
  return { mockSql };
});

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

// --- Import after mocks ---
import { hashApiKey, authenticateApiKey, _resetCaches } from "../lib/api-key-auth.js";

const TEST_RAW_KEY = "ns_live_sk_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6";
const TEST_USER_ID = "user-abc-123";
const TEST_KEY_ID = "550e8400-e29b-41d4-a716-446655440000";
const TEST_CONNECTION_STRING = "postgresql://postgres:postgres@db.example.com:5432/postgres";

const validRow = { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: {}, request_logging_enabled: false };

describe("hashApiKey", () => {
  it("produces the same hex output as Node.js crypto.createHash('sha256')", async () => {
    const expected = createHash("sha256").update(TEST_RAW_KEY).digest("hex");
    const result = await hashApiKey(TEST_RAW_KEY);
    expect(result).toBe(expected);
  });

  it("produces different hashes for different keys", async () => {
    const hash1 = await hashApiKey("ns_live_sk_key1");
    const hash2 = await hashApiKey("ns_live_sk_key2");
    expect(hash1).not.toBe(hash2);
  });

  it("produces 64-character hex string", async () => {
    const result = await hashApiKey(TEST_RAW_KEY);
    expect(result).toMatch(/^[0-9a-f]{64}$/);
  });

  it("handles unicode input", async () => {
    const rawKey = "ns_live_sk_unic\u00f6d\u00e9_k\u00e9y_\uD83D\uDD11";
    const expected = createHash("sha256").update(rawKey).digest("hex");
    const result = await hashApiKey(rawKey);
    expect(result).toBe(expected);
  });
});

describe("authenticateApiKey", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns identity on valid key (cache miss \u2192 DB hit)", async () => {
    mockSql.mockResolvedValueOnce([validRow]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);

    expect(result).toEqual({ userId: TEST_USER_ID, keyId: TEST_KEY_ID, hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {}, requestLoggingEnabled: false, allowedModels: null, allowedProviders: null, orgUpgradeUrl: null });
    expect(mockSql).toHaveBeenCalledTimes(1);
  });

  it("returns cached identity on second call (cache hit \u2192 no DB call)", async () => {
    mockSql.mockResolvedValueOnce([validRow]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1).toEqual({ userId: TEST_USER_ID, keyId: TEST_KEY_ID, hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {}, requestLoggingEnabled: false, allowedModels: null, allowedProviders: null, orgUpgradeUrl: null });
    expect(mockSql).toHaveBeenCalledTimes(1);

    // Second call — cache hit, no DB call
    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2).toEqual(result1);
    expect(mockSql).toHaveBeenCalledTimes(1); // Still 1
  });

  it("returns null for invalid key (no rows in DB)", async () => {
    mockSql.mockResolvedValueOnce([]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result).toBeNull();
    expect(mockSql).toHaveBeenCalledTimes(1);
  });

  it("negative-caches invalid keys (no DB call on second attempt)", async () => {
    mockSql.mockResolvedValueOnce([]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1).toBeNull();
    expect(mockSql).toHaveBeenCalledTimes(1);

    // Second call — negative cache hit
    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2).toBeNull();
    expect(mockSql).toHaveBeenCalledTimes(1); // Still 1
  });

  it("returns null for revoked key (query filters revoked_at IS NULL)", async () => {
    mockSql.mockResolvedValueOnce([]);
    const result = await authenticateApiKey("ns_live_sk_revoked_key", TEST_CONNECTION_STRING);
    expect(result).toBeNull();
  });

  it("throws when DB query fails (caller returns 503)", async () => {
    mockSql.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    await expect(
      authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING),
    ).rejects.toThrow("ECONNREFUSED");
  });

  it("passes hashed key as parameterized value to SQL query", async () => {
    const expectedHash = await hashApiKey(TEST_RAW_KEY);
    mockSql.mockResolvedValueOnce([validRow]);

    await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);

    // postgres.js tagged templates are called as (strings[], ...values)
    // The keyHash should be the first interpolated value
    const call = mockSql.mock.calls[0];
    const templateStrings = call[0] as string[];
    const keyHashParam = call[1] as string;

    expect(templateStrings.join("")).toContain("key_hash");
    expect(templateStrings.join("")).toContain("revoked_at IS NULL");
    expect(keyHashParam).toBe(expectedHash);
  });

  it("evicts oldest entry when positive cache exceeds 256 entries", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    for (let i = 0; i < 257; i++) {
      mockSql.mockResolvedValueOnce([
        { id: `key-${i}`, user_id: `user-${i}`, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: {}, request_logging_enabled: false },
      ]);
      await authenticateApiKey(`ns_live_sk_key_${i}`, TEST_CONNECTION_STRING);
    }

    expect(mockSql).toHaveBeenCalledTimes(257);

    // key_0 was evicted — looking it up should require a new DB call
    mockSql.mockResolvedValueOnce([
      { id: "key-0", user_id: "user-0", has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: {}, request_logging_enabled: false },
    ]);
    const result = await authenticateApiKey("ns_live_sk_key_0", TEST_CONNECTION_STRING);
    expect(result).toEqual({ userId: "user-0", keyId: "key-0", hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {}, requestLoggingEnabled: false, allowedModels: null, allowedProviders: null, orgUpgradeUrl: null });
    expect(mockSql).toHaveBeenCalledTimes(258);

    // key_2 should still be cached
    const result2 = await authenticateApiKey("ns_live_sk_key_2", TEST_CONNECTION_STRING);
    expect(result2).toEqual({ userId: "user-2", keyId: "key-2", hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {}, requestLoggingEnabled: false, allowedModels: null, allowedProviders: null, orgUpgradeUrl: null });
    expect(mockSql).toHaveBeenCalledTimes(258); // No additional DB call

    vi.restoreAllMocks();
  });

  it("does not negative-cache DB errors \u2014 next request retries", async () => {
    mockSql
      .mockRejectedValueOnce(new Error("ECONNREFUSED"))  // first call: DB down
      .mockResolvedValueOnce([validRow]);                  // second call: DB recovered

    // First call — DB error, throws
    await expect(
      authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING),
    ).rejects.toThrow("ECONNREFUSED");

    // Second call — retries DB (not negative-cached), succeeds
    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2).toEqual({ userId: TEST_USER_ID, keyId: TEST_KEY_ID, hasWebhooks: false, hasBudgets: false, orgId: null, apiVersion: "2026-04-01", defaultTags: {}, requestLoggingEnabled: false, allowedModels: null, allowedProviders: null, orgUpgradeUrl: null });
    expect(mockSql).toHaveBeenCalledTimes(2); // Both calls hit DB
  });

  it("negative cache allows more entries than positive cache (2048 vs 256)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    for (let i = 0; i < 257; i++) {
      mockSql.mockResolvedValueOnce([]);
      await authenticateApiKey(`ns_live_sk_neg_${i}`, TEST_CONNECTION_STRING);
    }

    expect(mockSql).toHaveBeenCalledTimes(257);

    // key_0 should still be in negative cache (not evicted at 256, limit is 2048)
    const result = await authenticateApiKey("ns_live_sk_neg_0", TEST_CONNECTION_STRING);
    expect(result).toBeNull();
    expect(mockSql).toHaveBeenCalledTimes(257); // No additional DB call

    vi.restoreAllMocks();
  });
});

describe("authenticateApiKey cache expiry", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    _resetCaches();
  });

  it("re-queries DB when positive cache entry expires", async () => {
    const now = Date.now();
    // Positive TTL is 120s \u00b110s jitter, so 131s guarantees expiry
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(now)
      .mockReturnValueOnce(now + 131_000);

    mockSql
      .mockResolvedValueOnce([validRow])
      .mockResolvedValueOnce([validRow]);

    await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(mockSql).toHaveBeenCalledTimes(1);

    // Second call — cache expired, hits DB again
    await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(mockSql).toHaveBeenCalledTimes(2);
  });
});

describe("authenticateApiKey defaultTags", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns defaultTags from DB query", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: { project: "openclaw", team: "backend" } },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.defaultTags).toEqual({ project: "openclaw", team: "backend" });
  });

  it("returns empty defaultTags when default_tags column is null", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: null },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.defaultTags).toEqual({});
  });

  it("returns empty defaultTags when default_tags is not an object", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: "not-an-object" },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.defaultTags).toEqual({});
  });

  it("returns empty defaultTags when default_tags is a JSONB array", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: ["a", "b"] },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.defaultTags).toEqual({});
  });

  it("defaultTags are cached in positive cache entry", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: { env: "prod" } },
    ]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1!.defaultTags).toEqual({ env: "prod" });

    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2!.defaultTags).toEqual({ env: "prod" });
    expect(mockSql).toHaveBeenCalledTimes(1); // Only one DB call
  });
});

describe("authenticateApiKey hasBudgets", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns hasBudgets=true when has_budgets is true in DB row", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: true, org_id: null, api_version: "2026-04-01", default_tags: {} },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.hasBudgets).toBe(true);
  });

  it("returns hasBudgets=false when has_budgets is false in DB row", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: {} },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.hasBudgets).toBe(false);
  });

  it("hasBudgets is cached in positive cache entry", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: true, org_id: null, api_version: "2026-04-01", default_tags: {} },
    ]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1!.hasBudgets).toBe(true);

    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2!.hasBudgets).toBe(true);
    expect(mockSql).toHaveBeenCalledTimes(1); // Only one DB call
  });
});

describe("authenticateApiKey orgId", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns orgId when org_id is present in DB row", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, org_id: "550e8400-e29b-41d4-a716-000000000099" },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.orgId).toBe("550e8400-e29b-41d4-a716-000000000099");
  });

  it("returns orgId=null when org_id is null in DB row", async () => {
    mockSql.mockResolvedValueOnce([validRow]); // validRow has org_id: null

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.orgId).toBeNull();
  });
});

describe("authenticateApiKey requestLoggingEnabled", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns requestLoggingEnabled=true when request_logging_enabled is true in DB row", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, request_logging_enabled: true },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.requestLoggingEnabled).toBe(true);
  });

  it("returns requestLoggingEnabled=false when request_logging_enabled is false in DB row", async () => {
    mockSql.mockResolvedValueOnce([validRow]); // validRow has request_logging_enabled: false

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.requestLoggingEnabled).toBe(false);
  });

  it("returns requestLoggingEnabled=false when field is absent (COALESCE fallback)", async () => {
    mockSql.mockResolvedValueOnce([
      { id: TEST_KEY_ID, user_id: TEST_USER_ID, has_webhooks: false, has_budgets: false, org_id: null, api_version: "2026-04-01", default_tags: {} },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.requestLoggingEnabled).toBe(false);
  });

  it("requestLoggingEnabled is cached in positive cache entry", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, request_logging_enabled: true },
    ]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1!.requestLoggingEnabled).toBe(true);

    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2!.requestLoggingEnabled).toBe(true);
    expect(mockSql).toHaveBeenCalledTimes(1);
  });
});

describe("authenticateApiKey allowedModels/allowedProviders", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    _resetCaches();
  });

  afterEach(() => {
    _resetCaches();
  });

  it("returns allowedModels from DB when allowed_models is a non-empty array", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: ["gpt-4o", "gpt-4o-mini"], allowed_providers: null },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toEqual(["gpt-4o", "gpt-4o-mini"]);
    expect(result!.allowedProviders).toBeNull();
  });

  it("returns allowedProviders from DB when allowed_providers is a non-empty array", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: null, allowed_providers: ["openai"] },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toBeNull();
    expect(result!.allowedProviders).toEqual(["openai"]);
  });

  it("returns empty array when allowed_models is empty (deny all)", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: [], allowed_providers: [] },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toEqual([]);
    expect(result!.allowedProviders).toEqual([]);
  });

  it("returns null when allowed_models/allowed_providers are null (unrestricted)", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: null, allowed_providers: null },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toBeNull();
    expect(result!.allowedProviders).toBeNull();
  });

  it("returns null when allowed_models/allowed_providers are absent from row", async () => {
    mockSql.mockResolvedValueOnce([validRow]); // validRow has no allowed_models/allowed_providers fields

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toBeNull();
    expect(result!.allowedProviders).toBeNull();
  });

  it("parses allowed_models from Postgres text array literal string (Hyperdrive/fetch_types:false)", async () => {
    // When fetch_types:false, postgres.js returns text[] as a raw string like "{gpt-4o,gpt-4o-mini}"
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: "{gpt-4o,gpt-4o-mini}", allowed_providers: "{openai}" },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toEqual(["gpt-4o", "gpt-4o-mini"]);
    expect(result!.allowedProviders).toEqual(["openai"]);
  });

  it("parses empty Postgres text array literal string as empty array", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: "{}", allowed_providers: "{}" },
    ]);

    const result = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result!.allowedModels).toEqual([]);
    expect(result!.allowedProviders).toEqual([]);
  });

  it("allowedModels/allowedProviders are cached in positive cache entry", async () => {
    mockSql.mockResolvedValueOnce([
      { ...validRow, allowed_models: ["gpt-4o"], allowed_providers: ["openai", "anthropic"] },
    ]);

    const result1 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result1!.allowedModels).toEqual(["gpt-4o"]);
    expect(result1!.allowedProviders).toEqual(["openai", "anthropic"]);

    const result2 = await authenticateApiKey(TEST_RAW_KEY, TEST_CONNECTION_STRING);
    expect(result2!.allowedModels).toEqual(["gpt-4o"]);
    expect(result2!.allowedProviders).toEqual(["openai", "anthropic"]);
    expect(mockSql).toHaveBeenCalledTimes(1); // Only one DB call
  });
});
