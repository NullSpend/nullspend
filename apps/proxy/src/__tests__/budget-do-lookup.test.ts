import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockSql, mockEmitMetric } = vi.hoisted(() => {
  const mockSql = vi.fn().mockResolvedValue([]);
  const mockEmitMetric = vi.fn();
  return { mockSql, mockEmitMetric };
});

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: mockEmitMetric,
}));

import { lookupBudgetsForDO, lookupCustomerUpgradeUrl } from "../lib/budget-do-lookup.js";

function makeBudgetRow(overrides: Record<string, unknown> = {}) {
  return {
    entity_type: "user",
    entity_id: "user-1",
    max_budget_microdollars: 50_000_000,
    spend_microdollars: 10_000_000,
    policy: "strict_block",
    reset_interval: "monthly",
    current_period_start: "2025-03-01T00:00:00Z",
    velocity_limit_microdollars: null,
    velocity_window_seconds: null,
    velocity_cooldown_seconds: null,
    threshold_percentages: null,
    session_limit_microdollars: null,
    ...overrides,
  };
}

describe("lookupBudgetsForDO", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("returns entities with all DO-required fields", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow()]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      entityType: "user",
      entityId: "user-1",
      maxBudget: 50_000_000,
      spend: 10_000_000,
      policy: "strict_block",
      resetInterval: "monthly",
      periodStart: new Date("2025-03-01T00:00:00Z").getTime(),
      velocityLimit: null,
      velocityWindow: 60_000,
      velocityCooldown: 60_000,
      thresholdPercentages: [50, 80, 90, 95],
      sessionLimit: null,
    });
  });

  it("converts velocity fields from seconds to ms", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow({
      velocity_limit_microdollars: 5_000_000,
      velocity_window_seconds: 120,
      velocity_cooldown_seconds: 90,
    })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result[0].velocityLimit).toBe(5_000_000);
    expect(result[0].velocityWindow).toBe(120_000);
    expect(result[0].velocityCooldown).toBe(90_000);
  });

  it("defaults velocity window and cooldown to 60s when null", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow()]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result[0].velocityLimit).toBeNull();
    expect(result[0].velocityWindow).toBe(60_000);
    expect(result[0].velocityCooldown).toBe(60_000);
  });

  it("converts timestamp to epoch ms", async () => {
    const dateStr = "2025-06-15T12:00:00Z";
    mockSql.mockResolvedValueOnce([makeBudgetRow({ current_period_start: dateStr })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: "key-1", orgId: null, userId: null, tags: {},
    });

    expect(result[0].periodStart).toBe(new Date(dateStr).getTime());
  });

  it("handles null currentPeriodStart (→ 0)", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow({ current_period_start: null })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result[0].periodStart).toBe(0);
  });

  it("handles null resetInterval (→ null)", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow({ reset_interval: null })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result[0].resetInterval).toBeNull();
  });

  it("returns empty array when no budgets found", async () => {
    mockSql.mockResolvedValueOnce([]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result).toEqual([]);
  });

  it("throws on Postgres error (fail-closed)", async () => {
    mockSql.mockRejectedValueOnce(new Error("connection refused"));

    await expect(
      lookupBudgetsForDO("postgres://test", { keyId: null, orgId: "user-1", userId: "user-1", tags: {} }),
    ).rejects.toThrow("connection refused");
  });

  it("skips entities where identity field is null", async () => {
    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: null, userId: null, tags: {},
    });

    expect(result).toEqual([]);
    expect(mockSql).not.toHaveBeenCalled();
  });

  it("returns tag budget entities when tags match", async () => {
    // First call: user entity
    mockSql.mockResolvedValueOnce([makeBudgetRow()]);
    // Second call: tag entities
    mockSql.mockResolvedValueOnce([makeBudgetRow({
      entity_type: "tag",
      entity_id: "project=openclaw",
      max_budget_microdollars: 50_000_000,
      spend_microdollars: 5_000_000,
      current_period_start: null,
    })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: { project: "openclaw" },
    });

    expect(result).toHaveLength(2);
    expect(result[0].entityType).toBe("user");
    expect(result[1].entityType).toBe("tag");
    expect(result[1].entityId).toBe("project=openclaw");
  });

  it("does not query tag budgets when tags is empty", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow()]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "user-1", userId: "user-1", tags: {},
    });

    expect(result).toHaveLength(1);
    expect(mockSql).toHaveBeenCalledTimes(1); // only user query
  });

  it("does not query tag budgets when orgId is null", async () => {
    mockSql.mockResolvedValueOnce([makeBudgetRow({ entity_type: "api_key", entity_id: "key-1" })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: "key-1", orgId: null, userId: null, tags: { project: "openclaw" },
    });

    expect(result).toHaveLength(1);
    expect(mockSql).toHaveBeenCalledTimes(1); // only api_key query
  });

  it("looks up customer budget when customer tag is present", async () => {
    // user query (no user budget) + tag query (no tags) + customer query (found)
    mockSql
      .mockResolvedValueOnce([]) // user query
      .mockResolvedValueOnce([]) // tag query
      .mockResolvedValueOnce([makeBudgetRow({ entity_type: "customer", entity_id: "acme-corp" })]);

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "org-1", userId: "user-1", tags: { customer: "acme-corp" },
    });

    expect(result).toHaveLength(1);
    expect(result[0].entityType).toBe("customer");
    expect(result[0].entityId).toBe("acme-corp");
  });

  it("does not look up customer budget when no customer tag", async () => {
    mockSql
      .mockResolvedValueOnce([]) // user query
      .mockResolvedValueOnce([]); // tag query (env=prod)

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "org-1", userId: "user-1", tags: { env: "prod" },
    });

    expect(result).toHaveLength(0);
    expect(mockSql).toHaveBeenCalledTimes(2); // user + tag, no customer query
  });

  it("returns both tag and customer budgets when both exist", async () => {
    mockSql
      .mockResolvedValueOnce([]) // user query
      .mockResolvedValueOnce([makeBudgetRow({ entity_type: "tag", entity_id: "customer=acme-corp" })]) // tag query
      .mockResolvedValueOnce([makeBudgetRow({ entity_type: "customer", entity_id: "acme-corp" })]); // customer query

    const result = await lookupBudgetsForDO("postgres://test", {
      keyId: null, orgId: "org-1", userId: "user-1", tags: { customer: "acme-corp" },
    });

    expect(result).toHaveLength(2);
    expect(result.map(r => r.entityType)).toEqual(["tag", "customer"]);
  });
});

// ─────────────────────────────────────────────────────────────────
// lookupCustomerUpgradeUrl — added by Phase 0 edge-case audit
//
// Cold-path Postgres query for per-customer upgrade URL. Must fail
// open on any error (returning null) so denial responses still ship,
// just without the upgrade_url field. Systematic failures should
// surface via the customer_upgrade_url_lookup_failed metric.
// ─────────────────────────────────────────────────────────────────
describe("lookupCustomerUpgradeUrl", () => {
  const CONN = "postgresql://test";
  const ORG_ID = "org-uuid-123";
  const CUSTOMER_ID = "acme-corp";

  beforeEach(() => {
    mockSql.mockReset();
    mockEmitMetric.mockReset();
  });

  it("returns the URL when a row exists with a non-empty upgrade_url", async () => {
    mockSql.mockResolvedValueOnce([{ upgrade_url: "https://acme.com/upgrade?id={customer_id}" }]);

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBe("https://acme.com/upgrade?id={customer_id}");
    expect(mockEmitMetric).not.toHaveBeenCalled();
  });

  it("returns null when no row matches", async () => {
    mockSql.mockResolvedValueOnce([]);

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
    expect(mockEmitMetric).not.toHaveBeenCalled();
  });

  it("returns null when row exists but upgrade_url column is null", async () => {
    mockSql.mockResolvedValueOnce([{ upgrade_url: null }]);

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
  });

  it("returns null for an empty-string upgrade_url (defensive)", async () => {
    mockSql.mockResolvedValueOnce([{ upgrade_url: "" }]);

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
  });

  it("fails open to null on DB throw (network error) and emits metric", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockSql.mockRejectedValueOnce(new Error("connection refused"));

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "customer_upgrade_url_lookup_failed",
      expect.objectContaining({ orgId: ORG_ID, error: "connection refused" }),
    );
  });

  it("fails open to null on non-Error rejection and emits metric with 'unknown'", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mockSql.mockRejectedValueOnce("string rejection");

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "customer_upgrade_url_lookup_failed",
      expect.objectContaining({ orgId: ORG_ID, error: "unknown" }),
    );
  });

  it("returns null for a non-string upgrade_url (defensive type check)", async () => {
    mockSql.mockResolvedValueOnce([{ upgrade_url: 42 }]);

    const result = await lookupCustomerUpgradeUrl(CONN, ORG_ID, CUSTOMER_ID);

    expect(result).toBeNull();
  });
});
