import { describe, it, expect, vi } from "vitest";

// shared.ts imports cloudflare:workers for waitUntil — mock it at the
// module boundary so the pure helper can be tested in isolation.
vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn(),
}));

import { buildBudgetHeaders } from "../routes/shared.js";
import type { BudgetEntity } from "../lib/budget-do-lookup.js";

/**
 * Tests for `buildBudgetHeaders` — the helper that stamps
 * `X-NullSpend-Budget-*` headers on proxy responses.
 *
 * Contract reminders:
 * - Empty entities → empty record (absence of headers means "no enforcement")
 * - Multi-entity → tightest remaining wins
 * - Ties broken deterministically by (entityType, entityId) ASCII order
 * - Microdollar integers, stringified
 * - `spent` = spend + reserved + reservedForThisRequest
 * - `remaining` = clamped max(0, limit - spend - reserved - reservedForThisRequest)
 * - `limit = spent + remaining` invariant on the winning entity
 */

function entity(
  type: string,
  id: string,
  maxBudget: number,
  spend: number,
  reserved: number,
): Pick<BudgetEntity, "entityType" | "entityId" | "maxBudget" | "spend" | "reserved"> {
  return { entityType: type, entityId: id, maxBudget, spend, reserved };
}

describe("buildBudgetHeaders", () => {
  it("returns empty record when there are no entities", () => {
    expect(buildBudgetHeaders([], 0)).toEqual({});
    expect(buildBudgetHeaders([], 1_000_000)).toEqual({});
  });

  it("single entity: reports maxBudget/spent/remaining/entity", () => {
    const entities = [entity("user", "u1", 100_000_000, 20_000_000, 5_000_000)];
    const headers = buildBudgetHeaders(entities, 1_000_000);

    expect(headers["X-NullSpend-Budget-Limit"]).toBe("100000000");
    expect(headers["X-NullSpend-Budget-Spent"]).toBe("26000000"); // 20M + 5M + 1M
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("74000000"); // 100M - 26M
    expect(headers["X-NullSpend-Budget-Entity"]).toBe("user:u1");
  });

  it("invariant: limit = spent + remaining on a single entity", () => {
    const entities = [entity("api_key", "k1", 50_000_000, 10_000_000, 2_500_000)];
    const headers = buildBudgetHeaders(entities, 7_500_000);

    const limit = Number(headers["X-NullSpend-Budget-Limit"]);
    const spent = Number(headers["X-NullSpend-Budget-Spent"]);
    const remaining = Number(headers["X-NullSpend-Budget-Remaining"]);

    expect(limit).toBe(spent + remaining);
  });

  it("multi-entity: picks the one with the lowest remaining", () => {
    const entities = [
      // Org: 500M limit, 100M spent, 0 reserved → 400M remaining
      entity("org", "o1", 500_000_000, 100_000_000, 0),
      // User: 100M limit, 50M spent, 10M reserved → 40M remaining — TIGHTEST
      entity("user", "u1", 100_000_000, 50_000_000, 10_000_000),
      // API key: 200M limit, 5M spent, 0 reserved → 195M remaining
      entity("api_key", "k1", 200_000_000, 5_000_000, 0),
    ];

    const headers = buildBudgetHeaders(entities, 0);

    expect(headers["X-NullSpend-Budget-Entity"]).toBe("user:u1");
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("40000000");
    expect(headers["X-NullSpend-Budget-Limit"]).toBe("100000000");
  });

  it("multi-entity: reservedForThisRequest is subtracted from all candidates during tightest selection", () => {
    const entities = [
      // Without subtracting estimate: both have 5M remaining. With 3M estimate: still tie.
      entity("user", "u1", 10_000_000, 5_000_000, 0),
      entity("org", "o1", 10_000_000, 5_000_000, 0),
    ];

    const headers = buildBudgetHeaders(entities, 3_000_000);

    // Both have remaining = max(0, 10M - 5M - 0 - 3M) = 2M — tie broken by
    // (entityType ASC): "org" < "user", so org wins.
    expect(headers["X-NullSpend-Budget-Entity"]).toBe("org:o1");
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("2000000");
  });

  it("tie-break: deterministic order by (entityType, entityId) ASCII", () => {
    // Three entities with identical remaining
    const entities = [
      entity("user", "u2", 100_000_000, 50_000_000, 0),
      entity("user", "u1", 100_000_000, 50_000_000, 0),
      entity("api_key", "k1", 100_000_000, 50_000_000, 0),
    ];

    const headers = buildBudgetHeaders(entities, 0);

    // api_key < user, so api_key:k1 wins
    expect(headers["X-NullSpend-Budget-Entity"]).toBe("api_key:k1");
  });

  it("tie-break: same entityType, different entityId — lower ID wins", () => {
    const entities = [
      entity("user", "u2", 100_000_000, 50_000_000, 0),
      entity("user", "u1", 100_000_000, 50_000_000, 0),
    ];

    const headers = buildBudgetHeaders(entities, 0);

    expect(headers["X-NullSpend-Budget-Entity"]).toBe("user:u1");
  });

  it("clamps negative remaining to 0", () => {
    // Edge case: entity has negative math because spend + reserved + estimate > limit
    // (shouldn't happen on the approved path but can occur on denial paths)
    const entities = [entity("user", "u1", 10_000_000, 5_000_000, 4_000_000)];

    const headers = buildBudgetHeaders(entities, 5_000_000); // 10M - 5M - 4M - 5M = -4M

    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("0");
  });

  it("live reserved from CheckedEntity is honored (not hardcoded 0)", () => {
    // Regression guard: under concurrent load, two in-flight requests each
    // see the other's reservation as non-zero via the DO's reserved column.
    // Pre-2026-04-08 the orchestrator hardcoded reserved:0 on the approved
    // path and this test would have reported stale "remaining" that didn't
    // account for concurrent in-flight reservations.
    const entities = [entity("user", "u1", 100_000_000, 20_000_000, 45_000_000)];

    const headers = buildBudgetHeaders(entities, 5_000_000);

    // Spent = 20M + 45M + 5M = 70M
    // Remaining = 100M - 70M = 30M
    expect(headers["X-NullSpend-Budget-Spent"]).toBe("70000000");
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("30000000");
  });

  it("denial semantics: reservedForThisRequest=0 reflects 'no reservation landed'", () => {
    // On denial the request's estimate never reserved against the DO, so
    // callers pass 0. Spent reflects state before the (rejected) request.
    const entities = [entity("user", "u1", 100_000_000, 99_500_000, 0)];

    const headers = buildBudgetHeaders(entities, 0);

    expect(headers["X-NullSpend-Budget-Spent"]).toBe("99500000");
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("500000");
  });

  it("returns exactly the four expected header keys (no extras, no misses)", () => {
    const entities = [entity("user", "u1", 100_000_000, 20_000_000, 5_000_000)];
    const headers = buildBudgetHeaders(entities, 1_000_000);

    expect(Object.keys(headers).sort()).toEqual([
      "X-NullSpend-Budget-Entity",
      "X-NullSpend-Budget-Limit",
      "X-NullSpend-Budget-Remaining",
      "X-NullSpend-Budget-Spent",
    ]);
  });

  it("handles zero-spend zero-reserved entity cleanly", () => {
    const entities = [entity("user", "u1", 100_000_000, 0, 0)];
    const headers = buildBudgetHeaders(entities, 0);

    expect(headers["X-NullSpend-Budget-Limit"]).toBe("100000000");
    expect(headers["X-NullSpend-Budget-Spent"]).toBe("0");
    expect(headers["X-NullSpend-Budget-Remaining"]).toBe("100000000");
  });
});
