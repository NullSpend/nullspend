import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

import { findOrphanedBudgets } from "../durable-objects/user-budget.js";

const NOW = 1_800_000_000_000;
const OLD = NOW - 10 * 60_000;
const RECENT = NOW - 10_000;

describe("findOrphanedBudgets (AUDIT-7 orphan sweep helper)", () => {
  it("returns empty when DO has no rows", () => {
    expect(findOrphanedBudgets([], [{ entity_type: "user", entity_id: "u1" }], NOW)).toEqual([]);
  });

  it("returns empty when every DO row has a matching Postgres row", () => {
    const doRows = [
      { entity_type: "user", entity_id: "u1", synced_at: OLD },
      { entity_type: "customer", entity_id: "acme", synced_at: OLD },
    ];
    const pgRows = [
      { entity_type: "user", entity_id: "u1" },
      { entity_type: "customer", entity_id: "acme" },
    ];
    expect(findOrphanedBudgets(doRows, pgRows, NOW)).toEqual([]);
  });

  it("flags a DO row with no Postgres counterpart as an orphan once past the safety window", () => {
    const doRows = [{ entity_type: "customer", entity_id: "ghost", synced_at: OLD }];
    expect(findOrphanedBudgets(doRows, [], NOW)).toEqual([
      { entity_type: "customer", entity_id: "ghost" },
    ]);
  });

  it("does NOT flag a recently-synced DO row even if Postgres is empty (race safety)", () => {
    const doRows = [{ entity_type: "customer", entity_id: "just-created", synced_at: RECENT }];
    expect(findOrphanedBudgets(doRows, [], NOW)).toEqual([]);
  });

  it("treats a synced_at exactly at the cutoff as safe (strict <)", () => {
    const doRows = [{ entity_type: "user", entity_id: "u1", synced_at: NOW - 60_000 }];
    expect(findOrphanedBudgets(doRows, [], NOW)).toEqual([]);
  });

  it("flags a synced_at one ms older than the cutoff", () => {
    const doRows = [{ entity_type: "user", entity_id: "u1", synced_at: NOW - 60_001 }];
    expect(findOrphanedBudgets(doRows, [], NOW)).toEqual([
      { entity_type: "user", entity_id: "u1" },
    ]);
  });

  it("mixed DO rows: some orphan, some current", () => {
    const doRows = [
      { entity_type: "user", entity_id: "alive", synced_at: OLD },
      { entity_type: "customer", entity_id: "ghost", synced_at: OLD },
      { entity_type: "customer", entity_id: "fresh", synced_at: RECENT },
    ];
    const pgRows = [{ entity_type: "user", entity_id: "alive" }];
    expect(findOrphanedBudgets(doRows, pgRows, NOW)).toEqual([
      { entity_type: "customer", entity_id: "ghost" },
    ]);
  });

  it("entity key compare is case-sensitive", () => {
    const doRows = [{ entity_type: "customer", entity_id: "ACME", synced_at: OLD }];
    const pgRows = [{ entity_type: "customer", entity_id: "acme" }];
    expect(findOrphanedBudgets(doRows, pgRows, NOW)).toEqual([
      { entity_type: "customer", entity_id: "ACME" },
    ]);
  });

  it("colon-containing entity IDs don't confuse the matcher", () => {
    const doRows = [
      { entity_type: "tag", entity_id: "env=prod", synced_at: OLD },
      { entity_type: "tag", entity_id: "env=staging", synced_at: OLD },
    ];
    const pgRows = [{ entity_type: "tag", entity_id: "env=prod" }];
    expect(findOrphanedBudgets(doRows, pgRows, NOW)).toEqual([
      { entity_type: "tag", entity_id: "env=staging" },
    ]);
  });

  it("honors a custom safetyMs window", () => {
    const doRows = [{ entity_type: "user", entity_id: "u1", synced_at: NOW - 120_000 }];
    expect(findOrphanedBudgets(doRows, [], NOW, 30_000)).toEqual([
      { entity_type: "user", entity_id: "u1" },
    ]);
    expect(findOrphanedBudgets(doRows, [], NOW, 180_000)).toEqual([]);
  });
});
