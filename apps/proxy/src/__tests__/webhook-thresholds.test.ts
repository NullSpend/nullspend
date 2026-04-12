import { describe, it, expect } from "vitest";
import { detectThresholdCrossings } from "../lib/webhook-thresholds.js";
import type { BudgetEntity } from "../lib/budget-do-lookup.js";

function makeBudgetEntity(overrides: Partial<BudgetEntity> = {}): BudgetEntity {
  return {
    entityKey: "{budget}:user:user-1",
    entityType: "user",
    entityId: "user-1",
    maxBudget: 100_000_000, // $100
    spend: 0,
    reserved: 0,
    policy: "strict_block",
    thresholdPercentages: [50, 80, 90, 95],
    ...overrides,
  };
}

describe("detectThresholdCrossings", () => {
  it("detects 50% threshold crossing", () => {
    const entity = makeBudgetEntity({ spend: 49_000_000 });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.warning");
    expect(events[0].api_version).toBe("2026-04-01");
    expect(typeof events[0].created_at).toBe("number");
    expect(events[0].data.object.threshold_percent).toBe(50);
  });

  it("detects 80% threshold crossing", () => {
    const entity = makeBudgetEntity({ spend: 79_000_000 });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.warning");
    expect(events[0].data.object.threshold_percent).toBe(80);
  });

  it("detects 90% threshold crossing as critical", () => {
    const entity = makeBudgetEntity({ spend: 89_000_000 });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
    expect(events[0].data.object.threshold_percent).toBe(90);
  });

  it("detects 95% threshold crossing as critical", () => {
    const entity = makeBudgetEntity({ spend: 94_000_000 });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
    expect(events[0].data.object.threshold_percent).toBe(95);
  });

  it("detects multiple threshold crossings in a single request", () => {
    // Spend goes from 0 to 85% in one big request
    const entity = makeBudgetEntity({ spend: 0 });
    const events = detectThresholdCrossings([entity], 85_000_000, "req_1");

    expect(events).toHaveLength(2);
    const types = events.map((e) => e.data.object.threshold_percent);
    expect(types).toContain(50);
    expect(types).toContain(80);
  });

  it("returns empty when no threshold is crossed", () => {
    const entity = makeBudgetEntity({ spend: 10_000_000 });
    // +5M = 15% — no threshold crossed
    const events = detectThresholdCrossings([entity], 5_000_000, "req_1");
    expect(events).toHaveLength(0);
  });

  it("returns empty when already above all thresholds", () => {
    const entity = makeBudgetEntity({ spend: 96_000_000 });
    const events = detectThresholdCrossings([entity], 1_000_000, "req_1");
    expect(events).toHaveLength(0);
  });

  it("handles multiple budget entities", () => {
    const entities = [
      makeBudgetEntity({ entityKey: "{budget}:user:u1", entityId: "u1", spend: 49_000_000 }),
      makeBudgetEntity({ entityKey: "{budget}:api_key:k1", entityId: "k1", spend: 89_000_000 }),
    ];
    const events = detectThresholdCrossings(entities, 2_000_000, "req_1");

    expect(events).toHaveLength(2);
    expect(events[0].data.object.threshold_percent).toBe(50);
    expect(events[0].data.object.budget_entity_id).toBe("u1");
    expect(events[1].data.object.threshold_percent).toBe(90);
    expect(events[1].data.object.budget_entity_id).toBe("k1");
  });

  it("skips entities with zero maxBudget", () => {
    const entity = makeBudgetEntity({ maxBudget: 0 });
    const events = detectThresholdCrossings([entity], 1_000_000, "req_1");
    expect(events).toHaveLength(0);
  });

  it("includes correct payload fields", () => {
    const entity = makeBudgetEntity({
      entityType: "api_key",
      entityId: "key_abc",
      maxBudget: 50_000_000,
      spend: 39_000_000,
    });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_xyz");

    expect(events).toHaveLength(1);
    expect(events[0].data.object).toEqual(expect.objectContaining({
      budget_entity_type: "api_key",
      budget_entity_id: "key_abc",
      budget_limit_microdollars: 50_000_000,
      budget_spend_microdollars: 41_000_000,
      threshold_percent: 80,
      budget_remaining_microdollars: 9_000_000,
      triggered_by_request_id: "req_xyz",
    }));
  });

  it("detects exact threshold boundary (spend equals threshold exactly)", () => {
    // Spend goes from 49% to exactly 50%
    const entity = makeBudgetEntity({ maxBudget: 100_000_000, spend: 49_000_000 });
    const events = detectThresholdCrossings([entity], 1_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].data.object.threshold_percent).toBe(50);
  });

  // ── Custom threshold tests ────────────────────────────────────────

  it("uses custom thresholds [25, 50, 75] — only those values trigger", () => {
    const entity = makeBudgetEntity({
      spend: 0,
      thresholdPercentages: [25, 50, 75],
    });
    // 0% → 80% — crosses 25, 50, 75
    const events = detectThresholdCrossings([entity], 80_000_000, "req_1");

    expect(events).toHaveLength(3);
    const thresholds = events.map((e) => e.data.object.threshold_percent);
    expect(thresholds).toEqual([25, 50, 75]);
  });

  it("last custom threshold is classified as critical", () => {
    const entity = makeBudgetEntity({
      spend: 74_000_000,
      thresholdPercentages: [25, 50, 75],
    });
    // 74% → 76% — crosses 75 (last = critical)
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
    expect(events[0].data.object.threshold_percent).toBe(75);
  });

  it("non-last custom threshold is classified as warning", () => {
    const entity = makeBudgetEntity({
      spend: 24_000_000,
      thresholdPercentages: [25, 50, 75],
    });
    // 24% → 26% — crosses 25 (not last, not >= 90 = warning)
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.warning");
  });

  it("empty thresholds [] produces zero events", () => {
    const entity = makeBudgetEntity({
      spend: 0,
      thresholdPercentages: [],
    });
    const events = detectThresholdCrossings([entity], 99_000_000, "req_1");
    expect(events).toHaveLength(0);
  });

  it("single threshold [90] is classified as critical", () => {
    const entity = makeBudgetEntity({
      spend: 89_000_000,
      thresholdPercentages: [90],
    });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
    expect(events[0].data.object.threshold_percent).toBe(90);
  });

  it("single threshold [60] below 90 is critical (last in array)", () => {
    const entity = makeBudgetEntity({
      spend: 59_000_000,
      thresholdPercentages: [60],
    });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
  });

  it("mixed entities: one custom, one default — independent thresholds", () => {
    const entities = [
      makeBudgetEntity({
        entityKey: "{budget}:user:u1",
        entityId: "u1",
        spend: 24_000_000,
        thresholdPercentages: [25, 50, 75],
      }),
      makeBudgetEntity({
        entityKey: "{budget}:api_key:k1",
        entityId: "k1",
        spend: 49_000_000,
        thresholdPercentages: [50, 80, 90, 95],
      }),
    ];
    const events = detectThresholdCrossings(entities, 2_000_000, "req_1");

    expect(events).toHaveLength(2);
    // u1 crosses 25% (custom)
    expect(events[0].data.object.budget_entity_id).toBe("u1");
    expect(events[0].data.object.threshold_percent).toBe(25);
    expect(events[0].type).toBe("budget.threshold.warning");
    // k1 crosses 50% (default)
    expect(events[1].data.object.budget_entity_id).toBe("k1");
    expect(events[1].data.object.threshold_percent).toBe(50);
    expect(events[1].type).toBe("budget.threshold.warning");
  });

  it("falls back to default thresholds when thresholdPercentages is undefined", () => {
    const entity = makeBudgetEntity({
      spend: 49_000_000,
      thresholdPercentages: undefined,
    });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].data.object.threshold_percent).toBe(50);
  });

  it("backward compat: default thresholds — 90% is still critical (not just last)", () => {
    // Regression test: with defaults [50,80,90,95], 90 is critical because >= 90
    const entity = makeBudgetEntity({ spend: 89_000_000 });
    const events = detectThresholdCrossings([entity], 2_000_000, "req_1");

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("budget.threshold.critical");
    expect(events[0].data.object.threshold_percent).toBe(90);
  });
});
