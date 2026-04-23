import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { resolvePeriodBounds } from "../lib/period-math.js";
import * as metrics from "../lib/metrics.js";

// Per PR-2a tests C1, C2, C3, C2b, C2c, C2d.
// Verifies the helper's fallback + metric emission contract per Decision #36 /
// plan-audit A5 / codex PR-2a-R3-2:
//   - Valid subscription period → use verbatim, NO metric.
//   - Free/unpaid (both fields null) → calendar month, NO metric.
//   - Paid-but-corrupt (one field null, other set) → calendar month + emit paid_partial.
//   - Paid-but-inverted (end <= start) → calendar month + emit paid_inverted.

describe("resolvePeriodBounds", () => {
  let emitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    emitSpy = vi.spyOn(metrics, "emitMetric").mockImplementation(() => {});
  });

  afterEach(() => {
    emitSpy.mockRestore();
  });

  // C1 — both subscription fields null → UTC calendar month; NO metric
  it("returns UTC calendar month for Free org (no subscription)", () => {
    const now = Date.UTC(2026, 4, 15, 10, 30, 0); // May 15, 10:30 UTC
    const bounds = resolvePeriodBounds(
      { orgId: "org-free", subscriptionPeriodStart: null, subscriptionPeriodEnd: null },
      now,
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 4, 1, 0, 0, 0));
    expect(bounds.periodEnd).toBe(Date.UTC(2026, 5, 1, 0, 0, 0));
    expect(emitSpy).not.toHaveBeenCalled();
  });

  // C2 — December → January year rollover for calendar fallback
  it("rolls over correctly at year boundary", () => {
    const now = Date.UTC(2026, 11, 31, 23, 59, 59); // Dec 31, 23:59:59 UTC
    const bounds = resolvePeriodBounds(
      { orgId: "org-free", subscriptionPeriodStart: null, subscriptionPeriodEnd: null },
      now,
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 11, 1, 0, 0, 0));
    expect(bounds.periodEnd).toBe(Date.UTC(2027, 0, 1, 0, 0, 0));
  });

  // C3 — exact midnight UTC on the 1st is the START of the new period
  it("midnight UTC on the 1st is start of new period", () => {
    const now = Date.UTC(2026, 5, 1, 0, 0, 0); // June 1, 00:00:00 UTC
    const bounds = resolvePeriodBounds(
      { orgId: "org-free", subscriptionPeriodStart: null, subscriptionPeriodEnd: null },
      now,
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 5, 1, 0, 0, 0));
    expect(bounds.periodEnd).toBe(Date.UTC(2026, 6, 1, 0, 0, 0));
  });

  // C2b — valid subscription period is used verbatim; NO metric
  it("uses valid subscription period verbatim", () => {
    const subStart = Date.UTC(2026, 3, 17, 0, 0, 0); // Apr 17
    const subEnd = Date.UTC(2026, 4, 17, 0, 0, 0); // May 17
    const bounds = resolvePeriodBounds(
      {
        orgId: "org-pro",
        subscriptionPeriodStart: subStart,
        subscriptionPeriodEnd: subEnd,
      },
      Date.UTC(2026, 3, 20),
    );
    expect(bounds.periodStart).toBe(subStart);
    expect(bounds.periodEnd).toBe(subEnd);
    expect(emitSpy).not.toHaveBeenCalled();
  });

  // C2c — paid-but-partial (one side null, other set) → fallback + emit paid_partial
  it("emits paid_partial metric when one period field is null", () => {
    const subStart = Date.UTC(2026, 3, 17);
    const now = Date.UTC(2026, 3, 20);
    const bounds = resolvePeriodBounds(
      { orgId: "org-pro-corrupt", subscriptionPeriodStart: subStart, subscriptionPeriodEnd: null },
      now,
    );
    // Fallback to calendar month
    expect(bounds.periodStart).toBe(Date.UTC(2026, 3, 1));
    expect(bounds.periodEnd).toBe(Date.UTC(2026, 4, 1));
    // Metric emitted with reason=paid_partial
    expect(emitSpy).toHaveBeenCalledWith("plan_counter_period_fallback", {
      reason: "paid_partial",
      orgId: "org-pro-corrupt",
    });
  });

  it("emits paid_partial metric when end is set but start is null", () => {
    const subEnd = Date.UTC(2026, 4, 17);
    const now = Date.UTC(2026, 3, 20);
    const bounds = resolvePeriodBounds(
      { orgId: "org-pro-corrupt2", subscriptionPeriodStart: null, subscriptionPeriodEnd: subEnd },
      now,
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 3, 1));
    expect(emitSpy).toHaveBeenCalledWith("plan_counter_period_fallback", {
      reason: "paid_partial",
      orgId: "org-pro-corrupt2",
    });
  });

  // C2d — paid-but-inverted (end <= start) → fallback + emit paid_inverted
  it("emits paid_inverted metric when subscriptionPeriodEnd <= subscriptionPeriodStart", () => {
    const subStart = Date.UTC(2026, 4, 17); // May 17
    const subEnd = Date.UTC(2026, 3, 17); // Apr 17 — INVERTED
    const now = Date.UTC(2026, 3, 20);
    const bounds = resolvePeriodBounds(
      { orgId: "org-pro-inverted", subscriptionPeriodStart: subStart, subscriptionPeriodEnd: subEnd },
      now,
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 3, 1));
    expect(bounds.periodEnd).toBe(Date.UTC(2026, 4, 1));
    expect(emitSpy).toHaveBeenCalledWith("plan_counter_period_fallback", {
      reason: "paid_inverted",
      orgId: "org-pro-inverted",
    });
  });

  // Zero-length period (end === start) is also "inverted" per the invariant
  it("treats zero-length period as paid_inverted", () => {
    const t = Date.UTC(2026, 3, 17);
    const bounds = resolvePeriodBounds(
      { orgId: "org-zero", subscriptionPeriodStart: t, subscriptionPeriodEnd: t },
      Date.UTC(2026, 3, 20),
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 3, 1));
    expect(emitSpy).toHaveBeenCalledWith("plan_counter_period_fallback", {
      reason: "paid_inverted",
      orgId: "org-zero",
    });
  });

  // Null orgId: metric tag is literal "null", not undefined
  it("uses 'null' sentinel for null orgId in metric tag", () => {
    const bounds = resolvePeriodBounds(
      {
        orgId: null,
        subscriptionPeriodStart: null,
        subscriptionPeriodEnd: Date.UTC(2026, 3, 17), // triggers paid_partial
      },
      Date.UTC(2026, 3, 20),
    );
    expect(bounds.periodStart).toBe(Date.UTC(2026, 3, 1));
    expect(emitSpy).toHaveBeenCalledWith("plan_counter_period_fallback", {
      reason: "paid_partial",
      orgId: "null",
    });
  });

  // C-METRIC1 regression — helper never emits the deprecated metric name
  it("NEVER emits plan_counter_invalid_period (regression guard)", () => {
    resolvePeriodBounds(
      { orgId: "org", subscriptionPeriodStart: null, subscriptionPeriodEnd: null },
      Date.now(),
    );
    resolvePeriodBounds(
      { orgId: "org", subscriptionPeriodStart: 1, subscriptionPeriodEnd: null },
      Date.now(),
    );
    resolvePeriodBounds(
      { orgId: "org", subscriptionPeriodStart: 2, subscriptionPeriodEnd: 1 },
      Date.now(),
    );
    const calls = emitSpy.mock.calls.map((c: unknown[]) => c[0]);
    expect(calls).not.toContain("plan_counter_invalid_period");
  });
});
