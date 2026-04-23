/**
 * PR-6a D1-D5: stampPeriodClose writer tests.
 *
 * Spec: `docs/plans/pricing-pr6a-overage-foundation.md` §5.
 *
 * Mocks `getSql` so we verify SQL shape + binding values without touching
 * a live Postgres. Pattern mirrors `upsert-plan-counter.test.ts`.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const sqlSpy = vi.fn();
sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);

vi.mock("../lib/db.js", () => ({
  getSql: vi.fn(() => sqlSpy),
}));

import {
  computeStampDisposition,
  stampPeriodClose,
} from "../lib/stamp-period-close.js";

const ORG = "00000000-0000-0000-0000-000000000001";
const PERIOD_START = 1_704_067_200_000;
const PERIOD_END = 1_706_745_600_000;

beforeEach(() => {
  sqlSpy.mockReset();
  sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);
});

/**
 * Queue results for the two SELECT reads the writer performs, in order:
 *   1. subscriptions SELECT (tier, status)
 *   2. org_period_usage SELECT (governed_requests_count, tier_at_period_end, disposition)
 */
function queueReads(
  sub: { tier: string; status: string } | null,
  opu: { governed_requests_count: number; tier_at_period_end: string | null; disposition?: string | null } | null,
): void {
  sqlSpy.mockResolvedValueOnce(sub ? [sub] : []);
  sqlSpy.mockResolvedValueOnce(
    opu
      ? [
          {
            governed_requests_count: opu.governed_requests_count,
            tier_at_period_end: opu.tier_at_period_end,
            disposition: opu.disposition ?? null,
          },
        ]
      : [],
  );
}

describe("computeStampDisposition (pure)", () => {
  it("Pro active + under cap → evaluated_skipped", () => {
    expect(computeStampDisposition("pro", "active", 400_000)).toBe("evaluated_skipped");
  });
  it("Pro active + over cap → billable_pending", () => {
    expect(computeStampDisposition("pro", "active", 600_000)).toBe("billable_pending");
  });
  it("Pro trialing → evaluated_skipped (known non-billable status)", () => {
    expect(computeStampDisposition("pro", "trialing", 10_000_000)).toBe("evaluated_skipped");
  });
  it("Pro past_due → evaluated_skipped", () => {
    expect(computeStampDisposition("pro", "past_due", 10_000_000)).toBe("evaluated_skipped");
  });
  it("Free active → evaluated_skipped (no rate)", () => {
    expect(computeStampDisposition("free", "active", 200_000)).toBe("evaluated_skipped");
  });
  it("Enterprise active → evaluated_skipped (null included)", () => {
    expect(computeStampDisposition("enterprise", "active", 10_000_000)).toBe("evaluated_skipped");
  });
  it("Pro paused → null (fail closed, plan R3 P0) — matches computeOverage unknown_status", () => {
    // paused is NOT in KNOWN_NON_BILLABLE (only trialing/past_due are; canceled
    // moved to BILLABLE_STATUSES per PR-6b CX-R1-4). Fail-closed so ops triages
    // before PR-6b's cron writes invoices on an unknown status.
    expect(computeStampDisposition("pro", "paused", 600_000)).toBeNull();
    expect(computeStampDisposition("pro", "incomplete", 600_000)).toBeNull();
    expect(computeStampDisposition("pro", "unpaid", 600_000)).toBeNull();
    expect(computeStampDisposition("pro", "wat_is_this", 600_000)).toBeNull();
  });
  it("Pro canceled + over cap → billable_pending (PR-6b CX-R1-4 inversion)", () => {
    // Cancel-mid-period customers still owe overage for prior usage.
    // Mirrors dashboard's computeOverage — both layers treat canceled billable.
    // T-OV4 / T-CXR21 iron-rule marker.
    expect(computeStampDisposition("pro", "canceled", 600_000)).toBe("billable_pending");
  });
  it("Pro canceled + under cap → evaluated_skipped (PR-6b CX-R1-4)", () => {
    expect(computeStampDisposition("pro", "canceled", 400_000)).toBe("evaluated_skipped");
  });
  it("Scale canceled + over cap → billable_pending (PR-6b CX-R1-4)", () => {
    expect(computeStampDisposition("scale", "canceled", 2_500_000)).toBe("billable_pending");
  });
  // F2 regression (audit finding #2): unknown tier must fail-closed.
  // Previously returned evaluated_skipped — silent underbill vector on any
  // future tier rollout, typo, or corrupt DB row.
  it("F2: unknown tier active → null (fail closed, NOT evaluated_skipped)", () => {
    expect(computeStampDisposition("platinum", "active", 600_000)).toBeNull();
    expect(computeStampDisposition("legacy_v1", "active", 600_000)).toBeNull();
    expect(computeStampDisposition("", "active", 600_000)).toBeNull();
  });
  it("F2: free + enterprise (known-not-overageable) → evaluated_skipped (distinct from unknown)", () => {
    expect(computeStampDisposition("free", "active", 10_000_000)).toBe("evaluated_skipped");
    expect(computeStampDisposition("enterprise", "active", 10_000_000)).toBe("evaluated_skipped");
  });
  it("Scale active at 2.5M → billable_pending", () => {
    expect(computeStampDisposition("scale", "active", 2_500_000)).toBe("billable_pending");
  });
});

describe("stampPeriodClose — D1-D5 integration paths", () => {
  it("D1: fresh opu + Pro active + under cap → evaluated_skipped; single UPDATE", async () => {
    queueReads(
      { tier: "pro", status: "active" },
      { governed_requests_count: 400_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 }); // UPDATE result

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result).toMatchObject({
      applied: true,
      tier: "pro",
      status: "active",
      disposition: "evaluated_skipped",
      deferred: false,
    });
    // Calls: 1 sub-SELECT + 1 opu-SELECT + 1 UPDATE = 3.
    expect(sqlSpy).toHaveBeenCalledTimes(3);

    const updateSql: string[] = sqlSpy.mock.calls[2][0];
    const updateJoined = updateSql.join(" ");
    expect(updateJoined).toContain("UPDATE org_period_usage");
    expect(updateJoined).toContain("SET tier_at_period_end");
    expect(updateJoined).toContain("disposition");
    expect(updateJoined).toContain("tier_at_period_end IS NULL");
    // Bindings: tier, status, disposition, orgId, periodStart.
    expect(sqlSpy.mock.calls[2][1]).toBe("pro");
    expect(sqlSpy.mock.calls[2][2]).toBe("active");
    expect(sqlSpy.mock.calls[2][3]).toBe("evaluated_skipped");
    expect(sqlSpy.mock.calls[2][4]).toBe(ORG);
    expect(sqlSpy.mock.calls[2][5]).toBe(PERIOD_START);
  });

  it("D2: Pro active + over cap → billable_pending", async () => {
    queueReads(
      { tier: "pro", status: "active" },
      { governed_requests_count: 600_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 });

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result.disposition).toBe("billable_pending");
    expect(sqlSpy.mock.calls[2][3]).toBe("billable_pending");
  });

  it("D3: idempotent — snapshot already present → no UPDATE, applied=false", async () => {
    queueReads(
      { tier: "pro", status: "active" },
      {
        governed_requests_count: 600_000,
        tier_at_period_end: "pro",
        disposition: "evaluated_skipped",
      },
    );
    // NOTE: no UPDATE mock — we assert the writer short-circuits before it.

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result.applied).toBe(false);
    expect(result.tier).toBe("pro");
    // 2 SELECTs, NO UPDATE.
    expect(sqlSpy).toHaveBeenCalledTimes(2);
  });

  it("D4: disposition='billed' (6b cron winner) → re-stamp MUST NOT overwrite", async () => {
    queueReads(
      { tier: "pro", status: "active" },
      {
        governed_requests_count: 600_000,
        tier_at_period_end: "pro",
        disposition: "billed",
      },
    );
    // No UPDATE expected — snapshot present triggers the idempotency short-circuit.

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result.applied).toBe(false);
    expect(result.disposition).toBe("billed"); // retained
    // Only the two SELECTs — no UPDATE issued.
    expect(sqlSpy).toHaveBeenCalledTimes(2);
  });

  it("D5: unknown status → stamps tier+status but NOT disposition; emits stamp_period_close_unknown_status", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    queueReads(
      { tier: "pro", status: "wat_is_this" }, // truly unknown (not in KNOWN_NON_BILLABLE)
      { governed_requests_count: 600_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 }); // UPDATE result (fewer columns set)

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result).toMatchObject({
      applied: true,
      tier: "pro",
      status: "wat_is_this",
      disposition: null, // FAIL-CLOSED: disposition stays NULL for manual triage
      deferred: false,
    });

    // UPDATE should NOT include a disposition binding — only tier_at_period_end
    // + status_at_period_end + last_updated_at.
    const updateSql: string[] = sqlSpy.mock.calls[2][0];
    const updateJoined = updateSql.join(" ");
    expect(updateJoined).toContain("tier_at_period_end = ");
    expect(updateJoined).toContain("status_at_period_end = ");
    expect(updateJoined).not.toContain("disposition = ");
    expect(updateJoined).toContain("tier_at_period_end IS NULL");

    // Observability: the status variant of the unknown metric emitted.
    // F2 discriminates tier-unknown from status-unknown so ops sees distinct alerts.
    const statusLines = logSpy.mock.calls
      .map((c) => String(c[0]))
      .filter((s) => s.includes("stamp_period_close_unknown_status"));
    expect(statusLines).toHaveLength(1);
    const tierLines = logSpy.mock.calls
      .map((c) => String(c[0]))
      .filter((s) => s.includes("stamp_period_close_unknown_tier"));
    expect(tierLines).toHaveLength(0);

    logSpy.mockRestore();
  });

  it("F2: unknown tier → stamps tier+status but NOT disposition; emits stamp_period_close_unknown_tier (audit finding #2)", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    queueReads(
      { tier: "platinum", status: "active" }, // tier not in TIER_CAPS
      { governed_requests_count: 600_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 });

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result).toMatchObject({
      applied: true,
      tier: "platinum",
      status: "active",
      disposition: null, // FAIL-CLOSED — PR-6b cron must NOT bill this row.
      deferred: false,
    });

    // Tier-variant metric fires; status-variant does NOT (active is a known status).
    const tierLines = logSpy.mock.calls
      .map((c) => String(c[0]))
      .filter((s) => s.includes("stamp_period_close_unknown_tier"));
    expect(tierLines).toHaveLength(1);
    const statusLines = logSpy.mock.calls
      .map((c) => String(c[0]))
      .filter((s) => s.includes("stamp_period_close_unknown_status"));
    expect(statusLines).toHaveLength(0);

    logSpy.mockRestore();
  });

  // Audit finding #3: "no sub row" previously returned deferred=true, which
  // made Free orgs permanently un-stamped. Now synthesizes Free snapshot.
  it("F3: Free org (no sub row) → synthesizes tier='free', status='active', disposition='evaluated_skipped' + metric", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    queueReads(
      null, // sub missing → NullSpend's Free-tier convention
      { governed_requests_count: 50_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 }); // UPDATE result

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result).toMatchObject({
      applied: true,
      tier: "free",
      status: "active",
      disposition: "evaluated_skipped",
      deferred: false, // Free is terminal, not transient.
    });

    // UPDATE bindings: tier="free", status="active", disposition="evaluated_skipped".
    expect(sqlSpy.mock.calls[2][1]).toBe("free");
    expect(sqlSpy.mock.calls[2][2]).toBe("active");
    expect(sqlSpy.mock.calls[2][3]).toBe("evaluated_skipped");

    // Observability: the Free synthesis metric fires (distinct from bridge
    // no-sub case so ops can see Free volume vs. genuine anomalies).
    const synthLines = logSpy.mock.calls
      .map((c) => String(c[0]))
      .filter((s) => s.includes("stamp_period_close_free_synthesized"));
    expect(synthLines).toHaveLength(1);

    logSpy.mockRestore();
  });

  it("F3: Free org over 100k cap (no sub row) → still evaluated_skipped (Free has no overage billing)", async () => {
    queueReads(
      null,
      { governed_requests_count: 150_000, tier_at_period_end: null },
    );
    sqlSpy.mockResolvedValueOnce({ count: 1 });

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    // Even though Free's cap is 100k and usage is 150k, there's no overage
    // billing for Free tier — enforcement already blocked these. Disposition
    // is evaluated_skipped, NOT billable_pending.
    expect(result.disposition).toBe("evaluated_skipped");
    expect(sqlSpy.mock.calls[2][3]).toBe("evaluated_skipped");
  });

  it("no-op: opu row missing → applied=false, deferred=false (nothing to stamp)", async () => {
    queueReads({ tier: "pro", status: "active" }, null);

    const result = await stampPeriodClose("postgres://fake", {
      orgId: ORG,
      periodStart: PERIOD_START,
      periodEnd: PERIOD_END,
    });

    expect(result).toMatchObject({
      applied: false,
      tier: null,
      status: null,
      disposition: null,
      deferred: false,
    });
    expect(sqlSpy).toHaveBeenCalledTimes(2);
  });
});
