/**
 * PR-2e /review R3 (testing specialist CRITICAL 9/10): `upsertPlanCounterDivergence`
 * had ZERO unit tests prior to this commit. It carries two launch-gating
 * invariants that must survive refactors:
 *
 * 1. Dedup via `plan_counter_divergence_dedup` (codex R7 Issue 1A) so
 *    retry-after-partial-commit doesn't insert a second divergence row.
 *    A regression dropping `ON CONFLICT DO NOTHING` inflates the watcher's
 *    15m COUNT(*) silently.
 * 2. Explicit `divergence_at` via `to_timestamp(ms/1000)` (codex R7 C1).
 *    Falling back to PG's `DEFAULT now()` breaks event-time semantics --
 *    events land in the wrong 15m window under HYPERDRIVE lag or retry
 *    backoff.
 *
 * Mirrors the mocking pattern of upsert-plan-counter.test.ts so refactors
 * that unify the two helpers don't silently lose coverage on either side.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const sqlSpy = vi.fn() as vi.Mock & {
  begin?: (fn: (tx: typeof sqlSpy) => Promise<unknown>) => Promise<unknown>;
};
sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);

vi.mock("../lib/db.js", () => ({
  getSql: vi.fn(() => sqlSpy),
}));

import { upsertPlanCounterDivergence } from "../lib/upsert-plan-counter-divergence.js";

function dedupInserted() {
  sqlSpy.mockResolvedValueOnce({ count: 1 });
}

function dedupHit() {
  sqlSpy.mockResolvedValueOnce({ count: 0 });
}

const BASE_ENTRY = {
  eventId: "11111111-2222-3333-4444-555555555555",
  orgId: "00000000-0000-0000-0000-00000000000a",
  tier: "free",
  divergenceAtMs: 1_704_067_200_000, // April 1, 2026
  storedPeriodStart: 1_704_067_200_000,
  incomingPeriodStart: 1_706_745_600_000,
};

describe("upsertPlanCounterDivergence", () => {
  beforeEach(() => {
    sqlSpy.mockReset();
    sqlSpy.begin = async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy);
  });

  it("emits composite dedup INSERT then divergence INSERT on first-time event", async () => {
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 }); // divergence INSERT returns row count

    const result = await upsertPlanCounterDivergence("postgres://fake", BASE_ENTRY);

    expect(result.applied).toBe(true);
    expect(sqlSpy).toHaveBeenCalledTimes(2);

    // Call 1 — dedup INSERT.
    const dedupSql: string[] = sqlSpy.mock.calls[0][0];
    const dedupJoined = dedupSql.join(" ");
    expect(dedupJoined).toContain("INSERT INTO plan_counter_divergence_dedup");
    expect(dedupJoined).toContain("ON CONFLICT (org_id, event_id) DO NOTHING");
    expect(sqlSpy.mock.calls[0][1]).toBe(BASE_ENTRY.orgId);
    expect(sqlSpy.mock.calls[0][2]).toBe(BASE_ENTRY.eventId);

    // Call 2 — divergence INSERT with explicit timestamps (not DEFAULT now()).
    const divergenceSql: string[] = sqlSpy.mock.calls[1][0];
    const divergenceJoined = divergenceSql.join(" ");
    expect(divergenceJoined).toContain("INSERT INTO plan_counter_divergences");
    expect(divergenceJoined).toContain("to_timestamp(");
    expect(divergenceJoined).toContain("/ 1000.0");
    // Bound values: orgId, divergence_at_ms, tier, storedPeriodStart, incomingPeriodStart.
    expect(sqlSpy.mock.calls[1][1]).toBe(BASE_ENTRY.orgId);
    expect(sqlSpy.mock.calls[1][2]).toBe(BASE_ENTRY.divergenceAtMs);
    expect(sqlSpy.mock.calls[1][3]).toBe(BASE_ENTRY.tier);
    expect(sqlSpy.mock.calls[1][4]).toBe(BASE_ENTRY.storedPeriodStart);
    expect(sqlSpy.mock.calls[1][5]).toBe(BASE_ENTRY.incomingPeriodStart);
  });

  it("dedup hit does NOT run the divergence INSERT — applied=false", async () => {
    dedupHit();

    const result = await upsertPlanCounterDivergence("postgres://fake", BASE_ENTRY);

    expect(result.applied).toBe(false);
    // Only the dedup INSERT ran — no divergence insert. Guards against the
    // regression that would double-count the watcher's 15m aggregate.
    expect(sqlSpy).toHaveBeenCalledTimes(1);
    const dedupSql: string[] = sqlSpy.mock.calls[0][0];
    expect(dedupSql.join(" ")).toContain("INSERT INTO plan_counter_divergence_dedup");
  });

  it("retry-idempotency: double-call with same eventId only applies once", async () => {
    // First call — dedup insert lands, divergence row written.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const r1 = await upsertPlanCounterDivergence("postgres://fake", BASE_ENTRY);
    expect(r1.applied).toBe(true);
    expect(sqlSpy).toHaveBeenCalledTimes(2);

    // Second call — dedup hit, divergence insert skipped.
    dedupHit();
    const r2 = await upsertPlanCounterDivergence("postgres://fake", BASE_ENTRY);
    expect(r2.applied).toBe(false);
    // Total = 2 (first pair) + 1 (second dedup only) = 3 (NOT 4). Regression
    // that lost the dedup would make this 4.
    expect(sqlSpy).toHaveBeenCalledTimes(3);
  });

  it("divergence_at_ms is passed verbatim as the PG bound value (not overwritten by now())", async () => {
    // Codex R7 C1 regression guard: a subtle bug would be adding a
    // NOW()-based default OR dropping the bound and letting PG's DEFAULT
    // now() take over. Either would land events in the wrong 15m window
    // during retry / HYPERDRIVE lag.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });

    const specificMs = 1_700_000_000_123; // arbitrary specific ms, not "now"
    await upsertPlanCounterDivergence("postgres://fake", {
      ...BASE_ENTRY,
      divergenceAtMs: specificMs,
    });

    // The divergence_at bind MUST be the exact ms we passed, not Date.now().
    const divergenceCall = sqlSpy.mock.calls[1];
    expect(divergenceCall).toBeDefined();
    // Second positional arg = divergence_at_ms bind.
    expect(divergenceCall[2]).toBe(specificMs);
  });

  it("EC-cross-org: same eventId across DIFFERENT orgs → both upserts apply", async () => {
    // Sanity: dedup key is (org_id, event_id). UUID collision is astronomical,
    // but the invariant "different orgs, different dedup namespaces" must hold.
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rA = await upsertPlanCounterDivergence("postgres://fake", {
      ...BASE_ENTRY,
      orgId: "00000000-0000-0000-0000-00000000000a",
    });
    expect(rA.applied).toBe(true);

    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });
    const rB = await upsertPlanCounterDivergence("postgres://fake", {
      ...BASE_ENTRY,
      orgId: "00000000-0000-0000-0000-00000000000b",
    });
    expect(rB.applied).toBe(true);
  });

  it("transactional: runs inside sql.begin() so dedup + divergence are atomic", async () => {
    // The implementation wraps both INSERTs in sql.begin(). Mock `begin` to
    // track that it's called, and that the SUT's work happens inside the
    // begin callback (not outside).
    const beginSpy = vi.fn(
      async (fn: (tx: typeof sqlSpy) => Promise<unknown>) => fn(sqlSpy),
    );
    sqlSpy.begin = beginSpy;
    dedupInserted();
    sqlSpy.mockResolvedValueOnce({ count: 1 });

    await upsertPlanCounterDivergence("postgres://fake", BASE_ENTRY);

    expect(beginSpy).toHaveBeenCalledTimes(1);
  });
});
