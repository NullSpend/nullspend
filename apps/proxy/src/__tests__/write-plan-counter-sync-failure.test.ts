/**
 * PR-2e /review R3 (testing specialist CRITICAL 8/10): `writePlanCounterSyncFailure`
 * had ZERO unit tests prior to this commit. The row it writes is the ONLY
 * input to the launch-watcher's `divergence_outbox_stalled` alert. A schema
 * drift (column rename, forgotten UUID cast, reason-string typo) silently
 * breaks the alert — no false-green is visible on the dashboard side.
 *
 * Keep this suite tight: it's a 2-line SQL statement. What we MUST pin:
 * (1) the reason string round-trips verbatim (per the open-set reason
 * taxonomy documented in the helper's docstring), (2) the ::uuid cast is
 * preserved on org_id, (3) the count bind is in the expected position.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const sqlSpy = vi.fn();

vi.mock("../lib/db.js", () => ({
  getSql: vi.fn(() => sqlSpy),
}));

import { writePlanCounterSyncFailure } from "../lib/write-plan-counter-sync-failure.js";

beforeEach(() => {
  sqlSpy.mockReset();
  sqlSpy.mockResolvedValue(undefined);
});

describe("writePlanCounterSyncFailure", () => {
  it("targets the plan_counter_sync_failures table", async () => {
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "divergence_outbox_abandoned",
      count: 3,
    });

    expect(sqlSpy).toHaveBeenCalledTimes(1);
    const tpl: string[] = sqlSpy.mock.calls[0][0];
    const joined = tpl.join(" ").toLowerCase();
    expect(joined).toContain("insert into plan_counter_sync_failures");
  });

  it("preserves the ::uuid cast on org_id (dashboard uuid column contract)", async () => {
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "anything",
      count: 1,
    });

    const tpl: string[] = sqlSpy.mock.calls[0][0];
    expect(tpl.join(" ")).toContain("::uuid");
  });

  it("binds (orgId, reason, count) in that order", async () => {
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "batch_sync_failed",
      count: 17,
    });

    const call = sqlSpy.mock.calls[0];
    expect(call[1]).toBe("00000000-0000-0000-0000-00000000000a");
    expect(call[2]).toBe("batch_sync_failed");
    expect(call[3]).toBe(17);
  });

  it("divergence_outbox_abandoned round-trips verbatim (alert-contract string regression guard)", async () => {
    // This exact string is the launch-watcher's `divergence_outbox_stalled`
    // alert-signal input — `/api/internal/metrics-summary` filters by
    // `reason = 'divergence_outbox_abandoned'`. Any drift in spelling kills
    // the alert silently. Pin the exact string as a regression guard.
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "divergence_outbox_abandoned",
      count: 1,
    });

    expect(sqlSpy.mock.calls[0][2]).toBe("divergence_outbox_abandoned");
  });

  it("batch_sync_failed round-trips verbatim (F1 dashboard fail-open input)", async () => {
    // Same rigor for the F1 dashboard fail-open path. `batch_sync_failed`
    // feeds the watcher's `dashboard_sync_failing` alert.
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "batch_sync_failed",
      count: 1,
    });

    expect(sqlSpy.mock.calls[0][2]).toBe("batch_sync_failed");
  });

  it("accepts arbitrary open-set reason strings (no server-side enum check)", async () => {
    // Helper docstring explicitly keeps reason as OPEN-SET. If someone
    // later adds an enum/switch inside the helper, this test fails loudly.
    await writePlanCounterSyncFailure("postgres://fake", {
      orgId: "00000000-0000-0000-0000-00000000000a",
      reason: "future_reason_not_yet_defined",
      count: 1,
    });

    expect(sqlSpy.mock.calls[0][2]).toBe("future_reason_not_yet_defined");
  });

  it("does not swallow errors — PG error propagates to caller for try/catch", async () => {
    // Best-effort-write semantics per the helper's docstring: callers MUST
    // wrap in try/catch. Verify the helper itself doesn't silently swallow
    // failures (which would mask an outbox stall from the DO alarm handler).
    sqlSpy.mockRejectedValueOnce(new Error("HYPERDRIVE unavailable"));

    await expect(
      writePlanCounterSyncFailure("postgres://fake", {
        orgId: "00000000-0000-0000-0000-00000000000a",
        reason: "divergence_outbox_abandoned",
        count: 1,
      }),
    ).rejects.toThrow(/HYPERDRIVE unavailable/);
  });
});
