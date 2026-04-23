/**
 * PR-2d tests for the plan-counter drift reconciliation cron.
 *
 * C42  per-event idempotency across runs (DO dedup via reconcile- keys).
 * C43  crash between DO call and tag clear — re-run is drift-free.
 * C44  tag clear is `id`-scoped (not request_id), atomic vs. concurrent inserts.
 * C44b `LIMIT 500` cap — batched runs drain the queue.
 * C56  metric emission on both success and failure paths.
 * C59  period override — DO increment uses STORED periods, not Date.now().
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const mockSql = Object.assign(vi.fn().mockResolvedValue([]), {});
const mockDoIncrementPlanCounter = vi.fn();
const mockEmitMetric = vi.fn();

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doIncrementPlanCounter: (...args: unknown[]) => mockDoIncrementPlanCounter(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import {
  handleReconcilePlanCounterCron,
  RECONCILE_BATCH_LIMIT,
} from "../routes/reconcile-plan-counter-cron.js";

const ORG_A = "11111111-2222-3333-4444-555555555555";
const ORG_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

/** Build the shape the SELECT returns. Ms-epoch for periods. */
function makeRow(overrides: Record<string, unknown> = {}) {
  return {
    event_id: "evt-001",
    request_id: "req-001",      // build-audit F1: cron now hashes this with provider
    provider: "openai",          // —
    org_id: ORG_A,
    period_start: 1743465600000, // 2026-04-01T00:00:00Z
    period_end: 1746057600000,   // 2026-05-01T00:00:00Z
    tier_label: "free",
    plan_limit_block_at: 100_000,
    plan_limit_mode: "hard",
    ...overrides,
  };
}

/**
 * Helper to compute the live-path SHA-256 idempotency key for assertions.
 * Mirrors `apps/proxy/src/lib/idempotency-key.ts::sha256IdempotencyKey` and
 * `lib/cost-events/idempotency-key.ts` on the dashboard side. If those drift
 * (build-audit F1 invariant), F5 below catches it.
 */
async function expectedKey(requestId: string, provider: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(JSON.stringify([requestId, provider])),
  );
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
}

// Captures CACHE_KV.put calls so Sub-PR 16 tests can assert heartbeat behavior.
const mockKvPut = vi.fn().mockResolvedValue(undefined);

function makeEnv(): Env {
  return {
    HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
    CACHE_KV: {
      put: mockKvPut,
      get: vi.fn(),
      delete: vi.fn(),
      list: vi.fn(),
      getWithMetadata: vi.fn(),
    } as unknown as KVNamespace,
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
  } as unknown as Env;
}

/**
 * Install a default SELECT response for one run. After `rows` returns, every
 * subsequent `mockSql` call (the per-row UPDATE) resolves to `[]`.
 */
function stubSelect(rows: Array<ReturnType<typeof makeRow>>) {
  mockSql.mockReset();
  mockSql.mockResolvedValueOnce(rows);
  mockSql.mockResolvedValue([]);
}

beforeEach(() => {
  vi.clearAllMocks();
  mockSql.mockResolvedValue([]);
  mockDoIncrementPlanCounter.mockResolvedValue({ status: "approved", count: 1 });
  // Restore the default KV success after clearAllMocks() (Sub-PR 16).
  mockKvPut.mockResolvedValue(undefined);
});

// ---------------------------------------------------------------------------
// C42 — per-event idempotency across runs
// ---------------------------------------------------------------------------

describe("C42: per-event idempotency (DO dedup across runs)", () => {
  it("replaying same tagged events → DO is called with same reconcile- key; counter doesn't double", async () => {
    // Simulate tag NOT being cleared between runs (e.g. UPDATE failure). Both
    // runs see the same event_id; DO dedup should suppress the second increment.
    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    const row = makeRow({ event_id: "evt-C42", request_id: "req-C42", provider: "openai" });
    const expected = await expectedKey("req-C42", "openai");

    stubSelect([row]);
    await handleReconcilePlanCounterCron(makeEnv());

    stubSelect([row]);
    await handleReconcilePlanCounterCron(makeEnv());

    // DO called twice, both with the SAME sha256(request_id, provider) key.
    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(2);
    const [, firstOpts] = mockDoIncrementPlanCounter.mock.calls[0];
    const [, secondOpts] = mockDoIncrementPlanCounter.mock.calls[1];
    expect((firstOpts as { idempotencyKey: string }).idempotencyKey).toBe(expected);
    expect((secondOpts as { idempotencyKey: string }).idempotencyKey).toBe(expected);

    // Dedup: counter stayed at 1.
    expect(currentCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// F5 (build audit) — partial-success regression: cron uses the SAME hash as
// the live path, so a DO already incremented by the dashboard's sync call
// stays at the same count when the cron picks the event up.
// ---------------------------------------------------------------------------

describe("F5: cron + live-path share idempotency-key namespace (build audit F1 regression)", () => {
  it("DO already incremented by live path → cron submits SAME key → count unchanged", async () => {
    // Stateful mock keyed by idempotencyKey — simulates DO dedup behavior.
    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    // Step 1: simulate the LIVE-PATH sync call (proxy completed the increment
    // BEFORE its HTTP response was lost in flight). Hash matches what the
    // dashboard's `/api/cost-events` route computes via `sha256IdempotencyKey`.
    const requestId = "req-partial-success";
    const provider = "openai";
    const liveKey = await expectedKey(requestId, provider);
    await mockDoIncrementPlanCounter({} as never, {
      orgId: ORG_A,
      planLimitBlockAt: 100_000,
      planLimitMode: "hard",
      tier: "free",
      periodStart: 1743465600000,
      periodEnd: 1746057600000,
      idempotencyKey: liveKey,
    });
    expect(currentCount).toBe(1);

    // Step 2: dashboard saw timeout/network → fail-opened the skip tag →
    // cron picks the same event up. Cron MUST hash with the SAME inputs so
    // the DO recognizes the replay.
    stubSelect([
      makeRow({
        event_id: "evt-partial",
        request_id: requestId,
        provider,
      }),
    ]);
    await handleReconcilePlanCounterCron(makeEnv());

    // Cron call should reuse the live-path key.
    const cronCall = mockDoIncrementPlanCounter.mock.calls[1];
    const cronKey = (cronCall[1] as { idempotencyKey: string }).idempotencyKey;
    expect(cronKey).toBe(liveKey);

    // Counter MUST stay at 1 — dedup catches the cron replay.
    expect(currentCount).toBe(1);

    // The cron treated this as a successful reconciliation (DO returned ok).
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_reconciled",
      expect.objectContaining({ orgId: ORG_A, eventId: "evt-partial" }),
    );
  });

  it("cron computes a DIFFERENT key for a DIFFERENT (request_id, provider) tuple", async () => {
    // Sanity check that the F1 invariant doesn't accidentally collapse all
    // cron events to a single hash (which would silently undercount).
    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    stubSelect([
      makeRow({ event_id: "e1", request_id: "r1", provider: "openai" }),
      makeRow({ event_id: "e2", request_id: "r2", provider: "openai" }),
      makeRow({ event_id: "e3", request_id: "r1", provider: "anthropic" }),  // same req, diff provider
    ]);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(3);
    const keys = mockDoIncrementPlanCounter.mock.calls.map((c) =>
      (c[1] as { idempotencyKey: string }).idempotencyKey,
    );
    // Three distinct tuples → three distinct hashes.
    expect(new Set(keys).size).toBe(3);
    // Counter advanced by 3.
    expect(currentCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// C43 — crash between DO call and tag clear
// ---------------------------------------------------------------------------

describe("C43: crash between DO call and tag clear", () => {
  it("UPDATE throws after DO succeeds → tag_clear_failed metric; next run retries DO safely", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    // Pass 1: SELECT returns row, UPDATE throws.
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([makeRow({ event_id: "evt-C43" })]); // SELECT
    mockSql.mockRejectedValueOnce(new Error("postgres hiccup"));        // UPDATE

    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    await handleReconcilePlanCounterCron(makeEnv());

    // DO fired once, tag clear failed.
    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(1);
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_tag_clear_failed",
      expect.objectContaining({ orgId: ORG_A, eventId: "evt-C43" }),
    );
    // No _reconciled metric (we didn't fully succeed).
    const reconciledCalls = mockEmitMetric.mock.calls.filter((c) => c[0] === "plan_counter_drift_reconciled");
    expect(reconciledCalls).toHaveLength(0);

    // Pass 2: SELECT returns the SAME event (tag wasn't cleared), UPDATE now succeeds.
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([makeRow({ event_id: "evt-C43" })]); // SELECT
    mockSql.mockResolvedValue([]);                                      // UPDATE

    await handleReconcilePlanCounterCron(makeEnv());

    // DO called again with SAME reconcile- key — dedup returns same count, no drift.
    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(2);
    expect(currentCount).toBe(1);
  });

  // PR-2e audit Finding 1: tag-clear failures must ALSO persist a row in
  // `plan_counter_sync_failures{reason='tag_clear_failed'}` so the
  // launch-watcher's `tag_clear_failing` alert can fire. Without this, the
  // condition is invisible until the DO's 24h idempotency prune opens the
  // double-count window.
  it("F1: tag-clear UPDATE failure inserts plan_counter_sync_failures{reason:'tag_clear_failed', count:1}", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([makeRow({ event_id: "evt-F1", org_id: ORG_B })]); // SELECT
    mockSql.mockRejectedValueOnce(new Error("UPDATE failed")); // UPDATE (tag clear)
    // Next SQL call is the writePlanCounterSyncFailure INSERT — let it succeed.
    mockSql.mockResolvedValueOnce([]);
    // Remaining calls (cron_history INSERT, 2 prune DELETEs) default to [].
    mockSql.mockResolvedValue([]);

    await handleReconcilePlanCounterCron(makeEnv());

    // Find the sync_failures INSERT among the mockSql calls.
    const insertCall = mockSql.mock.calls.find((c) => {
      const tpl = c[0] as ReadonlyArray<string> | undefined;
      return tpl?.join(" ").toLowerCase().includes("insert into plan_counter_sync_failures");
    });
    expect(insertCall).toBeDefined();
    // Bind order per writePlanCounterSyncFailure: (orgId, reason, count).
    expect(insertCall![1]).toBe(ORG_B);
    expect(insertCall![2]).toBe("tag_clear_failed");
    expect(insertCall![3]).toBe(1);
  });

  it("F1: if the sync_failures INSERT itself fails, cron emits secondary metric + continues", async () => {
    // Defense-in-depth: if PG is broken and the UPDATE AND the fallback
    // INSERT both fail, the cron must still continue cleanly. Emits a
    // dedicated `plan_counter_tag_clear_write_failed` metric so ops can
    // triage the secondary failure separately from the primary UPDATE
    // failure (both might be symptoms of the same PG outage).
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([makeRow({ event_id: "evt-F1b" })]); // SELECT
    mockSql.mockRejectedValueOnce(new Error("UPDATE failed")); // UPDATE (tag clear)
    mockSql.mockRejectedValueOnce(new Error("PG pool exhausted")); // sync_failures INSERT also fails
    mockSql.mockResolvedValue([]); // cron_history + prunes default to []

    await expect(
      handleReconcilePlanCounterCron(makeEnv()),
    ).resolves.toBeUndefined();

    // Primary tag-clear-failed metric fires.
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_tag_clear_failed",
      expect.objectContaining({ orgId: ORG_A, eventId: "evt-F1b" }),
    );
    // Secondary metric fires too so the PG-write failure is ALSO visible.
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_tag_clear_write_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
  });
});

// ---------------------------------------------------------------------------
// C44 — tag clear is id-scoped (not request_id)
// ---------------------------------------------------------------------------

describe("C44: tag clear is id-scoped (atomic vs. concurrent inserts)", () => {
  it("UPDATE uses cost_events.id in WHERE clause (not request_id)", async () => {
    stubSelect([makeRow({ event_id: "evt-C44" })]);
    await handleReconcilePlanCounterCron(makeEnv());

    // Collect the template strings passed to every mockSql invocation. The
    // second call (after the SELECT) is the UPDATE. Inspect it to prove the
    // scope column is `id`, not `request_id`.
    const updateCall = mockSql.mock.calls[1];
    const templateStrings = updateCall[0] as ReadonlyArray<string>;
    const combined = templateStrings.join(" ").toLowerCase();
    expect(combined).toContain("update cost_events");
    expect(combined).toContain("tags - '_ns_counter_sync_skipped'");
    expect(combined).toContain("where id =");
    expect(combined).not.toContain("where request_id");

    // The interpolated value is the event_id UUID (not request_id).
    const interpolated = updateCall.slice(1);
    expect(interpolated).toContain("evt-C44");
  });
});

// ---------------------------------------------------------------------------
// C44b — LIMIT 500 cap
// ---------------------------------------------------------------------------

describe("C44b: LIMIT 500 cap", () => {
  it("SELECT query includes the batch limit as a bound parameter", async () => {
    stubSelect([]);
    await handleReconcilePlanCounterCron(makeEnv());

    // First mockSql call is the SELECT. Verify the LIMIT value is interpolated.
    const selectCall = mockSql.mock.calls[0];
    const templateStrings = selectCall[0] as ReadonlyArray<string>;
    const combined = templateStrings.join(" ").toLowerCase();
    expect(combined).toContain("limit ");
    // The last interpolated value is the LIMIT bound — must match the exported constant.
    expect(selectCall[selectCall.length - 1]).toBe(RECONCILE_BATCH_LIMIT);
    expect(RECONCILE_BATCH_LIMIT).toBe(500);
  });

  it("a 500-row SELECT result emits summary metric with batchFull: true", async () => {
    const rows = Array.from({ length: 500 }, (_, i) => makeRow({ event_id: `evt-${i}` }));
    stubSelect(rows);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_cron_summary",
      expect.objectContaining({
        candidateCount: 500,
        reconciled: 500,
        batchFull: true,
      }),
    );
  });

  it("a 100-row SELECT result emits summary metric with batchFull: false", async () => {
    const rows = Array.from({ length: 100 }, (_, i) => makeRow({ event_id: `evt-${i}` }));
    stubSelect(rows);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_cron_summary",
      expect.objectContaining({
        candidateCount: 100,
        reconciled: 100,
        batchFull: false,
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// C56 — metric emission (success + failure)
// ---------------------------------------------------------------------------

describe("C56: metric emission", () => {
  it("emits plan_counter_drift_reconciled {orgId, eventId} for each successful row", async () => {
    stubSelect([
      makeRow({ event_id: "evt-1", org_id: ORG_A }),
      makeRow({ event_id: "evt-2", org_id: ORG_B }),
    ]);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_reconciled",
      { orgId: ORG_A, eventId: "evt-1" },
    );
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_reconciled",
      { orgId: ORG_B, eventId: "evt-2" },
    );
  });

  it("emits plan_counter_drift_reconcile_failed with {reason, orgId, eventId} when DO throws", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockDoIncrementPlanCounter.mockRejectedValue(new Error("DO unavailable"));

    stubSelect([makeRow({ event_id: "evt-fail" })]);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_reconcile_failed",
      expect.objectContaining({ reason: "do_error", orgId: ORG_A, eventId: "evt-fail" }),
    );
    // Corresponding _reconciled metric should NOT fire on failure.
    const reconciledCalls = mockEmitMetric.mock.calls.filter(
      (c) => c[0] === "plan_counter_drift_reconciled" && (c[1] as { eventId: string }).eventId === "evt-fail",
    );
    expect(reconciledCalls).toHaveLength(0);
  });

  it("emits plan_counter_drift_cron_run on each invocation with candidateCount", async () => {
    stubSelect([makeRow({ event_id: "a" }), makeRow({ event_id: "b" })]);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_cron_run",
      { candidateCount: 2 },
    );
  });

  it("emits plan_counter_drift_query_failed when the SELECT throws", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockRejectedValueOnce(new Error("connection refused"));

    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_drift_query_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
    // DO not called when SELECT fails.
    expect(mockDoIncrementPlanCounter).not.toHaveBeenCalled();
  });

  it("does not throw when SELECT throws (cron tick must ack cleanly)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockRejectedValueOnce(new Error("DB down"));

    await expect(handleReconcilePlanCounterCron(makeEnv())).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// C59 — period override: DO increment uses STORED periods, not Date.now()
// ---------------------------------------------------------------------------

describe("C59: period override (codex P0 #1)", () => {
  it("March event reconciled in April targets the MARCH period row", async () => {
    // Stored periods: March 2026.
    const MARCH_START = Date.UTC(2026, 2, 1);  // 2026-03-01
    const MARCH_END = Date.UTC(2026, 3, 1);    // 2026-04-01

    stubSelect([
      makeRow({
        event_id: "evt-march",
        period_start: MARCH_START,
        period_end: MARCH_END,
      }),
    ]);

    // Freeze now() to April 15 so a naive `Date.now()`-based implementation would
    // resolve to the APRIL period — this test catches that regression.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-15T12:00:00Z"));
    try {
      await handleReconcilePlanCounterCron(makeEnv());
    } finally {
      vi.useRealTimers();
    }

    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(1);
    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    // The DO call MUST carry the March-era bounds, not April-derived ones.
    expect((opts as { periodStart: number }).periodStart).toBe(MARCH_START);
    expect((opts as { periodEnd: number }).periodEnd).toBe(MARCH_END);
  });

  it("coerces BIGINT-as-string period values (postgres.js with fetch_types:false)", async () => {
    // BIGINT columns under fetch_types:false come back as string. toFiniteNumber
    // must coerce before handing off to the DO.
    stubSelect([
      makeRow({
        event_id: "evt-bigint",
        period_start: "1743465600000", // string form of April 1
        period_end: "1746057600000",   // string form of May 1
      }),
    ]);

    await handleReconcilePlanCounterCron(makeEnv());

    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    expect((opts as { periodStart: number }).periodStart).toBe(1743465600000);
    expect((opts as { periodEnd: number }).periodEnd).toBe(1746057600000);
  });
});

// ---------------------------------------------------------------------------
// DO argument shape — cross-cutting sanity
// ---------------------------------------------------------------------------

describe("DO call carries hydrated identity fields", () => {
  it("passes orgId, tier, planLimitBlockAt, planLimitMode from SQL row", async () => {
    stubSelect([
      makeRow({
        event_id: "evt-ids",
        org_id: ORG_B,
        tier_label: "pro",
        plan_limit_block_at: 500_000,
        plan_limit_mode: "soft",
      }),
    ]);

    await handleReconcilePlanCounterCron(makeEnv());

    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    expect(opts).toMatchObject({
      orgId: ORG_B,
      tier: "pro",
      planLimitBlockAt: 500_000,
      planLimitMode: "soft",
    });
    // Build-audit F1: idempotencyKey is now the live-path sha256 hash, not
    // `reconcile-${event_id}`. Verify it matches the helper output exactly.
    const expected = await expectedKey("req-001", "openai");
    expect((opts as { idempotencyKey: string }).idempotencyKey).toBe(expected);
  });

  it("defaults unknown tier_label to 'free' and unknown plan_limit_mode to 'soft'", async () => {
    stubSelect([
      makeRow({
        tier_label: "something-wrong",
        plan_limit_mode: "bogus",
      }),
    ]);

    await handleReconcilePlanCounterCron(makeEnv());

    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    expect((opts as { tier: string }).tier).toBe("free");
    // Unknown plan_limit_mode → "soft" per normalizePlanLimitMode idiom.
    expect((opts as { planLimitMode: string }).planLimitMode).toBe("soft");
  });
});

// ---------------------------------------------------------------------------
// Sub-PR 16 — KV heartbeat write (codex R5 #1)
// ---------------------------------------------------------------------------

describe("Sub-PR 16: KV heartbeat write (must NOT take down the cron)", () => {
  it("writes plan_counter_cron_last_tick_at to CACHE_KV with current timestamp", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-20T12:34:56.789Z"));
    try {
      stubSelect([]);
      await handleReconcilePlanCounterCron(makeEnv());
    } finally {
      vi.useRealTimers();
    }

    expect(mockKvPut).toHaveBeenCalledTimes(1);
    expect(mockKvPut).toHaveBeenCalledWith(
      "plan_counter_cron_last_tick_at",
      // Date.now() at the frozen time → stringified.
      new Date("2026-04-20T12:34:56.789Z").getTime().toString(),
    );
  });

  it("calls KV.put BEFORE the SELECT (so PG outages don't silence the heartbeat)", async () => {
    // Track call order across both mocks via a shared counter.
    const order: string[] = [];
    mockKvPut.mockImplementation(async () => {
      order.push("kv");
    });
    mockSql.mockReset();
    mockSql.mockImplementationOnce(async () => {
      order.push("sql");
      return [];
    });

    await handleReconcilePlanCounterCron(makeEnv());

    expect(order).toEqual(["kv", "sql"]);
  });

  it("writes the heartbeat EVEN WHEN the SELECT throws (cron handler ran proof)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockRejectedValueOnce(new Error("connection refused"));

    await handleReconcilePlanCounterCron(makeEnv());

    // Heartbeat fired despite the PG failure. This is the whole point of
    // putting it before the SELECT — the watcher needs to know the cron ran,
    // even if the work failed.
    expect(mockKvPut).toHaveBeenCalledTimes(1);
    expect(mockKvPut).toHaveBeenCalledWith(
      "plan_counter_cron_last_tick_at",
      expect.any(String),
    );
  });

  it("KV.put failure does NOT propagate to the cron (codex R5 #1 anti-pattern guard)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockKvPut.mockRejectedValueOnce(new Error("KV unavailable"));
    stubSelect([]);

    // Must resolve cleanly, NOT throw. A KV outage taking down the cron is
    // exactly the bug codex R5 #1 caught — monitoring code must not break
    // the path it's monitoring.
    await expect(
      handleReconcilePlanCounterCron(makeEnv()),
    ).resolves.toBeUndefined();

    // The reconciliation work continued despite the KV failure.
    expect(mockSql).toHaveBeenCalled(); // SELECT still ran
  });

  it("KV.put failure emits plan_counter_heartbeat_write_failed metric", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockKvPut.mockRejectedValueOnce(new Error("KV unavailable"));
    stubSelect([]);

    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_heartbeat_write_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
  });

  it("emits the failure metric even when KV.put throws synchronously (defense-in-depth)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockKvPut.mockImplementationOnce(() => {
      throw new Error("sync throw from KV stub");
    });
    stubSelect([]);

    await expect(
      handleReconcilePlanCounterCron(makeEnv()),
    ).resolves.toBeUndefined();

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_heartbeat_write_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
  });
});

// ---------------------------------------------------------------------------
// Sub-PR 16 — plan_counter_cron_history Postgres insert
// ---------------------------------------------------------------------------

describe("Sub-PR 16: plan_counter_cron_history Postgres insert", () => {
  it("inserts a row with batch_full=false + correct counts at end of run", async () => {
    stubSelect([
      makeRow({ event_id: "e1" }),
      makeRow({ event_id: "e2" }),
      makeRow({ event_id: "e3" }),
    ]);

    await handleReconcilePlanCounterCron(makeEnv());

    // mockSql calls: [SELECT, UPDATE_e1, UPDATE_e2, UPDATE_e3, INSERT,
    //                 DELETE_sync, DELETE_divergence]. Find the INSERT by
    //                 content search — its position is no longer at the end
    //                 since the PR-2e follow-up retry-state prune runs after.
    const insertCall = mockSql.mock.calls.find((c) => {
      const tpl = c[0] as ReadonlyArray<string> | undefined;
      return tpl?.join(" ").toLowerCase().includes("insert into plan_counter_cron_history");
    });
    expect(insertCall).toBeDefined();

    // Interpolated values: batch_full, candidate_count, reconciled, failed.
    // 3 rows total, all approved → [false, 3, 3, 0].
    const interpolated = insertCall!.slice(1);
    expect(interpolated).toContain(false); // batch_full
    expect(interpolated).toContain(3); // candidate_count
    expect(interpolated).toContain(3); // reconciled
    expect(interpolated).toContain(0); // failed
  });

  it("inserts batch_full=true when the LIMIT cap is hit", async () => {
    const rows = Array.from({ length: RECONCILE_BATCH_LIMIT }, (_, i) =>
      makeRow({ event_id: `evt-${i}` }),
    );
    stubSelect(rows);

    await handleReconcilePlanCounterCron(makeEnv());

    const insertCall = mockSql.mock.calls.find((c) => {
      const tpl = c[0] as ReadonlyArray<string> | undefined;
      return tpl?.join(" ").toLowerCase().includes("insert into plan_counter_cron_history");
    });
    expect(insertCall).toBeDefined();

    const interpolated = insertCall!.slice(1);
    expect(interpolated).toContain(true); // batch_full
  });

  it("INSERT failure does NOT propagate (codex R5 #1 same anti-pattern as KV)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([makeRow({ event_id: "evt-x" })]); // SELECT
    mockSql.mockResolvedValueOnce([]); // UPDATE
    mockSql.mockRejectedValueOnce(new Error("insert blew up")); // INSERT

    await expect(
      handleReconcilePlanCounterCron(makeEnv()),
    ).resolves.toBeUndefined();

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_cron_history_write_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
  });

  it("INSERT is NOT attempted when SELECT throws (early return path)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockRejectedValueOnce(new Error("SELECT failed"));

    await handleReconcilePlanCounterCron(makeEnv());

    // Only ONE mockSql call: the SELECT (which threw). No INSERT attempt.
    expect(mockSql).toHaveBeenCalledTimes(1);
    // No cron_history INSERT — the early return after SELECT failure short-circuits.
    const insertAttempted = mockSql.mock.calls.some((c) => {
      const tpl = c[0] as ReadonlyArray<string> | undefined;
      if (!tpl || !Array.isArray(tpl)) return false;
      return tpl.join(" ").toLowerCase().includes("insert into plan_counter_cron_history");
    });
    expect(insertAttempted).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// PR-2e follow-up: retry-state pruning (7-day cutoff)
// ---------------------------------------------------------------------------

describe("PR-2e follow-up: retry-state pruning", () => {
  // Helper: install the SELECT + UPDATE + INSERT cron_history responses
  // + the two DELETE responses for the prune step. Each mock call consumed
  // in order. `syncPrunedCount` and `divergencePrunedCount` are the number
  // of rows the DELETE returns via `RETURNING 1` (postgres.js returns one
  // object per deleted row).
  function stubFullRun(
    syncPrunedCount: number,
    divergencePrunedCount: number,
  ) {
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([]); // SELECT (no drift rows)
    mockSql.mockResolvedValueOnce([]); // cron_history INSERT
    mockSql.mockResolvedValueOnce(
      Array.from({ length: syncPrunedCount }, () => ({ "?column?": 1 })),
    ); // DELETE plan_counter_sync_requests
    mockSql.mockResolvedValueOnce(
      Array.from({ length: divergencePrunedCount }, () => ({ "?column?": 1 })),
    ); // DELETE plan_counter_divergence_dedup
  }

  it("DELETE queries use written_at < now() - INTERVAL '7 days' cutoff", async () => {
    stubFullRun(0, 0);
    await handleReconcilePlanCounterCron(makeEnv());

    // Last two SQL calls are the prune DELETEs (after SELECT + cron_history INSERT).
    const calls = mockSql.mock.calls;
    const syncDelete = calls[calls.length - 2];
    const divergenceDelete = calls[calls.length - 1];

    const syncTpl = (syncDelete[0] as ReadonlyArray<string>).join(" ").toLowerCase();
    const divergenceTpl = (divergenceDelete[0] as ReadonlyArray<string>)
      .join(" ")
      .toLowerCase();

    expect(syncTpl).toContain("delete from plan_counter_sync_requests");
    expect(syncTpl).toContain("where written_at < now() - interval '7 days'");
    expect(syncTpl).toContain("returning 1");

    expect(divergenceTpl).toContain("delete from plan_counter_divergence_dedup");
    expect(divergenceTpl).toContain("where written_at < now() - interval '7 days'");
    expect(divergenceTpl).toContain("returning 1");
  });

  it("emits plan_counter_retry_state_pruned with per-table deleted counts", async () => {
    stubFullRun(12, 3);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_retry_state_pruned",
      {
        syncRequestsDeleted: 12,
        divergenceDedupDeleted: 3,
      },
    );
  });

  it("emits zero counts when no rows are past the cutoff (steady state)", async () => {
    stubFullRun(0, 0);
    await handleReconcilePlanCounterCron(makeEnv());

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_retry_state_pruned",
      {
        syncRequestsDeleted: 0,
        divergenceDedupDeleted: 0,
      },
    );
  });

  it("DELETE failure emits prune_failed metric + does NOT propagate", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([]); // SELECT
    mockSql.mockResolvedValueOnce([]); // cron_history INSERT
    mockSql.mockRejectedValueOnce(new Error("PG lock timeout")); // First DELETE throws

    await expect(
      handleReconcilePlanCounterCron(makeEnv()),
    ).resolves.toBeUndefined();

    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_retry_state_prune_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
    // The success metric must NOT have fired on this tick.
    const successCalls = mockEmitMetric.mock.calls.filter(
      (c) => c[0] === "plan_counter_retry_state_pruned",
    );
    expect(successCalls).toHaveLength(0);
  });

  it("prune is NOT attempted when SELECT throws (same early-return as INSERT)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockRejectedValueOnce(new Error("SELECT failed"));

    await handleReconcilePlanCounterCron(makeEnv());

    // Only ONE mockSql call (the failing SELECT). Prune does not run.
    expect(mockSql).toHaveBeenCalledTimes(1);
    const pruneAttempted = mockSql.mock.calls.some((c) => {
      const tpl = c[0] as ReadonlyArray<string> | undefined;
      if (!tpl || !Array.isArray(tpl)) return false;
      return tpl.join(" ").toLowerCase().includes("delete from plan_counter_");
    });
    expect(pruneAttempted).toBe(false);
  });

  it("prune runs even when cron_history INSERT fails (independent try/catch)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockSql.mockReset();
    mockSql.mockResolvedValueOnce([]); // SELECT
    mockSql.mockRejectedValueOnce(new Error("INSERT exploded")); // cron_history
    mockSql.mockResolvedValueOnce([{ "?column?": 1 }]); // DELETE sync
    mockSql.mockResolvedValueOnce([]); // DELETE divergence

    await handleReconcilePlanCounterCron(makeEnv());

    // Both maintenance paths should have emitted their outcomes.
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_cron_history_write_failed",
      expect.objectContaining({ reason: expect.any(String) }),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith(
      "plan_counter_retry_state_pruned",
      { syncRequestsDeleted: 1, divergenceDedupDeleted: 0 },
    );
  });
});
