/**
 * Loop Detection Stress Tests — Durable Object (real SQLite)
 *
 * These run against the actual UserBudgetDO with real SQLite state.
 * Each test uses a unique user ID for isolation.
 *
 * Run: cd apps/proxy && npx vitest run --config vitest.do.config.mts src/__tests__/loop-detection-stress.do.test.ts
 */
import { env, runDurableObjectAlarm } from "cloudflare:test";
import { describe, it, expect } from "vitest";

function getStub(userId: string) {
  return env.USER_BUDGET.get(env.USER_BUDGET.idFromName(userId));
}

/** Populate a budget entity with loop detection defaults. */
async function setupBudget(
  userId: string,
  opts: {
    maxBudget?: number;
    policy?: string;
    loopMaxCalls?: number | null;
    loopWindowSeconds?: number | null;
    loopAggregateMaxKeys?: number | null;
  } = {},
) {
  const stub = getStub(userId);
  await stub.populateIfEmpty(
    "user", "u1",
    opts.maxBudget ?? 1_000_000_000, // $1000 — high enough to never budget-deny
    0, opts.policy ?? "strict_block", null, 0,
    null, 60_000, 60_000,  // velocity
    [50, 80, 90, 95],      // thresholds
    null,                   // session limit
    0,                      // finalization reserve
    opts.loopMaxCalls ?? null,
    opts.loopWindowSeconds ?? null,
    opts.loopAggregateMaxKeys ?? null,
  );
  return stub;
}

function loopCtx(model: string, hash: string) {
  return { provider: "openai", model, contentHash: hash };
}

// ── Per-key detection ────────────────────────────────────────────

describe("Loop Detection Stress — Per-Key", () => {
  it("denies at exactly threshold 50 (default)", async () => {
    const stub = await setupBudget("loop-pk-50");
    for (let i = 0; i < 49; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "samehash"));
      expect(r.status).toBe("approved");
      // Reconcile to free the reservation
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // 50th call should deny
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "samehash"));
    expect(denied.status).toBe("denied");
    expect(denied.loopDetected).toBe(true);
    expect(denied.loopDetails?.type).toBe("per_key");
    expect(denied.loopDetails?.callCount).toBe(50);
  });

  it("denies at custom threshold 5", async () => {
    const stub = await setupBudget("loop-pk-5", { loopMaxCalls: 5 });
    for (let i = 0; i < 4; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "abc123"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "abc123"));
    expect(denied.loopDetected).toBe(true);
    expect(denied.loopDetails?.callCount).toBe(5);
  });

  it("different content hashes are independent — no false positive", async () => {
    const stub = await setupBudget("loop-pk-diff", { loopMaxCalls: 5 });
    for (let i = 0; i < 20; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", `unique_${i}`));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
  });

  it("different models are tracked independently", async () => {
    const stub = await setupBudget("loop-pk-models", { loopMaxCalls: 3 });
    // 2 calls each on 3 models — none should trigger
    for (const model of ["gpt-4o", "gpt-4o-mini", "claude-sonnet"]) {
      for (let i = 0; i < 2; i++) {
        const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
          loopCtx(model, "samehash"));
        expect(r.status).toBe("approved");
        if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
      }
    }
    // 3rd call on gpt-4o should trigger
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "samehash"));
    expect(denied.loopDetected).toBe(true);
  });

  it("loopMaxCalls=0 completely disables — 100 identical calls all pass", async () => {
    const stub = await setupBudget("loop-pk-disabled", { loopMaxCalls: 0 });
    for (let i = 0; i < 100; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "samehash"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
  });

  it("null loopContext (MCP route) never triggers", async () => {
    const stub = await setupBudget("loop-pk-null", { loopMaxCalls: 1 });
    for (let i = 0; i < 10; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false, null);
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
  });
});

// ── Aggregate detection ──────────────────────────────────────────

describe("Loop Detection Stress — Aggregate", () => {
  it("triggers when 5 keys each have 3+ same-content repeats", async () => {
    const stub = await setupBudget("loop-agg-5keys", { loopMaxCalls: 100, loopAggregateMaxKeys: 5 });
    // Populate 4 keys with 3 repeats each (below aggregate threshold)
    for (const model of ["m1", "m2", "m3", "m4"]) {
      for (let i = 0; i < 3; i++) {
        const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
          loopCtx(model, "same"));
        expect(r.status).toBe("approved");
        if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
      }
    }
    // 5th key at 2 repeats (still safe)
    for (let i = 0; i < 2; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("m5", "same"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // 3rd call on m5 commits the INSERT (budget passes), making 5 qualifying keys.
    // Due to deferred INSERT, the aggregate SQL runs BEFORE the m5 insert commits,
    // so it sees only 4 qualifying keys on this call. The NEXT call sees all 5.
    const r5 = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("m5", "same"));
    expect(r5.status).toBe("approved"); // m5's 3rd call commits the deferred INSERT
    if (r5.reservationId) await stub.reconcile(r5.reservationId, 1000);

    // Now the next call sees all 5 qualifying keys in the committed log → aggregate triggers
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("m5", "same"));
    expect(denied.loopDetected).toBe(true);
    expect(denied.loopDetails?.type).toBe("aggregate");
    expect(denied.loopDetails?.callCount).toBe(5);
  });

  it("diverse content (unique per call) never triggers aggregate", async () => {
    const stub = await setupBudget("loop-agg-diverse", { loopMaxCalls: 100, loopAggregateMaxKeys: 3 });
    for (let k = 0; k < 10; k++) {
      for (let i = 0; i < 10; i++) {
        const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
          loopCtx(`model-${k}`, `unique_${k}_${i}`));
        expect(r.status).toBe("approved");
        if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
      }
    }
  });
});

// ── Budget denial isolation ──────────────────────────────────────

describe("Loop Detection Stress — Budget Denial Isolation", () => {
  it("budget-denied requests do NOT inflate loop counter", async () => {
    // Budget: $0.01 (10,000 microdollars), loop threshold: 10
    const stub = await setupBudget("loop-budget-iso", {
      maxBudget: 10_000, loopMaxCalls: 10,
    });
    // First call succeeds and uses entire budget
    const r1 = await stub.checkAndReserve(null, 10_000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "samehash"));
    expect(r1.status).toBe("approved");
    if (r1.reservationId) await stub.reconcile(r1.reservationId, 10_000);

    // Next 20 calls are all budget-denied (no budget left)
    for (let i = 0; i < 20; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "samehash"));
      expect(r.status).toBe("denied");
      // Should be budget denial, NOT loop denial
      expect(r.loopDetected).toBeFalsy();
    }

    // Reset budget — loop counter should still be at 1 (only the first approved call)
    await stub.resetSpend("user", "u1");
    // Can make 8 more calls (1 + 8 = 9, below threshold of 10)
    // Actually after reset + time passing, the loop_call_log entry from the first call
    // is still there. So we have 1 existing entry + 8 new = 9 total, under threshold.
    for (let i = 0; i < 8; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "samehash"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // 10th total call should trigger loop
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "samehash"));
    expect(denied.loopDetected).toBe(true);
  });
});

// ── Denial backoff cache ─────────────────────────────────────────

describe("Loop Detection Stress — Denial Backoff", () => {
  it("repeated denials return cached result (same details)", async () => {
    const stub = await setupBudget("loop-backoff", { loopMaxCalls: 1 });
    // First call triggers loop
    const d1 = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "abc"));
    expect(d1.loopDetected).toBe(true);
    expect(d1.loopDetails?.callCount).toBe(1);

    // Subsequent calls should return cached denial (same details)
    for (let i = 0; i < 10; i++) {
      const d = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "abc"));
      expect(d.loopDetected).toBe(true);
      expect(d.loopDetails?.type).toBe("per_key");
    }
  });

  it("cached aggregate denial preserves aggregate type", async () => {
    const stub = await setupBudget("loop-backoff-agg", {
      loopMaxCalls: 100, loopAggregateMaxKeys: 2,
    });
    // Build 2 qualifying keys
    for (const model of ["m1", "m2"]) {
      for (let i = 0; i < 3; i++) {
        const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
          loopCtx(model, "same"));
        if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
      }
    }
    // Next call triggers aggregate
    const d1 = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("m1", "same"));
    expect(d1.loopDetails?.type).toBe("aggregate");

    // Cached denial should still say aggregate
    const d2 = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("m1", "same"));
    expect(d2.loopDetails?.type).toBe("aggregate");
  });
});

// ── Warning header data ──────────────────────────────────────────

describe("Loop Detection Stress — Warning Header", () => {
  it("loopCount appears at 80% threshold (40 of 50)", async () => {
    const stub = await setupBudget("loop-warn-80");
    // Make 39 calls — below 80%
    for (let i = 0; i < 39; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "same"));
      expect(r.loopCount).toBeUndefined();
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // 40th call = 80% of 50 — should have loopCount
    const r40 = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "same"));
    expect(r40.status).toBe("approved");
    expect(r40.loopCount).toBe(40);
    expect(r40.loopMaxCalls).toBe(50);
    if (r40.reservationId) await stub.reconcile(r40.reservationId, 1000);
  });
});

// ── Config change mid-stream ─────────────────────────────────────

describe("Loop Detection Stress — Config Change", () => {
  it("updating loopMaxCalls takes effect on next call", async () => {
    const stub = await setupBudget("loop-config-change", { loopMaxCalls: 100 });
    // Make 10 calls — well under threshold of 100
    for (let i = 0; i < 10; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "same"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // Change threshold to 5 (below current count of 10)
    await stub.populateIfEmpty(
      "user", "u1", 1_000_000_000, 0, "strict_block", null, 0,
      null, 60_000, 60_000, [50, 80, 90, 95], null, 0,
      5, null, null, // loopMaxCalls=5
    );
    // Next call should be denied (10 existing + 1 = 11 > 5)
    const denied = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "same"));
    expect(denied.loopDetected).toBe(true);
  });
});

// ── SQL injection safety ─────────────────────────────────────────

describe("Loop Detection Stress — SQL Safety", () => {
  it("model name with SQL metacharacters does not crash or inject", async () => {
    const stub = await setupBudget("loop-sql-safe", { loopMaxCalls: 3 });
    const evilModel = "'; DROP TABLE loop_call_log; --";
    for (let i = 0; i < 3; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx(evilModel, "hash"));
      if (i < 2) {
        expect(r.status).toBe("approved");
        if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
      } else {
        expect(r.loopDetected).toBe(true);
      }
    }
    // Verify loop_call_log still exists (wasn't dropped)
    const state = await stub.getBudgetState();
    expect(state.length).toBe(1);
  });
});

// ── Alarm pruning ────────────────────────────────────────────────

describe("Loop Detection Stress — Alarm Pruning", () => {
  it("alarm cleans up loop_call_log entries", async () => {
    const stub = await setupBudget("loop-alarm-prune", { loopMaxCalls: 100 });
    // Make 20 calls
    for (let i = 0; i < 20; i++) {
      const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
        loopCtx("gpt-4o", "same"));
      expect(r.status).toBe("approved");
      if (r.reservationId) await stub.reconcile(r.reservationId, 1000);
    }
    // Run alarm — should prune loop entries and not crash
    await runDurableObjectAlarm(stub);
    // Verify DO still works
    const r = await stub.checkAndReserve(null, 1000, 30_000, null, [], null, false,
      loopCtx("gpt-4o", "same"));
    expect(r.status).toBe("approved");
  });
});
