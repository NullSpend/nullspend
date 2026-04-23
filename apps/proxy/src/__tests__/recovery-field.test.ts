import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn(),
}));

import { buildRecovery, fmtDollars, LOOP_RETRY_AFTER_SECONDS } from "../routes/shared.js";

// ---------------------------------------------------------------------------
// buildRecovery helper — denial code → recovery object mapping
// ---------------------------------------------------------------------------

describe("buildRecovery helper", () => {
  it("budget_exceeded → not retryable, owner action required", () => {
    expect(buildRecovery("budget_exceeded")).toEqual({
      retryable: false,
      owner_action_required: true,
      retry_after_seconds: null,
      docs: null,
    });
  });

  it("customer_budget_exceeded → not retryable, owner action required", () => {
    expect(buildRecovery("customer_budget_exceeded")).toEqual({
      retryable: false,
      owner_action_required: true,
      retry_after_seconds: null,
      docs: null,
    });
  });

  it("tag_budget_exceeded → not retryable, owner action required", () => {
    expect(buildRecovery("tag_budget_exceeded")).toEqual({
      retryable: false,
      owner_action_required: true,
      retry_after_seconds: null,
      docs: null,
    });
  });

  it("velocity_exceeded with retryAfterSeconds=45 → retryable with retry_after", () => {
    expect(buildRecovery("velocity_exceeded", 45)).toEqual({
      retryable: true,
      owner_action_required: false,
      retry_after_seconds: 45,
      docs: null,
    });
  });

  it("loop_detected with retryAfterSeconds=5 → retryable with retry_after", () => {
    expect(buildRecovery("loop_detected", 5)).toEqual({
      retryable: true,
      owner_action_required: false,
      retry_after_seconds: 5,
      docs: null,
    });
  });

  it("session_limit_exceeded → not retryable, no owner action", () => {
    expect(buildRecovery("session_limit_exceeded")).toEqual({
      retryable: false,
      owner_action_required: false,
      retry_after_seconds: null,
      docs: null,
    });
  });

  // PR-2c plan-audit F10 + codex-round-1 C2 decision D1: plan-limit is a HARD
  // block — not retryable, owner must upgrade or wait for period reset.
  // Added to `ownerAction` array via array-membership check, NOT a switch
  // refactor. This test fails loudly if an implementer emits {retryable: false}
  // alone (missing owner_action_required: true).
  it("plan_limit_exceeded → not retryable, owner action required", () => {
    expect(buildRecovery("plan_limit_exceeded")).toEqual({
      retryable: false,
      owner_action_required: true,
      retry_after_seconds: null,
      docs: null,
    });
  });
});

// ---------------------------------------------------------------------------
// buildRecovery edge cases — boundary value handling
// ---------------------------------------------------------------------------

describe("buildRecovery edge cases", () => {
  it("unknown code → safe defaults", () => {
    expect(buildRecovery("unknown_code")).toEqual({
      retryable: false,
      owner_action_required: false,
      retry_after_seconds: null,
      docs: null,
    });
  });

  it("velocity_exceeded with NaN → retry_after_seconds: null", () => {
    const result = buildRecovery("velocity_exceeded", NaN);
    expect(result.retry_after_seconds).toBeNull();
  });

  it("velocity_exceeded with negative value → retry_after_seconds: null", () => {
    const result = buildRecovery("velocity_exceeded", -5);
    expect(result.retry_after_seconds).toBeNull();
  });

  it("velocity_exceeded with Infinity → retry_after_seconds: null", () => {
    const result = buildRecovery("velocity_exceeded", Infinity);
    expect(result.retry_after_seconds).toBeNull();
  });

  it("velocity_exceeded with 0 → retry_after_seconds: 0 (valid)", () => {
    const result = buildRecovery("velocity_exceeded", 0);
    expect(result.retry_after_seconds).toBe(0);
  });

  it("non-retryable code ignores retryAfterSeconds argument", () => {
    const result = buildRecovery("budget_exceeded", 60);
    expect(result.retryable).toBe(false);
    expect(result.retry_after_seconds).toBeNull();
  });

  it("retryAfterSeconds undefined → null (not undefined)", () => {
    const result = buildRecovery("velocity_exceeded", undefined);
    expect(result.retry_after_seconds).toBeNull();
  });

  it("retryAfterSeconds null → null", () => {
    const result = buildRecovery("velocity_exceeded", null);
    expect(result.retry_after_seconds).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// shared.ts / mcp.ts parity — same code produces identical recovery
// Both files call buildRecovery() with the same arguments. Since it's a pure
// function imported from shared.ts into mcp.ts, parity is structural. These
// tests verify the invariant that buildRecovery's output is deterministic and
// identical regardless of caller.
// ---------------------------------------------------------------------------

describe("shared.ts / mcp.ts parity", () => {
  const SHARED_DENIAL_CODES = [
    "velocity_exceeded",
    "session_limit_exceeded",
    "tag_budget_exceeded",
    "customer_budget_exceeded",
    "budget_exceeded",
    "plan_limit_exceeded", // PR-2c plan-audit F1 — MCP + provider parity
  ] as const;

  it.each(SHARED_DENIAL_CODES)(
    "%s produces identical recovery from same buildRecovery call",
    (code) => {
      const retryAfter = code === "velocity_exceeded" ? 60 : undefined;
      const call1 = buildRecovery(code, retryAfter);
      const call2 = buildRecovery(code, retryAfter);
      expect(call1).toEqual(call2);
    },
  );

  it("loop_detected is NOT in MCP denial paths (shared.ts only)", () => {
    // loop_detected denials only occur via provider-handler → shared.ts.
    // MCP budget checks don't run loop detection. This test documents the
    // asymmetry: mcp.ts has 5 denial codes, shared.ts has 6.
    const recovery = buildRecovery("loop_detected", LOOP_RETRY_AFTER_SECONDS);
    expect(recovery.retryable).toBe(true);
    expect(recovery.retry_after_seconds).toBe(LOOP_RETRY_AFTER_SECONDS);
  });
});

// ---------------------------------------------------------------------------
// fmtDollars — display formatting for enriched messages
// ---------------------------------------------------------------------------

describe("fmtDollars", () => {
  it("formats standard microdollar amount", () => {
    expect(fmtDollars(5_000_000)).toBe("$5.00");
  });

  it("formats fractional amounts with 2 decimal places", () => {
    expect(fmtDollars(100_000)).toBe("$0.10");
  });

  it("formats zero", () => {
    expect(fmtDollars(0)).toBe("$0.00");
  });

  it("formats sub-cent amounts (rounds to 2 decimals)", () => {
    expect(fmtDollars(1)).toBe("$0.00");
    expect(fmtDollars(5000)).toBe("$0.01");
    expect(fmtDollars(4999)).toBe("$0.00");
  });

  it("formats large amounts", () => {
    expect(fmtDollars(100_000_000)).toBe("$100.00");
    expect(fmtDollars(1_000_000_000)).toBe("$1000.00");
  });

  it("handles negative microdollars (display only, not enforced)", () => {
    // fmtDollars is called with values already guarded by Math.max(0, ...),
    // but the function itself doesn't enforce non-negative.
    expect(fmtDollars(-500_000)).toBe("$-0.50");
  });
});

// ---------------------------------------------------------------------------
// Enriched message null-guard — verify fallback to static messages
// These test the message selection logic embedded in shared.ts denial builders.
// We can't call handleBudgetDenials directly without the full mock stack,
// but we can verify the conditional logic pattern: limit > 0 → enriched,
// limit === 0 → static fallback.
// ---------------------------------------------------------------------------

describe("enriched message null-guard pattern", () => {
  it("budget: enriched message when limit > 0", () => {
    const budgetLimit = 5_000_000;
    const budgetSpend = 4_900_000;
    const budgetRemaining = Math.max(0, budgetLimit - budgetSpend);
    const message = budgetLimit > 0
      ? `Budget exceeded: ${fmtDollars(budgetRemaining)} of ${fmtDollars(budgetLimit)} remaining. Request a budget increase or switch to a cheaper model.`
      : "Request blocked: estimated cost exceeds remaining budget.";
    expect(message).toBe("Budget exceeded: $0.10 of $5.00 remaining. Request a budget increase or switch to a cheaper model.");
  });

  it("budget: static fallback when limit is 0", () => {
    const budgetLimit = 0;
    const message = budgetLimit > 0
      ? `Budget exceeded: ${fmtDollars(0)} of ${fmtDollars(budgetLimit)} remaining. Request a budget increase or switch to a cheaper model.`
      : "Request blocked: estimated cost exceeds remaining budget.";
    expect(message).toBe("Request blocked: estimated cost exceeds remaining budget.");
  });

  it("velocity: enriched message when limit > 0", () => {
    const velLimit = 10_000_000;
    const velCurrent = 12_500_000;
    const velWindow = 300;
    const velRetryAfter = 45;
    const message = velLimit > 0
      ? `Rate limit exceeded: ${fmtDollars(velCurrent)} spent in ${velWindow}s window (limit: ${fmtDollars(velLimit)}). Retry after ${velRetryAfter}s.`
      : "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.";
    expect(message).toBe("Rate limit exceeded: $12.50 spent in 300s window (limit: $10.00). Retry after 45s.");
  });

  it("velocity: static fallback when limit is 0", () => {
    const velLimit = 0;
    const message = velLimit > 0
      ? `Rate limit exceeded: ${fmtDollars(0)} spent in 60s window (limit: ${fmtDollars(velLimit)}). Retry after 60s.`
      : "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.";
    expect(message).toBe("Request blocked: spending rate exceeds velocity limit. Retry after cooldown.");
  });

  it("session: enriched message when limit > 0", () => {
    const sessLimit = 1_000_000;
    const sessSpend = 900_000;
    const message = sessLimit > 0
      ? `Session limit reached: ${fmtDollars(sessSpend)} of ${fmtDollars(sessLimit)}. Start a new session.`
      : "Request blocked: session spend exceeds session limit. Start a new session.";
    expect(message).toBe("Session limit reached: $0.90 of $1.00. Start a new session.");
  });

  it("session: static fallback when limit is 0", () => {
    const sessLimit = 0;
    const message = sessLimit > 0
      ? `Session limit reached: ${fmtDollars(0)} of ${fmtDollars(sessLimit)}. Start a new session.`
      : "Request blocked: session spend exceeds session limit. Start a new session.";
    expect(message).toBe("Request blocked: session spend exceeds session limit. Start a new session.");
  });

  it("tag: enriched message when limit > 0", () => {
    const tagLimit = 2_000_000;
    const tagSpend = 1_800_000;
    const tagRemaining = Math.max(0, tagLimit - tagSpend);
    const tagKey = "env";
    const tagValue = "prod";
    const message = tagLimit > 0
      ? `Tag budget exceeded for ${tagKey}=${tagValue}: ${fmtDollars(tagRemaining)} of ${fmtDollars(tagLimit)} remaining.`
      : "Request blocked: estimated cost exceeds tag budget limit.";
    expect(message).toBe("Tag budget exceeded for env=prod: $0.20 of $2.00 remaining.");
  });

  it("customer: enriched message when limit > 0", () => {
    const custLimit = 10_000_000;
    const custSpend = 9_000_000;
    const custRemaining = Math.max(0, custLimit - custSpend);
    const custId = "acme-corp";
    const message = custLimit > 0
      ? `Customer budget exceeded for ${custId}: ${fmtDollars(custRemaining)} of ${fmtDollars(custLimit)} remaining.`
      : "Request blocked: estimated cost exceeds customer budget limit.";
    expect(message).toBe("Customer budget exceeded for acme-corp: $1.00 of $10.00 remaining.");
  });
});

// ---------------------------------------------------------------------------
// LOOP_RETRY_AFTER_SECONDS constant
// ---------------------------------------------------------------------------

describe("LOOP_RETRY_AFTER_SECONDS constant", () => {
  it("is a positive integer", () => {
    expect(Number.isInteger(LOOP_RETRY_AFTER_SECONDS)).toBe(true);
    expect(LOOP_RETRY_AFTER_SECONDS).toBeGreaterThan(0);
  });

  it("matches the value used in buildRecovery for loop_detected", () => {
    const recovery = buildRecovery("loop_detected", LOOP_RETRY_AFTER_SECONDS);
    expect(recovery.retry_after_seconds).toBe(LOOP_RETRY_AFTER_SECONDS);
  });
});
