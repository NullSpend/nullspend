import { describe, it, expect } from "vitest";
import {
  NullSpendError,
  TimeoutError,
  RejectedError,
  BudgetExceededError,
  SessionLimitExceededError,
  VelocityExceededError,
  TagBudgetExceededError,
} from "./errors.js";

// ---------------------------------------------------------------------------
// Constructor validation (Codex finding #7)
// ---------------------------------------------------------------------------

describe("TimeoutError constructor validation", () => {
  it("coerces non-string actionId to string", () => {
    const err = new TimeoutError(undefined as unknown as string, 5000);
    expect(err.actionId).toBe("unknown");
    expect(err.timeoutMs).toBe(5000);
  });

  it("clamps negative timeoutMs to 0", () => {
    const err = new TimeoutError("act-1", -100);
    expect(err.timeoutMs).toBe(0);
  });

  it("clamps NaN timeoutMs to 0", () => {
    const err = new TimeoutError("act-1", NaN);
    expect(err.timeoutMs).toBe(0);
  });

  it("clamps Infinity timeoutMs to 0", () => {
    const err = new TimeoutError("act-1", Infinity);
    expect(err.timeoutMs).toBe(0);
  });

  it("preserves valid values", () => {
    const err = new TimeoutError("act-1", 5000);
    expect(err.actionId).toBe("act-1");
    expect(err.timeoutMs).toBe(5000);
    expect(err.name).toBe("TimeoutError");
  });
});

describe("RejectedError constructor validation", () => {
  it("coerces non-string actionId", () => {
    const err = new RejectedError(null as unknown as string, "rejected");
    expect(err.actionId).toBe("unknown");
  });

  it("coerces non-string status", () => {
    const err = new RejectedError("act-1", undefined as unknown as string);
    expect(err.actionStatus).toBe("unknown");
  });

  it("preserves valid values", () => {
    const err = new RejectedError("act-1", "rejected");
    expect(err.actionId).toBe("act-1");
    expect(err.actionStatus).toBe("rejected");
    expect(err.name).toBe("RejectedError");
  });
});

describe("BudgetExceededError constructor validation", () => {
  it("clamps NaN remaining to 0", () => {
    const err = new BudgetExceededError(NaN);
    expect(err.remainingMicrodollars).toBe(0);
  });

  it("clamps negative remaining to 0", () => {
    const err = new BudgetExceededError(-500);
    expect(err.remainingMicrodollars).toBe(0);
  });

  it("clamps Infinity limit and spend to 0", () => {
    const err = new BudgetExceededError({
      remaining: 100,
      limit: Infinity,
      spend: -Infinity,
    });
    expect(err.remainingMicrodollars).toBe(100);
    expect(err.limitMicrodollars).toBe(0);
    expect(err.spendMicrodollars).toBe(0);
  });

  it("preserves valid values including upgradeUrl", () => {
    const err = new BudgetExceededError({
      remaining: 100_000,
      entityType: "api_key",
      entityId: "key-1",
      limit: 5_000_000,
      spend: 4_900_000,
      upgradeUrl: "https://example.com/upgrade",
    });
    expect(err.remainingMicrodollars).toBe(100_000);
    expect(err.entityType).toBe("api_key");
    expect(err.limitMicrodollars).toBe(5_000_000);
    expect(err.spendMicrodollars).toBe(4_900_000);
    expect(err.upgradeUrl).toBe("https://example.com/upgrade");
    expect(err.name).toBe("BudgetExceededError");
  });

  it("leaves limit/spend undefined when not provided", () => {
    const err = new BudgetExceededError({ remaining: 0 });
    expect(err.limitMicrodollars).toBeUndefined();
    expect(err.spendMicrodollars).toBeUndefined();
  });
});

describe("SessionLimitExceededError constructor validation", () => {
  it("clamps NaN values to 0", () => {
    const err = new SessionLimitExceededError(NaN, NaN);
    expect(err.sessionSpendMicrodollars).toBe(0);
    expect(err.sessionLimitMicrodollars).toBe(0);
  });

  it("clamps negative values to 0", () => {
    const err = new SessionLimitExceededError(-100, -200);
    expect(err.sessionSpendMicrodollars).toBe(0);
    expect(err.sessionLimitMicrodollars).toBe(0);
  });

  it("preserves valid values", () => {
    const err = new SessionLimitExceededError(500_000, 1_000_000);
    expect(err.sessionSpendMicrodollars).toBe(500_000);
    expect(err.sessionLimitMicrodollars).toBe(1_000_000);
    expect(err.name).toBe("SessionLimitExceededError");
  });
});

describe("VelocityExceededError constructor validation", () => {
  it("clamps NaN retryAfterSeconds to 0", () => {
    const err = new VelocityExceededError({ retryAfterSeconds: NaN });
    expect(err.retryAfterSeconds).toBe(0);
  });

  it("clamps negative values to 0", () => {
    const err = new VelocityExceededError({
      retryAfterSeconds: -5,
      limit: -100,
      window: -60,
      current: -50,
    });
    expect(err.retryAfterSeconds).toBe(0);
    expect(err.limitMicrodollars).toBe(0);
    expect(err.windowSeconds).toBe(0);
    expect(err.currentMicrodollars).toBe(0);
  });

  it("preserves valid values", () => {
    const err = new VelocityExceededError({
      retryAfterSeconds: 30,
      limit: 500_000,
      window: 60,
      current: 750_000,
    });
    expect(err.retryAfterSeconds).toBe(30);
    expect(err.limitMicrodollars).toBe(500_000);
    expect(err.windowSeconds).toBe(60);
    expect(err.currentMicrodollars).toBe(750_000);
    expect(err.name).toBe("VelocityExceededError");
  });

  it("leaves fields undefined when not provided", () => {
    const err = new VelocityExceededError();
    expect(err.retryAfterSeconds).toBeUndefined();
    expect(err.limitMicrodollars).toBeUndefined();
    expect(err.windowSeconds).toBeUndefined();
    expect(err.currentMicrodollars).toBeUndefined();
  });
});

describe("TagBudgetExceededError constructor validation", () => {
  it("clamps NaN values to 0", () => {
    const err = new TagBudgetExceededError({
      remaining: NaN,
      limit: NaN,
      spend: NaN,
    });
    expect(err.remainingMicrodollars).toBe(0);
    expect(err.limitMicrodollars).toBe(0);
    expect(err.spendMicrodollars).toBe(0);
  });

  it("preserves valid values", () => {
    const err = new TagBudgetExceededError({
      tagKey: "team",
      tagValue: "engineering",
      remaining: 100,
      limit: 1000,
      spend: 900,
    });
    expect(err.tagKey).toBe("team");
    expect(err.tagValue).toBe("engineering");
    expect(err.remainingMicrodollars).toBe(100);
    expect(err.name).toBe("TagBudgetExceededError");
  });

  it("leaves numeric fields undefined when not provided", () => {
    const err = new TagBudgetExceededError({ tagKey: "k" });
    expect(err.remainingMicrodollars).toBeUndefined();
    expect(err.limitMicrodollars).toBeUndefined();
    expect(err.spendMicrodollars).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------

describe("NullSpendError", () => {
  it("stores statusCode and code", () => {
    const err = new NullSpendError("test", 404, "not_found");
    expect(err.message).toBe("test");
    expect(err.statusCode).toBe(404);
    expect(err.code).toBe("not_found");
    expect(err.name).toBe("NullSpendError");
  });

  it("is an instance of Error", () => {
    const err = new NullSpendError("test");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(NullSpendError);
  });
});
