import { cloudflareWorkersMock, makeEnv } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";

beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as any).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.byteLength !== viewB.byteLength) return false;
      let result = 0;
      for (let i = 0; i < viewA.byteLength; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

const { mockGetBudgetState } = vi.hoisted(() => ({
  mockGetBudgetState: vi.fn().mockResolvedValue([]),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetGetState: (...args: unknown[]) => mockGetBudgetState(...args),
  doBudgetCheck: vi.fn(),
  doBudgetReconcile: vi.fn(),
}));

vi.mock("@nullspend/cost-engine", () => ({
  getAllPricing: vi.fn().mockReturnValue({
    "openai/gpt-4o": { inputPerMTok: 2.5, cachedInputPerMTok: 1.25, outputPerMTok: 10 },
    "openai/gpt-4o-mini": { inputPerMTok: 0.15, cachedInputPerMTok: 0.075, outputPerMTok: 0.60 },
    "anthropic/claude-sonnet-4-20250514": { inputPerMTok: 3.0, cachedInputPerMTok: 0.30, outputPerMTok: 15.0 },
    "anthropic/claude-haiku-4-5-20251001": { inputPerMTok: 0.80, cachedInputPerMTok: 0.08, outputPerMTok: 4.0 },
  }),
  isKnownModel: vi.fn().mockReturnValue(true),
  getModelPricing: vi.fn(),
}));

import { handlePolicy } from "../routes/policy.js";
import type { AuthResult } from "../lib/auth.js";

function makeAuth(overrides: Partial<AuthResult> = {}): AuthResult {
  return {
    userId: "user-1",
    orgId: "org-1",
    keyId: "key-1",
    hasWebhooks: false,
    hasBudgets: false,
    requestLoggingEnabled: false,
    apiVersion: "2026-04-01",
    defaultTags: {},
    allowedModels: null,
    allowedProviders: null,
    ...overrides,
  };
}

function makeRequest(): Request {
  return new Request("http://localhost/v1/policy", {
    method: "GET",
    headers: { "x-nullspend-key": "ns_live_sk_test" },
  });
}

describe("GET /v1/policy", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    mockGetBudgetState.mockReset().mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns policy with no budgets and no restrictions", async () => {
    const res = await handlePolicy(makeRequest(), makeEnv(), makeAuth(), "trace-1");

    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.budget).toBeNull();
    expect(json.allowed_models).toBeNull();
    expect(json.allowed_providers).toBeNull();
    expect(json.restrictions_active).toBe(false);
    expect(json.cheapest_overall).not.toBeNull();
    expect(json.cheapest_overall.model).toBe("gpt-4o-mini"); // cheapest in mock catalog
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-1");
  });

  it("returns budget state from DO", async () => {
    mockGetBudgetState.mockResolvedValue([{
      entity_type: "api_key",
      entity_id: "key-1",
      max_budget: 10_000_000,
      spend: 3_000_000,
      reserved: 500_000,
      policy: "strict_block",
      reset_interval: "monthly",
      period_start: new Date("2026-03-01").getTime(),
      velocity_limit: null,
      velocity_window: 60,
      velocity_cooldown: 60,
      threshold_percentages: null,
      session_limit: null,
    }]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-2",
    );

    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.budget).not.toBeNull();
    expect(json.budget.remaining_microdollars).toBe(6_500_000); // 10M - 3M - 500K
    expect(json.budget.max_microdollars).toBe(10_000_000);
    expect(json.budget.spend_microdollars).toBe(3_000_000);
    expect(json.budget.entity_type).toBe("api_key");
    expect(json.budget.period_end).toContain("2026-04"); // monthly reset
  });

  it("returns most restrictive budget when multiple exist", async () => {
    mockGetBudgetState.mockResolvedValue([
      {
        entity_type: "api_key", entity_id: "key-1",
        max_budget: 10_000_000, spend: 2_000_000, reserved: 0,
        policy: "strict_block", reset_interval: null, period_start: 0,
        velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
        threshold_percentages: null, session_limit: null,
      },
      {
        entity_type: "user", entity_id: "user-1",
        max_budget: 5_000_000, spend: 4_000_000, reserved: 0,
        policy: "strict_block", reset_interval: null, period_start: 0,
        velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
        threshold_percentages: null, session_limit: null,
      },
    ]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-3",
    );

    const json = await res.json();
    // User budget has lower remaining: 5M - 4M = 1M vs api_key: 10M - 2M = 8M
    expect(json.budget.entity_type).toBe("user");
    expect(json.budget.remaining_microdollars).toBe(1_000_000);
  });

  it("returns restrictions from auth", async () => {
    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({
        allowedModels: ["gpt-4o-mini", "claude-haiku-4-5-20251001"],
        allowedProviders: ["openai", "anthropic"],
      }),
      "trace-4",
    );

    const json = await res.json();
    expect(json.allowed_models).toEqual(["gpt-4o-mini", "claude-haiku-4-5-20251001"]);
    expect(json.allowed_providers).toEqual(["openai", "anthropic"]);
    expect(json.restrictions_active).toBe(true);
  });

  it("filters cheapest models by allowed models", async () => {
    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ allowedModels: ["gpt-4o", "claude-sonnet-4-20250514"] }),
      "trace-5",
    );

    const json = await res.json();
    // gpt-4o-mini and claude-haiku not in allowed list
    expect(json.cheapest_overall.model).toBe("gpt-4o"); // cheaper than claude-sonnet
    expect(json.cheapest_per_provider.openai.model).toBe("gpt-4o");
    expect(json.cheapest_per_provider.anthropic.model).toBe("claude-sonnet-4-20250514");
  });

  it("filters cheapest models by allowed providers", async () => {
    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ allowedProviders: ["openai"] }),
      "trace-6",
    );

    const json = await res.json();
    expect(json.cheapest_overall.model).toBe("gpt-4o-mini");
    expect(json.cheapest_per_provider.openai).toBeDefined();
    expect(json.cheapest_per_provider.anthropic).toBeUndefined();
  });

  it("returns null cheapest when allowed models is empty (deny all)", async () => {
    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ allowedModels: [] }),
      "trace-7",
    );

    const json = await res.json();
    expect(json.allowed_models).toEqual([]);
    expect(json.cheapest_overall).toBeNull();
    expect(json.cheapest_per_provider).toBeNull();
    expect(json.restrictions_active).toBe(true);
  });

  it("skips DO call when hasBudgets is false", async () => {
    await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: false }),
      "trace-8",
    );

    expect(mockGetBudgetState).not.toHaveBeenCalled();
  });

  it("returns policy without budget when DO fails", async () => {
    mockGetBudgetState.mockRejectedValue(new Error("DO unavailable"));

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-9",
    );

    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.budget).toBeNull(); // Graceful degradation
    expect(json.cheapest_overall).not.toBeNull(); // Models still work
  });

  it("clamps remaining to 0 when budget is overspent", async () => {
    mockGetBudgetState.mockResolvedValue([{
      entity_type: "api_key", entity_id: "key-1",
      max_budget: 1_000_000, spend: 1_200_000, reserved: 0,
      policy: "strict_block", reset_interval: null, period_start: 0,
      velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
      threshold_percentages: null, session_limit: null,
    }]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-10",
    );

    const json = await res.json();
    expect(json.budget.remaining_microdollars).toBe(0); // Clamped, not negative
  });

  it("sets Cache-Control: no-store", async () => {
    const res = await handlePolicy(makeRequest(), makeEnv(), makeAuth(), "trace-11");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
  });

  it("returns period_end null when period_start is 0 (uninitialized budget)", async () => {
    mockGetBudgetState.mockResolvedValue([{
      entity_type: "api_key", entity_id: "key-1",
      max_budget: 10_000_000, spend: 0, reserved: 0,
      policy: "strict_block", reset_interval: "daily", period_start: 0,
      velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
      threshold_percentages: null, session_limit: null,
    }]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-epoch",
    );

    const json = await res.json();
    // period_start: 0 means uninitialized (matches DO behavior which skips reset for period_start=0)
    expect(json.budget.period_end).toBeNull();
  });

  it("computes period_end for initialized period_start with reset interval", async () => {
    mockGetBudgetState.mockResolvedValue([{
      entity_type: "api_key", entity_id: "key-1",
      max_budget: 10_000_000, spend: 0, reserved: 0,
      policy: "strict_block", reset_interval: "weekly", period_start: new Date("2026-03-24").getTime(),
      velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
      threshold_percentages: null, session_limit: null,
    }]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-weekly",
    );

    const json = await res.json();
    expect(json.budget.period_end).toContain("2026-03-31");
  });

  it("returns period_end null when no reset interval", async () => {
    mockGetBudgetState.mockResolvedValue([{
      entity_type: "api_key", entity_id: "key-1",
      max_budget: 10_000_000, spend: 0, reserved: 0,
      policy: "strict_block", reset_interval: null, period_start: 0,
      velocity_limit: null, velocity_window: 60, velocity_cooldown: 60,
      threshold_percentages: null, session_limit: null,
    }]);

    const res = await handlePolicy(
      makeRequest(),
      makeEnv(),
      makeAuth({ hasBudgets: true }),
      "trace-12",
    );

    const json = await res.json();
    expect(json.budget.period_end).toBeNull();
  });
});
