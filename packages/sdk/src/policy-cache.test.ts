import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createPolicyCache } from "./policy-cache.js";
import type { PolicyResponse } from "./policy-cache.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePolicy(overrides: Partial<PolicyResponse> = {}): PolicyResponse {
  return {
    budget: null,
    allowed_models: null,
    allowed_providers: null,
    cheapest_per_provider: null,
    cheapest_overall: null,
    restrictions_active: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("createPolicyCache", () => {
  let fetchPolicy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    fetchPolicy = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // -----------------------------------------------------------------------
  // Cache behavior
  // -----------------------------------------------------------------------

  it("fetches policy on first call (cache miss)", async () => {
    const policy = makePolicy();
    fetchPolicy.mockResolvedValue(policy);

    const cache = createPolicyCache(fetchPolicy);
    const result = await cache.getPolicy();

    expect(result).toEqual(policy);
    expect(fetchPolicy).toHaveBeenCalledTimes(1);
  });

  it("returns cached policy within TTL without fetching again", async () => {
    const policy = makePolicy();
    fetchPolicy.mockResolvedValue(policy);

    const cache = createPolicyCache(fetchPolicy, 60_000);

    const first = await cache.getPolicy();
    expect(first).toEqual(policy);

    // Advance less than TTL
    vi.advanceTimersByTime(30_000);

    const second = await cache.getPolicy();
    expect(second).toEqual(policy);
    expect(fetchPolicy).toHaveBeenCalledTimes(1);
  });

  it("re-fetches policy after TTL expires", async () => {
    const policy1 = makePolicy({ restrictions_active: false });
    const policy2 = makePolicy({ restrictions_active: true });
    fetchPolicy.mockResolvedValueOnce(policy1).mockResolvedValueOnce(policy2);

    const cache = createPolicyCache(fetchPolicy, 60_000);

    const first = await cache.getPolicy();
    expect(first).toEqual(policy1);

    // Advance past TTL
    vi.advanceTimersByTime(61_000);

    const second = await cache.getPolicy();
    expect(second).toEqual(policy2);
    expect(fetchPolicy).toHaveBeenCalledTimes(2);
  });

  it("returns stale cache on fetch failure (fail-open)", async () => {
    const policy = makePolicy();
    fetchPolicy
      .mockResolvedValueOnce(policy)
      .mockRejectedValueOnce(new Error("network down"));

    const cache = createPolicyCache(fetchPolicy, 60_000);

    await cache.getPolicy(); // populates cache
    vi.advanceTimersByTime(61_000);

    const result = await cache.getPolicy();
    // Should return stale cached policy, not throw
    expect(result).toEqual(policy);
  });

  it("returns null when fetch fails with no prior cache", async () => {
    fetchPolicy.mockRejectedValue(new Error("network down"));

    const cache = createPolicyCache(fetchPolicy);
    const result = await cache.getPolicy();

    // cached is null, fetch failed, returns null
    expect(result).toBeNull();
  });

  it("deduplicates concurrent getPolicy calls (single fetch)", async () => {
    let resolvePromise: (v: PolicyResponse) => void;
    fetchPolicy.mockReturnValue(
      new Promise<PolicyResponse>((resolve) => {
        resolvePromise = resolve;
      }),
    );

    const cache = createPolicyCache(fetchPolicy);

    // Fire two concurrent calls
    const p1 = cache.getPolicy();
    const p2 = cache.getPolicy();

    // Resolve the single fetch
    resolvePromise!(makePolicy());

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual(r2);
    expect(fetchPolicy).toHaveBeenCalledTimes(1);
  });

  // -----------------------------------------------------------------------
  // invalidate
  // -----------------------------------------------------------------------

  it("invalidate clears cache, forcing a re-fetch", async () => {
    const policy1 = makePolicy({ restrictions_active: false });
    const policy2 = makePolicy({ restrictions_active: true });
    fetchPolicy.mockResolvedValueOnce(policy1).mockResolvedValueOnce(policy2);

    const cache = createPolicyCache(fetchPolicy, 60_000);

    await cache.getPolicy();
    expect(fetchPolicy).toHaveBeenCalledTimes(1);

    cache.invalidate();

    const result = await cache.getPolicy();
    expect(result).toEqual(policy2);
    expect(fetchPolicy).toHaveBeenCalledTimes(2);
  });

  // -----------------------------------------------------------------------
  // checkMandate
  // -----------------------------------------------------------------------

  describe("checkMandate", () => {
    it("allows when no cached policy (fail-open)", () => {
      const cache = createPolicyCache(fetchPolicy);
      // No getPolicy called yet => no cached policy
      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(true);
    });

    it("allows when no restrictions are set (null lists)", async () => {
      fetchPolicy.mockResolvedValue(makePolicy());
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(true);
    });

    it("denies model not in allowed_models", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ allowed_models: ["gpt-4o-mini", "gpt-3.5-turbo"] }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(false);
      expect(result.mandate).toBe("allowed_models");
      expect(result.requested).toBe("gpt-4o");
      expect(result.allowed_list).toEqual(["gpt-4o-mini", "gpt-3.5-turbo"]);
    });

    it("allows model in allowed_models", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ allowed_models: ["gpt-4o", "gpt-4o-mini"] }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(true);
    });

    it("denies provider not in allowed_providers", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ allowed_providers: ["anthropic"] }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(false);
      expect(result.mandate).toBe("allowed_providers");
      expect(result.requested).toBe("openai");
      expect(result.allowed_list).toEqual(["anthropic"]);
    });

    it("allows provider in allowed_providers", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ allowed_providers: ["openai", "anthropic"] }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(true);
    });

    it("checks providers before models (provider denied first)", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          allowed_providers: ["anthropic"],
          allowed_models: ["gpt-4o"],
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkMandate("openai", "gpt-4o");
      expect(result.allowed).toBe(false);
      expect(result.mandate).toBe("allowed_providers");
    });
  });

  // -----------------------------------------------------------------------
  // checkBudget
  // -----------------------------------------------------------------------

  describe("checkBudget", () => {
    it("allows when no cached policy (fail-open)", () => {
      const cache = createPolicyCache(fetchPolicy);
      const result = cache.checkBudget(1000);
      expect(result.allowed).toBe(true);
    });

    it("allows when no budget in policy", async () => {
      fetchPolicy.mockResolvedValue(makePolicy({ budget: null }));
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkBudget(5000);
      expect(result.allowed).toBe(true);
    });

    it("allows when sufficient budget remains", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          budget: {
            remaining_microdollars: 10_000,
            max_microdollars: 50_000,
            spend_microdollars: 40_000,
            period_end: null,
            entity_type: "api_key",
            entity_id: "key-1",
          },
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkBudget(5_000);
      expect(result.allowed).toBe(true);
      expect(result.remaining).toBe(10_000);
    });

    it("denies when estimate exceeds remaining budget", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          budget: {
            remaining_microdollars: 1_000,
            max_microdollars: 50_000,
            spend_microdollars: 49_000,
            period_end: null,
            entity_type: "api_key",
            entity_id: "key-1",
          },
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkBudget(5_000);
      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(1_000);
    });

    it("allows when estimate exactly equals remaining", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          budget: {
            remaining_microdollars: 5_000,
            max_microdollars: 50_000,
            spend_microdollars: 45_000,
            period_end: null,
            entity_type: "api_key",
            entity_id: "key-1",
          },
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      // estimate === remaining is NOT > remaining, so allowed
      const result = cache.checkBudget(5_000);
      expect(result.allowed).toBe(true);
    });

    it("checkBudget returns entity details when denied", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          budget: {
            remaining_microdollars: 500,
            max_microdollars: 10_000_000,
            spend_microdollars: 9_999_500,
            period_end: null,
            entity_type: "api_key",
            entity_id: "key-abc-123",
          },
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkBudget(1_000);
      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(500);
      expect(result.entityType).toBe("api_key");
      expect(result.entityId).toBe("key-abc-123");
      expect(result.limit).toBe(10_000_000);
      expect(result.spend).toBe(9_999_500);
    });

    it("checkBudget omits entity details when allowed", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({
          budget: {
            remaining_microdollars: 100_000,
            max_microdollars: 500_000,
            spend_microdollars: 400_000,
            period_end: null,
            entity_type: "api_key",
            entity_id: "key-xyz",
          },
        }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      const result = cache.checkBudget(5_000);
      expect(result.allowed).toBe(true);
      expect(result.remaining).toBe(100_000);
      // Entity details should NOT be present on allowed results
      expect(result.entityType).toBeUndefined();
      expect(result.entityId).toBeUndefined();
      expect(result.limit).toBeUndefined();
      expect(result.spend).toBeUndefined();
    });
  });

  // -----------------------------------------------------------------------
  // getSessionLimit
  // -----------------------------------------------------------------------

  describe("getSessionLimit", () => {
    it("returns null when no cached policy", () => {
      const cache = createPolicyCache(fetchPolicy);
      expect(cache.getSessionLimit()).toBeNull();
    });

    it("returns null when policy has no session_limit_microdollars", async () => {
      fetchPolicy.mockResolvedValue(makePolicy());
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      expect(cache.getSessionLimit()).toBeNull();
    });

    it("returns the value when present", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ session_limit_microdollars: 500_000 }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      expect(cache.getSessionLimit()).toBe(500_000);
    });

    it("returns null after invalidate()", async () => {
      fetchPolicy.mockResolvedValue(
        makePolicy({ session_limit_microdollars: 500_000 }),
      );
      const cache = createPolicyCache(fetchPolicy);
      await cache.getPolicy();

      expect(cache.getSessionLimit()).toBe(500_000);
      cache.invalidate();
      expect(cache.getSessionLimit()).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // onError callback (Codex finding #5)
  // -------------------------------------------------------------------------

  describe("onError callback (Codex finding #5)", () => {
    it("calls onError when fetch fails and no stale cache exists", async () => {
      const err = new Error("network down");
      fetchPolicy.mockRejectedValue(err);
      const onError = vi.fn();

      const cache = createPolicyCache(fetchPolicy, 1000, onError);
      const result = await cache.getPolicy();

      expect(result).toBeNull();
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledWith(err);
    });

    it("calls onError when fetch fails but returns stale cache", async () => {
      fetchPolicy
        .mockResolvedValueOnce(makePolicy({ session_limit_microdollars: 100 }))
        .mockRejectedValueOnce(new Error("timeout"));
      const onError = vi.fn();

      const cache = createPolicyCache(fetchPolicy, 0, onError); // TTL=0 to force refetch
      await cache.getPolicy(); // populate cache
      const result = await cache.getPolicy(); // should fail + return stale

      expect(result).not.toBeNull();
      expect(result!.session_limit_microdollars).toBe(100);
      expect(onError).toHaveBeenCalledTimes(1);
    });

    it("does not call onError on successful fetch", async () => {
      fetchPolicy.mockResolvedValue(makePolicy());
      const onError = vi.fn();

      const cache = createPolicyCache(fetchPolicy, 1000, onError);
      await cache.getPolicy();

      expect(onError).not.toHaveBeenCalled();
    });

    it("coerces non-Error rejections to Error", async () => {
      fetchPolicy.mockRejectedValue("string error");
      const onError = vi.fn();

      const cache = createPolicyCache(fetchPolicy, 1000, onError);
      await cache.getPolicy();

      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0][0]).toBeInstanceOf(Error);
      expect(onError.mock.calls[0][0].message).toBe("string error");
    });
  });
});
