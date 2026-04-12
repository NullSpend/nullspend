import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  suggestToolCost,
  EventBatcher,
  BudgetClient,
  CostTracker,
  ToolCostRegistry,
} from "./cost-tracker.js";

// ---------------------------------------------------------------------------
// suggestToolCost
// ---------------------------------------------------------------------------

describe("suggestToolCost", () => {
  it("returns TIER_READ (10000) when no annotations", () => {
    const cost = suggestToolCost(undefined);
    expect(cost).toBe(10_000);
  });

  it("returns TIER_FREE (0) for readOnly + not openWorld", () => {
    const cost = suggestToolCost({ readOnlyHint: true, openWorldHint: false });
    expect(cost).toBe(0);
  });

  it("returns TIER_READ (10000) for readOnly with openWorldHint undefined (spec defaults to true)", () => {
    const cost = suggestToolCost({ readOnlyHint: true });
    expect(cost).toBe(10_000);
  });

  it("returns TIER_FREE (0) for readOnly with openWorldHint explicitly false", () => {
    const cost = suggestToolCost({ readOnlyHint: true, openWorldHint: false });
    expect(cost).toBe(0);
  });

  it("returns TIER_WRITE (100000) for destructive + openWorld", () => {
    const cost = suggestToolCost({ destructiveHint: true, openWorldHint: true });
    expect(cost).toBe(100_000);
  });

  it("returns TIER_READ (10000) for openWorld + not destructive", () => {
    const cost = suggestToolCost({ openWorldHint: true });
    expect(cost).toBe(10_000);
  });

  it("returns TIER_READ (10000) for destructive without openWorld", () => {
    const cost = suggestToolCost({ destructiveHint: true });
    expect(cost).toBe(10_000);
  });

  it("returns TIER_READ for empty annotations object", () => {
    const cost = suggestToolCost({});
    expect(cost).toBe(10_000);
  });

  it("returns TIER_READ for readOnly + openWorld: true (read-only API call)", () => {
    const cost = suggestToolCost({ readOnlyHint: true, openWorldHint: true });
    expect(cost).toBe(10_000);
  });

  it("returns TIER_READ for readOnly: false without explicit destructive/openWorld", () => {
    const cost = suggestToolCost({ readOnlyHint: false });
    expect(cost).toBe(10_000);
  });

  it("returns TIER_WRITE only when destructive AND openWorld are both explicit", () => {
    const cost = suggestToolCost({ destructiveHint: true, openWorldHint: true });
    expect(cost).toBe(100_000);
    const cost2 = suggestToolCost({ destructiveHint: true });
    expect(cost2).toBe(10_000);
  });
});

// ---------------------------------------------------------------------------
// EventBatcher
// ---------------------------------------------------------------------------

describe("EventBatcher", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  function makeBatcher(opts?: { batchSize?: number; flushIntervalMs?: number }) {
    return new EventBatcher({
      backendUrl: "http://localhost:8787",
      apiKey: "ns_live_sk_test0001",
      batchSize: opts?.batchSize ?? 5,
      flushIntervalMs: opts?.flushIntervalMs ?? 60_000, // high to avoid auto-flush in tests
    });
  }

  function makeEvent(name = "tool_a") {
    return {
      toolName: name,
      serverName: "test-server",
      durationMs: 100,
      costMicrodollars: 10_000,
      status: "success" as const,
    };
  }

  it("does not flush before batchSize is reached", () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 5 });
    batcher.push(makeEvent());
    batcher.push(makeEvent());

    expect(mockFetch).not.toHaveBeenCalled();

    // cleanup
    batcher.shutdown();
  });

  it("flushes when batchSize is reached", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ accepted: 5 })));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 3 });
    batcher.push(makeEvent("t1"));
    batcher.push(makeEvent("t2"));
    batcher.push(makeEvent("t3"));

    // Allow flush to complete
    await new Promise((r) => setTimeout(r, 10));

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8787/v1/mcp/events");
    expect(opts.method).toBe("POST");

    const body = JSON.parse(opts.body);
    expect(body.events).toHaveLength(3);

    await batcher.shutdown();
  });

  it("sends correct auth headers", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 1 });
    batcher.push(makeEvent());

    await new Promise((r) => setTimeout(r, 10));

    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers["x-nullspend-key"]).toBe("ns_live_sk_test0001");

    await batcher.shutdown();
  });

  it("drops oldest when queue exceeds MAX_QUEUE_SIZE", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 5000 }); // high batchSize to prevent auto flush

    // Push 4097 events (MAX_QUEUE_SIZE is 4096)
    for (let i = 0; i < 4097; i++) {
      batcher.push(makeEvent(`tool_${i}`));
    }

    // Manually trigger flush
    batcher.flush();
    await new Promise((r) => setTimeout(r, 10));

    // The first event (tool_0) should have been dropped
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.events[0].toolName).toBe("tool_1");

    await batcher.shutdown();
  });

  it("shutdown flushes all remaining events", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 100 }); // high batchSize
    batcher.push(makeEvent("t1"));
    batcher.push(makeEvent("t2"));

    expect(mockFetch).not.toHaveBeenCalled();

    await batcher.shutdown();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.events).toHaveLength(2);
  });

  it("handles fetch errors gracefully", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network error"));

    const batcher = makeBatcher({ batchSize: 1 });
    batcher.push(makeEvent());

    // Should not throw
    await batcher.shutdown();
  });

  it("handles non-ok response gracefully", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response("err", { status: 500 }));

    const batcher = makeBatcher({ batchSize: 1 });
    batcher.push(makeEvent());

    await batcher.shutdown();
    // No throw expected
  });

  it("re-queues failed batches for retry on next flush", async () => {
    let callCount = 0;
    const mockFetch = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.reject(new Error("network error"));
      }
      return Promise.resolve(new Response("ok"));
    });
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 2, flushIntervalMs: 60_000 });
    batcher.push(makeEvent("t1"));
    batcher.push(makeEvent("t2")); // triggers flush → fails → re-queued

    await new Promise((r) => setTimeout(r, 20));
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Manually flush again — should retry the re-queued batch
    batcher.flush();
    await new Promise((r) => setTimeout(r, 20));

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const body = JSON.parse(mockFetch.mock.calls[1][1].body);
    expect(body.events).toHaveLength(2);
    expect(body.events[0].toolName).toBe("t1");

    await batcher.shutdown();
  });

  it("re-queues on non-ok response", async () => {
    let callCount = 0;
    const mockFetch = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve(new Response("err", { status: 500 }));
      }
      return Promise.resolve(new Response("ok"));
    });
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 1, flushIntervalMs: 60_000 });
    batcher.push(makeEvent("t1")); // triggers flush → 500 → re-queued

    await new Promise((r) => setTimeout(r, 20));

    // Retry via manual flush
    batcher.flush();
    await new Promise((r) => setTimeout(r, 20));

    expect(mockFetch).toHaveBeenCalledTimes(2);

    await batcher.shutdown();
  });

  it("does not re-queue during shutdown (isRetry=true)", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("down"));

    const batcher = makeBatcher({ batchSize: 100 });
    batcher.push(makeEvent("t1"));

    // shutdown will try to send and fail, but should NOT re-queue
    await batcher.shutdown();

    // Queue should be empty (not re-queued infinitely)
    // Verify by checking fetch was called exactly once for the shutdown flush
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it("rejects push() calls during shutdown", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher({ batchSize: 100 });
    batcher.push(makeEvent("before-shutdown"));

    // Start shutdown (which will flush the existing event)
    const shutdownPromise = batcher.shutdown();

    // Try to push during shutdown — should be silently rejected
    batcher.push(makeEvent("during-shutdown"));

    await shutdownPromise;

    // Only the pre-shutdown event should have been sent
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.events).toHaveLength(1);
    expect(body.events[0].toolName).toBe("before-shutdown");
  });

  it("flush is a no-op when queue is empty", () => {
    const mockFetch = vi.fn();
    globalThis.fetch = mockFetch;

    const batcher = makeBatcher();
    batcher.flush();

    expect(mockFetch).not.toHaveBeenCalled();
    batcher.shutdown();
  });
});

// ---------------------------------------------------------------------------
// BudgetClient
// ---------------------------------------------------------------------------

describe("BudgetClient", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  function makeClient() {
    return new BudgetClient({
      backendUrl: "http://localhost:8787",
      apiKey: "ns_live_sk_test0001",
    });
  }

  it("returns allowed: true on successful budget check", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ allowed: true, reservationId: "rsv-1" })),
    );

    const client = makeClient();
    const result = await client.check("run_query", "supabase", 10_000);

    expect(result.allowed).toBe(true);
    expect(result.reservationId).toBe("rsv-1");
  });

  it("sends correct request to budget check endpoint", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ allowed: true })),
    );
    globalThis.fetch = mockFetch;

    const client = makeClient();
    await client.check("run_query", "supabase", 10_000);

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8787/v1/mcp/budget/check",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "x-nullspend-key": "ns_live_sk_test0001",
        }),
      }),
    );

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({
      toolName: "run_query",
      serverName: "supabase",
      estimateMicrodollars: 10_000,
    });
  });

  it("returns denied response when proxy emits 429 with budget_exceeded envelope", async () => {
    // Post-2026-04-08 migration: proxy emits 429 + { error: { code, message, details } }
    // envelope shape. Status 429, NOT 200.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: "Request blocked: estimated cost exceeds remaining budget.",
          details: {
            entity_type: "user",
            entity_id: "user-1",
            budget_limit_microdollars: 1_000_000,
            budget_spend_microdollars: 999_995,
            estimated_cost_microdollars: 100_000,
          },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("expensive", "server", 100_000);

    expect(result.allowed).toBe(false);
    expect(result.denied).toBe(true);
    expect(result.code).toBe("budget_exceeded");
    // remaining = max(0, limit - spend) = 1M - 999.995k = 5 microdollars
    expect(result.remaining).toBe(5);
  });

  it("surfaces upgrade_url from envelope on budget_exceeded denial", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: "Budget exceeded",
          upgrade_url: "https://acme.com/billing?customer=c1",
          details: {
            budget_limit_microdollars: 1_000_000,
            budget_spend_microdollars: 1_000_000,
          },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100_000);

    expect(result.allowed).toBe(false);
    expect(result.code).toBe("budget_exceeded");
    expect(result.upgradeUrl).toBe("https://acme.com/billing?customer=c1");
  });

  it("surfaces upgrade_url on customer_budget_exceeded denial", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "customer_budget_exceeded",
          message: "Customer budget exceeded",
          upgrade_url: "https://acme.com/upgrade?customer=acme-corp",
          details: {
            customer_id: "acme-corp",
            budget_limit_microdollars: 500_000,
            budget_spend_microdollars: 500_000,
          },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("expensive", "server", 100_000);

    expect(result.code).toBe("customer_budget_exceeded");
    expect(result.upgradeUrl).toBe("https://acme.com/upgrade?customer=acme-corp");
    expect(result.remaining).toBe(0);
  });

  it("429 envelope without upgrade_url returns undefined upgradeUrl", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: "Budget exceeded",
          details: {
            budget_limit_microdollars: 1_000,
            budget_spend_microdollars: 1_000,
          },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100);
    expect(result.upgradeUrl).toBeUndefined();
  });

  it("429 WITH X-NullSpend-Denied + malformed body returns a denial (defensive parser)", async () => {
    // Header gates the denial parser. Body is malformed — parser falls back
    // to safe defaults WITHOUT throwing / tripping the circuit breaker.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("not json", { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100);
    expect(result.allowed).toBe(false);
    expect(result.denied).toBe(true);
    // Parser couldn't extract code/remaining/upgradeUrl — all undefined
    expect(result.code).toBeUndefined();
    expect(result.remaining).toBeUndefined();
    expect(result.upgradeUrl).toBeUndefined();
  });

  it("E2: 429 WITHOUT X-NullSpend-Denied header is NOT treated as a budget denial", async () => {
    // Rate-limit 429 from Cloudflare IP limiter or upstream gateway never
    // carries X-NullSpend-Denied. Pre-fix, the parser mis-classified these
    // as budget denials. Post-fix: falls through to the non-ok / throw
    // path and hits the fail-open circuit-breaker fallback.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("Rate limited", { status: 429 }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100);

    // Fail-open — rate limit shouldn't block the user
    expect(result.allowed).toBe(true);
    expect(result.denied).toBeUndefined();
    expect(result.code).toBeUndefined();
  });

  it("E2: 429 from Cloudflare with CF-Ray header (no denial header) falls through", async () => {
    // Real-world: Cloudflare responses carry CF-Ray but not X-NullSpend-Denied.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: "rate limited" }), {
        status: 429,
        headers: { "CF-Ray": "abc123", "Content-Type": "application/json" },
      }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100);
    expect(result.allowed).toBe(true); // fail-open
  });

  it("429 does NOT trip the circuit breaker (valid denial ≠ failure)", async () => {
    // Five consecutive 429s should NOT open the circuit breaker — denials
    // are normal, only actual failures (network errors, 5xx) count.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: { code: "budget_exceeded", message: "denied" },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    for (let i = 0; i < 6; i++) {
      await client.check("tool", "server", 100_000);
    }

    // Still hitting fetch — circuit not open
    expect(globalThis.fetch).toHaveBeenCalledTimes(6);
  });

  it("velocity_exceeded 429 returns code but no limit/spend math", async () => {
    // Velocity details use camelCase + don't have budget_limit_microdollars,
    // so `remaining` should be undefined (no false zero).
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "velocity_exceeded",
          message: "Velocity limit",
          details: {
            limitMicrodollars: 10_000_000,
            windowSeconds: 60,
            currentMicrodollars: 12_500_000,
          },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 100);
    expect(result.allowed).toBe(false);
    expect(result.code).toBe("velocity_exceeded");
    expect(result.remaining).toBeUndefined();
    expect(result.upgradeUrl).toBeUndefined();
  });

  it("fails open on fetch error", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network down"));

    const client = makeClient();
    const result = await client.check("tool", "server", 10_000);

    expect(result.allowed).toBe(true);
  });

  it("fails open on non-ok response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("error", { status: 500 }),
    );

    const client = makeClient();
    const result = await client.check("tool", "server", 10_000);

    expect(result.allowed).toBe(true);
  });

  it("opens circuit breaker after 5 consecutive failures", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fail"));

    const client = makeClient();

    for (let i = 0; i < 5; i++) {
      await client.check("tool", "server", 10_000);
    }

    // 6th call should not hit fetch (circuit open)
    const mockFetch = vi.fn();
    globalThis.fetch = mockFetch;

    const result = await client.check("tool", "server", 10_000);
    expect(result.allowed).toBe(true);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("uses cached response during circuit breaker cooldown", async () => {
    // First call succeeds
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ allowed: true, reservationId: "rsv-cached" })),
    );

    const client = makeClient();
    await client.check("tool", "server", 10_000);

    // Next 5 calls fail
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fail"));
    for (let i = 0; i < 5; i++) {
      await client.check("tool", "server", 10_000);
    }

    // Circuit is now open; should return cached response
    const result = await client.check("tool", "server", 10_000);
    expect(result.allowed).toBe(true);
    expect(result.reservationId).toBe("rsv-cached");
  });

  it("does not cache denied responses for fail-open fallback", async () => {
    // First call: 429 envelope denial (post-migration shape)
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "budget_exceeded",
          message: "denied",
          details: { budget_limit_microdollars: 0, budget_spend_microdollars: 0 },
        },
      }), { status: 429, headers: { "X-NullSpend-Denied": "1" } }),
    );

    const client = makeClient();
    const denied = await client.check("tool", "server", 100_000);
    expect(denied.allowed).toBe(false);

    // Now 5 failures to trip circuit breaker. 429 doesn't count as a
    // failure (post-migration) so we need real network errors here.
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fail"));
    for (let i = 0; i < 5; i++) {
      await client.check("tool", "server", 10_000);
    }

    // Circuit is open — fallback should be fail-OPEN, not return cached denial
    const fallbackResult = await client.check("tool", "server", 10_000);
    expect(fallbackResult.allowed).toBe(true); // fail-open, not cached denial
  });

  it("caches allowed responses for fallback", async () => {
    // First call: budget allowed
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ allowed: true, reservationId: "rsv-cached" })),
    );

    const client = makeClient();
    await client.check("tool", "server", 10_000);

    // Single failure (circuit not open)
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("blip"));
    const result = await client.check("tool", "server", 10_000);

    // Should use cached allowed response
    expect(result.allowed).toBe(true);
    expect(result.reservationId).toBe("rsv-cached");
  });

  it("resets consecutive failures on success", async () => {
    const mockFetch = vi.fn();
    globalThis.fetch = mockFetch;

    const client = makeClient();

    // 4 failures
    mockFetch.mockRejectedValue(new Error("fail"));
    for (let i = 0; i < 4; i++) {
      await client.check("tool", "server", 10_000);
    }

    // 1 success
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ allowed: true })),
    );
    await client.check("tool", "server", 10_000);

    // 4 more failures — should not trip circuit (counter reset)
    mockFetch.mockRejectedValue(new Error("fail"));
    for (let i = 0; i < 4; i++) {
      await client.check("tool", "server", 10_000);
    }

    // Circuit should still be closed (only 4 consecutive failures)
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ allowed: true })),
    );
    const result = await client.check("tool", "server", 10_000);
    expect(result.allowed).toBe(true);
    expect(mockFetch).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// CostTracker (facade)
// ---------------------------------------------------------------------------

describe("CostTracker", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  function makeTracker(overrides?: Partial<import("./cost-tracker.js").CostTrackerConfig>) {
    return new CostTracker({
      backendUrl: "http://localhost:8787",
      apiKey: "ns_live_sk_test0001",
      serverName: "test-server",
      budgetEnforcementEnabled: true,
      toolCostOverrides: {},
      ...overrides,
    });
  }

  it("resolveToolCost returns override when present", () => {
    const tracker = makeTracker({ toolCostOverrides: { special: 42 } });
    expect(tracker.resolveToolCost("special")).toBe(42);
    tracker.shutdown();
  });

  it("resolveToolCost returns 0 for unknown tool (unpriced)", () => {
    const tracker = makeTracker();
    expect(tracker.resolveToolCost("unknown")).toBe(0);
    tracker.shutdown();
  });

  it("resolveToolCost uses registry when no env override exists", async () => {
    const tracker = makeTracker({ serverName: "supabase" });
    const registry = new ToolCostRegistry({
      nullspendUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      serverName: "supabase",
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        data: [
          { serverName: "supabase", toolName: "execute_sql", costMicrodollars: 50_000 },
        ],
      })),
    );
    await registry.fetchCosts();
    tracker.setRegistry(registry);

    expect(tracker.resolveToolCost("execute_sql")).toBe(50_000);
    expect(tracker.resolveToolCost("unknown_tool")).toBe(0);

    await tracker.shutdown();
  });

  it("resolveToolCost env override takes precedence over registry", async () => {
    const tracker = makeTracker({
      serverName: "supabase",
      toolCostOverrides: { execute_sql: 999 },
    });
    const registry = new ToolCostRegistry({
      nullspendUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      serverName: "supabase",
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        data: [
          { serverName: "supabase", toolName: "execute_sql", costMicrodollars: 50_000 },
        ],
      })),
    );
    await registry.fetchCosts();
    tracker.setRegistry(registry);

    expect(tracker.resolveToolCost("execute_sql")).toBe(999);

    await tracker.shutdown();
  });

  it("resolveToolCost registry takes precedence over $0 default", async () => {
    const tracker = makeTracker({ serverName: "supabase" });
    const registry = new ToolCostRegistry({
      nullspendUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      serverName: "supabase",
    });

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        data: [
          { serverName: "supabase", toolName: "read_file", costMicrodollars: 77_000 },
        ],
      })),
    );
    await registry.fetchCosts();
    tracker.setRegistry(registry);

    expect(tracker.resolveToolCost("read_file")).toBe(77_000);

    await tracker.shutdown();
  });

  it("checkBudget returns allowed when budget enforcement is disabled", async () => {
    globalThis.fetch = vi.fn();

    const tracker = makeTracker({ budgetEnforcementEnabled: false });
    const result = await tracker.checkBudget("tool", 10_000);

    expect(result.allowed).toBe(true);
    expect(globalThis.fetch).not.toHaveBeenCalled();

    await tracker.shutdown();
  });

  it("checkBudget calls BudgetClient when enforcement is enabled", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ allowed: true, reservationId: "rsv-1" })),
    );

    const tracker = makeTracker();
    const result = await tracker.checkBudget("tool", 10_000);

    expect(result.allowed).toBe(true);
    expect(result.reservationId).toBe("rsv-1");
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);

    await tracker.shutdown();
  });

  it("reportEvent pushes to batcher", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const tracker = makeTracker();
    tracker.reportEvent({
      toolName: "tool_a",
      serverName: "test-server",
      durationMs: 100,
      costMicrodollars: 10_000,
      status: "success",
    });

    await tracker.shutdown();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.events[0].toolName).toBe("tool_a");
  });

  it("shutdown flushes batcher", async () => {
    const mockFetch = vi.fn().mockResolvedValue(new Response("ok"));
    globalThis.fetch = mockFetch;

    const tracker = makeTracker();
    tracker.reportEvent({
      toolName: "t",
      serverName: "s",
      durationMs: 50,
      costMicrodollars: 5000,
      status: "success",
    });

    await tracker.shutdown();

    expect(mockFetch).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// ToolCostRegistry
// ---------------------------------------------------------------------------

describe("ToolCostRegistry", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  function makeRegistry(opts?: { serverName?: string }) {
    return new ToolCostRegistry({
      nullspendUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      serverName: opts?.serverName ?? "test-server",
    });
  }

  // --- fetchCosts ---

  describe("fetchCosts", () => {
    it("populates cost map on successful fetch", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          data: [
            { serverName: "test-server", toolName: "run_query", costMicrodollars: 50_000 },
            { serverName: "test-server", toolName: "list_tables", costMicrodollars: 10_000 },
          ],
        })),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("run_query")).toBe(50_000);
      expect(registry.getCost("list_tables")).toBe(10_000);
    });

    it("sends correct auth header", async () => {
      const mockFetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ data: [] })),
      );
      globalThis.fetch = mockFetch;

      const registry = makeRegistry();
      await registry.fetchCosts();

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers["x-nullspend-key"]).toBe("ns_live_sk_test0001");
    });

    it("logs actionable message on 401 and falls back gracefully", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response("Unauthorized", { status: 401 }),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      // Should not throw, falls back to empty map
      expect(registry.getCost("anything")).toBeUndefined();

      // Check the 401 log message mentions Settings page
      const logCalls = vi.mocked(process.stderr.write).mock.calls;
      const logText = logCalls.map((c) => String(c[0])).join("");
      expect(logText).toContain("API key not recognized");
      expect(logText).toContain("/app/settings");
    });

    it("falls back gracefully on non-ok response", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response("error", { status: 500 }),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("anything")).toBeUndefined();
    });

    it("falls back gracefully on network error", async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error("ECONNREFUSED"));

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("anything")).toBeUndefined();
    });

    it("handles response with missing data field gracefully", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "something went wrong" })),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("anything")).toBeUndefined();
    });

    it("handles response with null data gracefully", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ data: null })),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("anything")).toBeUndefined();
    });

    it("handles response with non-JSON body gracefully", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response("<html>Gateway Timeout</html>", {
          status: 200,
          headers: { "content-type": "text/html" },
        }),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("anything")).toBeUndefined();
    });

    it("skips rows with missing or invalid fields", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          data: [
            { serverName: "test-server", toolName: "valid_tool", costMicrodollars: 10_000 },
            { serverName: "test-server", toolName: null, costMicrodollars: 20_000 },
            { serverName: "test-server", costMicrodollars: 30_000 },
            { serverName: "test-server", toolName: "no_cost" },
          ],
        })),
      );

      const registry = makeRegistry();
      await registry.fetchCosts();

      expect(registry.getCost("valid_tool")).toBe(10_000);
      // Invalid rows are silently skipped
      expect(registry.getCost("null")).toBeUndefined();
      expect(registry.getCost("no_cost")).toBeUndefined();
    });

    it("stores costs from multiple servers but getCost scopes to own server", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          data: [
            { serverName: "test-server", toolName: "tool_a", costMicrodollars: 10_000 },
            { serverName: "other-server", toolName: "tool_b", costMicrodollars: 20_000 },
          ],
        })),
      );

      const registry = makeRegistry({ serverName: "test-server" });
      await registry.fetchCosts();

      expect(registry.getCost("tool_a")).toBe(10_000);
      // tool_b is on other-server, not accessible via getCost
      expect(registry.getCost("tool_b")).toBeUndefined();
    });
  });

  // --- discoverTools ---

  describe("discoverTools", () => {
    it("sends correct request payload", async () => {
      const mockFetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ registered: 2 }), { status: 201 }),
      );
      globalThis.fetch = mockFetch;

      const registry = makeRegistry({ serverName: "supabase" });
      await registry.discoverTools([
        { name: "execute_sql", description: "Run SQL", annotations: { openWorldHint: true }, tierCost: 0, suggestedCost: 10_000 },
        { name: "list_tables", description: null, annotations: null, tierCost: 0, suggestedCost: 10_000 },
      ]);

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, opts] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:3000/api/tool-costs/discover");
      expect(opts.method).toBe("POST");
      expect(opts.headers["x-nullspend-key"]).toBe("ns_live_sk_test0001");

      const body = JSON.parse(opts.body);
      expect(body.serverName).toBe("supabase");
      expect(body.tools).toHaveLength(2);
      expect(body.tools[0].name).toBe("execute_sql");
    });

    it("skips call when tools array is empty", async () => {
      const mockFetch = vi.fn();
      globalThis.fetch = mockFetch;

      const registry = makeRegistry();
      await registry.discoverTools([]);

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("logs actionable message on 401", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response("Unauthorized", { status: 401 }),
      );

      const registry = makeRegistry();
      await registry.discoverTools([
        { name: "tool_a", tierCost: 0, suggestedCost: 10_000 },
      ]);

      const logCalls = vi.mocked(process.stderr.write).mock.calls;
      const logText = logCalls.map((c) => String(c[0])).join("");
      expect(logText).toContain("API key not recognized");
    });

    it("falls back gracefully on network error", async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error("ECONNREFUSED"));

      const registry = makeRegistry();
      // Should not throw
      await registry.discoverTools([
        { name: "tool_a", tierCost: 0, suggestedCost: 10_000 },
      ]);
    });

    it("chunks batches larger than 500 tools", async () => {
      const mockFetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ registered: 500 }), { status: 201 }),
      );
      globalThis.fetch = mockFetch;

      const registry = makeRegistry();
      const tools = Array.from({ length: 750 }, (_, i) => ({
        name: `tool_${i}`,
        tierCost: 0,
        suggestedCost: 10_000,
      }));

      await registry.discoverTools(tools);

      // Should make 2 calls: 500 + 250
      expect(mockFetch).toHaveBeenCalledTimes(2);

      const body1 = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body1.tools).toHaveLength(500);

      const body2 = JSON.parse(mockFetch.mock.calls[1][1].body);
      expect(body2.tools).toHaveLength(250);
    });

    it("stops chunking on first failure", async () => {
      let callCount = 0;
      globalThis.fetch = vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve(new Response(JSON.stringify({ registered: 500 }), { status: 201 }));
        }
        return Promise.resolve(new Response("error", { status: 500 }));
      });

      const registry = makeRegistry();
      const tools = Array.from({ length: 1000 }, (_, i) => ({
        name: `tool_${i}`,
        tierCost: 0,
        suggestedCost: 10_000,
      }));

      await registry.discoverTools(tools);

      // First chunk succeeds, second fails, no third attempt
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });
  });

  // --- getCost ---

  describe("getCost", () => {
    it("returns undefined for unknown tool", () => {
      const registry = makeRegistry();
      expect(registry.getCost("nonexistent")).toBeUndefined();
    });

    it("uses serverName/toolName key format", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          data: [
            { serverName: "my-server", toolName: "my-tool", costMicrodollars: 42_000 },
          ],
        })),
      );

      const registry = makeRegistry({ serverName: "my-server" });
      await registry.fetchCosts();

      expect(registry.getCost("my-tool")).toBe(42_000);
    });
  });
});
