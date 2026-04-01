import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { withNullSpend, withNullSpendAsync, _resetPolicyCache } from "./with-nullspend.js";

describe("withNullSpend", () => {
  const BASE = { apiKey: "ns_test_sk_abc123" } as const;

  // --- URL ---

  it("sets ANTHROPIC_BASE_URL to default proxy URL", () => {
    const result = withNullSpend({ ...BASE });
    expect(result.env?.ANTHROPIC_BASE_URL).toBe(
      "https://proxy.nullspend.com",
    );
  });

  it("uses custom proxyUrl when provided", () => {
    const result = withNullSpend({
      ...BASE,
      proxyUrl: "https://custom.proxy.dev",
    });
    expect(result.env?.ANTHROPIC_BASE_URL).toBe("https://custom.proxy.dev");
  });

  // --- Headers ---

  it("sets x-nullspend-key header", () => {
    const result = withNullSpend({ ...BASE });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-key: ns_test_sk_abc123",
    );
  });

  it("sets x-nullspend-session header from budgetSessionId", () => {
    const result = withNullSpend({
      ...BASE,
      budgetSessionId: "sess-001",
    });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-session: sess-001",
    );
  });

  it("auto-generates session ID when budgetSessionId is not provided", () => {
    const result = withNullSpend(BASE);
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).toContain("x-nullspend-session: ses_");
  });

  it("does not auto-generate session ID when autoSession is false", () => {
    const result = withNullSpend({ ...BASE, autoSession: false });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).not.toContain("x-nullspend-session");
  });

  it("prefers explicit budgetSessionId over auto-generated", () => {
    const result = withNullSpend({ ...BASE, budgetSessionId: "my-session" });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).toContain("x-nullspend-session: my-session");
    expect(headers).not.toContain("ses_");
  });

  it("sets x-nullspend-tags header as JSON", () => {
    const result = withNullSpend({
      ...BASE,
      tags: { team: "platform", env: "staging" },
    });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).toContain("x-nullspend-tags:");
    const match = headers.match(/x-nullspend-tags: (.+)/);
    expect(match).not.toBeNull();
    expect(JSON.parse(match![1])).toEqual({ team: "platform", env: "staging" });
  });

  it("sets x-nullspend-trace-id header", () => {
    const result = withNullSpend({
      ...BASE,
      traceId: "abcdef0123456789abcdef0123456789",
    });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-trace-id: abcdef0123456789abcdef0123456789",
    );
  });

  it("sets x-nullspend-action-id header", () => {
    const result = withNullSpend({
      ...BASE,
      actionId: "ns_act_550e8400-e29b-41d4-a716-446655440000",
    });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-action-id: ns_act_550e8400-e29b-41d4-a716-446655440000",
    );
  });

  it("omits tags header when tags is empty object", () => {
    const result = withNullSpend({ ...BASE, tags: {} });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).not.toContain(
      "x-nullspend-tags",
    );
  });

  it("joins headers with newline delimiter", () => {
    const result = withNullSpend({
      ...BASE,
      budgetSessionId: "sess-001",
      traceId: "abcdef0123456789abcdef0123456789",
    });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    const lines = headers.split("\n");
    expect(lines.length).toBe(3);
    expect(lines[0]).toMatch(/^x-nullspend-key:/);
    expect(lines[1]).toMatch(/^x-nullspend-session:/);
    expect(lines[2]).toMatch(/^x-nullspend-trace-id:/);
  });

  it("merges existing ANTHROPIC_CUSTOM_HEADERS", () => {
    const result = withNullSpend({
      ...BASE,
      env: { ANTHROPIC_CUSTOM_HEADERS: "x-custom: existing-value" },
    });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).toContain("x-custom: existing-value");
    expect(headers).toContain("x-nullspend-key: ns_test_sk_abc123");
    // Existing headers come first
    expect(headers.indexOf("x-custom")).toBeLessThan(
      headers.indexOf("x-nullspend-key"),
    );
  });

  // --- Passthrough ---

  it("preserves SDK options (maxTurns, allowedTools, model)", () => {
    const result = withNullSpend({
      ...BASE,
      maxTurns: 5,
      allowedTools: ["Read", "Write"],
      model: "claude-sonnet-4-6",
    });
    expect(result.maxTurns).toBe(5);
    expect(result.allowedTools).toEqual(["Read", "Write"]);
    expect(result.model).toBe("claude-sonnet-4-6");
  });

  it("preserves SDK sessionId separately from budgetSessionId", () => {
    const result = withNullSpend({
      ...BASE,
      sessionId: "550e8400-e29b-41d4-a716-446655440000",
      budgetSessionId: "budget-sess-001",
    });
    expect(result.sessionId).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-session: budget-sess-001",
    );
    // SDK sessionId must not leak into headers
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).not.toContain(
      "550e8400-e29b-41d4-a716-446655440000",
    );
  });

  it("preserves existing env vars", () => {
    const result = withNullSpend({
      ...BASE,
      env: { MY_VAR: "hello", ANOTHER: "world" },
    });
    expect(result.env?.MY_VAR).toBe("hello");
    expect(result.env?.ANOTHER).toBe("world");
  });

  // --- Validation: apiKey ---

  it("throws on missing apiKey", () => {
    expect(() => withNullSpend({ apiKey: "" })).toThrow(
      "withNullSpend: apiKey is required",
    );
  });

  // --- Validation: tags ---

  it("throws on >10 tags", () => {
    const tags: Record<string, string> = {};
    for (let i = 0; i < 11; i++) tags[`key${i}`] = "val";
    expect(() => withNullSpend({ ...BASE, tags })).toThrow(
      "tags must have at most 10 keys",
    );
  });

  it("throws on long tag key", () => {
    const longKey = "k".repeat(65);
    expect(() => withNullSpend({ ...BASE, tags: { [longKey]: "v" } })).toThrow(
      `tag key "${longKey}" exceeds 64 chars`,
    );
  });

  it("throws on long tag value", () => {
    const longVal = "v".repeat(257);
    expect(() =>
      withNullSpend({ ...BASE, tags: { short: longVal } }),
    ).toThrow('tag value for "short" exceeds 256 chars');
  });

  it("throws on tag key with invalid characters", () => {
    expect(() =>
      withNullSpend({ ...BASE, tags: { "my.team": "platform" } }),
    ).toThrow('tag key "my.team" must match [a-zA-Z0-9_-]+');
  });

  it("throws on tag key with spaces", () => {
    expect(() =>
      withNullSpend({ ...BASE, tags: { "cost center": "eng" } }),
    ).toThrow('tag key "cost center" must match [a-zA-Z0-9_-]+');
  });

  it("throws on empty tag key", () => {
    expect(() =>
      withNullSpend({ ...BASE, tags: { "": "val" } }),
    ).toThrow('tag key "" must match [a-zA-Z0-9_-]+');
  });

  it("accepts tag keys with underscores and hyphens", () => {
    const result = withNullSpend({
      ...BASE,
      tags: { "my-team": "platform", cost_center: "eng" },
    });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    expect(headers).toContain("x-nullspend-tags:");
  });

  // --- Validation: traceId ---

  it("throws on invalid traceId (uppercase)", () => {
    expect(() =>
      withNullSpend({ ...BASE, traceId: "ABCDEF0123456789ABCDEF0123456789" }),
    ).toThrow("traceId must be a 32-char lowercase hex string");
  });

  it("throws on invalid traceId (wrong length)", () => {
    expect(() =>
      withNullSpend({ ...BASE, traceId: "abcdef" }),
    ).toThrow("traceId must be a 32-char lowercase hex string");
  });

  it("throws on invalid traceId (non-hex)", () => {
    expect(() =>
      withNullSpend({ ...BASE, traceId: "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" }),
    ).toThrow("traceId must be a 32-char lowercase hex string");
  });

  // --- Validation: actionId ---

  it("throws on actionId without ns_act_ prefix", () => {
    expect(() =>
      withNullSpend({ ...BASE, actionId: "act_123" }),
    ).toThrow("actionId must be in ns_act_<UUID> format");
  });

  it("throws on actionId with invalid UUID after prefix", () => {
    expect(() =>
      withNullSpend({ ...BASE, actionId: "ns_act_not-a-uuid" }),
    ).toThrow("actionId must be in ns_act_<UUID> format");
  });

  it("accepts valid actionId with ns_act_ prefix and UUID", () => {
    const result = withNullSpend({
      ...BASE,
      actionId: "ns_act_550e8400-e29b-41d4-a716-446655440000",
    });
    expect(result.env?.ANTHROPIC_CUSTOM_HEADERS).toContain(
      "x-nullspend-action-id: ns_act_550e8400-e29b-41d4-a716-446655440000",
    );
  });

  // --- Validation: newline injection ---

  it("throws on apiKey containing newline", () => {
    expect(() =>
      withNullSpend({ apiKey: "ns_test_sk_abc\nx-injected: evil" }),
    ).toThrow("apiKey must not contain newline characters");
  });

  it("throws on budgetSessionId containing carriage return", () => {
    expect(() =>
      withNullSpend({ ...BASE, budgetSessionId: "sess\r\ninjected" }),
    ).toThrow("budgetSessionId must not contain newline characters");
  });

  // --- Env ---

  it("spreads process.env when no env provided", () => {
    const result = withNullSpend({ ...BASE });
    expect(result.env?.ANTHROPIC_BASE_URL).toBe(
      "https://proxy.nullspend.com",
    );
    expect(result.env).toBeDefined();
  });

  it("merges process.env as base when partial env is provided", () => {
    // User-provided env should override process.env, not replace it.
    // The child process needs PATH, HOME, etc. to function.
    const result = withNullSpend({
      ...BASE,
      env: { MY_CUSTOM_VAR: "test" },
    });
    expect(result.env?.MY_CUSTOM_VAR).toBe("test");
    // process.env.PATH should still be present (child process needs it)
    expect(result.env?.PATH ?? result.env?.Path).toBeDefined();
  });

  it("user-provided env overrides process.env values", () => {
    const result = withNullSpend({
      ...BASE,
      env: { NODE_ENV: "custom-override" },
    });
    expect(result.env?.NODE_ENV).toBe("custom-override");
  });

  // --- Edge cases: tag values ---

  it("escapes newlines in tag values via JSON.stringify", () => {
    const result = withNullSpend({
      ...BASE,
      tags: { note: "line1\nline2" },
    });
    const headers = result.env?.ANTHROPIC_CUSTOM_HEADERS ?? "";
    // JSON.stringify escapes the newline so it doesn't split the header
    expect(headers.split("\n")).toHaveLength(3); // key header + session (auto) + tags header
    const match = headers.match(/x-nullspend-tags: (.+)/);
    expect(match).not.toBeNull();
    expect(JSON.parse(match![1])).toEqual({ note: "line1\nline2" });
  });

  // --- Edge cases: proxyUrl trailing slash ---

  it("preserves proxyUrl trailing slash as-is", () => {
    const result = withNullSpend({
      ...BASE,
      proxyUrl: "https://proxy.nullspend.com/",
    });
    expect(result.env?.ANTHROPIC_BASE_URL).toBe(
      "https://proxy.nullspend.com/",
    );
  });
});

describe("withNullSpendAsync", () => {
  const BASE = { apiKey: "ns_test_sk_abc123" } as const;

  const POLICY_RESPONSE = {
    budget: {
      remaining_microdollars: 4_200_000,
      max_microdollars: 10_000_000,
      spend_microdollars: 5_800_000,
      period_end: "2026-04-01T00:00:00.000Z",
      entity_type: "api_key",
      entity_id: "key-1",
    },
    allowed_models: ["gpt-4o-mini", "claude-haiku-4-5-20251001"],
    allowed_providers: ["openai", "anthropic"],
    cheapest_per_provider: {
      openai: { model: "gpt-4o-mini", input_per_mtok: 0.15, output_per_mtok: 0.60 },
    },
    cheapest_overall: { model: "gpt-4o-mini", provider: "openai", input_per_mtok: 0.15, output_per_mtok: 0.60 },
    restrictions_active: true,
  };

  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    _resetPolicyCache();
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(POLICY_RESPONSE), { status: 200 }),
    );
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    _resetPolicyCache();
  });

  it("fetches policy and sets appendSystemPrompt", async () => {
    const result = await withNullSpendAsync({ ...BASE });
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const appendPrompt = (result as any).appendSystemPrompt as string;
    expect(appendPrompt).toContain("[NullSpend Budget Context]");
    expect(appendPrompt).toContain("$4.20 remaining");
    expect(appendPrompt).toContain("gpt-4o-mini");
    expect(appendPrompt).toContain("Allowed models:");
  });

  it("skips policy fetch when budgetAwareness is false", async () => {
    const result = await withNullSpendAsync({ ...BASE, budgetAwareness: false });
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect((result as any).appendSystemPrompt).toBeUndefined();
  });

  it("proceeds without budget context on fetch failure", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network error"));
    const result = await withNullSpendAsync({ ...BASE });
    // Should not throw, should return base options without appendSystemPrompt
    expect(result.env?.ANTHROPIC_BASE_URL).toBe("https://proxy.nullspend.com");
    expect((result as any).appendSystemPrompt).toBeUndefined();
  });

  it("proceeds without budget context on non-200 response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "unauthorized" } }), { status: 401 }),
    );
    const result = await withNullSpendAsync({ ...BASE });
    expect((result as any).appendSystemPrompt).toBeUndefined();
  });

  it("caches policy response and does not re-fetch within TTL", async () => {
    await withNullSpendAsync({ ...BASE });
    await withNullSpendAsync({ ...BASE });
    expect(globalThis.fetch).toHaveBeenCalledTimes(1); // Only one fetch
  });

  it("uses different cache keys for different API keys", async () => {
    await withNullSpendAsync({ ...BASE });
    await withNullSpendAsync({ ...BASE, apiKey: "ns_test_sk_other" });
    expect(globalThis.fetch).toHaveBeenCalledTimes(2); // Two different keys
  });

  it("preserves existing SDK options", async () => {
    const result = await withNullSpendAsync({
      ...BASE,
      env: { MY_VAR: "test" },
    });
    expect(result.env?.MY_VAR).toBe("test");
    expect(result.env?.ANTHROPIC_BASE_URL).toBe("https://proxy.nullspend.com");
  });

  it("includes budget remaining and period end in prompt", async () => {
    const result = await withNullSpendAsync({ ...BASE });
    const prompt = (result as any).appendSystemPrompt as string;
    expect(prompt).toContain("$4.20 remaining");
    expect(prompt).toContain("resets 2026-04-01");
  });

  it("includes cheapest model recommendation in prompt", async () => {
    const result = await withNullSpendAsync({ ...BASE });
    const prompt = (result as any).appendSystemPrompt as string;
    expect(prompt).toContain("Preferred model (cheapest): gpt-4o-mini");
    expect(prompt).toContain("$0.15/MTok input");
    expect(prompt).toContain("$0.6/MTok output");
  });

  it("concatenates with existing appendSystemPrompt", async () => {
    const result = await withNullSpendAsync({
      ...BASE,
      appendSystemPrompt: "You are a helpful research assistant.",
    } as any);
    const prompt = (result as any).appendSystemPrompt as string;
    expect(prompt).toContain("You are a helpful research assistant.");
    expect(prompt).toContain("[NullSpend Budget Context]");
    // Existing prompt comes first, budget context appended after
    expect(prompt.indexOf("research assistant")).toBeLessThan(prompt.indexOf("[NullSpend"));
  });

  it("handles policy with no budget (restrictions only)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        ...POLICY_RESPONSE,
        budget: null,
      }), { status: 200 }),
    );
    const result = await withNullSpendAsync({ ...BASE });
    const prompt = (result as any).appendSystemPrompt as string;
    expect(prompt).toContain("[NullSpend Budget Context]");
    expect(prompt).toContain("Allowed models:");
    expect(prompt).not.toContain("remaining");
  });

  it("handles policy with no cheapest model (empty allowlist)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        ...POLICY_RESPONSE,
        allowed_models: [],
        cheapest_overall: null,
        cheapest_per_provider: null,
      }), { status: 200 }),
    );
    const result = await withNullSpendAsync({ ...BASE });
    const prompt = (result as any).appendSystemPrompt as string;
    expect(prompt).toContain("[NullSpend Budget Context]");
    expect(prompt).not.toContain("Preferred model");
    // Empty allowed_models means no "Allowed models:" line either
    expect(prompt).not.toContain("Allowed models:");
  });
});
