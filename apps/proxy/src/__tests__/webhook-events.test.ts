import { describe, it, expect } from "vitest";
import {
  buildCostEventPayload,
  buildBudgetExceededPayload,
  buildThresholdPayload,
  buildBudgetResetPayload,
  buildRequestBlockedPayload,
  buildTestPingPayload,
  buildThinCostEventPayload,
  buildTagBudgetExceededPayload,
  buildCustomerBudgetExceededPayload,
  CURRENT_API_VERSION,
} from "../lib/webhook-events.js";

describe("buildCostEventPayload", () => {
  it("builds a valid cost_event.created payload", () => {
    const event = buildCostEventPayload({
      requestId: "req_123",
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 1000,
      outputTokens: 500,
      cachedInputTokens: 200,
      costMicrodollars: 25000,
      durationMs: 1500,
      apiKeyId: "key_abc",
      eventType: "llm",
      sessionId: "session-1",
      upstreamDurationMs: 1400,
      toolName: "search",
      toolServer: "mcp-server",
      toolCallsRequested: [{ name: "search", id: "call_1" }],
      toolDefinitionTokens: 800,
      source: "proxy",
    });

    expect(event.type).toBe("cost_event.created");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.request_id).toBe("req_123");
    expect(event.data.object.provider).toBe("openai");
    expect(event.data.object.model).toBe("gpt-4o");
    expect(event.data.object.input_tokens).toBe(1000);
    expect(event.data.object.output_tokens).toBe(500);
    expect(event.data.object.cached_input_tokens).toBe(200);
    expect(event.data.object.cost_microdollars).toBe(25000);
    expect(event.data.object.duration_ms).toBe(1500);
    expect(event.data.object.api_key_id).toBe("key_abc");
    expect(event.data.object.session_id).toBe("session-1");
    expect(event.data.object.upstream_duration_ms).toBe(1400);
    expect(event.data.object.tool_name).toBe("search");
    expect(event.data.object.tool_server).toBe("mcp-server");
    expect(event.data.object.tool_calls_requested).toEqual([{ name: "search", id: "call_1" }]);
    expect(event.data.object.tool_definition_tokens).toBe(800);
    expect(event.data.object.source).toBe("proxy");
  });

  it("handles null optional fields", () => {
    const event = buildCostEventPayload({
      requestId: "req_456",
      provider: "anthropic",
      model: "claude-3-5-sonnet",
      inputTokens: 500,
      outputTokens: 200,
      cachedInputTokens: 0,
      costMicrodollars: 3000,
      durationMs: null,
      apiKeyId: null,
      eventType: "llm",
    });

    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.duration_ms).toBeNull();
    expect(event.data.object.api_key_id).toBeNull();
    expect(event.data.object.session_id).toBeNull();
    expect(event.data.object.upstream_duration_ms).toBeNull();
    expect(event.data.object.tool_name).toBeNull();
    expect(event.data.object.tool_server).toBeNull();
    expect(event.data.object.tool_calls_requested).toBeNull();
    expect(event.data.object.tool_definition_tokens).toBe(0);
    expect(event.data.object.source).toBeNull();
  });

  it("generates unique event IDs", () => {
    const event1 = buildCostEventPayload({
      requestId: "req_1", provider: "openai", model: "gpt-4o",
      inputTokens: 100, outputTokens: 50, cachedInputTokens: 0,
      costMicrodollars: 1000, durationMs: 500, apiKeyId: "k1", eventType: "llm",
    });
    const event2 = buildCostEventPayload({
      requestId: "req_2", provider: "openai", model: "gpt-4o",
      inputTokens: 100, outputTokens: 50, cachedInputTokens: 0,
      costMicrodollars: 1000, durationMs: 500, apiKeyId: "k1", eventType: "llm",
    });
    expect(event1.id).not.toBe(event2.id);
  });

  it("accepts custom apiVersion", () => {
    const event = buildCostEventPayload({
      requestId: "req_1", provider: "openai", model: "gpt-4o",
      inputTokens: 100, outputTokens: 50, cachedInputTokens: 0,
      costMicrodollars: 1000, durationMs: 500, apiKeyId: "k1", eventType: "llm",
    }, "2027-01-01");
    expect(event.api_version).toBe("2027-01-01");
  });

  it("data.object.created_at is ISO string (cost event's own timestamp)", () => {
    const event = buildCostEventPayload({
      requestId: "req_1", provider: "openai", model: "gpt-4o",
      inputTokens: 100, outputTokens: 50, cachedInputTokens: 0,
      costMicrodollars: 1000, durationMs: 500, apiKeyId: "k1", eventType: "llm",
      createdAt: "2026-03-19T00:00:00Z",
    });
    expect(event.data.object.created_at).toBe("2026-03-19T00:00:00Z");
  });
});

describe("buildBudgetExceededPayload", () => {
  it("builds a valid budget.exceeded payload", () => {
    const event = buildBudgetExceededPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key_xxx",
      budgetLimitMicrodollars: 50_000_000,
      budgetSpendMicrodollars: 48_200_000,
      estimatedRequestCostMicrodollars: 3_500_000,
      model: "gpt-4o",
      provider: "openai",
    });

    expect(event.type).toBe("budget.exceeded");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.budget_entity_type).toBe("api_key");
    expect(event.data.object.budget_entity_id).toBe("key_xxx");
    expect(event.data.object.budget_limit_microdollars).toBe(50_000_000);
    expect(event.data.object.budget_spend_microdollars).toBe(48_200_000);
    expect(event.data.object.estimated_request_cost_microdollars).toBe(3_500_000);
    expect(event.data.object.model).toBe("gpt-4o");
    expect(event.data.object.provider).toBe("openai");
    expect(event.data.object.blocked_at).toBeTruthy();
  });
});

describe("buildThresholdPayload", () => {
  it("builds a warning event for threshold < 90", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 50_000_000,
      budgetSpendMicrodollars: 40_100_000,
      thresholdPercent: 80,
      triggeredByRequestId: "req_xyz",
    });

    expect(event.type).toBe("budget.threshold.warning");
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.threshold_percent).toBe(80);
    expect(event.data.object.budget_remaining_microdollars).toBe(9_900_000);
    expect(event.data.object.triggered_by_request_id).toBe("req_xyz");
  });

  it("builds a critical event for threshold >= 90", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "api_key",
      budgetEntityId: "key_yyy",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 95_000_000,
      thresholdPercent: 95,
      triggeredByRequestId: "req_abc",
    });

    expect(event.type).toBe("budget.threshold.critical");
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.threshold_percent).toBe(95);
  });

  it("builds a critical event for exactly 90%", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_def",
      budgetLimitMicrodollars: 10_000_000,
      budgetSpendMicrodollars: 9_000_000,
      thresholdPercent: 90,
      triggeredByRequestId: "req_ghi",
    });

    expect(event.type).toBe("budget.threshold.critical");
  });

  it("builds a warning event for 50% threshold", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_jkl",
      budgetLimitMicrodollars: 20_000_000,
      budgetSpendMicrodollars: 10_000_000,
      thresholdPercent: 50,
      triggeredByRequestId: "req_mno",
    });

    expect(event.type).toBe("budget.threshold.warning");
  });

  it("isCritical: true overrides threshold < 90 to critical", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 75_000_000,
      thresholdPercent: 75,
      triggeredByRequestId: "req_1",
      isCritical: true,
    });

    expect(event.type).toBe("budget.threshold.critical");
  });

  it("isCritical: false overrides threshold >= 90 to warning", () => {
    const event = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 95_000_000,
      thresholdPercent: 95,
      triggeredByRequestId: "req_1",
      isCritical: false,
    });

    expect(event.type).toBe("budget.threshold.warning");
  });

  it("omitted isCritical falls back to >= 90 logic (backward compat)", () => {
    const warning = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 80_000_000,
      thresholdPercent: 80,
      triggeredByRequestId: "req_1",
    });
    expect(warning.type).toBe("budget.threshold.warning");

    const critical = buildThresholdPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 90_000_000,
      thresholdPercent: 90,
      triggeredByRequestId: "req_1",
    });
    expect(critical.type).toBe("budget.threshold.critical");
  });
});

describe("buildBudgetResetPayload", () => {
  it("builds a valid budget.reset payload", () => {
    const event = buildBudgetResetPayload({
      budgetEntityType: "user",
      budgetEntityId: "user_abc",
      budgetLimitMicrodollars: 50_000_000,
      previousSpendMicrodollars: 45_000_000,
      newPeriodStart: "2026-04-01T00:00:00Z",
      resetInterval: "monthly",
    });

    expect(event.type).toBe("budget.reset");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.budget_entity_type).toBe("user");
    expect(event.data.object.budget_entity_id).toBe("user_abc");
    expect(event.data.object.budget_limit_microdollars).toBe(50_000_000);
    expect(event.data.object.previous_spend_microdollars).toBe(45_000_000);
    expect(event.data.object.new_period_start).toBe("2026-04-01T00:00:00Z");
    expect(event.data.object.reset_interval).toBe("monthly");
  });
});

describe("buildRequestBlockedPayload", () => {
  it("builds a valid request.blocked payload with budget reason", () => {
    const event = buildRequestBlockedPayload({
      reason: "budget",
      model: "gpt-4o",
      provider: "openai",
      apiKeyId: "key_abc",
      details: "Budget exceeded for api_key key_abc",
    });

    expect(event.type).toBe("request.blocked");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.reason).toBe("budget");
    expect(event.data.object.model).toBe("gpt-4o");
    expect(event.data.object.provider).toBe("openai");
    expect(event.data.object.api_key_id).toBe("key_abc");
    expect(event.data.object.details).toBe("Budget exceeded for api_key key_abc");
  });

  it("builds payload with rate_limit reason", () => {
    const event = buildRequestBlockedPayload({
      reason: "rate_limit",
      model: "claude-3-5-sonnet",
      provider: "anthropic",
      apiKeyId: null,
      details: null,
    });

    expect(event.data.object.reason).toBe("rate_limit");
    expect(event.data.object.api_key_id).toBeNull();
    expect(event.data.object.details).toBeNull();
  });

  it("builds payload with policy reason", () => {
    const event = buildRequestBlockedPayload({
      reason: "policy",
      model: "gpt-4o",
      provider: "openai",
      apiKeyId: "key_xyz",
      details: "Model not allowed by policy",
    });

    expect(event.data.object.reason).toBe("policy");
  });
});

describe("buildTestPingPayload", () => {
  it("builds a valid test.ping payload", () => {
    const event = buildTestPingPayload();

    expect(event.type).toBe("test.ping");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe("2026-04-01");
    expect(typeof event.created_at).toBe("number");
    expect(event.data.object.message).toBe("Test webhook event");
  });

  it("accepts custom apiVersion", () => {
    const event = buildTestPingPayload("2027-01-01");
    expect(event.api_version).toBe("2027-01-01");
  });
});

describe("buildThinCostEventPayload", () => {
  it("returns correct shape with related_object (id, type, url)", () => {
    const event = buildThinCostEventPayload("req_123", "openai");

    expect(event.related_object).toBeDefined();
    expect(event.related_object.id).toBe("req_123");
    expect(event.related_object.type).toBe("cost_event");
    expect(event.related_object.url).toBe(
      "/api/cost-events?requestId=req_123&provider=openai",
    );
  });

  it("has no data field present", () => {
    const event = buildThinCostEventPayload("req_123", "openai");
    expect(event).not.toHaveProperty("data");
  });

  it("type is 'cost_event.created'", () => {
    const event = buildThinCostEventPayload("req_123", "openai");
    expect(event.type).toBe("cost_event.created");
  });

  it("related_object.url is correctly encoded", () => {
    const event = buildThinCostEventPayload("req_123", "openai");
    expect(event.related_object.url).toBe(
      "/api/cost-events?requestId=req_123&provider=openai",
    );
    expect(event.api_version).toBe(CURRENT_API_VERSION);
    expect(typeof event.created_at).toBe("number");
  });

  it("each call produces a unique id", () => {
    const event1 = buildThinCostEventPayload("req_1", "openai");
    const event2 = buildThinCostEventPayload("req_1", "openai");
    expect(event1.id).toMatch(/^evt_/);
    expect(event2.id).toMatch(/^evt_/);
    expect(event1.id).not.toBe(event2.id);
  });

  it("special characters in requestId/provider are URL-encoded", () => {
    const event = buildThinCostEventPayload("req/foo bar", "open&ai");
    expect(event.related_object.url).toBe(
      "/api/cost-events?requestId=req%2Ffoo%20bar&provider=open%26ai",
    );
  });

  it("related_object.url parses as a valid URL with correct query params", () => {
    const event = buildThinCostEventPayload("req_abc123", "anthropic");
    const parsed = new URL(event.related_object.url, "https://placeholder.invalid");

    expect(parsed.pathname).toBe("/api/cost-events");
    expect(parsed.searchParams.get("requestId")).toBe("req_abc123");
    expect(parsed.searchParams.get("provider")).toBe("anthropic");
    // No extra query params beyond requestId and provider
    expect([...parsed.searchParams.keys()]).toEqual(["requestId", "provider"]);
  });
});

describe("buildTagBudgetExceededPayload", () => {
  it("builds valid tag_budget.exceeded event", () => {
    const event = buildTagBudgetExceededPayload({
      tagKey: "project",
      tagValue: "openclaw",
      budgetEntityId: "project=openclaw",
      budgetLimitMicrodollars: 50_000_000,
      budgetSpendMicrodollars: 49_500_000,
      estimatedRequestCostMicrodollars: 1_000_000,
      model: "gpt-4o",
      provider: "openai",
    });

    expect(event.type).toBe("tag_budget.exceeded");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe(CURRENT_API_VERSION);
    expect(typeof event.created_at).toBe("number");
  });

  it("includes tag_key and tag_value in data.object", () => {
    const event = buildTagBudgetExceededPayload({
      tagKey: "env",
      tagValue: "prod",
      budgetEntityId: "env=prod",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 90_000_000,
      estimatedRequestCostMicrodollars: 5_000_000,
      model: "claude-sonnet-4-20250514",
      provider: "anthropic",
    });

    const obj = event.data.object;
    expect(obj.tag_key).toBe("env");
    expect(obj.tag_value).toBe("prod");
    expect(obj.budget_entity_type).toBe("tag");
    expect(obj.budget_entity_id).toBe("env=prod");
    expect(obj.budget_limit_microdollars).toBe(100_000_000);
    expect(obj.budget_spend_microdollars).toBe(90_000_000);
    expect(obj.estimated_request_cost_microdollars).toBe(5_000_000);
    expect(obj.model).toBe("claude-sonnet-4-20250514");
    expect(obj.provider).toBe("anthropic");
    expect(obj.blocked_at).toBeDefined();
  });

  it("standard fields: id, type, api_version, created_at", () => {
    const event = buildTagBudgetExceededPayload({
      tagKey: "project",
      tagValue: "openclaw",
      budgetEntityId: "project=openclaw",
      budgetLimitMicrodollars: 50_000_000,
      budgetSpendMicrodollars: 50_000_000,
      estimatedRequestCostMicrodollars: 500_000,
      model: "gpt-4o-mini",
      provider: "openai",
    }, "2025-01-01");

    expect(event.id).toMatch(/^evt_[0-9a-f-]+$/);
    expect(event.type).toBe("tag_budget.exceeded");
    expect(event.api_version).toBe("2025-01-01");
    expect(event.created_at).toBeGreaterThan(0);
  });
});

describe("buildCustomerBudgetExceededPayload", () => {
  it("builds valid customer_budget.exceeded event", () => {
    const event = buildCustomerBudgetExceededPayload({
      customerId: "acme-corp",
      budgetLimitMicrodollars: 50_000_000,
      budgetSpendMicrodollars: 49_500_000,
      estimatedRequestCostMicrodollars: 1_000_000,
      model: "gpt-4o",
      provider: "openai",
    });

    expect(event.type).toBe("customer_budget.exceeded");
    expect(event.id).toMatch(/^evt_/);
    expect(event.api_version).toBe(CURRENT_API_VERSION);
    expect(typeof event.created_at).toBe("number");
  });

  it("includes customer_id in data.object", () => {
    const event = buildCustomerBudgetExceededPayload({
      customerId: "acme-corp",
      budgetLimitMicrodollars: 100_000_000,
      budgetSpendMicrodollars: 95_000_000,
      estimatedRequestCostMicrodollars: 10_000_000,
      model: "claude-sonnet-4-20250514",
      provider: "anthropic",
    });

    const obj = event.data.object;
    expect(obj.customer_id).toBe("acme-corp");
    expect(obj.budget_entity_type).toBe("customer");
    expect(obj.budget_entity_id).toBe("acme-corp");
    expect(obj.budget_limit_microdollars).toBe(100_000_000);
    expect(obj.budget_spend_microdollars).toBe(95_000_000);
    expect(obj.estimated_request_cost_microdollars).toBe(10_000_000);
    expect(obj.model).toBe("claude-sonnet-4-20250514");
    expect(obj.provider).toBe("anthropic");
    expect(obj.blocked_at).toBeDefined();
  });
});

describe("CURRENT_API_VERSION", () => {
  it("is 2026-04-01", () => {
    expect(CURRENT_API_VERSION).toBe("2026-04-01");
  });
});
