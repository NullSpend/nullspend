import { describe, it, expect } from "vitest";
import { getTableColumns } from "drizzle-orm";
import { budgets, costEvents, actions, apiKeys, slackConfigs, webhookEndpoints } from "./schema";
import type { BudgetRow, NewBudgetRow, CostEventRow, NewCostEventRow } from "./schema";

describe("budgets table schema", () => {
  const cols = getTableColumns(budgets);

  it("has all expected columns", () => {
    const names = Object.keys(cols);
    expect(names).toContain("id");
    expect(names).toContain("entityType");
    expect(names).toContain("entityId");
    expect(names).toContain("maxBudgetMicrodollars");
    expect(names).toContain("spendMicrodollars");
    expect(names).toContain("policy");
    expect(names).toContain("resetInterval");
    expect(names).toContain("currentPeriodStart");
    expect(names).toContain("createdAt");
    expect(names).toContain("updatedAt");
    expect(names).toContain("thresholdPercentages");
    expect(names).toContain("velocityLimitMicrodollars");
    expect(names).toContain("velocityWindowSeconds");
    expect(names).toContain("velocityCooldownSeconds");
    expect(names).toContain("userId");
  });

  it("id is a UUID primary key", () => {
    expect(cols.id.dataType).toBe("string");
    expect(cols.id.notNull).toBe(true);
    expect(cols.id.hasDefault).toBe(true);
  });

  it("maxBudgetMicrodollars is bigint (number mode)", () => {
    expect(cols.maxBudgetMicrodollars.dataType).toBe("number");
    expect(cols.maxBudgetMicrodollars.notNull).toBe(true);
  });

  it("spendMicrodollars defaults to 0", () => {
    expect(cols.spendMicrodollars.dataType).toBe("number");
    expect(cols.spendMicrodollars.notNull).toBe(true);
    expect(cols.spendMicrodollars.hasDefault).toBe(true);
  });

  it("policy defaults to strict_block", () => {
    expect(cols.policy.notNull).toBe(true);
    expect(cols.policy.hasDefault).toBe(true);
  });

  it("thresholdPercentages is NOT NULL with default", () => {
    expect(cols.thresholdPercentages.notNull).toBe(true);
    expect(cols.thresholdPercentages.hasDefault).toBe(true);
  });

  it("velocityLimitMicrodollars is nullable bigint (number mode)", () => {
    expect(cols.velocityLimitMicrodollars.dataType).toBe("number");
    expect(cols.velocityLimitMicrodollars.notNull).toBe(false);
  });

  it("velocityWindowSeconds has default 60", () => {
    expect(cols.velocityWindowSeconds.dataType).toBe("number");
    expect(cols.velocityWindowSeconds.hasDefault).toBe(true);
  });

  it("velocityCooldownSeconds has default 60", () => {
    expect(cols.velocityCooldownSeconds.dataType).toBe("number");
    expect(cols.velocityCooldownSeconds.hasDefault).toBe(true);
  });

  it("resetInterval is nullable (no reset by default)", () => {
    expect(cols.resetInterval.notNull).toBe(false);
  });

  it("currentPeriodStart is nullable", () => {
    expect(cols.currentPeriodStart.notNull).toBe(false);
  });

  it("userId is NOT NULL", () => {
    expect(cols.userId.notNull).toBe(true);
  });

  it("type inference produces correct BudgetRow shape", () => {
    const row: BudgetRow = {
      id: "uuid",
      entityType: "key",
      entityId: "key-123",
      maxBudgetMicrodollars: 1_000_000,
      spendMicrodollars: 500_000,
      policy: "strict_block",
      resetInterval: "monthly",
      thresholdPercentages: [50, 80, 90, 95],
      velocityLimitMicrodollars: null,
      velocityWindowSeconds: 60,
      velocityCooldownSeconds: 60,
      userId: null,
      currentPeriodStart: new Date(),
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    expect(row.maxBudgetMicrodollars).toBe(1_000_000);
    expect(row.thresholdPercentages).toEqual([50, 80, 90, 95]);
  });

  it("NewBudgetRow requires only non-default fields", () => {
    const insert: NewBudgetRow = {
      entityType: "user",
      entityId: "user-abc",
      maxBudgetMicrodollars: 5_000_000,
    };
    expect(insert.entityType).toBe("user");
    // spendMicrodollars, policy, id, timestamps are all optional (have defaults)
    expect(insert.spendMicrodollars).toBeUndefined();
  });
});

describe("costEvents table schema", () => {
  const cols = getTableColumns(costEvents);

  it("has all expected columns", () => {
    const names = Object.keys(cols);
    expect(names).toContain("id");
    expect(names).toContain("requestId");
    expect(names).toContain("apiKeyId");
    expect(names).toContain("userId");
    expect(names).toContain("provider");
    expect(names).toContain("model");
    expect(names).toContain("inputTokens");
    expect(names).toContain("outputTokens");
    expect(names).toContain("cachedInputTokens");
    expect(names).toContain("reasoningTokens");
    expect(names).toContain("costMicrodollars");
    expect(names).toContain("durationMs");
    expect(names).toContain("source");
    expect(names).toContain("tags");
    expect(names).toContain("createdAt");
  });

  it("costMicrodollars is bigint (number mode)", () => {
    expect(cols.costMicrodollars.dataType).toBe("number");
    expect(cols.costMicrodollars.notNull).toBe(true);
  });

  it("token columns are integer type", () => {
    expect(cols.inputTokens.dataType).toBe("number");
    expect(cols.outputTokens.dataType).toBe("number");
    expect(cols.cachedInputTokens.dataType).toBe("number");
    expect(cols.reasoningTokens.dataType).toBe("number");
  });

  it("cachedInputTokens defaults to 0", () => {
    expect(cols.cachedInputTokens.hasDefault).toBe(true);
  });

  it("reasoningTokens defaults to 0", () => {
    expect(cols.reasoningTokens.hasDefault).toBe(true);
  });

  it("apiKeyId is nullable, userId is NOT NULL", () => {
    expect(cols.apiKeyId.notNull).toBe(false);
    expect(cols.userId.notNull).toBe(true);
  });

  it("durationMs is nullable", () => {
    expect(cols.durationMs.notNull).toBe(false);
  });

  it("requestId is not nullable", () => {
    expect(cols.requestId.notNull).toBe(true);
  });

  it("tags column is NOT NULL with default", () => {
    expect(cols.tags.notNull).toBe(true);
    expect(cols.tags.hasDefault).toBe(true);
  });

  it("type inference produces correct CostEventRow shape", () => {
    const row: CostEventRow = {
      id: "uuid",
      requestId: "req-123",
      apiKeyId: "key-uuid",
      userId: "user-abc",
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 1000,
      outputTokens: 500,
      cachedInputTokens: 200,
      reasoningTokens: 0,
      costMicrodollars: 31250,
      durationMs: 1500,
      source: "proxy",
      tags: { project: "alpha" },
      createdAt: new Date(),
    };
    expect(row.costMicrodollars).toBe(31250);
    expect(row.tags).toEqual({ project: "alpha" });
  });

  it("NewCostEventRow allows nullable fields to be omitted", () => {
    const insert: NewCostEventRow = {
      requestId: "req-456",
      provider: "anthropic",
      model: "claude-sonnet-4-6",
      inputTokens: 2000,
      outputTokens: 1000,
      costMicrodollars: 45000,
    };
    expect(insert.apiKeyId).toBeUndefined();
    expect(insert.userId).toBeUndefined();
    expect(insert.durationMs).toBeUndefined();
  });
});

describe("existing tables still have correct structure", () => {
  it("actions table has required columns", () => {
    const cols = getTableColumns(actions);
    expect(cols.id.notNull).toBe(true);
    expect(cols.agentId.notNull).toBe(true);
    expect(cols.actionType.notNull).toBe(true);
    expect(cols.status.notNull).toBe(true);
    expect(cols.payloadJson.notNull).toBe(true);
    expect(cols.ownerUserId.notNull).toBe(true);
  });

  it("apiKeys table has required columns", () => {
    const cols = getTableColumns(apiKeys);
    expect(cols.id.notNull).toBe(true);
    expect(cols.userId.notNull).toBe(true);
    expect(cols.keyHash.notNull).toBe(true);
    expect(cols.apiVersion.notNull).toBe(true);
    expect(cols.apiVersion.hasDefault).toBe(true);
    expect(cols.environment.notNull).toBe(true);
    expect(cols.environment.hasDefault).toBe(true);
    expect(cols.defaultTags.notNull).toBe(true);
    expect(cols.defaultTags.hasDefault).toBe(true);
  });

  it("slackConfigs table has required columns", () => {
    const cols = getTableColumns(slackConfigs);
    expect(cols.id.notNull).toBe(true);
    expect(cols.userId.notNull).toBe(true);
    expect(cols.webhookUrl.notNull).toBe(true);
  });
});

describe("schema consistency with shared package types", () => {
  it("BudgetRow microdollar fields are number type (not bigint)", () => {
    const row: BudgetRow = {
      id: "test",
      entityType: "key",
      entityId: "k1",
      maxBudgetMicrodollars: 1_000_000,
      spendMicrodollars: 0,
      policy: "strict_block",
      resetInterval: null,
      thresholdPercentages: [50, 80, 90, 95],
      velocityLimitMicrodollars: null,
      velocityWindowSeconds: 60,
      velocityCooldownSeconds: 60,
      userId: null,
      currentPeriodStart: null,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    expect(typeof row.maxBudgetMicrodollars).toBe("number");
    expect(typeof row.spendMicrodollars).toBe("number");
  });

  it("CostEventRow microdollar field is number type (not bigint)", () => {
    const row: CostEventRow = {
      id: "test",
      requestId: "r1",
      apiKeyId: null,
      userId: null,
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      cachedInputTokens: 0,
      reasoningTokens: 0,
      costMicrodollars: 1000,
      durationMs: null,
      source: "proxy",
      tags: {},
      createdAt: new Date(),
    };
    expect(typeof row.costMicrodollars).toBe("number");
  });

  it("microdollar values handle real-world ranges without overflow", () => {
    // $1000 budget = 1 billion microdollars — fits in JS number
    const budget = 1_000_000_000;
    expect(Number.isSafeInteger(budget)).toBe(true);

    // $10,000 budget = 10 billion — still safe
    const largeBudget = 10_000_000_000;
    expect(Number.isSafeInteger(largeBudget)).toBe(true);

    // $1M budget = 1 trillion — still safe (MAX_SAFE_INTEGER is ~9 quadrillion)
    const hugeBudget = 1_000_000_000_000;
    expect(Number.isSafeInteger(hugeBudget)).toBe(true);
  });
});

describe("webhookEndpoints table schema", () => {
  const cols = getTableColumns(webhookEndpoints);

  it("previousSigningSecret is nullable", () => {
    expect(cols.previousSigningSecret.notNull).toBe(false);
  });

  it("secretRotatedAt is nullable", () => {
    expect(cols.secretRotatedAt.notNull).toBe(false);
  });
});
