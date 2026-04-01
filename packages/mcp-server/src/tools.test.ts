import { describe, it, expect, vi, beforeEach } from "vitest";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerTools } from "./tools.js";
import type { McpServerConfig } from "./config.js";

let mockCreateAction = vi.fn();
let mockGetAction = vi.fn();
const mockListBudgets = vi.fn();
const mockGetCostSummary = vi.fn();
const mockListCostEvents = vi.fn();

vi.mock("@nullspend/sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@nullspend/sdk")>();
  class MockNullSpend {
    constructor() {}
    createAction(...args: unknown[]) { return mockCreateAction(...args); }
    getAction(...args: unknown[]) { return mockGetAction(...args); }
    listBudgets(...args: unknown[]) { return mockListBudgets(...args); }
    getCostSummary(...args: unknown[]) { return mockGetCostSummary(...args); }
    listCostEvents(...args: unknown[]) { return mockListCostEvents(...args); }
  }
  return {
    ...actual,
    NullSpend: MockNullSpend,
  };
});

const TEST_CONFIG: McpServerConfig = {
  nullspendUrl: "http://localhost:3000",
  nullspendApiKey: "ns_live_sk_test0001",
  agentId: "test-agent",
};

function mockAction(overrides: Record<string, unknown> = {}) {
  return {
    id: "act-1",
    agentId: "test-agent",
    actionType: "send_email",
    status: "pending",
    payload: { to: "a@b.com" },
    metadata: null,
    createdAt: "2026-01-01T00:00:00Z",
    approvedAt: null,
    rejectedAt: null,
    executedAt: null,
    expiresAt: null,
    expiredAt: null,
    approvedBy: null,
    rejectedBy: null,
    result: null,
    errorMessage: null,
    environment: null,
    sourceFramework: null,
    ...overrides,
  };
}

interface ToolRegistration {
  name: string;
  description: string;
  cb: (params: Record<string, unknown>) => Promise<{
    content: Array<{ type: string; text: string }>;
    isError?: boolean;
  }>;
}

function captureTools(config: McpServerConfig, signal?: AbortSignal) {
  const tools: ToolRegistration[] = [];
  const fakeServer = {
    tool: vi.fn((...args: unknown[]) => {
      const name = args[0] as string;
      const description = args[1] as string;
      const cb = args[args.length - 1] as ToolRegistration["cb"];
      tools.push({ name, description, cb });
    }),
  } as unknown as McpServer;

  const abortController = new AbortController();
  registerTools(fakeServer, config, signal ?? abortController.signal);
  return { tools, abortController };
}

function getToolByName(tools: ToolRegistration[], name: string) {
  const tool = tools.find((t) => t.name === name);
  if (!tool) throw new Error(`Tool ${name} not found`);
  return tool;
}

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return {
    data: JSON.parse(result.content[0].text),
    isError: result.isError ?? false,
  };
}

describe("registerTools", () => {
  beforeEach(() => {
    mockCreateAction = vi.fn();
    mockGetAction = vi.fn();
  });

  it("registers propose_action and check_action", () => {
    const { tools } = captureTools(TEST_CONFIG);
    const names = tools.map((t) => t.name);
    expect(names).toContain("propose_action");
    expect(names).toContain("check_action");
  });

  describe("propose_action", () => {
    it("returns pending immediately when waitForDecision=false", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      const result = await tool.cb({
        actionType: "send_email",
        payload: { to: "test@example.com" },
        summary: "Send an email",
        waitForDecision: false,
      });

      const { data, isError } = parseResult(result);
      expect(isError).toBe(false);
      expect(data.actionId).toBe("act-1");
      expect(data.status).toBe("pending");
      expect(data.approved).toBe(false);
      expect(data.timedOut).toBe(false);
    });

    it("returns approved when action is approved during wait", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });
      mockGetAction.mockResolvedValue(
        mockAction({ status: "approved", approvedAt: "2026-01-01T00:01:00Z" }),
      );

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      const result = await tool.cb({
        actionType: "send_email",
        payload: { to: "test@example.com" },
        summary: "Send an email",
        waitForDecision: true,
        timeoutSeconds: 5,
      });

      const { data, isError } = parseResult(result);
      expect(isError).toBe(false);
      expect(data.actionId).toBe("act-1");
      expect(data.status).toBe("approved");
      expect(data.approved).toBe(true);
    });

    it("returns rejected when action is rejected during wait", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });
      mockGetAction.mockResolvedValue(
        mockAction({ status: "rejected", rejectedAt: "2026-01-01T00:01:00Z" }),
      );

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      const result = await tool.cb({
        actionType: "send_email",
        payload: { to: "test@example.com" },
        summary: "Send an email",
        waitForDecision: true,
        timeoutSeconds: 5,
      });

      const { data, isError } = parseResult(result);
      expect(isError).toBe(false);
      expect(data.status).toBe("rejected");
      expect(data.rejected).toBe(true);
      expect(data.approved).toBe(false);
    });

    it("stamps MCP metadata on created action", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      await tool.cb({
        actionType: "send_email",
        payload: { to: "test@example.com" },
        summary: "Send an email",
        waitForDecision: false,
      });

      expect(mockCreateAction).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            sourceFramework: "mcp",
            transport: "stdio",
            summary: "Send an email",
          }),
        }),
      );
    });

    it("MCP metadata cannot be overwritten by user metadata", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        metadata: { sourceFramework: "custom", transport: "http", extra: "kept" },
        waitForDecision: false,
      });

      const calledWith = mockCreateAction.mock.calls[0][0];
      expect(calledWith.metadata.sourceFramework).toBe("mcp");
      expect(calledWith.metadata.transport).toBe("stdio");
      expect(calledWith.metadata.extra).toBe("kept");
    });

    it("uses config.agentId when agentId is not provided", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        waitForDecision: false,
      });

      expect(mockCreateAction).toHaveBeenCalledWith(
        expect.objectContaining({ agentId: "test-agent" }),
      );
    });

    it("uses provided agentId when specified", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        agentId: "custom-agent",
        waitForDecision: false,
      });

      expect(mockCreateAction).toHaveBeenCalledWith(
        expect.objectContaining({ agentId: "custom-agent" }),
      );
    });

    it("returns error result when SDK throws", async () => {
      mockCreateAction.mockRejectedValue(new Error("Connection refused"));

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      const result = await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        waitForDecision: false,
      });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Connection refused");
    });

    it("returns structured timeout response when deadline passes", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });
      mockGetAction.mockResolvedValue(mockAction({ status: "pending" }));

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "propose_action");

      const result = await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        waitForDecision: true,
        timeoutSeconds: 0,
      });

      const { data, isError } = parseResult(result);
      expect(isError).toBe(false);
      expect(data.actionId).toBe("act-1");
      expect(data.status).toBe("pending");
      expect(data.timedOut).toBe(true);
      expect(data.approved).toBe(false);
      expect(data.message).toContain("Timed out");
    });

    it("aborts polling when shutdown signal fires", async () => {
      mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });
      mockGetAction.mockResolvedValue(mockAction({ status: "pending" }));

      const abortController = new AbortController();
      const { tools } = captureTools(TEST_CONFIG, abortController.signal);
      const tool = getToolByName(tools, "propose_action");

      // Abort after a short delay so the polling loop starts then gets interrupted
      setTimeout(() => abortController.abort(), 50);

      const result = await tool.cb({
        actionType: "send_email",
        payload: {},
        summary: "Test",
        waitForDecision: true,
        timeoutSeconds: 60,
      });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Aborted");
    });
  });

  describe("check_action", () => {
    it("returns current action status", async () => {
      mockGetAction.mockResolvedValue(
        mockAction({ id: "act-42", status: "approved", approvedAt: "2026-01-01T00:01:00Z" }),
      );

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "check_action");

      const result = await tool.cb({ actionId: "act-42" });

      const { data, isError } = parseResult(result);
      expect(isError).toBe(false);
      expect(data.actionId).toBe("act-42");
      expect(data.status).toBe("approved");
      expect(data.approved).toBe(true);
    });

    it("returns error result when SDK throws", async () => {
      mockGetAction.mockRejectedValue(new Error("Not found"));

      const { tools } = captureTools(TEST_CONFIG);
      const tool = getToolByName(tools, "check_action");

      const result = await tool.cb({ actionId: "act-missing" });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Not found");
    });
  });

  // -------------------------------------------------------------------------
  // get_budgets
  // -------------------------------------------------------------------------
  describe("get_budgets", () => {
    it("returns formatted budget list", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_budgets")!;

      mockListBudgets.mockResolvedValue({
        data: [{
          id: "b-1",
          entityType: "user",
          entityId: "user-1",
          maxBudgetMicrodollars: 10_000_000_000,
          spendMicrodollars: 3_500_000_000,
          policy: "strict_block",
          resetInterval: "monthly",
          currentPeriodStart: "2026-03-01T00:00:00Z",
          thresholdPercentages: [50, 80, 90, 95],
          velocityLimitMicrodollars: null,
          velocityWindowSeconds: null,
          velocityCooldownSeconds: null,
          sessionLimitMicrodollars: null,
          createdAt: "2026-01-01T00:00:00Z",
          updatedAt: "2026-03-01T00:00:00Z",
        }],
      });

      const result = await tool.cb({});
      expect(result.isError).toBeUndefined();

      const data = JSON.parse(result.content[0].text);
      expect(data.budgets).toHaveLength(1);
      expect(data.budgets[0].limitDollars).toBe(10_000);
      expect(data.budgets[0].spendDollars).toBe(3_500);
      expect(data.budgets[0].remainingDollars).toBe(6_500);
      expect(data.budgets[0].percentUsed).toBe(35);
    });

    it("returns empty message when no budgets", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_budgets")!;
      mockListBudgets.mockResolvedValue({ data: [] });

      const result = await tool.cb({});
      const data = JSON.parse(result.content[0].text);
      expect(data.budgets).toEqual([]);
      expect(data.message).toContain("No budgets configured");
    });

    it("returns error on API failure", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_budgets")!;
      mockListBudgets.mockRejectedValue(new Error("connection refused"));

      const result = await tool.cb({});
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("NullSpend API error");
    });
  });

  // -------------------------------------------------------------------------
  // get_spend_summary
  // -------------------------------------------------------------------------
  describe("get_spend_summary", () => {
    it("returns spend summary with default period", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_spend_summary")!;

      mockGetCostSummary.mockResolvedValue({
        daily: [{ date: "2026-03-01", totalCostMicrodollars: 1_000_000 }],
        models: { "gpt-4o": 800_000, "gpt-4o-mini": 200_000 },
        providers: { openai: 1_000_000 },
        totals: {
          totalCostMicrodollars: 1_000_000,
          totalRequests: 50,
          totalInputTokens: 10_000,
          totalOutputTokens: 5_000,
        },
      });

      const result = await tool.cb({});
      expect(result.isError).toBeUndefined();

      const data = JSON.parse(result.content[0].text);
      expect(data.period).toBe("30d");
      expect(data.totalCostDollars).toBe(1);
      expect(data.totalRequests).toBe(50);
      expect(data.costByModel["gpt-4o"]).toBe(0.8);
      expect(mockGetCostSummary).toHaveBeenCalledWith("30d");
    });

    it("passes explicit period", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_spend_summary")!;

      mockGetCostSummary.mockResolvedValue({
        daily: [],
        models: {},
        providers: {},
        totals: { totalCostMicrodollars: 0, totalRequests: 0, totalInputTokens: 0, totalOutputTokens: 0 },
      });

      await tool.cb({ period: "7d" });
      expect(mockGetCostSummary).toHaveBeenCalledWith("7d");
    });

    it("returns error on API failure", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_spend_summary")!;
      mockGetCostSummary.mockRejectedValue(new Error("timeout"));

      const result = await tool.cb({});
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("timeout");
    });
  });

  // -------------------------------------------------------------------------
  // get_recent_costs
  // -------------------------------------------------------------------------
  describe("get_recent_costs", () => {
    it("returns recent cost events with default limit", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_recent_costs")!;

      mockListCostEvents.mockResolvedValue({
        data: [
          { id: "e-1", model: "gpt-4o", provider: "openai", inputTokens: 500, outputTokens: 150, costMicrodollars: 4625, durationMs: 800, createdAt: "2026-03-01T12:00:00Z" },
          { id: "e-2", model: "gpt-4o-mini", provider: "openai", inputTokens: 1000, outputTokens: 300, costMicrodollars: 225, durationMs: 400, createdAt: "2026-03-01T11:00:00Z" },
        ],
        cursor: null,
      });

      const result = await tool.cb({});
      expect(result.isError).toBeUndefined();

      const data = JSON.parse(result.content[0].text);
      expect(data.count).toBe(2);
      expect(data.events[0].model).toBe("gpt-4o");
      expect(data.events[0].costDollars).toBeCloseTo(0.004625);
      expect(data.totalCostDollars).toBeCloseTo(0.00485);
      expect(mockListCostEvents).toHaveBeenCalledWith({ limit: 10 });
    });

    it("clamps limit to max 50", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_recent_costs")!;
      mockListCostEvents.mockResolvedValue({ data: [], cursor: null });

      await tool.cb({ limit: 100 });
      expect(mockListCostEvents).toHaveBeenCalledWith({ limit: 50 });
    });

    it("returns error on API failure", async () => {
      const { tools } = captureTools(TEST_CONFIG);
      const tool = tools.find((t) => t.name === "get_recent_costs")!;
      mockListCostEvents.mockRejectedValue(new Error("network error"));

      const result = await tool.cb({});
      expect(result.isError).toBe(true);
    });
  });
});
