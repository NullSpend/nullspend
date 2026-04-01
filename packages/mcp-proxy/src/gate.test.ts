import { describe, it, expect, vi, beforeEach } from "vitest";
import { isToolGated, gateToolCall } from "./gate.js";
import type { ProxyConfig } from "./config.js";
import type { GateResult } from "./gate.js";

let mockCreateAction = vi.fn();
let mockGetAction = vi.fn();

vi.mock("@nullspend/sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@nullspend/sdk")>();
  class MockNullSpend {
    constructor() {}
    createAction(...args: unknown[]) {
      return mockCreateAction(...args);
    }
    getAction(...args: unknown[]) {
      return mockGetAction(...args);
    }
    markResult() {
      return Promise.resolve({ id: "act-1", status: "executed" });
    }
  }
  return {
    ...actual,
    NullSpend: MockNullSpend,
  };
});

function makeConfig(overrides: Partial<ProxyConfig> = {}): ProxyConfig {
  return {
    nullspendUrl: "http://127.0.0.1:3000",
    nullspendApiKey: "ns_live_sk_test0001",
    agentId: "test-proxy",
    upstreamCommand: "node",
    upstreamArgs: [],
    upstreamEnv: {},
    gatedTools: "*",
    passthroughTools: new Set<string>(),
    approvalTimeoutSeconds: 300,
    backendUrl: "http://127.0.0.1:3000",
    serverName: "test-server",
    costTrackingEnabled: true,
    budgetEnforcementEnabled: true,
    toolCostOverrides: {},
    ...overrides,
  };
}

describe("isToolGated", () => {
  it("gates all tools when gatedTools is *", () => {
    const config = makeConfig();
    expect(isToolGated("any_tool", config)).toBe(true);
    expect(isToolGated("another_tool", config)).toBe(true);
  });

  it("passthrough overrides * wildcard", () => {
    const config = makeConfig({ passthroughTools: new Set(["safe_tool"]) });
    expect(isToolGated("safe_tool", config)).toBe(false);
    expect(isToolGated("risky_tool", config)).toBe(true);
  });

  it("gates only specific tools when gatedTools is a set", () => {
    const config = makeConfig({
      gatedTools: new Set(["run_query", "delete_file"]),
    });
    expect(isToolGated("run_query", config)).toBe(true);
    expect(isToolGated("delete_file", config)).toBe(true);
    expect(isToolGated("read_file", config)).toBe(false);
  });

  it("passthrough overrides explicit gate list", () => {
    const config = makeConfig({
      gatedTools: new Set(["run_query", "delete_file"]),
      passthroughTools: new Set(["run_query"]),
    });
    expect(isToolGated("run_query", config)).toBe(false);
    expect(isToolGated("delete_file", config)).toBe(true);
  });

  it("does not gate tools not in explicit gate list", () => {
    const config = makeConfig({
      gatedTools: new Set(["run_query"]),
    });
    expect(isToolGated("list_tables", config)).toBe(false);
  });
});

describe("gateToolCall", () => {
  beforeEach(() => {
    mockCreateAction = vi.fn();
    mockGetAction = vi.fn();
  });

  it("returns approved when action is approved", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-1", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "approved" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig();

    const result: GateResult = await gateToolCall(
      sdk,
      "run_query",
      { sql: "SELECT 1" },
      config,
      new AbortController().signal,
    );

    expect(result.actionId).toBe("act-1");
    expect(result.decision).toBe("approved");
  });

  it("returns rejected when action is rejected", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-2", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "rejected" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig();

    const result = await gateToolCall(
      sdk,
      "delete_file",
      { path: "/important" },
      config,
      new AbortController().signal,
    );

    expect(result.actionId).toBe("act-2");
    expect(result.decision).toBe("rejected");
  });

  it("returns timedOut when approval times out", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-3", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "pending" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig({ approvalTimeoutSeconds: 0 });

    const result = await gateToolCall(
      sdk,
      "run_query",
      { sql: "DROP TABLE" },
      config,
      new AbortController().signal,
    );

    expect(result.actionId).toBe("act-3");
    expect(result.decision).toBe("timedOut");
  });

  it("includes correct metadata in created action", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-4", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "approved" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig();

    await gateToolCall(
      sdk,
      "send_email",
      { to: "user@test.com" },
      config,
      new AbortController().signal,
    );

    expect(mockCreateAction).toHaveBeenCalledWith(
      expect.objectContaining({
        agentId: "test-proxy",
        actionType: "send_email",
        payload: { to: "user@test.com" },
        metadata: expect.objectContaining({
          sourceFramework: "mcp-proxy",
          transport: "stdio",
          upstreamTool: "send_email",
          summary: expect.stringContaining("send_email"),
        }),
      }),
    );
  });

  it("handles undefined args gracefully", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-5", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "approved" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig();

    const result = await gateToolCall(
      sdk,
      "no_args_tool",
      undefined,
      config,
      new AbortController().signal,
    );

    expect(result.decision).toBe("approved");
    expect(mockCreateAction).toHaveBeenCalledWith(
      expect.objectContaining({ payload: {} }),
    );
  });

  it("throws when SDK createAction fails", async () => {
    mockCreateAction.mockRejectedValue(new Error("Connection refused"));

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig();

    await expect(
      gateToolCall(sdk, "tool", {}, config, new AbortController().signal),
    ).rejects.toThrow("Connection refused");
  });

  it("stops polling when abort signal fires", async () => {
    mockCreateAction.mockResolvedValue({ id: "act-abort", status: "pending" });
    mockGetAction.mockResolvedValue({ status: "pending" });

    const { NullSpend } = await import("@nullspend/sdk");
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });
    const config = makeConfig({ approvalTimeoutSeconds: 600 });
    const controller = new AbortController();

    setTimeout(() => controller.abort(), 50);

    await expect(
      gateToolCall(sdk, "tool", {}, config, controller.signal),
    ).rejects.toThrow("Aborted");
  });
});
