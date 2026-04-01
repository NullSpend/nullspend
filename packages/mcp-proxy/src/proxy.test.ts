import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ProxyConfig } from "./config.js";
import type { GateResult } from "./gate.js";

const mockCreateAction = vi.fn();
const mockGetAction = vi.fn();
const mockMarkResult = vi.fn();

vi.mock("@nullspend/sdk", () => {
  class MockNullSpend {
    constructor() {}
    createAction(...args: unknown[]) {
      return mockCreateAction(...args);
    }
    getAction(...args: unknown[]) {
      return mockGetAction(...args);
    }
    markResult(...args: unknown[]) {
      return mockMarkResult(...args);
    }
  }
  class MockTimeoutError extends Error {
    constructor(actionId: string, timeoutMs: number) {
      super(`Timed out on ${actionId} after ${timeoutMs}ms`);
      this.name = "TimeoutError";
    }
  }
  return {
    NullSpend: MockNullSpend,
    TimeoutError: MockTimeoutError,
  };
});

const mockCallTool = vi.fn();
const mockListTools = vi.fn();

vi.mock("./gate.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("./gate.js")>();
  return {
    ...original,
    isToolGated: vi.fn(),
    gateToolCall: vi.fn(),
  };
});

import { discoverUpstreamTools, registerProxyHandlers } from "./proxy.js";
import { isToolGated, gateToolCall } from "./gate.js";
import { NullSpend } from "@nullspend/sdk";

const mockedIsToolGated = vi.mocked(isToolGated);
const mockedGateToolCall = vi.mocked(gateToolCall);

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

function makeFakeUpstreamClient() {
  return {
    listTools: mockListTools,
    callTool: mockCallTool,
  } as unknown as import("@modelcontextprotocol/sdk/client/index.js").Client;
}

interface RequestHandler {
  (request: { params: Record<string, unknown> }): Promise<unknown>;
}

function makeFakeServer() {
  const registeredHandlers: RequestHandler[] = [];
  return {
    setRequestHandler: vi.fn((_schema: unknown, handler: RequestHandler) => {
      registeredHandlers.push(handler);
    }),
    get listToolsHandler(): RequestHandler | undefined {
      return registeredHandlers[0];
    },
    get callToolHandler(): RequestHandler | undefined {
      return registeredHandlers[1];
    },
  };
}

describe("discoverUpstreamTools", () => {
  beforeEach(() => {
    mockListTools.mockReset();
  });

  it("returns upstream tools from listTools()", async () => {
    const tools = [
      { name: "read_file", description: "Read a file", inputSchema: { type: "object" } },
      { name: "write_file", description: "Write a file", inputSchema: { type: "object" } },
    ];
    mockListTools.mockResolvedValue({ tools });

    const result = await discoverUpstreamTools(makeFakeUpstreamClient());
    expect(result).toEqual(tools);
  });

  it("fetches all pages when upstream returns nextCursor", async () => {
    const page1Tools = [
      { name: "tool_a", inputSchema: { type: "object" } },
      { name: "tool_b", inputSchema: { type: "object" } },
    ];
    const page2Tools = [
      { name: "tool_c", inputSchema: { type: "object" } },
    ];

    mockListTools
      .mockResolvedValueOnce({ tools: page1Tools, nextCursor: "cursor-1" })
      .mockResolvedValueOnce({ tools: page2Tools });

    const result = await discoverUpstreamTools(makeFakeUpstreamClient());

    expect(result).toEqual([...page1Tools, ...page2Tools]);
    expect(mockListTools).toHaveBeenCalledTimes(2);
    expect(mockListTools).toHaveBeenNthCalledWith(1, undefined);
    expect(mockListTools).toHaveBeenNthCalledWith(2, { cursor: "cursor-1" });
  });

  it("handles multiple pages of pagination", async () => {
    mockListTools
      .mockResolvedValueOnce({ tools: [{ name: "t1", inputSchema: { type: "object" } }], nextCursor: "c1" })
      .mockResolvedValueOnce({ tools: [{ name: "t2", inputSchema: { type: "object" } }], nextCursor: "c2" })
      .mockResolvedValueOnce({ tools: [{ name: "t3", inputSchema: { type: "object" } }] });

    const result = await discoverUpstreamTools(makeFakeUpstreamClient());

    expect(result).toHaveLength(3);
    expect(result.map((t) => t.name)).toEqual(["t1", "t2", "t3"]);
    expect(mockListTools).toHaveBeenCalledTimes(3);
  });
});

describe("registerProxyHandlers", () => {
  beforeEach(() => {
    mockCallTool.mockReset();
    mockMarkResult.mockReset();
    mockedIsToolGated.mockReset();
    mockedGateToolCall.mockReset();
  });

  it("registers tools/list and tools/call handlers", () => {
    const fakeServer = makeFakeServer();
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

    registerProxyHandlers(
      fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
      makeFakeUpstreamClient(),
      sdk,
      makeConfig(),
      [],
      new AbortController().signal,
    );

    expect(fakeServer.setRequestHandler).toHaveBeenCalledTimes(2);
  });

  it("tools/list returns cached tools verbatim", async () => {
    const cachedTools = [
      { name: "read_file", description: "Read", inputSchema: { type: "object", properties: { path: { type: "string" } } } },
    ];
    const fakeServer = makeFakeServer();
    const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

    registerProxyHandlers(
      fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
      makeFakeUpstreamClient(),
      sdk,
      makeConfig(),
      cachedTools,
      new AbortController().signal,
    );

    const handler = fakeServer.listToolsHandler!;
    const result = await handler({ params: {} });
    expect(result).toEqual({ tools: cachedTools });
  });

  describe("tools/call - forward path", () => {
    it("forwards non-gated tool directly to upstream", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "file contents" }],
      });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = await handler({
        params: { name: "read_file", arguments: { path: "/test.txt" } },
      });

      expect(mockCallTool).toHaveBeenCalledWith({
        name: "read_file",
        arguments: { path: "/test.txt" },
      });
      expect(result).toEqual({
        content: [{ type: "text", text: "file contents" }],
      });
    });

    it("returns error when upstream throws on forward", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockRejectedValue(new Error("upstream crash"));

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "read_file", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("upstream crash");
    });
  });

  describe("tools/call - gate path", () => {
    it("gates tool, marks executing, forwards on approval, marks executed", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-1",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "query result" }],
      });
      mockMarkResult.mockResolvedValue({ id: "act-1", status: "executed" });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = await handler({
        params: { name: "run_query", arguments: { sql: "SELECT 1" } },
      });

      expect(mockMarkResult).toHaveBeenCalledWith("act-1", { status: "executing" });
      expect(mockCallTool).toHaveBeenCalledWith({
        name: "run_query",
        arguments: { sql: "SELECT 1" },
      });
      expect(mockMarkResult).toHaveBeenCalledWith("act-1", {
        status: "executed",
        result: expect.objectContaining({
          content: [{ type: "text", text: "query result" }],
        }),
      });
      expect(result).toMatchObject({
        content: [{ type: "text", text: "query result" }],
        actionId: "act-1",
      });
    });

    it("marks failed when upstream throws after approval", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-2",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockRejectedValue(new Error("upstream exploded"));
      mockMarkResult.mockResolvedValue({ id: "act-2", status: "failed" });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: { sql: "DROP TABLE" } },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(mockMarkResult).toHaveBeenCalledWith("act-2", { status: "executing" });
      expect(mockMarkResult).toHaveBeenCalledWith("act-2", {
        status: "failed",
        errorMessage: "upstream exploded",
      });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("upstream exploded");
    });

    it("marks failed when upstream returns isError: true", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-3",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "permission denied" }],
        isError: true,
      });
      mockMarkResult.mockResolvedValue({ id: "act-3", status: "failed" });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(mockMarkResult).toHaveBeenCalledWith("act-3", {
        status: "failed",
        errorMessage: "permission denied",
      });
      expect(result.isError).toBe(true);
    });

    it("returns rejection text when action is rejected", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-4",
        decision: "rejected",
      } satisfies GateResult);

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "delete_file", arguments: { path: "/important" } },
      })) as { content: Array<{ text: string }>; isError?: boolean };

      expect(result.content[0].text).toContain("rejected");
      expect(result.isError).toBe(true);
      expect(mockCallTool).not.toHaveBeenCalled();
    });

    it("returns timeout text when approval times out", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-5",
        decision: "timedOut",
      } satisfies GateResult);

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: {} },
      })) as { content: Array<{ text: string }>; isError?: boolean };

      expect(result.content[0].text).toContain("timed out");
      expect(result.isError).toBe(true);
      expect(mockCallTool).not.toHaveBeenCalled();
    });

    it("returns isError when NullSpend API is unreachable", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockRejectedValue(new Error("Connection refused"));

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("approval service");
      expect(mockCallTool).not.toHaveBeenCalled();
    });

    it("still forwards upstream call when markResult(executing) fails", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-mr1",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "success" }],
      });

      let callCount = 0;
      mockMarkResult.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.reject(new Error("API down"));
        }
        return Promise.resolve({ id: "act-mr1", status: "executed" });
      });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: { sql: "SELECT 1" } },
      })) as { content: Array<{ text: string }>; isError?: boolean };

      expect(mockCallTool).toHaveBeenCalled();
      expect(result.content[0].text).toBe("success");
      expect(result.isError).toBeUndefined();
    });

    it("still returns result when markResult(executed) fails", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-mr2",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "data here" }],
      });
      mockMarkResult.mockImplementation((_id: string, input: { status: string }) => {
        if (input.status === "executed") {
          return Promise.reject(new Error("Audit trail failed"));
        }
        return Promise.resolve({ id: "act-mr2", status: input.status });
      });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: {} },
      })) as { content: Array<{ text: string }>; isError?: boolean };

      expect(result.content[0].text).toBe("data here");
      expect(result.isError).toBeUndefined();
    });

    it("still returns error when markResult(failed) fails after upstream throw", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-mr3",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockRejectedValue(new Error("upstream boom"));
      mockMarkResult.mockRejectedValue(new Error("Audit trail also down"));

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "run_query", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("upstream boom");
    });
  });

  describe("tools/call - cost tracking", () => {
    it("blocks tool call when budget is denied", async () => {
      mockedIsToolGated.mockReturnValue(false);

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({
          allowed: false,
          denied: true,
          remaining: 5,
        }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "expensive_tool", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("budget exceeded");
      expect(mockCallTool).not.toHaveBeenCalled();
    });

    it("forwards tool call and reports event when budget allows", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "result" }],
      });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({
          allowed: true,
          reservationId: "rsv-1",
        }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "read_file", arguments: { path: "/test" } },
      })) as { content: Array<{ text: string }> };

      expect(result.content[0].text).toBe("result");
      expect(mockCostTracker.reportEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          toolName: "read_file",
          serverName: "test-server",
          costMicrodollars: 10_000,
          status: "success",
          reservationId: "rsv-1",
        }),
      );
    });

    it("reports error status when tool call fails", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockRejectedValue(new Error("upstream crash"));

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "read_file", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(mockCostTracker.reportEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          status: "error",
        }),
      );
    });

    it("skips budget check for free tools (cost 0)", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "data" }],
      });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(0),
        checkBudget: vi.fn(),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "free_tool", arguments: {} } });

      expect(mockCostTracker.checkBudget).not.toHaveBeenCalled();
      expect(mockCallTool).toHaveBeenCalled();
      expect(mockCostTracker.reportEvent).toHaveBeenCalled();
    });

    it("uses tool annotations from cachedTools for cost estimation", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "ok" }],
      });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(0),
        checkBudget: vi.fn(),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const cachedTools = [
        {
          name: "read_file",
          description: "Read a file",
          inputSchema: { type: "object" as const },
          annotations: { readOnlyHint: true },
        },
      ];

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        cachedTools,
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "read_file", arguments: {} } });

      expect(mockCostTracker.resolveToolCost).toHaveBeenCalledWith("read_file");
    });

    it("works without cost tracker (backward compatible)", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "result" }],
      });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        // No costTracker
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "tool", arguments: {} },
      })) as { content: Array<{ text: string }> };

      expect(result.content[0].text).toBe("result");
    });

    it("reports denied event when budget blocks the call", async () => {
      mockedIsToolGated.mockReturnValue(false);

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({
          allowed: false,
          denied: true,
          remaining: 5,
        }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({
        params: { name: "expensive_tool", arguments: {} },
      });

      expect(mockCostTracker.reportEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          toolName: "expensive_tool",
          serverName: "test-server",
          durationMs: 0,
          costMicrodollars: 0,
          status: "denied",
        }),
      );
    });

    it("includes actionId from gated call in cost event", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-cost-1",
        decision: "approved",
      } satisfies GateResult);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "ok" }],
      });
      mockMarkResult.mockResolvedValue({ id: "act-cost-1", status: "executed" });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({
          allowed: true,
          reservationId: "rsv-g1",
        }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({
        params: { name: "run_query", arguments: { sql: "SELECT 1" } },
      });

      expect(mockCostTracker.reportEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          toolName: "run_query",
          actionId: "act-cost-1",
          reservationId: "rsv-g1",
          status: "success",
        }),
      );
    });

    it("reports durationMs that measures upstream time for non-gated calls", async () => {
      mockedIsToolGated.mockReturnValue(false);

      // Upstream takes ~10ms
      mockCallTool.mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 10));
        return { content: [{ type: "text", text: "ok" }] };
      });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(0),
        checkBudget: vi.fn(),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "tool", arguments: {} } });

      const event = mockCostTracker.reportEvent.mock.calls[0][0];
      expect(event.durationMs).toBeGreaterThanOrEqual(5);
      expect(event.durationMs).toBeLessThan(5000);
    });

    it("reports only upstream durationMs for gated calls, not gate wait", async () => {
      mockedIsToolGated.mockReturnValue(true);

      // Gate wait: ~30ms (simulated)
      mockedGateToolCall.mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 30));
        return { actionId: "act-dur", decision: "approved" as const };
      });

      // Upstream: ~10ms
      mockCallTool.mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 10));
        return { content: [{ type: "text", text: "ok" }] };
      });
      mockMarkResult.mockResolvedValue({});

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "tool", arguments: {} } });

      const event = mockCostTracker.reportEvent.mock.calls[0][0];
      // Duration should be ~10ms (upstream), not ~40ms+ (gate + upstream)
      expect(event.durationMs).toBeLessThan(25);
    });

    it("reports durationMs=0 for rejected gated calls (no upstream execution)", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-rej-dur",
        decision: "rejected",
      } satisfies GateResult);

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "tool", arguments: {} } });

      const event = mockCostTracker.reportEvent.mock.calls[0][0];
      // No upstream call happened, so duration should be 0
      expect(event.durationMs).toBe(0);
    });

    it("includes actionId for rejected gated calls", async () => {
      mockedIsToolGated.mockReturnValue(true);
      mockedGateToolCall.mockResolvedValue({
        actionId: "act-rej-1",
        decision: "rejected",
      } satisfies GateResult);

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({
        params: { name: "delete_file", arguments: {} },
      });

      expect(mockCostTracker.reportEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          actionId: "act-rej-1",
          status: "error",
        }),
      );
    });

    it("handles tool not in cachedTools (annotations undefined, TIER_READ default)", async () => {
      mockedIsToolGated.mockReturnValue(false);
      mockCallTool.mockResolvedValue({
        content: [{ type: "text", text: "ok" }],
      });

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(10_000),
        checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      // cachedTools does NOT contain "dynamic_tool"
      const cachedTools = [
        { name: "other_tool", description: "Other", inputSchema: { type: "object" as const } },
      ];

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        cachedTools,
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      await handler({ params: { name: "dynamic_tool", arguments: {} } });

      expect(mockCostTracker.resolveToolCost).toHaveBeenCalledWith("dynamic_tool");
      expect(mockCallTool).toHaveBeenCalled();
    });

    it("budget check happens before gate check", async () => {
      const callOrder: string[] = [];

      const mockCostTracker = {
        resolveToolCost: vi.fn().mockReturnValue(100_000),
        checkBudget: vi.fn().mockImplementation(async () => {
          callOrder.push("budget");
          return { allowed: false, denied: true, remaining: 0 };
        }),
        reportEvent: vi.fn(),
        config: { serverName: "test-server" },
        shutdown: vi.fn(),
      };

      mockedIsToolGated.mockImplementation(() => {
        callOrder.push("gate");
        return true;
      });
      mockedGateToolCall.mockImplementation(async () => {
        callOrder.push("gateCall");
        return { actionId: "act-1", decision: "approved" as const };
      });

      const fakeServer = makeFakeServer();
      const sdk = new NullSpend({ baseUrl: "http://test", apiKey: "key" });

      registerProxyHandlers(
        fakeServer as unknown as import("@modelcontextprotocol/sdk/server/index.js").Server,
        makeFakeUpstreamClient(),
        sdk,
        makeConfig(),
        [],
        new AbortController().signal,
        mockCostTracker as any,
      );

      const handler = fakeServer.callToolHandler!;
      const result = (await handler({
        params: { name: "tool", arguments: {} },
      })) as { content: Array<{ text: string }>; isError: boolean };

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("budget exceeded");
      // Budget check should have been called, but gate check should NOT
      expect(callOrder).toEqual(["budget"]);
      expect(mockedGateToolCall).not.toHaveBeenCalled();
    });
  });
});
