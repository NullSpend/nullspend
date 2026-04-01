import { describe, it, expect, vi, beforeEach } from "vitest";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

// Mock the content module with test fixtures
vi.mock("./content.js", () => {
  const docs = [
    {
      path: "quickstart/openai",
      title: "OpenAI Quickstart",
      description: "Get cost tracking for your OpenAI calls in under 2 minutes.",
      content: "Set two environment variables and add an X-NullSpend-Key header.",
    },
    {
      path: "features/budgets",
      title: "Budget Enforcement",
      description: "Hard spending ceilings.",
      content: "Budget enforcement prevents runaway costs.",
    },
    {
      path: "features/human-in-the-loop",
      title: "Human-in-the-Loop",
      description: "Approval workflow for high-cost or sensitive operations.",
      content: "HITL allows human review before agents execute risky actions.",
    },
    {
      path: "llms.txt",
      title: "LLM-Readable API Reference",
      description: "Machine-readable NullSpend overview for AI agents.",
      content: "NullSpend is a FinOps layer for AI agents.",
    },
  ];

  return {
    DOCS: docs,
    DOCS_BY_PATH: new Map(docs.map((d: { path: string }) => [d.path, d])),
  };
});

import { registerTools } from "./tools.js";

interface ToolRegistration {
  name: string;
  description: string;
  cb: (params: Record<string, unknown>) => Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }>;
}

function captureTools() {
  const tools: ToolRegistration[] = [];
  const fakeServer = {
    tool: vi.fn((...args: unknown[]) => {
      const name = args[0] as string;
      const description = args[1] as string;
      // Schema is args[2], handler is args[3]
      const cb = args[args.length - 1] as ToolRegistration["cb"];
      tools.push({ name, description, cb });
    }),
  } as unknown as McpServer;

  registerTools(fakeServer);
  return tools;
}

function getToolByName(tools: ToolRegistration[], name: string) {
  const tool = tools.find((t) => t.name === name);
  if (!tool) throw new Error(`Tool ${name} not found`);
  return tool;
}

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  const text = result.content[0].text;
  try {
    return { data: JSON.parse(text), isError: result.isError ?? false };
  } catch {
    return { data: text, isError: result.isError ?? false };
  }
}

describe("registerTools", () => {
  it("registers exactly 2 tools", () => {
    const tools = captureTools();
    expect(tools).toHaveLength(2);
  });

  it("registers nullspend_search_docs and nullspend_fetch_doc", () => {
    const tools = captureTools();
    const names = tools.map((t) => t.name);
    expect(names).toContain("nullspend_search_docs");
    expect(names).toContain("nullspend_fetch_doc");
  });
});

describe("nullspend_search_docs", () => {
  let tools: ToolRegistration[];

  beforeEach(() => {
    tools = captureTools();
  });

  it("returns ranked results for known query", async () => {
    const tool = getToolByName(tools, "nullspend_search_docs");
    const result = await tool.cb({ query: "openai quickstart" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.results.length).toBeGreaterThan(0);
    expect(data.results[0].path).toBe("quickstart/openai");
    expect(data.query).toBe("openai quickstart");
  });

  it("returns results with path field usable by fetch_doc", async () => {
    const searchTool = getToolByName(tools, "nullspend_search_docs");
    const fetchTool = getToolByName(tools, "nullspend_fetch_doc");

    const searchResult = await searchTool.cb({ query: "budget" });
    const { data: searchData } = parseResult(searchResult);

    expect(searchData.results.length).toBeGreaterThan(0);

    // Use the path from search result to fetch
    const fetchResult = await fetchTool.cb({ path: searchData.results[0].path });
    const { data: fetchData, isError } = parseResult(fetchResult);

    expect(isError).toBe(false);
    expect(fetchData.content).toBeDefined();
    expect(typeof fetchData.content).toBe("string");
  });

  it("returns empty results with message for no matches", async () => {
    const tool = getToolByName(tools, "nullspend_search_docs");
    const result = await tool.cb({ query: "xyzzy foobarbaz nonsense" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.results).toHaveLength(0);
    expect(data.message).toContain("No relevant docs found");
  });

  it("respects limit parameter", async () => {
    const tool = getToolByName(tools, "nullspend_search_docs");
    const result = await tool.cb({ query: "cost", limit: 1 });
    const { data } = parseResult(result);

    expect(data.results.length).toBeLessThanOrEqual(1);
  });

  it("handles NaN limit gracefully (falls back to default)", async () => {
    const tool = getToolByName(tools, "nullspend_search_docs");
    const result = await tool.cb({ query: "budget", limit: NaN });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    // Should not return empty — NaN should fall back to default limit
    expect(data.results.length).toBeGreaterThan(0);
  });

  it("handles Infinity limit (clamped to max)", async () => {
    const tool = getToolByName(tools, "nullspend_search_docs");
    const result = await tool.cb({ query: "budget", limit: Infinity });
    const { data } = parseResult(result);

    expect(data.results.length).toBeGreaterThan(0);
  });
});

describe("nullspend_fetch_doc", () => {
  let tools: ToolRegistration[];

  beforeEach(() => {
    tools = captureTools();
  });

  it("returns content for valid path", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "quickstart/openai" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
    expect(data.description).toBe("Get cost tracking for your OpenAI calls in under 2 minutes.");
    expect(data.path).toBe("quickstart/openai");
    expect(data.content).toContain("X-NullSpend-Key");
  });

  it("normalizes path with leading slash and .md extension", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "/quickstart/openai.md" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });

  it("returns llms.txt content", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "llms.txt" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("LLM-Readable API Reference");
    expect(data.content).toContain("FinOps");
  });

  it("returns error for unknown path with helpful message", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "nonexistent/page" });
    const { isError } = parseResult(result);

    expect(isError).toBe(true);
    expect(result.content[0].text).toContain("Doc not found");
    expect(result.content[0].text).toContain("nullspend_search_docs");
  });

  it("returns not-found for path traversal attempt", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "../../package.json" });
    const { isError } = parseResult(result);

    expect(isError).toBe(true);
  });

  it("handles path with extra whitespace", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "  quickstart/openai  " });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });

  it("normalizes Windows backslash paths", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "quickstart\\openai" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });

  it("handles empty string path", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "" });
    const { isError } = parseResult(result);

    expect(isError).toBe(true);
  });

  it("normalizes uppercase paths (case-insensitive)", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "Quickstart/OpenAI" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });

  it("collapses double slashes in path", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "quickstart//openai" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });

  it("strips ./ prefix from path", async () => {
    const tool = getToolByName(tools, "nullspend_fetch_doc");
    const result = await tool.cb({ path: "./quickstart/openai" });
    const { data, isError } = parseResult(result);

    expect(isError).toBe(false);
    expect(data.title).toBe("OpenAI Quickstart");
  });
});
