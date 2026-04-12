import { describe, it, expect } from "vitest";
import { searchDocs, buildIndex, tokenize } from "./search.js";
import type { DocPage } from "./content.js";
import docsData from "./generated/docs.json" with { type: "json" };

// --- Test fixtures ---

function makeDocs(): DocPage[] {
  return [
    {
      path: "features/budgets",
      title: "Budget Enforcement",
      description: "Hard spending ceilings. The proxy returns 429 before the request reaches the provider.",
      content: "Budget enforcement prevents runaway costs. Set limits per API key, team, or organization.",
    },
    {
      path: "quickstart/openai",
      title: "OpenAI Quickstart",
      description: "Get cost tracking for your OpenAI calls in under 2 minutes.",
      content: "Set two environment variables and add an X-NullSpend-Key header.",
    },
    {
      path: "features/cost-tracking",
      title: "Cost Tracking",
      description: "Per-request cost calculation for every model.",
      content: "NullSpend calculates cost from input, output, cached, and reasoning tokens.",
    },
    {
      path: "webhooks/security",
      title: "Webhook Security",
      description: "HMAC-SHA256 signing for webhook payloads.",
      content: "Every webhook delivery includes an X-NullSpend-Signature header.",
    },
    {
      path: "features/human-in-the-loop",
      title: "Human-in-the-Loop",
      description: "Approval workflow for high-cost or sensitive operations.",
      content: "HITL allows human review before agents execute risky actions.",
    },
    {
      path: "features/tracing",
      title: "Tracing",
      description: "W3C traceparent propagation and custom trace IDs.",
      content: "Correlate requests across services with distributed tracing.",
    },
    {
      path: "sdks/python",
      title: "Python SDK",
      description: "Python client for the NullSpend API.",
      content: "Install with pip install nullspend. Supports async and sync clients.",
    },
    {
      path: "api-reference/errors",
      title: "Error Reference",
      description: "Error codes and HTTP status semantics.",
      content: "429 Too Many Requests — budget_exceeded, velocity_exceeded, or rate_limited.",
    },
    {
      path: "llms.txt",
      title: "LLM-Readable API Reference",
      description: "Machine-readable NullSpend overview for AI agents.",
      content: "NullSpend is a FinOps layer for AI agents.",
    },
    {
      path: "features/velocity-limits",
      title: "Velocity Limits",
      description: "Detect runaway loops with spend rate thresholds.",
      content: "Block when spend rate exceeds a threshold within a time window.",
    },
  ];
}

describe("tokenize", () => {
  it("lowercases and splits on non-alphanumeric", () => {
    expect(tokenize("Hello World")).toEqual(["hello", "world"]);
  });

  it("preserves hyphens", () => {
    expect(tokenize("human-in-the-loop")).toEqual(["human-in-the-loop"]);
  });

  it("returns empty array for empty string", () => {
    expect(tokenize("")).toEqual([]);
  });

  it("handles numbers", () => {
    expect(tokenize("429 error")).toEqual(["429", "error"]);
  });

  it("strips leading/trailing hyphens from tokens", () => {
    expect(tokenize("- list item")).toEqual(["list", "item"]);
    expect(tokenize("--help")).toEqual(["help"]);
  });

  it("filters hyphen-only tokens from markdown separators", () => {
    expect(tokenize("---|---|---")).toEqual([]);
    expect(tokenize("- - -")).toEqual([]);
  });

  it("preserves internal hyphens in compound words", () => {
    expect(tokenize("api-key self-hosted")).toEqual(["api-key", "self-hosted"]);
  });
});

describe("searchDocs", () => {
  const docs = makeDocs();
  const index = buildIndex(docs);

  it("'budget' matches Budget Enforcement in top 3", () => {
    const results = searchDocs("budget", index);
    const paths = results.map((r) => r.path);
    expect(paths.slice(0, 3)).toContain("features/budgets");
  });

  it("'openai quickstart' returns quickstart/openai as #1", () => {
    const results = searchDocs("openai quickstart", index);
    expect(results[0].path).toBe("quickstart/openai");
  });

  it("'webhook security' returns webhooks/security in top 3", () => {
    const results = searchDocs("webhook security", index);
    const paths = results.map((r) => r.path);
    expect(paths.slice(0, 3)).toContain("webhooks/security");
  });

  it("'HITL' matches human-in-the-loop via synonym", () => {
    const results = searchDocs("HITL", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("features/human-in-the-loop");
  });

  it("'tracing' matches features/tracing via substring", () => {
    const results = searchDocs("tracing", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("features/tracing");
  });

  it("'python SDK' matches sdks/python", () => {
    const results = searchDocs("python SDK", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("sdks/python");
  });

  it("empty query returns all docs limited", () => {
    const results = searchDocs("", index, 5);
    expect(results).toHaveLength(5);
    // All should have score 0
    expect(results.every((r) => r.score === 0)).toBe(true);
  });

  it("returns empty for gibberish query (below threshold)", () => {
    const results = searchDocs("xyzzy foobarbaz", index);
    expect(results).toHaveLength(0);
  });

  it("is case insensitive", () => {
    const upper = searchDocs("WEBHOOKS", index);
    const lower = searchDocs("webhooks", index);
    expect(upper.map((r) => r.path)).toEqual(lower.map((r) => r.path));
  });

  it("respects limit parameter", () => {
    const results = searchDocs("cost", index, 2);
    expect(results.length).toBeLessThanOrEqual(2);
  });

  it("results have deterministic ordering for same scores", () => {
    const r1 = searchDocs("cost", index);
    const r2 = searchDocs("cost", index);
    expect(r1.map((r) => r.path)).toEqual(r2.map((r) => r.path));
  });

  it("special characters in query don't crash", () => {
    expect(() => searchDocs("what's the $cost?!?", index)).not.toThrow();
    expect(() => searchDocs("budget > 100 && true", index)).not.toThrow();
  });

  it("'429' matches error reference via synonym expansion", () => {
    const results = searchDocs("429", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("api-reference/errors");
  });

  it("'spend limits' finds budgets via synonym", () => {
    const results = searchDocs("spend limits", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("features/budgets");
  });

  it("each result has all required fields", () => {
    const results = searchDocs("budget", index);
    for (const r of results) {
      expect(r).toHaveProperty("path");
      expect(r).toHaveProperty("title");
      expect(r).toHaveProperty("description");
      expect(r).toHaveProperty("score");
      expect(typeof r.path).toBe("string");
      expect(typeof r.score).toBe("number");
    }
  });

  it("synonym is bidirectional: 'spend' finds cost docs", () => {
    const results = searchDocs("spend", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("features/cost-tracking");
  });

  it("synonym is bidirectional: 'cost' finds budget/spend docs", () => {
    const results = searchDocs("cost", index);
    const paths = results.map((r) => r.path);
    expect(paths).toContain("features/budgets");
  });

  it("short token 'a' requires exact match, does not match everything", () => {
    const results = searchDocs("a", index);
    // "a" is too short for substring matching — only exact token matches
    // No doc token is exactly "a", so results should be empty or very few
    expect(results.length).toBeLessThan(index.length);
  });

  it("short token 'in' does not match 'input', 'within', etc.", () => {
    const results = searchDocs("in", index);
    // "in" requires exact match — shouldn't match most docs
    expect(results.length).toBeLessThan(5);
  });

  it("per-result threshold filters low-scoring noise", () => {
    // A query where some docs match strongly and others weakly
    const results = searchDocs("openai", index);
    // All returned results should have score >= 2
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(2);
    }
  });

  it("empty index returns empty", () => {
    const emptyIndex = buildIndex([]);
    expect(searchDocs("budget", emptyIndex)).toHaveLength(0);
    expect(searchDocs("", emptyIndex)).toHaveLength(0);
  });

  it("long query tokens don't match short content tokens like 'a' or 'e'", () => {
    // Build a doc where content has only short tokens
    const shortTokenDocs: DocPage[] = [
      {
        path: "noise",
        title: "Noise Doc",
        description: "A e i o u.",
        content: "a e i o u s t m n p r x",
      },
      {
        path: "real-match",
        title: "Rate Limiting",
        description: "Rate limits protect the system.",
        content: "Configure rate limits per API key.",
      },
    ];
    const shortIndex = buildIndex(shortTokenDocs);

    const results = searchDocs("rate limiting", shortIndex);
    // "rate" should NOT match the "a" or "e" tokens in the noise doc
    const paths = results.map((r) => r.path);
    expect(paths).toContain("real-match");
    expect(paths).not.toContain("noise");
  });
});

// --- Smoke tests against real generated docs ---

describe("searchDocs (real data)", () => {
  const realDocs = docsData as DocPage[];
  const realIndex = buildIndex(realDocs);

  it("has all 46 docs indexed", () => {
    expect(realIndex).toHaveLength(46);
  });

  it("no doc has CRLF in content", () => {
    for (const entry of realIndex) {
      expect(entry.doc.content).not.toContain("\r");
    }
  });

  it("no doc has empty title", () => {
    for (const entry of realIndex) {
      expect(entry.doc.title.length).toBeGreaterThan(0);
    }
  });

  it("no doc has empty description", () => {
    for (const entry of realIndex) {
      expect(entry.doc.description.length).toBeGreaterThan(0);
    }
  });

  it("no-frontmatter docs don't start with # heading in content", () => {
    const noFmPaths = [
      "sdks/javascript", "sdks/python", "sdks/mcp-server",
      "api-reference/custom-headers", "api-reference/proxy-endpoints", "api-reference/rate-limits",
    ];
    for (const p of noFmPaths) {
      const entry = realIndex.find((e) => e.doc.path === p);
      expect(entry, `Missing doc: ${p}`).toBeDefined();
      expect(entry!.doc.content.startsWith("# ")).toBe(false);
    }
  });

  it("all paths use forward slashes, no .md extension", () => {
    for (const entry of realIndex) {
      expect(entry.doc.path).not.toContain("\\");
      if (entry.doc.path !== "llms.txt") {
        expect(entry.doc.path).not.toMatch(/\.md$/);
      }
    }
  });

  it("'budget' finds features/budgets in real data", () => {
    const results = searchDocs("budget", realIndex);
    expect(results.map((r) => r.path)).toContain("features/budgets");
  });

  it("'openai quickstart' finds quickstart/openai in real data", () => {
    const results = searchDocs("openai quickstart", realIndex);
    expect(results[0].path).toBe("quickstart/openai");
  });

  it("'HITL' finds human-in-the-loop in real data", () => {
    const results = searchDocs("HITL", realIndex);
    expect(results.map((r) => r.path)).toContain("features/human-in-the-loop");
  });

  it("llms.txt is searchable", () => {
    const results = searchDocs("llm api reference", realIndex);
    expect(results.map((r) => r.path)).toContain("llms.txt");
  });
});
