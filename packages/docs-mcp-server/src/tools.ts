import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { DOCS, DOCS_BY_PATH } from "./content.js";
import type { DocPage } from "./content.js";
import { buildIndex, searchDocs } from "./search.js";
import type { DocIndex } from "./search.js";
import { successResult, errorResult } from "./output.js";

const MAX_LIMIT = 20;
const DEFAULT_LIMIT = 10;

/** Normalize a fetch path: backslashes, case, slashes, ./ prefix, trailing .md */
function normalizePath(raw: string): string {
  let p = raw.trim().toLowerCase().replace(/\\/g, "/");
  // Collapse consecutive slashes
  p = p.replace(/\/+/g, "/");
  // Strip leading ./ or /
  p = p.replace(/^(\.\/|\/)+/, "");
  // Strip trailing .md
  if (p.endsWith(".md")) {
    p = p.slice(0, -3);
  }
  return p;
}

export function registerTools(server: McpServer): void {
  const index: DocIndex[] = buildIndex(DOCS);

  server.tool(
    "nullspend_search_docs",
    "Search NullSpend documentation. Returns ranked results with title, description, and path. " +
      "Use nullspend_fetch_doc with the returned path to read the full page.",
    {
      query: z.string().describe("Search query — e.g. 'budget enforcement', 'openai quickstart', 'webhook security'"),
      limit: z.number().optional().describe(`Max results to return (default: ${DEFAULT_LIMIT}, max: ${MAX_LIMIT})`),
    },
    async (params) => {
      const rawLimit = params.limit ?? DEFAULT_LIMIT;
      const limit = Math.min(Math.max(Number.isFinite(rawLimit) ? rawLimit : DEFAULT_LIMIT, 1), MAX_LIMIT);
      const results = searchDocs(params.query, index, limit);

      if (results.length === 0) {
        return successResult({
          results: [],
          total: 0,
          query: params.query,
          message: "No relevant docs found. Try different search terms.",
        });
      }

      return successResult({
        results,
        total: results.length,
        query: params.query,
      });
    },
  );

  server.tool(
    "nullspend_fetch_doc",
    "Fetch the full content of a NullSpend documentation page. " +
      "Use the path from nullspend_search_docs results.",
    {
      path: z.string().describe("Doc path — e.g. 'quickstart/openai', 'features/budgets', 'llms.txt'"),
    },
    async (params) => {
      const normalized = normalizePath(params.path);
      const doc: DocPage | undefined = DOCS_BY_PATH.get(normalized);

      if (!doc) {
        return errorResult(
          `Doc not found: "${params.path}". Use nullspend_search_docs to find available pages.`,
        );
      }

      return successResult({
        title: doc.title,
        description: doc.description,
        path: doc.path,
        content: doc.content,
      });
    },
  );
}
