import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { NullSpend, NullSpendError, TimeoutError, waitWithAbort } from "@nullspend/sdk";
import type { McpServerConfig } from "./config.js";
import {
  successResult,
  errorResult,
  formatActionForResponse,
} from "./output.js";

function classifiedError(err: unknown): string {
  if (err instanceof NullSpendError) {
    if (err.statusCode === 401 || err.statusCode === 403) return `NullSpend API error (auth): ${err.message}`;
    if (err.statusCode === 429) return `NullSpend API error (rate_limit): ${err.message}`;
    if (err.statusCode && err.statusCode >= 500) return `NullSpend API error (server): ${err.message}`;
  }
  if (err instanceof Error && /timeout|abort/i.test(err.message)) return `NullSpend API error (timeout): ${err.message}`;
  return `NullSpend API error: ${err instanceof Error ? err.message : String(err)}`;
}

const DEFAULT_TIMEOUT_SECONDS = 300;

const MCP_METADATA = {
  sourceFramework: "mcp",
  transport: "stdio",
} as const;

export function registerTools(
  server: McpServer,
  config: McpServerConfig,
  shutdownSignal: AbortSignal,
) {
  const sdk = new NullSpend({
    baseUrl: config.nullspendUrl,
    apiKey: config.nullspendApiKey,
  });

  registerProposeAction(server, sdk, config, shutdownSignal);
  registerCheckAction(server, sdk);
  registerGetBudgets(server, sdk);
  registerGetSpendSummary(server, sdk);
  registerGetRecentCosts(server, sdk);
}

function registerProposeAction(
  server: McpServer,
  sdk: NullSpend,
  config: McpServerConfig,
  shutdownSignal: AbortSignal,
) {
  server.tool(
    "propose_action",
    "Propose a risky action for human approval before execution. " +
      "Returns the approval decision (approved/rejected/expired) or pending status.",
    {
      actionType: z.string().describe("Type of action (e.g. send_email, http_post, db_write)"),
      payload: z.record(z.string(), z.unknown()).describe("Action payload with relevant details"),
      summary: z.string().describe("Human-readable summary of what this action will do"),
      agentId: z.string().optional().describe("Identifier for the agent proposing this action"),
      metadata: z.record(z.string(), z.unknown()).optional().describe("Additional metadata"),
      timeoutSeconds: z.number().optional().describe(
        `Seconds to wait for a decision (default: ${DEFAULT_TIMEOUT_SECONDS})`,
      ),
      waitForDecision: z.boolean().optional().describe(
        "If true (default), block until approved/rejected/expired. If false, return immediately with pending status.",
      ),
    },
    async (params) => {
      try {
        const agentId = params.agentId ?? config.agentId;
        const wait = params.waitForDecision ?? true;
        const timeoutMs = (params.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS) * 1_000;

        const { id } = await sdk.createAction({
          agentId,
          actionType: params.actionType,
          payload: params.payload,
          metadata: {
            ...params.metadata,
            ...MCP_METADATA,
            summary: params.summary,
          },
          expiresInSeconds: params.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS,
        });

        if (!wait) {
          return successResult({
            actionId: id,
            status: "pending",
            approved: false,
            rejected: false,
            timedOut: false,
            message: `Action ${id} created. Use check_action to poll for the decision.`,
          });
        }

        try {
          const action = await waitWithAbort(sdk, id, timeoutMs, shutdownSignal);
          const data = formatActionForResponse(action);
          const statusLabel =
            action.status === "approved"
              ? "approved"
              : action.status === "rejected"
                ? "rejected"
                : action.status;

          return successResult({
            ...data,
            message: `Action ${id} was ${statusLabel}.`,
          });
        } catch (err) {
          if (err instanceof TimeoutError) {
            return successResult({
              actionId: id,
              status: "pending",
              approved: false,
              rejected: false,
              timedOut: true,
              message: `Timed out waiting for decision on action ${id}. Use check_action to poll later.`,
            });
          }
          throw err;
        }
      } catch (err) {
        return errorResult(classifiedError(err));
      }
    },
  );
}

function registerCheckAction(server: McpServer, sdk: NullSpend) {
  server.tool(
    "check_action",
    "Check the current status of a previously proposed action.",
    {
      actionId: z.string().describe("The ID of the action to check"),
    },
    async (params) => {
      try {
        const action = await sdk.getAction(params.actionId);
        const data = formatActionForResponse(action);
        return successResult({
          ...data,
          message: `Action ${params.actionId} is currently ${action.status}.`,
        });
      } catch (err) {
        return errorResult(classifiedError(err));
      }
    },
  );
}

// ---------------------------------------------------------------------------
// Cost awareness tools
// ---------------------------------------------------------------------------

function registerGetBudgets(server: McpServer, sdk: NullSpend) {
  server.tool(
    "get_budgets",
    "Get current budget limits and spend for this API key's organization. " +
      "Shows how much budget remains before requests are blocked.",
    {},
    async () => {
      try {
        const { data: budgets } = await sdk.listBudgets();

        if (budgets.length === 0) {
          return successResult({
            budgets: [],
            message: "No budgets configured. All requests are allowed without spending limits.",
          });
        }

        const formatted = budgets.map((b) => ({
          entityType: b.entityType,
          entityId: b.entityId,
          limitDollars: b.maxBudgetMicrodollars / 1_000_000,
          spendDollars: b.spendMicrodollars / 1_000_000,
          remainingDollars: Math.max(0, (b.maxBudgetMicrodollars - b.spendMicrodollars)) / 1_000_000,
          percentUsed: b.maxBudgetMicrodollars > 0
            ? Math.min(100, Math.round((b.spendMicrodollars / b.maxBudgetMicrodollars) * 100))
            : 0,
          policy: b.policy,
          resetInterval: b.resetInterval,
        }));

        return successResult({
          budgets: formatted,
          message: `${budgets.length} budget(s) found.`,
        });
      } catch (err) {
        return errorResult(classifiedError(err));
      }
    },
  );
}

function registerGetSpendSummary(server: McpServer, sdk: NullSpend) {
  server.tool(
    "get_spend_summary",
    "Get aggregated spending data for a time period — total cost, request count, " +
      "and breakdown by model and provider.",
    {
      period: z.enum(["7d", "30d", "90d"]).optional().describe(
        "Time period to summarize. Default: 30d",
      ),
    },
    async (params) => {
      try {
        const period = params.period ?? "30d";
        const summary = await sdk.getCostSummary(period);

        return successResult({
          period,
          totalCostDollars: summary.totals.totalCostMicrodollars / 1_000_000,
          totalRequests: summary.totals.totalRequests,
          totalInputTokens: summary.totals.totalInputTokens,
          totalOutputTokens: summary.totals.totalOutputTokens,
          costByModel: Object.fromEntries(
            Object.entries(summary.models).map(([model, cost]) => [model, cost / 1_000_000]),
          ),
          costByProvider: Object.fromEntries(
            Object.entries(summary.providers).map(([provider, cost]) => [provider, cost / 1_000_000]),
          ),
          message: `Spend summary for the last ${period}: $${(summary.totals.totalCostMicrodollars / 1_000_000).toFixed(2)} across ${summary.totals.totalRequests} requests.`,
        });
      } catch (err) {
        return errorResult(classifiedError(err));
      }
    },
  );
}

const MAX_RECENT_COSTS = 50;

function registerGetRecentCosts(server: McpServer, sdk: NullSpend) {
  server.tool(
    "get_recent_costs",
    "List the most recent API call costs with model, tokens, and cost for each request.",
    {
      limit: z.number().optional().describe(
        "Number of recent cost events to return. Default: 10, max: 50.",
      ),
    },
    async (params) => {
      try {
        const limit = Math.min(Math.max(params.limit ?? 10, 1), MAX_RECENT_COSTS);
        const { data: events } = await sdk.listCostEvents({ limit });

        const formatted = events.map((e) => ({
          model: e.model,
          provider: e.provider,
          inputTokens: e.inputTokens,
          outputTokens: e.outputTokens,
          costDollars: e.costMicrodollars / 1_000_000,
          durationMs: e.durationMs,
          createdAt: e.createdAt,
        }));

        const totalCost = events.reduce((sum, e) => sum + e.costMicrodollars, 0) / 1_000_000;

        return successResult({
          events: formatted,
          count: formatted.length,
          totalCostDollars: totalCost,
          message: `${formatted.length} recent cost event(s). Total: $${totalCost.toFixed(2)}.`,
        });
      } catch (err) {
        return errorResult(classifiedError(err));
      }
    },
  );
}
