import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import type { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { NullSpend } from "@nullspend/sdk";
import type { ProxyConfig } from "./config.js";
import { isToolGated, gateToolCall } from "./gate.js";
import type { CostTracker } from "./cost-tracker.js";

export async function discoverUpstreamTools(
  upstreamClient: Client,
): Promise<Tool[]> {
  const allTools: Tool[] = [];
  let cursor: string | undefined;

  do {
    const response = await upstreamClient.listTools(
      cursor ? { cursor } : undefined,
    );
    allTools.push(...response.tools);
    cursor = response.nextCursor;
  } while (cursor);

  return allTools;
}

export function registerProxyHandlers(
  server: Server,
  upstreamClient: Client,
  sdk: NullSpend,
  config: ProxyConfig,
  cachedTools: Tool[],
  shutdownSignal: AbortSignal,
  costTracker?: CostTracker,
): void {
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: cachedTools,
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    // --- Cost tracking: estimate & budget check ---
    let costEstimate = 0;
    let reservationId: string | undefined;

    if (costTracker) {
      costEstimate = costTracker.resolveToolCost(name);

      if (costEstimate > 0) {
        const budgetResult = await costTracker.checkBudget(name, costEstimate);
        if (!budgetResult.allowed) {
          costTracker.reportEvent({
            toolName: name,
            serverName: costTracker.config.serverName,
            durationMs: 0,
            costMicrodollars: 0,
            status: "denied",
          });
          return {
            content: [
              {
                type: "text",
                text: `Tool "${name}" blocked: budget exceeded. Remaining: ${budgetResult.remaining ?? 0} microdollars.`,
              },
            ],
            isError: true,
          };
        }
        reservationId = budgetResult.reservationId;
      }
    }

    // --- Gate check (existing approval flow) ---
    let result: { content: unknown[]; isError?: boolean };
    let actionId: string | undefined;
    let durationMs: number;

    if (!isToolGated(name, config)) {
      const startTime = performance.now();
      result = await forwardToUpstream(upstreamClient, name, args);
      durationMs = Math.round(performance.now() - startTime);
    } else {
      const gatedResult = await handleGatedCall(upstreamClient, sdk, name, args, config, shutdownSignal);
      result = gatedResult;
      actionId = gatedResult.actionId;
      durationMs = gatedResult.upstreamDurationMs ?? 0;
    }

    // --- Cost tracking: report event ---
    if (costTracker) {
      costTracker.reportEvent({
        toolName: name,
        serverName: costTracker.config.serverName,
        durationMs,
        costMicrodollars: costEstimate,
        status: result.isError ? "error" : "success",
        reservationId,
        actionId,
      });
    }

    return result;
  });
}

async function forwardToUpstream(
  upstreamClient: Client,
  name: string,
  args: Record<string, unknown> | undefined,
): Promise<{ content: unknown[]; isError?: boolean }> {
  try {
    const result = await upstreamClient.callTool({
      name,
      arguments: args,
    });
    return result as { content: unknown[]; isError?: boolean };
  } catch (err) {
    return {
      content: [
        {
          type: "text",
          text: `Upstream error: ${err instanceof Error ? err.message : String(err)}`,
        },
      ],
      isError: true,
    };
  }
}

async function handleGatedCall(
  upstreamClient: Client,
  sdk: NullSpend,
  name: string,
  args: Record<string, unknown> | undefined,
  config: ProxyConfig,
  signal: AbortSignal,
): Promise<{ content: unknown[]; isError?: boolean; actionId?: string; upstreamDurationMs?: number }> {
  let gateResult;
  try {
    gateResult = await gateToolCall(sdk, name, args, config, signal);
  } catch (err) {
    return {
      content: [
        {
          type: "text",
          text: `Failed to reach approval service: ${err instanceof Error ? err.message : String(err)}`,
        },
      ],
      isError: true,
    };
  }

  if (gateResult.decision === "rejected") {
    return {
      content: [
        {
          type: "text",
          text: `Action "${name}" was rejected by a human reviewer.`,
        },
      ],
      isError: true,
      actionId: gateResult.actionId,
    };
  }

  if (gateResult.decision === "timedOut") {
    return {
      content: [
        {
          type: "text",
          text: `Approval for "${name}" timed out after ${config.approvalTimeoutSeconds} seconds. The action was not executed.`,
        },
      ],
      isError: true,
      actionId: gateResult.actionId,
    };
  }

  try {
    await sdk.markResult(gateResult.actionId, { status: "executing" });
  } catch {
    // best-effort; don't block the call if this fails
  }

  const upstreamStart = performance.now();
  try {
    const result = await upstreamClient.callTool({
      name,
      arguments: args,
    });
    const upstreamDurationMs = Math.round(performance.now() - upstreamStart);

    const typedResult = result as { content: unknown[]; isError?: boolean };

    if (typedResult.isError) {
      const errorText =
        typedResult.content?.[0] &&
        typeof typedResult.content[0] === "object" &&
        typedResult.content[0] !== null &&
        "text" in typedResult.content[0]
          ? String((typedResult.content[0] as { text: string }).text)
          : "Unknown upstream error";

      try {
        await sdk.markResult(gateResult.actionId, {
          status: "failed",
          errorMessage: errorText,
        });
      } catch {
        // best-effort audit trail
      }

      return { ...typedResult, actionId: gateResult.actionId, upstreamDurationMs };
    }

    try {
      await sdk.markResult(gateResult.actionId, {
        status: "executed",
        result: typedResult as unknown as Record<string, unknown>,
      });
    } catch {
      // best-effort audit trail
    }

    return { ...typedResult, actionId: gateResult.actionId, upstreamDurationMs };
  } catch (err) {
    const upstreamDurationMs = Math.round(performance.now() - upstreamStart);
    const errorMessage = err instanceof Error ? err.message : String(err);

    try {
      await sdk.markResult(gateResult.actionId, {
        status: "failed",
        errorMessage,
      });
    } catch {
      // best-effort audit trail
    }

    return {
      content: [
        {
          type: "text",
          text: `Upstream call failed after approval: ${errorMessage}`,
        },
      ],
      isError: true,
      actionId: gateResult.actionId,
      upstreamDurationMs,
    };
  }
}
