import { NullSpend, TimeoutError, waitWithAbort } from "@nullspend/sdk";
import type { ProxyConfig } from "./config.js";

export interface GateResult {
  actionId: string;
  decision: "approved" | "rejected" | "timedOut";
}

const PROXY_METADATA = {
  sourceFramework: "mcp-proxy",
  transport: "stdio",
} as const;

export function isToolGated(toolName: string, config: ProxyConfig): boolean {
  if (config.passthroughTools.has(toolName)) return false;
  if (config.gatedTools === "*") return true;
  return config.gatedTools.has(toolName);
}

export async function gateToolCall(
  sdk: NullSpend,
  toolName: string,
  args: Record<string, unknown> | undefined,
  config: ProxyConfig,
  signal: AbortSignal,
): Promise<GateResult> {
  const argsPreview = JSON.stringify(args ?? {}).slice(0, 200);
  const summary = `Tool call: ${toolName}(${argsPreview})`;

  const { id } = await sdk.createAction({
    agentId: config.agentId,
    actionType: toolName,
    payload: args ?? {},
    metadata: {
      ...PROXY_METADATA,
      upstreamTool: toolName,
      summary,
    },
    expiresInSeconds: config.approvalTimeoutSeconds,
  });

  const timeoutMs = config.approvalTimeoutSeconds * 1_000;

  try {
    const action = await waitWithAbort(sdk, id, timeoutMs, signal);

    if (action.status === "approved") {
      return { actionId: id, decision: "approved" };
    }
    if (action.status === "expired") {
      return { actionId: id, decision: "timedOut" };
    }
    return { actionId: id, decision: "rejected" };
  } catch (err) {
    if (err instanceof TimeoutError) {
      return { actionId: id, decision: "timedOut" };
    }
    throw err;
  }
}
