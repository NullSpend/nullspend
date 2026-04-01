import type { ActionRecord } from "@nullspend/sdk";

export interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
}

export function successResult(data: Record<string, unknown>): ToolResult {
  return {
    content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
  };
}

export function errorResult(message: string): ToolResult {
  return {
    content: [{ type: "text", text: message }],
    isError: true,
  };
}

export function formatActionForResponse(action: ActionRecord) {
  return {
    actionId: action.id,
    status: action.status,
    actionType: action.actionType,
    approved: action.status === "approved",
    rejected: action.status === "rejected",
    timedOut: action.status === "expired",
    createdAt: action.createdAt,
    approvedAt: action.approvedAt,
    rejectedAt: action.rejectedAt,
    executedAt: action.executedAt,
    result: action.result,
    errorMessage: action.errorMessage,
  };
}
