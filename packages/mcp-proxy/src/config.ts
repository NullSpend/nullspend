export interface ProxyConfig {
  nullspendUrl: string;
  nullspendApiKey: string;
  agentId: string;
  upstreamCommand: string;
  upstreamArgs: string[];
  upstreamEnv: Record<string, string>;
  gatedTools: Set<string> | "*";
  passthroughTools: Set<string>;
  approvalTimeoutSeconds: number;
  // Cost tracking
  backendUrl: string;
  serverName: string;
  costTrackingEnabled: boolean;
  budgetEnforcementEnabled: boolean;
  toolCostOverrides: Record<string, number>;
}

const DEFAULT_AGENT_ID = "mcp-proxy";
const DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300;

export function loadConfig(): ProxyConfig {
  const nullspendUrl = process.env.NULLSPEND_URL;
  const nullspendApiKey = process.env.NULLSPEND_API_KEY;
  const upstreamCommand = process.env.UPSTREAM_COMMAND;

  const missing: string[] = [];
  if (!nullspendUrl) missing.push("NULLSPEND_URL");
  if (!nullspendApiKey) missing.push("NULLSPEND_API_KEY");
  if (!upstreamCommand) missing.push("UPSTREAM_COMMAND");

  if (missing.length > 0) {
    throw new ConfigError(
      `Missing required environment variables: ${missing.join(", ")}. ` +
        `Set them before starting the MCP proxy.`,
    );
  }

  const upstreamArgs = parseUpstreamArgs(process.env.UPSTREAM_ARGS);
  const upstreamEnv = parseUpstreamEnv(process.env.UPSTREAM_ENV);
  const gatedTools = parseGatedTools(process.env.GATED_TOOLS);
  const passthroughTools = parseToolSet(process.env.PASSTHROUGH_TOOLS);

  const approvalTimeoutRaw = process.env.APPROVAL_TIMEOUT_SECONDS;
  const approvalTimeoutSeconds = approvalTimeoutRaw
    ? Number(approvalTimeoutRaw)
    : DEFAULT_APPROVAL_TIMEOUT_SECONDS;

  if (isNaN(approvalTimeoutSeconds) || approvalTimeoutSeconds <= 0) {
    throw new ConfigError(
      `APPROVAL_TIMEOUT_SECONDS must be a positive number, got: "${approvalTimeoutRaw}"`,
    );
  }

  const costTrackingEnabled =
    process.env.NULLSPEND_COST_TRACKING !== "false";
  const budgetEnforcementEnabled =
    process.env.NULLSPEND_BUDGET_ENFORCEMENT !== "false";

  const serverNameRaw =
    process.env.NULLSPEND_SERVER_NAME ?? upstreamCommand!;
  const serverName = serverNameRaw.trim();

  if (!serverName) {
    throw new ConfigError(
      "NULLSPEND_SERVER_NAME must not be empty (resolved to empty string after trimming whitespace).",
    );
  }

  if (serverName.includes("/")) {
    throw new ConfigError(
      `NULLSPEND_SERVER_NAME must not contain '/' (got: "${serverName}"). ` +
        `The '/' character is reserved for separating server/tool in analytics.`,
    );
  }

  const toolCostOverrides = parseToolCostOverrides(
    process.env.NULLSPEND_TOOL_COSTS,
  );

  return {
    nullspendUrl: nullspendUrl!,
    nullspendApiKey: nullspendApiKey!,
    agentId: process.env.NULLSPEND_AGENT_ID ?? DEFAULT_AGENT_ID,
    upstreamCommand: upstreamCommand!,
    upstreamArgs,
    upstreamEnv,
    gatedTools,
    passthroughTools,
    approvalTimeoutSeconds,
    backendUrl: nullspendUrl!,
    serverName,
    costTrackingEnabled,
    budgetEnforcementEnabled,
    toolCostOverrides,
  };
}

function parseUpstreamArgs(raw: string | undefined): string[] {
  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      throw new ConfigError(
        `UPSTREAM_ARGS must be a JSON array of strings, got: ${typeof parsed}`,
      );
    }
    return parsed.map(String);
  } catch (err) {
    if (err instanceof ConfigError) throw err;
    throw new ConfigError(
      `UPSTREAM_ARGS is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function parseUpstreamEnv(raw: string | undefined): Record<string, string> {
  if (!raw) return {};

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new ConfigError(
        `UPSTREAM_ENV must be a JSON object, got: ${Array.isArray(parsed) ? "array" : typeof parsed}`,
      );
    }
    const result: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed)) {
      result[key] = String(value);
    }
    return result;
  } catch (err) {
    if (err instanceof ConfigError) throw err;
    throw new ConfigError(
      `UPSTREAM_ENV is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function parseGatedTools(raw: string | undefined): Set<string> | "*" {
  if (raw === undefined) return "*";
  if (raw.trim() === "*") return "*";
  return parseToolSet(raw);
}

function parseToolSet(raw: string | undefined): Set<string> {
  if (!raw) return new Set();
  return new Set(
    raw
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0),
  );
}

function parseToolCostOverrides(
  raw: string | undefined,
): Record<string, number> {
  if (!raw) return {};

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new ConfigError(
        `NULLSPEND_TOOL_COSTS must be a JSON object, got: ${Array.isArray(parsed) ? "array" : typeof parsed}`,
      );
    }
    const result: Record<string, number> = {};
    for (const [key, value] of Object.entries(parsed)) {
      const num = Number(value);
      if (!Number.isFinite(num) || num < 0) {
        throw new ConfigError(
          `NULLSPEND_TOOL_COSTS: value for "${key}" must be a non-negative number, got: ${String(value)}`,
        );
      }
      result[key] = num;
    }
    return result;
  } catch (err) {
    if (err instanceof ConfigError) throw err;
    throw new ConfigError(
      `NULLSPEND_TOOL_COSTS is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}
