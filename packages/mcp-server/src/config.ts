export interface McpServerConfig {
  nullspendUrl: string;
  nullspendApiKey: string;
  agentId: string;
}

export function loadConfig(): McpServerConfig {
  const nullspendUrl = process.env.NULLSPEND_URL;
  const nullspendApiKey = process.env.NULLSPEND_API_KEY;
  const agentId = process.env.NULLSPEND_AGENT_ID ?? "mcp-agent";

  const missing: string[] = [];
  if (!nullspendUrl) missing.push("NULLSPEND_URL");
  if (!nullspendApiKey) missing.push("NULLSPEND_API_KEY");

  if (missing.length > 0) {
    throw new ConfigError(
      `Missing required environment variables: ${missing.join(", ")}. ` +
        `Set them before starting the MCP server.`,
    );
  }

  return {
    nullspendUrl: nullspendUrl!,
    nullspendApiKey: nullspendApiKey!,
    agentId,
  };
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}
