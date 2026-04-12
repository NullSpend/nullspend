import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.js";
import { registerTools } from "./tools.js";
import { DOCS } from "./content.js";

async function main() {
  const config = loadConfig();

  const mcpServer = new McpServer(
    { name: config.serverName, version: config.version },
    { capabilities: { tools: {} } },
  );

  registerTools(mcpServer);

  const transport = new StdioServerTransport();

  let isShuttingDown = false;
  async function shutdown() {
    if (isShuttingDown) return;
    isShuttingDown = true;
    try {
      await mcpServer.close();
    } catch {
      // best-effort close
    }
    process.exit(0);
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  process.stdin.on("close", shutdown);

  await mcpServer.connect(transport);
  process.stderr.write(
    `[${config.serverName}] Server running (stdio). ${DOCS.length} docs indexed.\n`,
  );
}

main().catch((err) => {
  process.stderr.write(`[nullspend-docs] Fatal: ${String(err)}\n`);
  process.exit(1);
});
