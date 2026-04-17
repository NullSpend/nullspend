import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig, ConfigError } from "./config.js";
import { registerTools } from "./tools.js";
import { runSetup } from "./setup.js";

async function main() {
  // Subcommand dispatch — `nullspend-mcp setup [...]` runs the config-snippet
  // generator instead of starting the server. (D-5)
  const argv = process.argv.slice(2);
  if (argv[0] === "setup") {
    const code = runSetup(argv.slice(1));
    process.exit(code);
  }

  let config;
  try {
    config = loadConfig();
  } catch (err) {
    if (err instanceof ConfigError) {
      process.stderr.write(`[nullspend-mcp] ${err.message}\n`);
      process.exit(1);
    }
    throw err;
  }

  const shutdownController = new AbortController();

  const mcpServer = new McpServer(
    { name: "nullspend", version: "0.1.0" },
    { capabilities: { tools: {} } },
  );

  registerTools(mcpServer, config, shutdownController.signal);

  const transport = new StdioServerTransport();

  let isShuttingDown = false;
  async function shutdown() {
    if (isShuttingDown) return;
    isShuttingDown = true;
    shutdownController.abort();
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
    `[nullspend-mcp] Server running (stdio). API → ${config.nullspendUrl}\n`,
  );
}

main().catch((err) => {
  process.stderr.write(`[nullspend-mcp] Fatal: ${String(err)}\n`);
  process.exit(1);
});
