import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { NullSpend } from "@nullspend/sdk";
import { loadConfig, ConfigError } from "./config.js";
import { discoverUpstreamTools, registerProxyHandlers } from "./proxy.js";
import { isToolGated } from "./gate.js";
import { CostTracker, ToolCostRegistry, suggestToolCost } from "./cost-tracker.js";

const LOG_PREFIX = "[nullspend-proxy]";

function log(message: string): void {
  process.stderr.write(`${LOG_PREFIX} ${message}\n`);
}

async function main() {
  let config;
  try {
    config = loadConfig();
  } catch (err) {
    if (err instanceof ConfigError) {
      log(err.message);
      process.exit(1);
    }
    throw err;
  }

  const shutdownController = new AbortController();

  const upstreamTransport = new StdioClientTransport({
    command: config.upstreamCommand,
    args: config.upstreamArgs,
    env: { ...process.env, ...config.upstreamEnv } as Record<string, string>,
  });

  const upstreamClient = new Client(
    { name: "nullspend-proxy", version: "0.1.0" },
    {
      capabilities: {},
      listChanged: {
        tools: {
          onChanged: (error: Error | null) => {
            if (error) {
              log(`ERROR: Failed to refresh tool list: ${error.message}`);
              return;
            }
            log("WARNING: Upstream tool list changed. Restart the proxy to pick up new tools.");
          },
        },
      },
    },
  );

  log(`Connecting to upstream: ${config.upstreamCommand} ${config.upstreamArgs.join(" ")}`);
  await upstreamClient.connect(upstreamTransport);
  log("Upstream connected. Discovering tools...");

  const cachedTools = await discoverUpstreamTools(upstreamClient);
  log(`Discovered ${cachedTools.length} upstream tool(s): ${cachedTools.map((t) => t.name).join(", ")}`);

  const discoveredNames = new Set(cachedTools.map((t) => t.name));
  const gated = cachedTools.filter((t) => isToolGated(t.name, config)).map((t) => t.name);
  const passthrough = cachedTools.filter((t) => !isToolGated(t.name, config)).map((t) => t.name);

  if (gated.length > 0) {
    log(`Gated (${gated.length}): ${gated.join(", ")}`);
  }
  if (passthrough.length > 0) {
    log(`Passthrough (${passthrough.length}): ${passthrough.join(", ")}`);
  }

  if (config.gatedTools !== "*") {
    for (const name of config.gatedTools) {
      if (!discoveredNames.has(name)) {
        log(`WARNING: GATED_TOOLS contains "${name}" which was not found in upstream tools`);
      }
    }
  }
  for (const name of config.passthroughTools) {
    if (!discoveredNames.has(name)) {
      log(`WARNING: PASSTHROUGH_TOOLS contains "${name}" which was not found in upstream tools`);
    }
  }

  const sdk = new NullSpend({
    baseUrl: config.nullspendUrl,
    apiKey: config.nullspendApiKey,
  });

  let costTracker: CostTracker | undefined;
  if (config.costTrackingEnabled) {
    log("Resolving identity from API key...");
    try {
      const resp = await fetch(`${config.nullspendUrl}/api/auth/introspect`, {
        headers: { "x-nullspend-key": config.nullspendApiKey },
        signal: AbortSignal.timeout(5_000),
      });
      if (!resp.ok) {
        log(`API key introspection failed (${resp.status}). Create a managed key at ${config.nullspendUrl}/app/settings.`);
        process.exit(1);
      }
      const identity = (await resp.json()) as {
        userId: string; keyId: string; dev?: boolean;
      };
      log(`Authenticated as userId=${identity.userId}, keyId=${identity.keyId}${identity.dev ? " (dev mode)" : ""}`);
    } catch (err) {
      log(`API key introspection failed: ${err instanceof Error ? err.message : String(err)}`);
      process.exit(1);
    }

    costTracker = new CostTracker({
      backendUrl: config.backendUrl,
      apiKey: config.nullspendApiKey,
      serverName: config.serverName,
      budgetEnforcementEnabled: config.budgetEnforcementEnabled,
      toolCostOverrides: config.toolCostOverrides,
    });
    log(`Cost tracking enabled (server: ${config.serverName})`);

    // Register discovered tools with dashboard and fetch user-configured costs
    const registry = new ToolCostRegistry({
      nullspendUrl: config.nullspendUrl,
      apiKey: config.nullspendApiKey,
      serverName: config.serverName,
    });

    const discoverPayloads = cachedTools.map((t) => ({
      name: t.name,
      description: t.description ?? null,
      annotations: (t.annotations as Record<string, unknown> | undefined) ?? null,
      tierCost: 0,
      suggestedCost: suggestToolCost(t.annotations),
    }));

    await registry.discoverTools(discoverPayloads);
    await registry.fetchCosts();
    costTracker.setRegistry(registry);
  }

  let isShuttingDown = false;

  async function shutdown() {
    if (isShuttingDown) return;
    isShuttingDown = true;

    log("Shutting down...");
    shutdownController.abort();

    if (costTracker) {
      try {
        await costTracker.shutdown();
      } catch {
        // best-effort flush
      }
    }

    try {
      await server.close();
    } catch {
      // best-effort close
    }

    try {
      await upstreamClient.close();
    } catch {
      // best-effort close
    }

    process.exit(0);
  }

  // Register signal handlers BEFORE server setup so costTracker is
  // always cleaned up, even if registerProxyHandlers or connect throws.
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  process.stdin.on("close", shutdown);

  const server = new Server(
    { name: "nullspend-proxy", version: "0.1.0" },
    { capabilities: { tools: {} } },
  );

  try {
    registerProxyHandlers(
      server,
      upstreamClient,
      sdk,
      config,
      cachedTools,
      shutdownController.signal,
      costTracker,
    );

    const serverTransport = new StdioServerTransport();
    await server.connect(serverTransport);
    log(`Proxy running (stdio). API → ${config.nullspendUrl}`);
  } catch (err) {
    // Clean up costTracker before re-throwing startup error
    if (costTracker) {
      try { await costTracker.shutdown(); } catch { /* best-effort */ }
    }
    throw err;
  }

  upstreamTransport.onclose = () => {
    log("Upstream server disconnected. Shutting down.");
    shutdown();
  };
}

main().catch((err) => {
  log(`Fatal: ${String(err)}`);
  process.exit(1);
});
