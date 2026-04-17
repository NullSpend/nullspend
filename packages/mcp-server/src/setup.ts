import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir, platform } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * `nullspend-mcp setup` subcommand — generates a Claude Desktop config snippet
 * pre-filled with the absolute path to this binary, removing the manual
 * "find the right path" + "guess the JSON shape" friction that is the #1
 * MCP onboarding complaint. (D-5)
 *
 * Default behavior: print the snippet to stdout. Pass `--write` to merge
 * into the platform-specific Claude Desktop config (refuses to clobber an
 * existing `mcpServers.nullspend` entry — print-only fallback in that case).
 */

interface SetupOptions {
  apiKey?: string;
  nullspendUrl?: string;
  agentId?: string;
  write?: boolean;
  print?: boolean;
}

const DEFAULT_NULLSPEND_URL = "https://nullspend.dev";
const DEFAULT_AGENT_ID = "mcp-agent";

/**
 * Resolve the absolute path to this package's compiled `dist/index.js` —
 * the value Claude Desktop needs in its `args[0]`. Computing this from
 * import.meta keeps the snippet correct regardless of where the package
 * is installed (global, project-local, etc).
 */
function resolveBinaryPath(): string {
  // After build, this file lives at packages/mcp-server/dist/setup.js (CJS)
  // or .../dist/setup.js (ESM). The `bin` entry points at dist/index.js,
  // so we compute the sibling path.
  // import.meta.url works in ESM; for CJS we'd use __filename. tsup builds
  // both — this branch handles ESM builds.
  const here = fileURLToPath(import.meta.url);
  return resolve(dirname(here), "index.js");
}

function claudeDesktopConfigPath(): string | null {
  const home = homedir();
  switch (platform()) {
    case "darwin":
      return join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json");
    case "win32": {
      const appdata = process.env.APPDATA;
      if (!appdata) return null;
      return join(appdata, "Claude", "claude_desktop_config.json");
    }
    case "linux":
      return join(home, ".config", "Claude", "claude_desktop_config.json");
    default:
      return null;
  }
}

function buildSnippet(options: SetupOptions): {
  config: Record<string, unknown>;
  binaryPath: string;
} {
  const binaryPath = resolveBinaryPath();
  const env: Record<string, string> = {
    NULLSPEND_URL: options.nullspendUrl ?? DEFAULT_NULLSPEND_URL,
  };
  if (options.apiKey) {
    env.NULLSPEND_API_KEY = options.apiKey;
  } else {
    env.NULLSPEND_API_KEY = "ns_live_sk_REPLACE_ME";
  }
  if (options.agentId && options.agentId !== DEFAULT_AGENT_ID) {
    env.NULLSPEND_AGENT_ID = options.agentId;
  }

  return {
    binaryPath,
    config: {
      command: "node",
      args: [binaryPath],
      env,
    },
  };
}

function parseArgs(argv: readonly string[]): SetupOptions {
  const opts: SetupOptions = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--api-key":
      case "-k":
        opts.apiKey = argv[++i];
        break;
      case "--url":
        opts.nullspendUrl = argv[++i];
        break;
      case "--agent-id":
        opts.agentId = argv[++i];
        break;
      case "--write":
        opts.write = true;
        break;
      case "--print":
        opts.print = true;
        break;
    }
  }
  // Default to print when neither flag is set.
  if (!opts.write && !opts.print) opts.print = true;
  return opts;
}

function printUsage(stream: NodeJS.WriteStream): void {
  stream.write(
    [
      "Usage: nullspend-mcp setup [options]",
      "",
      "Generates a Claude Desktop MCP server snippet for NullSpend, pre-filled",
      "with the absolute path to this binary.",
      "",
      "Options:",
      "  -k, --api-key <KEY>     NullSpend API key to embed (default: placeholder)",
      "      --url <URL>         NullSpend dashboard URL (default: https://nullspend.dev)",
      "      --agent-id <ID>     Agent ID to attribute MCP-tool actions to",
      "      --print             Print the snippet to stdout (default)",
      "      --write             Merge into the platform-specific Claude Desktop config",
      "                          (refuses to overwrite an existing nullspend entry)",
      "",
      "Examples:",
      "  nullspend-mcp setup --api-key ns_live_sk_xxx",
      "  nullspend-mcp setup --write --api-key ns_live_sk_xxx",
      "",
    ].join("\n"),
  );
}

function mergeAndWrite(
  configPath: string,
  entry: Record<string, unknown>,
): { written: boolean; reason?: string } {
  let existing: Record<string, unknown> = {};
  if (existsSync(configPath)) {
    try {
      existing = JSON.parse(readFileSync(configPath, "utf8")) as Record<string, unknown>;
    } catch (err) {
      return {
        written: false,
        reason: `existing config at ${configPath} is not valid JSON: ${(err as Error).message}`,
      };
    }
  }

  const mcpServers = (existing.mcpServers as Record<string, unknown> | undefined) ?? {};
  if (mcpServers.nullspend) {
    return {
      written: false,
      reason:
        `existing 'nullspend' entry detected at ${configPath} — refusing to overwrite. ` +
        "Remove or rename the existing entry, or run without --write and paste manually.",
    };
  }

  const next = {
    ...existing,
    mcpServers: { ...mcpServers, nullspend: entry },
  };

  mkdirSync(dirname(configPath), { recursive: true });
  writeFileSync(configPath, JSON.stringify(next, null, 2) + "\n", "utf8");
  return { written: true };
}

export function runSetup(argv: readonly string[], streams = {
  stdout: process.stdout,
  stderr: process.stderr,
}): number {
  if (argv.includes("--help") || argv.includes("-h")) {
    printUsage(streams.stdout);
    return 0;
  }

  const opts = parseArgs(argv);
  const { config, binaryPath } = buildSnippet(opts);
  const snippet = {
    mcpServers: { nullspend: config },
  };
  const formatted = JSON.stringify(snippet, null, 2);

  if (opts.print) {
    streams.stdout.write(
      `# NullSpend MCP server setup\n` +
        `# Binary: ${binaryPath}\n` +
        `# Paste the JSON below into your Claude Desktop config.\n\n` +
        `${formatted}\n`,
    );
  }

  if (opts.write) {
    const configPath = claudeDesktopConfigPath();
    if (!configPath) {
      streams.stderr.write(
        `[nullspend-mcp] --write not supported on platform '${platform()}'. Use --print and paste manually.\n`,
      );
      return 1;
    }
    const result = mergeAndWrite(configPath, config);
    if (!result.written) {
      streams.stderr.write(`[nullspend-mcp] ${result.reason}\n`);
      return 1;
    }
    streams.stdout.write(
      `[nullspend-mcp] Wrote 'nullspend' MCP server entry to ${configPath}\n` +
        `[nullspend-mcp] Restart Claude Desktop to pick it up.\n`,
    );
  }

  return 0;
}
