# @nullspend/mcp-proxy

MCP proxy that sits between an LLM and any upstream MCP server — gating risky tool calls through human approval, tracking cost per invocation, and enforcing budgets before execution.

## How it works

```
LLM / MCP Client  ──stdio──▶  NullSpend MCP Proxy  ──stdio──▶  Upstream MCP Server
                                      │                          (e.g. Supabase, filesystem)
                                      │ HTTP
                                      ▼
                                NullSpend API
                          (approval, budgets, cost events)
                                      ▲
                          Human reviews in Dashboard
```

1. The proxy spawns the upstream MCP server as a child process and discovers all its tools.
2. It re-exposes those tools to the LLM under the same names and schemas.
3. On every tool call, the proxy:
   - **Checks the budget** — if the estimated cost would exceed the limit, the call is blocked (`budget exceeded`)
   - **Gates through approval** (if configured) — creates an action, waits for human approval, then forwards or rejects
   - **Tracks cost** — reports the tool invocation cost to the NullSpend dashboard
4. **Passthrough** tools skip approval but still get cost tracking and budget checks.

## Quick start (local)

### 1. Build

From the repo root:

```bash
pnpm --filter @nullspend/sdk build
pnpm --filter @nullspend/mcp-proxy build
```

### 2. Configure and connect

#### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "supabase-gated": {
      "command": "node",
      "args": ["C:/path/to/NullSpend/packages/mcp-proxy/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "http://127.0.0.1:3000",
        "NULLSPEND_API_KEY": "ns_live_sk_your-api-key-here",
        "UPSTREAM_COMMAND": "npx",
        "UPSTREAM_ARGS": "[\"-y\", \"@supabase/mcp-server\"]",
        "UPSTREAM_ENV": "{\"SUPABASE_ACCESS_TOKEN\": \"sbp_your-token\"}",
        "GATED_TOOLS": "*",
        "PASSTHROUGH_TOOLS": "list_tables,get_table_schema"
      }
    }
  }
}
```

#### Cursor

Add to your Cursor MCP settings (`.cursor/mcp.json` or global settings):

```json
{
  "mcpServers": {
    "supabase-gated": {
      "command": "node",
      "args": ["C:/path/to/NullSpend/packages/mcp-proxy/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "http://127.0.0.1:3000",
        "NULLSPEND_API_KEY": "ns_live_sk_your-api-key-here",
        "UPSTREAM_COMMAND": "npx",
        "UPSTREAM_ARGS": "[\"-y\", \"@supabase/mcp-server\"]",
        "UPSTREAM_ENV": "{\"SUPABASE_ACCESS_TOKEN\": \"sbp_your-token\"}",
        "GATED_TOOLS": "*",
        "PASSTHROUGH_TOOLS": "list_tables,get_table_schema"
      }
    }
  }
}
```

## Configuration reference

All configuration is via environment variables.

### Required

| Variable | Description |
|----------|-------------|
| `NULLSPEND_URL` | Base URL of your NullSpend API (e.g. `http://127.0.0.1:3000`) |
| `NULLSPEND_API_KEY` | API key created from the NullSpend dashboard |
| `UPSTREAM_COMMAND` | Command to spawn the upstream MCP server (e.g. `npx`, `node`, `python`) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_ARGS` | `[]` | JSON array of arguments passed to the upstream command. Example: `["--port", "3001"]` or `["-y", "@supabase/mcp-server"]` |
| `UPSTREAM_ENV` | `{}` | JSON object of extra environment variables for the upstream process. **Merged** with `process.env` — the upstream inherits all current env vars, with `UPSTREAM_ENV` values taking precedence for any overlapping keys. |
| `GATED_TOOLS` | `*` | Which tools require approval. `*` (or unset) gates all tools, a comma-separated list gates only those tools (e.g. `execute_sql,insert_row`), and an empty string `""` gates nothing. |
| `PASSTHROUGH_TOOLS` | (none) | Comma-separated list of tools that are **always forwarded** without approval. **Passthrough always wins** — if a tool appears in both `GATED_TOOLS` and `PASSTHROUGH_TOOLS`, it is passed through. |
| `NULLSPEND_AGENT_ID` | `mcp-proxy` | Agent identifier attached to actions created by this proxy. |
| `APPROVAL_TIMEOUT_SECONDS` | `300` | How long (seconds) to wait for a human decision before timing out. |
| `NULLSPEND_COST_TRACKING` | `true` | Set to `false` to disable cost event reporting. |
| `NULLSPEND_BUDGET_ENFORCEMENT` | `true` | Set to `false` to disable pre-call budget checks. |
| `NULLSPEND_SERVER_NAME` | `UPSTREAM_COMMAND` | Server name for cost events and analytics. Must not contain `/`. |
| `NULLSPEND_TOOL_COSTS` | `{}` | JSON object mapping tool names to cost in microdollars (e.g. `{"write_file": 50000}`). |

### Gating examples

| Scenario | `GATED_TOOLS` | `PASSTHROUGH_TOOLS` | Result |
|----------|---------------|---------------------|--------|
| Gate everything | `*` | (unset) | All tools require approval |
| Gate everything except reads | `*` | `list_tables,get_schema` | Read tools pass through, everything else gated |
| Gate only writes | `execute_sql,delete_row` | (unset) | Only those two tools require approval |
| Passthrough overrides gate | `execute_sql` | `execute_sql` | `execute_sql` passes through (passthrough wins) |

## What happens during a gated call

1. The LLM calls a gated tool (e.g. `execute_sql`).
2. The proxy creates an NullSpend action with the tool name, arguments, and a summary. The action's server-side TTL matches the proxy's `APPROVAL_TIMEOUT_SECONDS`.
3. The action appears in the NullSpend dashboard for human review.
4. The proxy polls until the action is **approved**, **rejected**, **expired** (server-side TTL elapsed), or the client-side timeout expires.
5. On **approval**: the proxy marks the action `executing`, forwards the call to the upstream server, then marks it `executed` (or `failed` if the upstream errors).
6. On **rejection**, **expiration**, or **timeout**: the proxy returns an error message to the LLM explaining the tool call was blocked.

## Cost Tracking & Budget Enforcement

Cost tracking and budget enforcement are enabled by default.

**Cost estimation** uses a three-tier priority:

1. `NULLSPEND_TOOL_COSTS` env var — per-tool overrides in microdollars
2. Dashboard-configured costs — fetched from the API at startup
3. MCP annotation tiers — inferred from the tool's `annotations` field (free / $0.01 / $0.10)

**Budget enforcement** checks the budget before each tool call. If the estimated cost would exceed the remaining budget, the call is blocked with `Tool "<name>" blocked: budget exceeded.` The budget client includes a circuit breaker (5 consecutive failures → fail-open, 30s cooldown).

**Cost events** are batched (20 events / 5s flush interval) and reported to the NullSpend API for dashboard visibility.

Set `NULLSPEND_BUDGET_ENFORCEMENT=false` to disable budget checks while keeping cost tracking active.

## Development

```bash
# Run tests
pnpm --filter @nullspend/mcp-proxy test

# Watch mode
pnpm --filter @nullspend/mcp-proxy test:watch

# Rebuild
pnpm --filter @nullspend/mcp-proxy build
```
