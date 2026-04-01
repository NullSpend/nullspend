# MCP Proxy

MCP proxy that gates risky tool calls through NullSpend approval before forwarding to an upstream MCP server. Adds cost tracking and budget enforcement for every tool invocation.

## Architecture

```
LLM Client ──► NullSpend MCP Proxy ──► Upstream MCP Server
                    │                         │
                    │  ◄── tool result ───────┘
                    │
                    ├──► NullSpend API (approval, cost events, budget checks)
                    │
                    ▼
              NullSpend Dashboard
```

The proxy sits between the LLM client and the upstream MCP server. It discovers upstream tools at startup, optionally gates tool calls through human approval, tracks cost per invocation, and enforces budgets.

## Installation

Build from source (private package):

```bash
cd packages/mcp-proxy
npm install
npm run build
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `NULLSPEND_URL` | Yes | — | NullSpend dashboard URL (e.g. `https://nullspend.com`) |
| `NULLSPEND_API_KEY` | Yes | — | API key (`ns_live_sk_...`) |
| `UPSTREAM_COMMAND` | Yes | — | Command to start the upstream MCP server |
| `UPSTREAM_ARGS` | No | `[]` | JSON array of arguments for the upstream command |
| `UPSTREAM_ENV` | No | `{}` | JSON object of additional env vars for the upstream process |
| `GATED_TOOLS` | No | `"*"` | Which tools require approval (see [Gating](#gating)) |
| `PASSTHROUGH_TOOLS` | No | `""` | Tools that always skip approval |
| `NULLSPEND_AGENT_ID` | No | `"mcp-proxy"` | Agent ID for created actions |
| `APPROVAL_TIMEOUT_SECONDS` | No | `300` | Seconds to wait for human approval |
| `NULLSPEND_COST_TRACKING` | No | `"true"` | Set to `"false"` to disable cost event reporting |
| `NULLSPEND_BUDGET_ENFORCEMENT` | No | `"true"` | Set to `"false"` to disable budget checks |
| `NULLSPEND_SERVER_NAME` | No | `UPSTREAM_COMMAND` | Server name for cost events and analytics. Must not contain `/` |
| `NULLSPEND_TOOL_COSTS` | No | `{}` | JSON object mapping tool names to cost in microdollars |

## Gating

Tool gating controls which upstream tools require human approval before execution.

### How It Works

1. If a tool is in `PASSTHROUGH_TOOLS`, it is **never gated** (passthrough always wins)
2. If `GATED_TOOLS` is `"*"`, **all** non-passthrough tools require approval
3. If `GATED_TOOLS` is a comma-separated list, only those tools require approval
4. If `GATED_TOOLS` is `""` (empty), **no** tools require approval

### Examples

| `GATED_TOOLS` | `PASSTHROUGH_TOOLS` | Effect |
|---|---|---|
| `*` | _(empty)_ | All tools gated |
| `*` | `read_file,list_dir` | All tools gated except `read_file` and `list_dir` |
| `write_file,delete_file` | _(empty)_ | Only `write_file` and `delete_file` gated |
| _(empty string)_ | _(empty)_ | No tools gated (approval disabled, cost tracking still active) |

### Approval Flow

When a gated tool is called:

1. The proxy creates an action via `POST /api/actions` with the tool name, arguments, and a summary
2. It polls for a human decision (approved/rejected/expired)
3. On **approval**: forwards the call to the upstream server, then reports the result
4. On **rejection**: returns an error to the LLM client: `Action "<tool>" was rejected by a human reviewer.`
5. On **timeout**: returns an error: `Approval for "<tool>" timed out after N seconds.`

## Cost Tracking

The proxy tracks cost for every tool invocation (gated or not) when `NULLSPEND_COST_TRACKING` is `"true"` (default).

### Cost Estimation

Cost per tool call is estimated using a three-tier priority system:

| Priority | Source | Description |
|---|---|---|
| 1 | `NULLSPEND_TOOL_COSTS` env var | Per-tool overrides in microdollars |
| 2 | Dashboard-configured costs | Fetched from `/api/tool-costs` at startup |
| 3 | MCP annotation tiers | Inferred from the tool's `annotations` field |

### Annotation Tiers

When no explicit cost is configured, the proxy uses the tool's MCP annotations:

| Condition | Tier | Cost |
|---|---|---|
| `readOnlyHint: true` AND `openWorldHint: false` | FREE | $0.00 |
| `destructiveHint: true` AND `openWorldHint: true` | WRITE | $0.10 |
| Everything else (default) | READ | $0.01 |

To override costs for specific tools:

```bash
NULLSPEND_TOOL_COSTS='{"write_file": 50000, "run_query": 200000}'
```

Values are in microdollars (1,000,000 = $1.00).

### Budget Enforcement

When `NULLSPEND_BUDGET_ENFORCEMENT` is `"true"` (default), the proxy checks the budget before each tool call:

1. Estimates the cost using the priority system above
2. Calls `POST /v1/mcp/budget/check` with the tool name, server name, and estimate
3. If the budget is exceeded, returns an error: `Tool "<name>" blocked: budget exceeded.`
4. If the check fails (network error, timeout), falls back to fail-open after 5 consecutive failures (circuit breaker with 30s cooldown)

### Event Reporting

Cost events are batched and sent to `POST /v1/mcp/events`:

- Batch size: 20 events
- Flush interval: 5 seconds
- Max queue: 4,096 events (oldest dropped on overflow)
- Failed batches are re-queued once for retry

### Tool Discovery

At startup, the proxy registers all upstream tools with the dashboard via `POST /api/tool-costs/discover`. This populates the tool catalog for per-tool cost configuration in the UI.

## Claude Desktop Setup

Example `claude_desktop_config.json` with gating and passthrough:

```json
{
  "mcpServers": {
    "gated-filesystem": {
      "command": "node",
      "args": ["path/to/packages/mcp-proxy/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "https://nullspend.com",
        "NULLSPEND_API_KEY": "ns_live_sk_your-key-here",
        "UPSTREAM_COMMAND": "npx",
        "UPSTREAM_ARGS": "[\"@modelcontextprotocol/server-filesystem\", \"/home/user/projects\"]",
        "GATED_TOOLS": "write_file,delete_file",
        "PASSTHROUGH_TOOLS": "read_file,list_dir",
        "NULLSPEND_SERVER_NAME": "filesystem"
      }
    }
  }
}
```

### Gate-Everything Example

```json
{
  "mcpServers": {
    "gated-everything": {
      "command": "node",
      "args": ["path/to/packages/mcp-proxy/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "https://nullspend.com",
        "NULLSPEND_API_KEY": "ns_live_sk_your-key-here",
        "UPSTREAM_COMMAND": "npx",
        "UPSTREAM_ARGS": "[\"some-mcp-server\"]",
        "NULLSPEND_TOOL_COSTS": "{\"expensive_tool\": 500000}"
      }
    }
  }
}
```

## Error Messages

| Scenario | Error Text |
|---|---|
| Budget exceeded | `Tool "<name>" blocked: budget exceeded. Remaining: <n> microdollars.` |
| Action rejected | `Action "<name>" was rejected by a human reviewer.` |
| Approval timeout | `Approval for "<name>" timed out after <n> seconds. The action was not executed.` |
| Upstream error | `Upstream error: <message>` |
| Upstream error after approval | `Upstream call failed after approval: <message>` |
| Approval service unreachable | `Failed to reach approval service: <message>` |

## Graceful Shutdown

The proxy handles `SIGINT`, `SIGTERM`, and stdin close. On shutdown:

1. Aborts any in-flight approval polls
2. Flushes remaining cost events (no re-queue during shutdown)
3. Waits for in-flight HTTP requests to complete
4. Closes the upstream MCP client connection

## Related

- [MCP Server](mcp-server.md) — standalone MCP server for approval tools (no upstream proxy)
- [Human-in-the-Loop](../features/human-in-the-loop.md) — approval workflow concepts
- [Budgets](../features/budgets.md) — budget enforcement and spending limits
- [Cost Tracking](../features/cost-tracking.md) — how cost events are recorded
- [JavaScript SDK](javascript.md) — programmatic API client
