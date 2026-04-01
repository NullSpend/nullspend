# MCP Server

MCP server that makes AI agents cost-aware and safety-conscious. Exposes tools for human-in-the-loop approval and real-time spend visibility to any MCP client (Claude Desktop, Cursor, Windsurf, etc.).

**5 tools:**
- **Approval:** `propose_action`, `check_action` — human-in-the-loop safety
- **Cost awareness:** `get_budgets`, `get_spend_summary`, `get_recent_costs` — agents that know their spend

## Installation

Build from source (private package):

```bash
cd packages/mcp-server
npm install
npm run build
```

## Configuration

Set environment variables before starting the server:

| Variable | Required | Default | Description |
|---|---|---|---|
| `NULLSPEND_URL` | Yes | — | NullSpend dashboard URL (e.g. `https://nullspend.com`) |
| `NULLSPEND_API_KEY` | Yes | — | API key (`ns_live_sk_...`) |
| `NULLSPEND_AGENT_ID` | No | `"mcp-agent"` | Default agent ID for actions created by this server |

## Tools

The server registers five MCP tools across two categories.

### Approval Tools

### `propose_action`

Propose a risky action for human approval. By default, blocks until a decision is made.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `actionType` | `string` | Yes | — | Type of action (e.g. `send_email`, `http_post`, `db_write`) |
| `payload` | `object` | Yes | — | Action payload with relevant details |
| `summary` | `string` | Yes | — | Human-readable summary of what this action will do |
| `agentId` | `string` | No | `NULLSPEND_AGENT_ID` | Identifier for the agent proposing the action |
| `metadata` | `object` | No | — | Additional metadata attached to the action |
| `timeoutSeconds` | `number` | No | `300` | Seconds to wait for a decision |
| `waitForDecision` | `boolean` | No | `true` | If `true`, block until decided. If `false`, return immediately |

**Response (blocking mode — `waitForDecision: true`):**

```json
{
  "actionId": "ns_act_550e8400-...",
  "status": "approved",
  "approved": true,
  "rejected": false,
  "timedOut": false,
  "message": "Action ns_act_550e8400-... was approved."
}
```

**Response (non-blocking mode — `waitForDecision: false`):**

```json
{
  "actionId": "ns_act_550e8400-...",
  "status": "pending",
  "approved": false,
  "rejected": false,
  "timedOut": false,
  "message": "Action ns_act_550e8400-... created. Use check_action to poll for the decision."
}
```

**Response (timeout):**

```json
{
  "actionId": "ns_act_550e8400-...",
  "status": "pending",
  "approved": false,
  "rejected": false,
  "timedOut": true,
  "message": "Timed out waiting for decision on action ns_act_550e8400-.... Use check_action to poll later."
}
```

### `check_action`

Check the current status of a previously proposed action.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `actionId` | `string` | Yes | The ID of the action to check |

**Response:**

```json
{
  "actionId": "ns_act_550e8400-...",
  "status": "approved",
  "approved": true,
  "rejected": false,
  "timedOut": false,
  "message": "Action ns_act_550e8400-... is currently approved."
}
```

### Cost Awareness Tools

These tools let agents query their own spend data — enabling cost-conscious behavior like choosing cheaper models when approaching a budget limit.

### `get_budgets`

Get current budget limits and spend for this API key's organization.

**Parameters:** None.

**Response:**

```json
{
  "budgets": [
    {
      "entityType": "user",
      "entityId": "user-123",
      "limitDollars": 10000,
      "spendDollars": 3500,
      "remainingDollars": 6500,
      "percentUsed": 35,
      "policy": "strict_block",
      "resetInterval": "monthly"
    }
  ],
  "message": "1 budget(s) found."
}
```

When no budgets are configured: `{ "budgets": [], "message": "No budgets configured. All requests are allowed without spending limits." }`

### `get_spend_summary`

Get aggregated spending data for a time period.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `period` | `"7d" \| "30d" \| "90d"` | No | `"30d"` | Time period to summarize |

**Response:**

```json
{
  "period": "30d",
  "totalCostDollars": 142.50,
  "totalRequests": 3847,
  "totalInputTokens": 2150000,
  "totalOutputTokens": 890000,
  "costByModel": {
    "gpt-4o": 95.20,
    "gpt-4o-mini": 12.30,
    "claude-sonnet-4-6": 35.00
  },
  "costByProvider": {
    "openai": 107.50,
    "anthropic": 35.00
  },
  "message": "Spend summary for the last 30d: $142.50 across 3847 requests."
}
```

### `get_recent_costs`

List the most recent API call costs.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | `number` | No | `10` | Number of events to return (max: 50) |

**Response:**

```json
{
  "events": [
    {
      "model": "gpt-4o",
      "provider": "openai",
      "inputTokens": 500,
      "outputTokens": 150,
      "costDollars": 0.004625,
      "durationMs": 800,
      "createdAt": "2026-03-25T12:00:00Z"
    }
  ],
  "count": 1,
  "totalCostDollars": 0.004625,
  "message": "1 recent cost event(s). Total: $0.0046."
}
```

## Claude Desktop Setup

Add the server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nullspend": {
      "command": "node",
      "args": ["path/to/packages/mcp-server/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "https://nullspend.com",
        "NULLSPEND_API_KEY": "ns_live_sk_your-key-here"
      }
    }
  }
}
```

The server communicates via stdio transport. It logs startup and error messages to stderr.

## Graceful Shutdown

The server handles `SIGINT`, `SIGTERM`, and stdin close. In-flight `propose_action` calls waiting for a decision are aborted when the server shuts down.

## Related

- [Human-in-the-Loop](../features/human-in-the-loop.md) — approval workflow concepts and state machine
- [Cost Tracking](../features/cost-tracking.md) — how cost events are recorded
- [Budgets](../features/budgets.md) — budget enforcement and policies
- [Actions API](../api-reference/actions-api.md) — raw HTTP endpoint reference
- [Cost Events API](../api-reference/cost-events-api.md) — cost event query endpoints
- [Budgets API](../api-reference/budgets-api.md) — budget management endpoints
- [MCP Proxy](mcp-proxy.md) — proxy that gates upstream MCP tool calls through approval
- [JavaScript SDK](javascript.md) — programmatic API client used internally
