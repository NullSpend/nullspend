# @nullspend/mcp-server

MCP (Model Context Protocol) server that exposes NullSpend approval tools to any MCP client — Claude Desktop, Cursor, or any other MCP-compatible host.

## How it works

```
LLM / MCP Client  ──stdio──▶  NullSpend MCP Server  ──HTTP──▶  NullSpend API
                                                                     ▲
                                                       Human reviews in Dashboard
```

The MCP server exposes seven tools:

| Tool | Category | Purpose |
|------|----------|---------|
| `request_budget_increase` | Budget negotiation | Request more budget from a human approver. Blocks until approved/rejected. |
| `check_budget` | Budget negotiation | Preflight check — remaining budget, policy, and whether the next request will be blocked. |
| `propose_action` | Approval | Propose a risky action for human approval. Blocks until approved/rejected (or returns immediately). |
| `check_action` | Approval | Check the current status of a previously proposed action. |
| `get_budgets` | Cost awareness | Get current budget limits and spend. |
| `get_spend_summary` | Cost awareness | Aggregated spending by model and provider. |
| `get_recent_costs` | Cost awareness | List recent API call costs. |

## Quick start (local)

### 1. Build

From the repo root:

```bash
pnpm --filter @nullspend/sdk build
pnpm --filter @nullspend/mcp-server build
```

### 2. Configure environment

The server requires two environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `NULLSPEND_URL` | Yes | Base URL of your NullSpend API (e.g. `http://127.0.0.1:3000`) |
| `NULLSPEND_API_KEY` | Yes | API key created from the NullSpend dashboard |
| `NULLSPEND_AGENT_ID` | No | Default agent ID for actions (default: `mcp-agent`) |

### 3. Connect to an MCP client

#### Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nullspend": {
      "command": "node",
      "args": ["C:/path/to/NullSpend/packages/mcp-server/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "http://127.0.0.1:3000",
        "NULLSPEND_API_KEY": "ns_live_sk_your-api-key-here"
      }
    }
  }
}
```

#### Cursor

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "nullspend": {
      "command": "node",
      "args": ["C:/path/to/NullSpend/packages/mcp-server/dist/index.js"],
      "env": {
        "NULLSPEND_URL": "http://127.0.0.1:3000",
        "NULLSPEND_API_KEY": "ns_live_sk_your-api-key-here"
      }
    }
  }
}
```

## Tool reference

### `request_budget_increase`

Request a budget increase from a human approver. The request is sent to Slack (if configured) or the NullSpend dashboard. Blocks until approved, rejected, or timed out.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `amount` | number | Yes | Amount to request in dollars (e.g. `5` for $5) |
| `reason` | string | Yes | Why you need more budget — shown to the human approver |
| `entityType` | string | No | Budget entity type (e.g. `api_key`, `user`). Default: `api_key` |
| `entityId` | string | No | Budget entity ID. Default: inferred from API key |
| `currentLimitDollars` | number | No | Current budget limit in dollars (for context) |
| `currentSpendDollars` | number | No | Current spend in dollars (for context) |
| `agentId` | string | No | Agent identifier |
| `timeoutSeconds` | number | No | Seconds to wait for a decision (default: 300) |

**Typical flow:**

```
1. Agent calls check_budget → sees $0.50 remaining, willBlock: true
2. Agent calls request_budget_increase → "$5, finishing document processing"
3. Human approves in Slack or dashboard
4. Tool returns { approved: true }
5. Agent retries the original request
```

### `check_budget`

Check your current budget status before making an expensive request. Returns remaining budget, spend, and policy for each budget entity.

**Parameters:** None.

**Response:**

```json
{
  "hasBudgets": true,
  "budgets": [
    {
      "entityType": "api_key",
      "entityId": "key-123",
      "limitDollars": 10,
      "spendDollars": 9.50,
      "remainingDollars": 0.50,
      "percentUsed": 95,
      "policy": "strict_block",
      "resetInterval": "monthly",
      "willBlock": false
    }
  ],
  "mostConstrained": {
    "entityType": "api_key",
    "entityId": "key-123",
    "remainingDollars": 0.50,
    "willBlock": false
  },
  "message": "$0.50 remaining on most constrained budget (api_key/key-123)."
}
```

### `propose_action`

Propose a risky action for human approval before execution.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actionType` | string | Yes | Type of action (e.g. `send_email`, `http_post`, `db_write`) |
| `payload` | object | Yes | Action payload with relevant details |
| `summary` | string | Yes | Human-readable summary of what this action will do |
| `agentId` | string | No | Identifier for the agent proposing this action |
| `metadata` | object | No | Additional metadata |
| `timeoutSeconds` | number | No | Seconds to wait for a decision (default: 300) |
| `waitForDecision` | boolean | No | If `true` (default), block until decided. If `false`, return immediately. |

**Blocking mode** (`waitForDecision: true`, the default): The tool blocks and polls the NullSpend API until the action is approved, rejected, or the timeout expires. The LLM receives the final decision and can act on it.

**Non-blocking mode** (`waitForDecision: false`): The tool returns immediately with the `actionId` and `pending` status. The LLM can use `check_action` to poll for the decision later. Use this when the MCP client has strict tool timeout limits.

### `check_action`

Check the current status of a previously proposed action.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actionId` | string | Yes | The ID of the action to check |

## Development

```bash
# Run tests
pnpm --filter @nullspend/mcp-server test

# Watch mode
pnpm --filter @nullspend/mcp-server test:watch

# Rebuild
pnpm --filter @nullspend/mcp-server build
```
