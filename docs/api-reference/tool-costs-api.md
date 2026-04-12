# Tool Costs API

Manage cost assignments for MCP tools discovered by the proxy. Tools are automatically registered when the MCP proxy sees them; use this API to view and override their per-invocation costs.

## List tool costs

```
GET /api/tool-costs
```

Returns all tool cost entries for your organization.

**Auth:** API key or session (viewer role).

### Response

```json
{
  "data": [
    {
      "id": "tc_550e8400-e29b-41d4-a716-446655440000",
      "userId": "usr_...",
      "serverName": "weather-server",
      "toolName": "get_forecast",
      "costMicrodollars": 50000,
      "suggestedCost": 25000,
      "source": "discovered",
      "description": "Get weather forecast for a location",
      "annotations": { "category": "external-api" },
      "lastSeenAt": "2026-04-10T15:30:00.000Z",
      "createdAt": "2026-04-08T10:00:00.000Z",
      "updatedAt": "2026-04-10T15:30:00.000Z"
    }
  ]
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Prefixed ID (`tc_...`) |
| `serverName` | string | MCP server name |
| `toolName` | string | Tool name within the server |
| `costMicrodollars` | integer | Cost per invocation in microdollars (1,000,000 = $1.00) |
| `suggestedCost` | integer | Cost suggested by the MCP server at discovery time |
| `source` | string | `"discovered"` (auto) or `"manual"` (user override) |
| `description` | string or null | Tool description from the MCP server |
| `annotations` | object or null | Arbitrary metadata from the MCP server |
| `lastSeenAt` | string or null | Last time the proxy saw this tool |

## Update tool cost

```
POST /api/tool-costs
```

Set a manual cost override for a discovered tool. The tool must already exist (created via proxy discovery) — you cannot create phantom entries.

**Auth:** Session only (admin role).

### Request body

```json
{
  "serverName": "weather-server",
  "toolName": "get_forecast",
  "costMicrodollars": 100000
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `serverName` | string | Yes | MCP server name (no `/` characters) |
| `toolName` | string | Yes | Tool name |
| `costMicrodollars` | integer | Yes | Cost in microdollars (>= 0) |

Setting a manual cost changes the `source` to `"manual"`. Future proxy discoveries will update metadata (description, annotations, lastSeenAt) but will **not** overwrite the manual cost.

### Response

Returns the updated tool cost object (same shape as list response items), or `404` if the tool hasn't been discovered yet.

## Reset tool cost

```
DELETE /api/tool-costs/:id
```

Reset a tool's cost back to 0 and change its source back to `"discovered"`. This effectively removes a manual override.

**Auth:** Session only (admin role).

### Response

```json
{
  "deleted": true
}
```

Returns `404` if the tool cost doesn't exist or doesn't belong to your organization.

## Discover tools (proxy-only)

```
POST /api/tool-costs/discover
```

Bulk-register tools from an MCP server. Called by the proxy automatically — you typically don't need to call this directly.

**Auth:** API key only.

### Request body

```json
{
  "serverName": "weather-server",
  "tools": [
    {
      "name": "get_forecast",
      "tierCost": 50000,
      "suggestedCost": 25000,
      "description": "Get weather forecast for a location",
      "annotations": { "category": "external-api" }
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `serverName` | string | Yes | MCP server name |
| `tools` | array | Yes | 1–500 tools, no duplicate names |
| `tools[].name` | string | Yes | Tool name |
| `tools[].tierCost` | integer | Yes | Default cost in microdollars |
| `tools[].suggestedCost` | integer | No | Server-suggested cost |
| `tools[].description` | string | No | Tool description |
| `tools[].annotations` | object | No | Arbitrary metadata (max 10 levels of nesting) |

### Response

```json
{
  "registered": 3
}
```

This endpoint is idempotent. Re-discovering the same tools updates metadata but preserves manual cost overrides.

## Related

- [MCP Proxy SDK](../sdks/mcp-proxy.md) — the MCP proxy that discovers and reports tool costs
- [Cost Tracking](../features/cost-tracking.md) — how costs flow through the system
