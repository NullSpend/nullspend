# Cost Events API

Create, list, and analyze cost events. Cost events are the core data model — every AI API call tracked by NullSpend produces one.

See [API Overview](overview.md) for authentication, pagination, errors, and ID formats.

---

## Ingest Single Event

`POST /api/cost-events`

Record a cost event from your agent or SDK. The proxy creates these automatically; use this endpoint for custom integrations.

### Authentication

API key

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `provider` | body | string | Yes | Provider name (e.g., `"openai"`, `"anthropic"`). 1–100 chars. |
| `model` | body | string | Yes | Model identifier (e.g., `"gpt-4o"`, `"claude-sonnet-4-5-20250514"`). 1–200 chars. |
| `inputTokens` | body | integer | Yes | Input token count. Min 0. |
| `outputTokens` | body | integer | Yes | Output token count. Min 0. |
| `costMicrodollars` | body | integer | Yes | Cost in microdollars (1 microdollar = $0.000001). Min 0. |
| `cachedInputTokens` | body | integer | No | Cached input tokens. Default 0. |
| `reasoningTokens` | body | integer | No | Reasoning/thinking tokens. Default 0. |
| `durationMs` | body | integer | No | Request duration in milliseconds. |
| `sessionId` | body | string | No | Session identifier. Max 200 chars. |
| `traceId` | body | string | No | 128-bit hex trace ID (`^[0-9a-f]{32}$`). |
| `eventType` | body | string | No | `"llm"`, `"tool"`, or `"custom"`. Default `"custom"`. Stored but not returned in list/detail responses — used for internal filtering and analytics. |
| `toolName` | body | string | No | Tool name for tool-use events. Max 200 chars. |
| `toolServer` | body | string | No | Tool server name. Max 200 chars. |
| `customer` | body | string | No | Customer identifier for per-customer profitability. Max 256 chars, min 1 char. |
| `tags` | body | object | No | Key-value metadata. Max 10 keys, key 1–64 chars (`^[a-zA-Z0-9_-]+$`), value max 256 chars. |
| `idempotencyKey` | body | string | No | Deduplication key. Max 200 chars. Alternative to `Idempotency-Key` header. |
| `Idempotency-Key` | header | string | No | Deduplication key. Takes priority over body field. |

### Request

```typescript
const res = await fetch("https://nullspend.dev/api/cost-events", {
  method: "POST",
  headers: {
    "X-NullSpend-Key": "ns_live_sk_abc123...",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    provider: "openai",
    model: "gpt-4o",
    inputTokens: 1200,
    outputTokens: 350,
    costMicrodollars: 5250,
    tags: { environment: "production", agent: "support-bot" },
  }),
});
```

```python
import requests

resp = requests.post(
    "https://nullspend.dev/api/cost-events",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
    json={
        "provider": "openai",
        "model": "gpt-4o",
        "inputTokens": 1200,
        "outputTokens": 350,
        "costMicrodollars": 5250,
        "tags": {"environment": "production", "agent": "support-bot"},
    },
)
```

```bash
curl -X POST https://nullspend.dev/api/cost-events \
  -H "X-NullSpend-Key: ns_live_sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "inputTokens": 1200,
    "outputTokens": 350,
    "costMicrodollars": 5250,
    "tags": {"environment": "production", "agent": "support-bot"}
  }'
```

### Response

**201 Created** (new event) or **200 OK** (deduplicated):

```json
{
  "data": {
    "id": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "createdAt": "2026-03-20T14:30:00.000Z"
  }
}
```

### Idempotency

Deduplication key resolution (first match wins):

1. `Idempotency-Key` header
2. `idempotencyKey` body field
3. Auto-generated `sdk_<uuid>`

Duplicate detection uses the `(requestId, provider)` unique index. If a duplicate is found, the endpoint returns `200` with the original event's ID and timestamp.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid or missing required fields |
| `invalid_json` | 400 | Malformed JSON body |
| `unsupported_media_type` | 415 | Content-Type is not `application/json` |
| `payload_too_large` | 413 | Body exceeds 1 MB |
| `authentication_required` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## Batch Ingest

`POST /api/cost-events/batch`

Insert up to 100 cost events in a single request. Duplicates are silently skipped.

### Authentication

API key

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `events` | body | array | Yes | Array of cost event objects (same schema as single ingest). 1–100 items. |

### Request

```typescript
const res = await fetch("https://nullspend.dev/api/cost-events/batch", {
  method: "POST",
  headers: {
    "X-NullSpend-Key": "ns_live_sk_abc123...",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    events: [
      {
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 500,
        outputTokens: 200,
        costMicrodollars: 2100,
      },
      {
        provider: "anthropic",
        model: "claude-sonnet-4-5-20250514",
        inputTokens: 800,
        outputTokens: 300,
        costMicrodollars: 4950,
      },
    ],
  }),
});
```

```python
import requests

resp = requests.post(
    "https://nullspend.dev/api/cost-events/batch",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
    json={
        "events": [
            {
                "provider": "openai",
                "model": "gpt-4o",
                "inputTokens": 500,
                "outputTokens": 200,
                "costMicrodollars": 2100,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250514",
                "inputTokens": 800,
                "outputTokens": 300,
                "costMicrodollars": 4950,
            },
        ]
    },
)
```

```bash
curl -X POST https://nullspend.dev/api/cost-events/batch \
  -H "X-NullSpend-Key: ns_live_sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"provider":"openai","model":"gpt-4o","inputTokens":500,"outputTokens":200,"costMicrodollars":2100},
      {"provider":"anthropic","model":"claude-sonnet-4-5-20250514","inputTokens":800,"outputTokens":300,"costMicrodollars":4950}
    ]
  }'
```

### Response

**201 Created**:

```json
{
  "inserted": 2,
  "ids": [
    "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "ns_evt_b2c3d4e5-f6a7-8901-bcde-f12345678901"
  ]
}
```

`inserted` reflects only newly created events — duplicates are silently skipped via `ON CONFLICT DO NOTHING`.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Empty array, >100 events, or invalid event fields |
| `invalid_json` | 400 | Malformed JSON body |
| `unsupported_media_type` | 415 | Content-Type is not `application/json` |
| `payload_too_large` | 413 | Body exceeds 1 MB |
| `authentication_required` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## List Cost Events

`GET /api/cost-events`

Retrieve cost events for the current organization with filtering and pagination.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `limit` | query | integer | No | Page size. 1–100, default 25. |
| `cursor` | query | string | No | JSON-encoded cursor from a previous response. |
| `requestId` | query | string | No | Filter by request ID. 1–200 chars. |
| `apiKeyId` | query | string | No | Filter by API key (`ns_key_*`). |
| `model` | query | string | No | Filter by model name. |
| `provider` | query | string | No | Filter by provider. |
| `source` | query | string | No | Filter by source: `"proxy"`, `"api"`, or `"mcp"`. |
| `traceId` | query | string | No | Filter by trace ID (32 hex chars). |
| `sessionId` | query | string | No | Filter by session ID. 1–200 chars. Returns events for a specific [session](../features/cost-tracking.md#session-replay). |
| `tag.*` | query | string | No | JSONB containment filter. Example: `tag.environment=production`. |
| `budgetStatus` | query | string | No | Filter by budget status: `"skipped"`, `"approved"`, or `"denied"`. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/cost-events?limit=10&provider=openai \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "requestId": "sdk_f8e7d6c5-b4a3-2190-fedc-ba0987654321",
      "apiKeyId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
      "provider": "openai",
      "model": "gpt-4o",
      "inputTokens": 1200,
      "outputTokens": 350,
      "cachedInputTokens": 0,
      "reasoningTokens": 0,
      "costMicrodollars": 5250,
      "durationMs": 1340,
      "createdAt": "2026-03-20T14:30:00.000Z",
      "source": "proxy",
      "traceId": "a1b2c3d4e5f67890a1b2c3d4e5f67890",
      "sessionId": "research-task-47",
      "tags": { "environment": "production" },
      "customerId": "acme-corp",
      "budgetStatus": "approved",
      "keyName": "production-key"
    }
  ],
  "cursor": {
    "createdAt": "2026-03-20T14:30:00.000Z",
    "id": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

Headers: `NullSpend-Version: 2026-04-01`

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid query parameters |
| `authentication_required` | 401 | No valid session |

---

## Get Single Event

`GET /api/cost-events/:id`

Retrieve a single cost event by ID. The event must belong to the authenticated user's organization.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Event ID. Accepts `ns_evt_*` or raw UUID. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/cost-events/ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": {
    "id": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "requestId": "sdk_f8e7d6c5-b4a3-2190-fedc-ba0987654321",
    "apiKeyId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "provider": "openai",
    "model": "gpt-4o",
    "inputTokens": 1200,
    "outputTokens": 350,
    "cachedInputTokens": 0,
    "reasoningTokens": 0,
    "costMicrodollars": 5250,
    "durationMs": 1340,
    "createdAt": "2026-03-20T14:30:00.000Z",
    "source": "proxy",
    "traceId": "a1b2c3d4e5f67890a1b2c3d4e5f67890",
    "tags": { "environment": "production" },
    "keyName": "production-key"
  }
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid ID format |
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Event not found or not in user's organization |

---

## Cost Analytics

`GET /api/cost-events/summary`

Aggregate cost breakdown by day, model, provider, key, tool, source, and trace.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `period` | query | string | No | `"7d"`, `"30d"`, or `"90d"`. Default `"30d"`. |
| `excludeEstimated` | query | string | No | `"true"` or `"false"`. Default `"false"`. |

### Request

```bash
# Requires dashboard session cookie
curl "https://nullspend.dev/api/cost-events/summary?period=7d" \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

| Key | Description |
|---|---|
| `daily` | Cost per day over the period |
| `models` | Cost, token counts, and request count per model |
| `providers` | Cost and request count per provider |
| `keys` | Cost and request count per API key |
| `tools` | Cost, request count, and avg duration per tool |
| `sources` | Cost and request count per source (proxy/api/mcp) |
| `traces` | Cost and request count per trace ID |
| `totals` | Total cost, total requests, and the period value |
| `costBreakdown` | Input, output, cached, and reasoning cost breakdown |

```json
{
  "daily": [
    { "date": "2026-03-20", "totalCostMicrodollars": 125000 },
    { "date": "2026-03-19", "totalCostMicrodollars": 98500 }
  ],
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "totalCostMicrodollars": 85000,
      "requestCount": 42,
      "inputTokens": 50000,
      "outputTokens": 15000,
      "cachedInputTokens": 0,
      "reasoningTokens": 0
    }
  ],
  "providers": [
    { "provider": "openai", "totalCostMicrodollars": 85000, "requestCount": 42 }
  ],
  "keys": [
    {
      "apiKeyId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
      "keyName": "production-key",
      "totalCostMicrodollars": 85000,
      "requestCount": 42
    }
  ],
  "tools": [
    {
      "model": "gpt-4o",
      "totalCostMicrodollars": 12000,
      "requestCount": 8,
      "avgDurationMs": 1250
    }
  ],
  "sources": [
    { "source": "proxy", "totalCostMicrodollars": 80000, "requestCount": 38 }
  ],
  "traces": [
    {
      "traceId": "a1b2c3d4e5f67890a1b2c3d4e5f67890",
      "totalCostMicrodollars": 15000,
      "requestCount": 5
    }
  ],
  "totals": {
    "totalCostMicrodollars": 223500,
    "totalRequests": 84,
    "period": "7d"
  },
  "costBreakdown": {
    "inputCost": 120000,
    "outputCost": 95000,
    "cachedCost": 3500,
    "reasoningCost": 5000
  }
}
```

Headers: `NullSpend-Version: 2026-04-01`

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid period or excludeEstimated value |
| `authentication_required` | 401 | No valid session |

---

## Get Session

`GET /api/cost-events/sessions/:sessionId`

Retrieve all cost events for a session, in chronological order, with aggregate summary stats. Use this to build session replay views.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `sessionId` | path | string | Yes | Session ID. 1–200 chars. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/cost-events/sessions/research-task-47 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "sessionId": "research-task-47",
  "summary": {
    "eventCount": 12,
    "totalCostMicrodollars": 43000,
    "totalInputTokens": 15200,
    "totalOutputTokens": 4800,
    "totalDurationMs": 8340,
    "startedAt": "2026-03-20T14:21:05.000Z",
    "endedAt": "2026-03-20T14:23:39.000Z"
  },
  "events": [
    {
      "id": "ns_evt_...",
      "requestId": "req-001",
      "provider": "openai",
      "model": "gpt-4o",
      "inputTokens": 1200,
      "outputTokens": 350,
      "costMicrodollars": 5250,
      "durationMs": 680,
      "createdAt": "2026-03-20T14:21:05.000Z",
      "sessionId": "research-task-47",
      "tags": {},
      "keyName": "production-key"
    }
  ]
}
```

Events are ordered chronologically (oldest first). Maximum 200 events per session. If a session has no events, `events` is an empty array and `summary.startedAt`/`endedAt` are null.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid or empty session ID |
| `authentication_required` | 401 | No valid session |

---

## Export Cost Events

`GET /api/cost-events/export`

Export cost events as a CSV file. Returns up to 10,000 rows sorted by `createdAt DESC`. Supports the same filters as the list endpoint.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `provider` | query | string | No | Filter by provider. |
| `model` | query | string | No | Filter by model name. |
| `apiKeyId` | query | string | No | Filter by API key (`ns_key_*`). |
| `source` | query | string | No | Filter by source: `"proxy"`, `"api"`, or `"mcp"`. |
| `sessionId` | query | string | No | Filter by session ID. |
| `traceId` | query | string | No | Filter by trace ID (32 hex chars). |
| `tag.*` | query | string | No | JSONB containment filter. Example: `tag.environment=production`. |

### Request

```bash
# Requires dashboard session cookie
curl "https://nullspend.dev/api/cost-events/export?provider=openai" \
  -H "Cookie: session=..." \
  -o cost-events.csv
```

### Response

**200 OK** — CSV file download.

Headers: `Content-Type: text/csv; charset=utf-8`, `Content-Disposition: attachment; filename="nullspend-cost-events-2026-03-28.csv"`

CSV columns: `id`, `request_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`, `cost_microdollars`, `cost_usd`, `duration_ms`, `source`, `session_id`, `trace_id`, `key_name`, `created_at`.

The `cost_usd` column is a convenience conversion (`cost_microdollars / 1,000,000`).

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `forbidden` | 403 | User lacks viewer role |

---

## Cost Attribution

`GET /api/cost-events/attribution`

Group cost events by API key or any tag value. Returns ranked groups with total cost, request count, and average cost per request. Supports JSON and CSV export.

See [Cost Attribution](../features/cost-attribution.md) for the feature overview and common patterns.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `groupBy` | query | string | Yes | `"api_key"` for API key grouping, or any tag key name (e.g., `"customer_id"`). 1–100 chars. |
| `period` | query | string | No | `"7d"`, `"30d"`, or `"90d"`. Default `"30d"`. |
| `limit` | query | integer | No | Max groups returned. 1–500, default 100. |
| `excludeEstimated` | query | string | No | `"true"` or `"false"`. Default `"false"`. Excludes cancelled stream estimates. |
| `format` | query | string | No | `"json"` (default) or `"csv"`. CSV returns a downloadable file. |

### Request

```bash
# Group by API key (requires dashboard session)
curl "https://nullspend.dev/api/cost-events/attribution?groupBy=api_key&period=30d" \
  -H "Cookie: session=..."

# Group by customer_id tag
curl "https://nullspend.dev/api/cost-events/attribution?groupBy=customer_id&period=30d" \
  -H "Cookie: session=..."

# CSV export
curl "https://nullspend.dev/api/cost-events/attribution?groupBy=api_key&format=csv" \
  -H "Cookie: session=..." \
  -o attribution.csv
```

### Response

**200 OK** (JSON):

```json
{
  "data": {
    "groups": [
      {
        "key": "production-key",
        "keyId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
        "totalCostMicrodollars": 8500000,
        "requestCount": 4200,
        "avgCostMicrodollars": 2024
      },
      {
        "key": "staging-key",
        "keyId": "ns_key_aabbccdd-eeff-0011-2233-445566778899",
        "totalCostMicrodollars": 1200000,
        "requestCount": 800,
        "avgCostMicrodollars": 1500
      }
    ],
    "period": "30d",
    "groupBy": "api_key",
    "totalGroups": 2,
    "hasMore": false,
    "totals": {
      "totalCostMicrodollars": 9700000,
      "totalRequests": 5000
    }
  }
}
```

When `groupBy` is a tag key, `keyId` is `null` for all groups:

```json
{
  "data": {
    "groups": [
      {
        "key": "acme-corp",
        "keyId": null,
        "totalCostMicrodollars": 6000000,
        "requestCount": 3000,
        "avgCostMicrodollars": 2000
      }
    ],
    "period": "30d",
    "groupBy": "customer_id",
    "totalGroups": 1,
    "hasMore": false,
    "totals": {
      "totalCostMicrodollars": 9700000,
      "totalRequests": 5000
    }
  }
}
```

`totals` contains org-wide aggregates for the period (not just the visible groups). `hasMore` is `true` when more groups exist beyond the limit.

**200 OK** (CSV, when `format=csv`):

```
key,key_id,total_cost_microdollars,total_cost_usd,request_count,avg_cost_microdollars,avg_cost_usd
production-key,ns_key_11223344-5566-7788-99aa-bbccddeeff00,8500000,8.500000,4200,2024,0.002024
staging-key,ns_key_aabbccdd-eeff-0011-2233-445566778899,1200000,1.200000,800,1500,0.001500
```

Headers: `Content-Type: text/csv; charset=utf-8`, `Content-Disposition: attachment; filename="nullspend-attribution-api_key-2026-03-28.csv"`

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Missing groupBy, invalid period, limit out of range, invalid format |
| `authentication_required` | 401 | No valid session |
| `forbidden` | 403 | User lacks viewer role |

---

## Attribution Detail

`GET /api/cost-events/attribution/:key`

Retrieve daily spend trend and model breakdown for a single attribution group.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `key` | path | string | Yes | API key ID (`ns_key_*`) or tag value. Use `(no key)` for events without an API key. |
| `groupBy` | query | string | Yes | Must match the groupBy used in the list endpoint. `"api_key"` or a tag key name. |
| `period` | query | string | No | `"7d"`, `"30d"`, or `"90d"`. Default `"30d"`. |
| `excludeEstimated` | query | string | No | `"true"` or `"false"`. Default `"false"`. |

### Request

```bash
# API key detail (requires dashboard session)
curl "https://nullspend.dev/api/cost-events/attribution/ns_key_11223344-5566-7788-99aa-bbccddeeff00?groupBy=api_key" \
  -H "Cookie: session=..."

# Tag value detail
curl "https://nullspend.dev/api/cost-events/attribution/acme-corp?groupBy=customer_id" \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": {
    "key": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "totalCostMicrodollars": 8500000,
    "requestCount": 4200,
    "avgCostMicrodollars": 2024,
    "daily": [
      { "date": "2026-03-27", "cost": 285000, "count": 140 },
      { "date": "2026-03-26", "cost": 310000, "count": 155 }
    ],
    "models": [
      { "model": "gpt-4o", "cost": 7200000, "count": 3600 },
      { "model": "gpt-4o-mini", "cost": 1300000, "count": 600 }
    ]
  }
}
```

`daily` is sorted by date ascending. `models` is sorted by cost descending.

### Errors

| Code | HTTP | When |
|---|---|---|
| `invalid_key` | 400 | Key contains `/` or `..` (path traversal), or invalid `ns_key_*` format |
| `validation_error` | 400 | Missing groupBy or invalid period |
| `authentication_required` | 401 | No valid session |
| `forbidden` | 403 | User lacks viewer role |

---

## Tag Keys

`GET /api/cost-events/tag-keys`

Returns distinct non-internal tag key names from the last 7 days of cost events. Used to populate the Attribution page's groupBy dropdown.

### Authentication

Session (dashboard)

### Request

```bash
# Requires dashboard session
curl "https://nullspend.dev/api/cost-events/tag-keys" \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": ["customer_id", "environment", "feature", "team"]
}
```

Keys starting with `_ns_` (internal tags) are excluded. Maximum 50 keys returned. Sorted alphabetically.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `forbidden` | 403 | User lacks viewer role |

---

## Related

- [Cost Attribution](../features/cost-attribution.md) — feature overview and common patterns
- [Cost Tracking](../features/cost-tracking.md) — how costs are calculated
- [Tags](../features/tags.md) — tagging and filtering
- [Error Reference](errors.md) — full error catalog
- [Custom Headers](custom-headers.md) — header reference
