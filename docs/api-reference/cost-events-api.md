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
| `eventType` | body | string | No | `"llm"`, `"tool"`, or `"custom"`. Default `"custom"`. |
| `toolName` | body | string | No | Tool name for tool-use events. Max 200 chars. |
| `toolServer` | body | string | No | Tool server name. Max 200 chars. |
| `tags` | body | object | No | Key-value metadata. Max 10 keys, key 1–64 chars (`^[a-zA-Z0-9_-]+$`), value max 256 chars. |
| `idempotencyKey` | body | string | No | Deduplication key. Max 200 chars. Alternative to `Idempotency-Key` header. |
| `Idempotency-Key` | header | string | No | Deduplication key. Takes priority over body field. |

### Request

```typescript
const res = await fetch("https://nullspend.com/api/cost-events", {
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
    "https://nullspend.com/api/cost-events",
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
curl -X POST https://nullspend.com/api/cost-events \
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
  "id": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "createdAt": "2026-03-20T14:30:00.000Z"
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
const res = await fetch("https://nullspend.com/api/cost-events/batch", {
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
    "https://nullspend.com/api/cost-events/batch",
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
curl -X POST https://nullspend.com/api/cost-events/batch \
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

Retrieve cost events for the authenticated user with filtering and pagination.

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
| `tag.*` | query | string | No | JSONB containment filter. Example: `tag.environment=production`. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.com/api/cost-events?limit=10&provider=openai \
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
      "tags": { "environment": "production" },
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

Retrieve a single cost event by ID. The event must belong to the authenticated user.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Event ID. Accepts `ns_evt_*` or raw UUID. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.com/api/cost-events/ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
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
| `not_found` | 404 | Event not found or not owned by user |

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
curl "https://nullspend.com/api/cost-events/summary?period=7d" \
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

## Related

- [Cost Tracking](../features/cost-tracking.md) — feature overview
- [Tags](../features/tags.md) — tagging and filtering
- [Error Reference](errors.md) — full error catalog
- [Custom Headers](custom-headers.md) — header reference
