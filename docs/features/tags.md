# Tags

Tags let you attribute costs to teams, environments, features, or anything else. Attach a JSON object to any request and query costs by those dimensions in the dashboard, API, or webhooks.

## Sending Tags

Add the `X-NullSpend-Tags` header with a JSON object:

### TypeScript

```typescript
const response = await openai.chat.completions.create(
  {
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  },
  {
    headers: {
      "X-NullSpend-Tags": JSON.stringify({
        team: "billing",
        env: "production",
        feature: "summarizer",
      }),
    },
  }
);
```

### Python

```python
import json

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-NullSpend-Tags": json.dumps({
            "team": "billing",
            "env": "production",
            "feature": "summarizer",
        }),
    },
)
```

### cURL

```bash
curl https://proxy.nullspend.dev/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H 'X-NullSpend-Tags: {"team":"billing","env":"production"}' \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Validation Rules

Tags are **supplementary** — they never cause a request to be rejected. Invalid tags are silently dropped.

| Rule | Limit |
|---|---|
| Max keys per request | 10 |
| Key pattern | `[a-zA-Z0-9_-]+` |
| Key max length | 64 characters |
| Value max length | 256 characters |
| Reserved prefix | `_ns_` — keys starting with this are silently dropped |
| Null bytes in values | Tag is silently dropped |
| Invalid JSON | All tags silently dropped (request proceeds with no tags) |
| Single invalid key/value | That key is dropped; valid keys are kept |

## System Tags

NullSpend uses the `_ns_` prefix for internal tags. You cannot set these — they are added automatically when applicable.

| Tag | Value | When |
|---|---|---|
| `_ns_estimated` | `"true"` | Cost is an estimate (stream was cancelled before completion) |
| `_ns_cancelled` | `"true"` | The streaming response was cancelled by the client |

## Querying by Tags

### Dashboard

Filter cost events by tag key-value pairs in the analytics view.

### API

Use `tag.*` query parameters on [`GET /api/cost-events`](../api-reference/cost-events-api.md#list-cost-events):

```bash
# All cost events tagged with team=billing (requires dashboard session)
curl "https://nullspend.dev/api/cost-events?tag.team=billing" \
  -H "Cookie: session=..."

# Multiple tag filters (AND logic)
curl "https://nullspend.dev/api/cost-events?tag.team=billing&tag.env=production" \
  -H "Cookie: session=..."
```

Tag queries use PostgreSQL JSONB containment (`@>`), so they are indexed and fast.

## Tags in Webhooks

Tags are included in the `cost_event.created` webhook payload under `data.object.tags`:

```json
{
  "id": "evt_abc123",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "request_id": "req_xyz",
      "provider": "openai",
      "model": "gpt-4o",
      "cost_microdollars": 45,
      "tags": {
        "team": "billing",
        "env": "production"
      }
    }
  }
}
```

See [Webhook Event Types](../webhooks/event-types.md#cost_eventcreated) for the full payload.

## Tag Budgets

You can create budgets scoped to a specific tag key-value pair. When spend for that tag exceeds the budget, the proxy blocks requests carrying that tag with `429`:

```json
{
  "error": {
    "code": "tag_budget_exceeded",
    "message": "Request blocked: tag budget exceeded",
    "details": {
      "tag_key": "team",
      "tag_value": "billing",
      "budget_limit_microdollars": 50000000,
      "budget_spend_microdollars": 49500000
    }
  }
}
```

Tag budgets support the same features as user and API key budgets: reset intervals, threshold alerts, and velocity limits. See [Budgets](budgets.md) for configuration details.

## Customer Attribution

NullSpend supports two ways to identify the end customer a request serves:

### Dedicated header (recommended)

```bash
X-NullSpend-Customer: acme-corp
```

The `X-NullSpend-Customer` header writes to a dedicated indexed column, is auto-injected into tags for budget compatibility, and powers the [Customers](margins.md) dashboard. See [Custom Headers](../api-reference/custom-headers.md#x-nullspend-customer) for validation rules.

### Tag fallback (backward compatible)

```
X-NullSpend-Tags: {"customer": "acme-corp", "env": "production"}
```

If no `X-NullSpend-Customer` header is set, NullSpend falls back to `tags["customer"]` for customer attribution. This works identically for margins, budgets, and the Customers dashboard.

### Customer budgets

You can create budgets scoped to a specific customer (entity type `customer`) or via the tag convention (entity type `tag`, entity ID `customer=acme-corp`). Both enforce spending limits. Do not create both for the same customer.

## Related

- [Custom Headers Reference](../api-reference/custom-headers.md#x-nullspend-tags) — header format and validation
- [Cost Tracking](cost-tracking.md) — how costs are calculated
- [Budgets](budgets.md) — enforce spending limits including per-tag budgets
- [Margins](margins.md) — per-customer profitability using the `customer` tag
- [Webhook Event Types](../webhooks/event-types.md) — tags in webhook payloads
