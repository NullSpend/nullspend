---
title: "Cost Attribution"
description: "Break down AI spend by customer, team, feature, or any dimension. Answer 'how much does each customer cost me?' with per-key and per-tag grouping."
---

Break down AI spend by customer, team, feature, or any dimension. Answer "how much does each customer cost me?" with per-key and per-tag grouping, daily trends, model breakdowns, and CSV export.

## The Problem

You're running an AI-powered product. Your CEO asks: "How much does each customer cost us in AI spend?" Without per-customer attribution, you're stuck dividing total spend by customer count and hoping the average is close enough. It never is.

## How It Works

The Attribution page groups your existing cost events by **API key** or by any **tag value**, then ranks them by spend.

```
┌──────────────────────────────────────────────────────────┐
│  Attribution                                              │
│                                                          │
│  Group by: [▾ Tag: customer_id ]   Period: [30 days]     │
│                                                          │
│  Key              Cost        % of Total   Requests      │
│  acme-corp        $842.30     20.1%        12,341        │
│  globex           $623.50     14.8%         8,921        │
│  initech          $198.20      4.7%         3,102        │
│  ...                                                     │
│                                                          │
│  Click any row → daily trend + model breakdown           │
└──────────────────────────────────────────────────────────┘
```

No new integration required. Attribution works with the cost event data you're already collecting.

## Two Grouping Modes

### API Key Grouping (default)

Groups by the API key that made each request. Works immediately if you use one key per customer or per environment.

This mode includes a "(no key)" group for cost events without an associated API key, so totals always match the analytics page.

### Tag Grouping

Groups by any [tag](tags.md) value you've attached to requests via `X-NullSpend-Tags`. Tag your requests with `customer_id`, `team`, `feature`, or any dimension, then select that tag in the Attribution dropdown.

```typescript
// Tag every request with a customer identifier
const response = await openai.chat.completions.create(
  {
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  },
  {
    headers: {
      "X-NullSpend-Tags": JSON.stringify({
        customer_id: "acme-corp",
        team: "engineering",
      }),
    },
  }
);
```

```python
import json

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-NullSpend-Tags": json.dumps({
            "customer_id": "acme-corp",
            "team": "engineering",
        }),
    },
)
```

The Attribution page auto-detects which tag keys exist in your recent data and populates the dropdown.

## Dashboard

### List View

Navigate to **Attribution** in the FinOps sidebar section. The page shows:

- **Summary stats** — Total spend, total requests, and average cost per request for the entire period (not just visible groups)
- **Sortable table** — Key name, cost, % of total, request count, average cost. Click any column header to sort.
- **Period selector** — 7 days, 30 days, or 90 days
- **GroupBy dropdown** — "API Key" plus any tag keys detected in your last 7 days of data
- **CSV export** — Download the attribution table as a CSV file for finance teams

### Detail View

Click any row to see that group's detail page:

- **Daily spend trend** — Area chart showing cost per day over the selected period
- **Model breakdown** — Which models this group is using, with cost and % of total per model
- **Summary stats** — Total spend, requests, and average cost for this specific group

## API

Three new endpoints power the Attribution feature. All require session authentication (dashboard).

### List Attribution Groups

`GET /api/cost-events/attribution`

See [Cost Events API — Attribution](../api-reference/cost-events-api.md#cost-attribution) for full parameters, request examples, and response format.

### Attribution Detail

`GET /api/cost-events/attribution/:key`

See [Cost Events API — Attribution Detail](../api-reference/cost-events-api.md#attribution-detail) for full parameters and response format.

### Tag Key Detection

`GET /api/cost-events/tag-keys`

See [Cost Events API — Tag Keys](../api-reference/cost-events-api.md#tag-keys) for response format.

## Common Patterns

### 1 Key Per Customer

If you already issue one API key per customer, attribution works out of the box with the default "API Key" grouping. No tags needed.

### Shared Keys + Customer Tags

Most companies use a few shared keys (production, staging) and tag requests with customer metadata. Use `X-NullSpend-Tags` with a `customer_id` key, then select "Tag: customer_id" in the Attribution dropdown.

### Multi-Dimensional Attribution

Tag requests with multiple dimensions simultaneously:

```json
{
  "customer_id": "acme-corp",
  "team": "engineering",
  "feature": "chatbot"
}
```

Then switch between groupings in the dropdown to answer different questions:
- "How much does each **customer** cost?" → Group by `customer_id`
- "How much does each **team** spend?" → Group by `team`
- "How much does each **feature** cost?" → Group by `feature`

## Performance

API key grouping is fast — it uses a compound index on `(api_key_id, created_at)`.

Tag-based grouping requires JSONB extraction and is slower at scale. Expected performance:

| Events in period | Approximate query time |
|---|---|
| < 100K | < 500ms |
| 100K–1M | 500ms–2s |
| 1M–5M | 2–5s |

If tag-based queries become slow for your usage, contact support — we can add a targeted index for your most-used tag keys.

## Related

- [Tags](tags.md) — how to send tags with requests
- [Cost Tracking](cost-tracking.md) — how costs are calculated
- [Cost Events API](../api-reference/cost-events-api.md#cost-attribution) — API reference for attribution endpoints
