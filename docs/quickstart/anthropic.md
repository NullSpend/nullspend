# Anthropic Quickstart

Get cost tracking for your Anthropic calls in under 2 minutes.

## Prerequisites

- A NullSpend account ([sign up](https://nullspend.dev/signup))
- An existing app that calls the Anthropic API

## Step 1: Create an API Key

1. Log in to the [NullSpend dashboard](https://nullspend.dev/app/analytics)
2. Go to **Settings** → **Create API Key**
3. Copy the key (starts with `ns_live_sk_`) — you won't see it again

## Step 2: Point Your SDK at the Proxy

Set two environment variables:

```bash
# Point Anthropic SDK at NullSpend
ANTHROPIC_BASE_URL=https://proxy.nullspend.dev

# Your real Anthropic key — unchanged
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# NullSpend API key
NULLSPEND_API_KEY=ns_live_sk_your-key-here
```

Then add the `X-NullSpend-Key` header to your client:

### TypeScript

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  baseURL: "https://proxy.nullspend.dev",
  defaultHeaders: {
    "X-NullSpend-Key": process.env.NULLSPEND_API_KEY!,
  },
});

// Use exactly as before — no other code changes
const message = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello" }],
});
```

### Python

```python
from anthropic import Anthropic
import os

client = Anthropic(
    base_url="https://proxy.nullspend.dev",
    default_headers={
        "X-NullSpend-Key": os.environ["NULLSPEND_API_KEY"],
    },
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### cURL

```bash
curl https://proxy.nullspend.dev/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Tip:** Use the full model name including the date suffix (e.g., `claude-sonnet-4-20250514`, not just `claude-sonnet-4`). The proxy resolves model aliases, but the full name ensures accurate pricing.

## Step 3: Check the Dashboard

Open the [NullSpend dashboard](https://nullspend.dev/app/analytics). Cost events appear within seconds of each request completing. You'll see:

- **Daily spend chart** — cost over time
- **Model breakdown** — which models cost the most
- **Per-key breakdown** — costs attributed to each API key

### Cache Token Tracking

NullSpend tracks Anthropic cache tokens separately. If you use prompt caching, the dashboard breaks down cost into:

- **Input tokens** — standard input pricing
- **Cache read tokens** — discounted (typically 90% cheaper than input)
- **Cache creation tokens** — premium (typically 25% more than input)

This gives you accurate cost attribution even with heavy caching.

## What's Next

- **[Set a budget](../features/budgets.md)** — Go to Budgets → Create Budget. The proxy blocks requests with `429` when the ceiling is hit.
- **[Track per-customer costs](../features/tags.md#customer-attribution)** — Add the `X-NullSpend-Customer` header to see profitability per customer.
- **[Add tags](../features/tags.md)** — Attribute costs to teams or features with the [`X-NullSpend-Tags`](../api-reference/custom-headers.md#x-nullspend-tags) header.
- **[Configure webhooks](../webhooks/overview.md)** — Get notified on cost events, budget thresholds, and velocity alerts.
- **OpenAI too?** — [OpenAI Quickstart](openai.md)
- **Claude Agent SDK?** — [Claude Code Quickstart](claude-code.md)

## Troubleshooting

**401 Unauthorized**
Your `X-NullSpend-Key` header is missing, malformed, or the key has been revoked. Verify the key in Settings.

**429 Too Many Requests**
Either a budget ceiling was hit (check the `error.code` field — `budget_exceeded`, `velocity_exceeded`, or `rate_limited`) or you've exceeded the rate limit (600 req/min per key). See the [error reference](../api-reference/errors.md).

**Costs don't appear in the dashboard**
Cost logging is asynchronous. Wait a few seconds and refresh. If costs still don't appear, verify the request completed successfully in your application logs.

**Streaming doesn't work**
The proxy transparently passes through SSE streams. Anthropic uses named-event SSE (`event: message_start`, `event: content_block_delta`, etc.) — the proxy handles this natively. Cost is calculated from the final `message_delta` event's usage data.
