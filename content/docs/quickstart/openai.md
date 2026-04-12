---
title: "OpenAI Quickstart"
description: "Get cost tracking for your OpenAI calls in under 2 minutes."
---

Get cost tracking for your OpenAI calls in under 2 minutes.

## Prerequisites

- A NullSpend account ([sign up](https://nullspend.dev/signup))
- An existing app that calls the OpenAI API

## Step 1: Create an API Key

1. Log in to the [NullSpend dashboard](https://nullspend.dev/app/analytics)
2. Go to **Settings** → **Create API Key**
3. Copy the key (starts with `ns_live_sk_`) — you won't see it again

## Step 2: Point Your SDK at the Proxy

Set two environment variables:

```bash
# Point OpenAI SDK at NullSpend
OPENAI_BASE_URL=https://proxy.nullspend.dev/v1

# Your real OpenAI key — unchanged
OPENAI_API_KEY=sk-your-openai-key

# NullSpend API key
NULLSPEND_API_KEY=ns_live_sk_your-key-here
```

Then add the `X-NullSpend-Key` header to your client:

### TypeScript

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://proxy.nullspend.dev/v1",
  defaultHeaders: {
    "X-NullSpend-Key": process.env.NULLSPEND_API_KEY,
  },
});

// Use exactly as before — no other code changes
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

### Python

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://proxy.nullspend.dev/v1",
    default_headers={
        "X-NullSpend-Key": os.environ["NULLSPEND_API_KEY"],
    },
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### cURL

```bash
curl https://proxy.nullspend.dev/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Step 3: Check the Dashboard

Open the [NullSpend dashboard](https://nullspend.dev/app/analytics). Cost events appear within seconds of each request completing. You'll see:

- **Daily spend chart** — cost over time
- **Model breakdown** — which models cost the most
- **Per-key breakdown** — costs attributed to each API key

## What's Next

- **[Set a budget](../features/budgets.md)** — Go to Budgets → Create Budget. The proxy blocks requests with `429` when the ceiling is hit.
- **[Add tags](../features/tags.md)** — Attribute costs to teams or features with the [`X-NullSpend-Tags`](../api-reference/custom-headers.md#x-nullspend-tags) header.
- **[Configure webhooks](../webhooks/overview.md)** — Get notified on cost events, budget thresholds, and velocity alerts.
- **Anthropic too?** — [Anthropic Quickstart](anthropic.md)
- **Claude Agent SDK?** — [Claude Code Quickstart](claude-code.md)

## Troubleshooting

**401 Unauthorized**
Your `X-NullSpend-Key` header is missing, malformed, or the key has been revoked. Verify the key in Settings.

**429 Too Many Requests**
Either a budget ceiling was hit (check the `error.code` field — `budget_exceeded`, `velocity_exceeded`, or `rate_limited`) or you've exceeded the rate limit (600 req/min per key). See the [error reference](../api-reference/errors.md).

**Costs don't appear in the dashboard**
Cost logging is asynchronous. Wait a few seconds and refresh. If costs still don't appear, verify the request completed successfully in your application logs.

**Streaming doesn't work**
The proxy transparently passes through SSE streams. If streaming worked before pointing at NullSpend, it works identically after. The proxy calculates cost from the final `usage` chunk.
