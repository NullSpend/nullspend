---
title: "Claude Code Quickstart"
description: "Get cost tracking for Claude Agent SDK calls in under 2 minutes."
---

Get cost tracking for Claude Agent SDK calls in under 2 minutes.

## Prerequisites

- A NullSpend account ([sign up](https://nullspend.com/signup))
- An existing app using the Claude Agent SDK (`@anthropic-ai/claude-agent-sdk`)

## Step 1: Create an API Key

1. Log in to the [NullSpend dashboard](https://nullspend.com/app/analytics)
2. Go to **Settings** → **Create API Key**
3. Copy the key (starts with `ns_live_sk_`) — you won't see it again

## Step 2: Install the Adapter

```bash
npm install @nullspend/claude-agent
```

## Step 3: Wrap Your Config

`withNullSpend()` takes your Claude Agent SDK options, adds NullSpend headers, and returns the modified options. No other code changes needed.

```typescript
import { withNullSpend } from "@nullspend/claude-agent";

const options = withNullSpend({
  apiKey: process.env.NULLSPEND_API_KEY!,

  // Optional: attribute costs to a project
  tags: { project: "my-project" },

  // ... your existing Claude Agent SDK options
  model: "claude-sonnet-4-20250514",
  prompt: "You are a helpful assistant.",
});

// Pass options to the Claude Agent SDK as usual
```

Under the hood, `withNullSpend` sets `ANTHROPIC_BASE_URL` to the NullSpend proxy and injects `ANTHROPIC_CUSTOM_HEADERS` with your API key. Your Anthropic API key passes through unchanged.

## Step 4: Check the Dashboard

Open the [NullSpend dashboard](https://nullspend.com/app/analytics). Cost events appear within seconds of each request completing. You'll see:

- **Daily spend chart** — cost over time
- **Model breakdown** — which models cost the most
- **Per-key breakdown** — costs attributed to each API key

## Optional: Add Session & Trace

Track per-conversation costs and correlate multi-step agent runs:

```typescript
const options = withNullSpend({
  apiKey: process.env.NULLSPEND_API_KEY!,

  // Group costs by conversation
  budgetSessionId: "session-abc-123",

  // Correlate requests in a multi-step run
  traceId: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",

  // ... other options
});
```

- `budgetSessionId` enables [session limits](../features/budgets.md#session-limits) — per-conversation spend caps
- `traceId` must be a 32-char lowercase hex string — see [Tracing](../features/tracing.md)

## What's Next

- **[Set a budget](../features/budgets.md)** — spending ceilings that block requests with `429`
- **[Add tags](../features/tags.md)** — attribute costs to teams, environments, or features
- **[Configure webhooks](../webhooks/overview.md)** — get notified on cost events and budget thresholds
- **[SDK reference](../sdks/claude-agent.md)** — full `withNullSpend()` option reference

## Troubleshooting

**401 Unauthorized**
Your NullSpend API key is missing or invalid. Verify the key in Settings. The key must start with `ns_live_sk_` or `ns_test_sk_`.

**Costs don't appear in the dashboard**
Cost logging is asynchronous. Wait a few seconds and refresh. If costs still don't appear, verify that `ANTHROPIC_BASE_URL` is set to `https://proxy.nullspend.com` in the subprocess environment — `withNullSpend` sets this automatically.

**429 Budget Exceeded**
A budget ceiling was hit. Check the dashboard for which budget was exceeded. See the [error reference](../api-reference/errors.md).

**Validation errors on startup**
`withNullSpend` validates inputs eagerly. Common issues:
- `traceId` must be exactly 32 lowercase hex characters
- `tags` must have at most 10 keys, with keys matching `[a-zA-Z0-9_-]+`
- `apiKey` and `budgetSessionId` must not contain newline characters
