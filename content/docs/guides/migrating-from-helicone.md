---
title: "Migrating from Helicone to NullSpend"
description: "Migrating from Helicone to NullSpend documentation"
---

Helicone was acquired by Mintlify on March 3, 2026 and is entering maintenance
mode. If you're one of the 16,000 organizations on Helicone, this guide shows
you how to migrate to NullSpend in under 5 minutes.

## Why migrate now

- **Helicone is in maintenance mode.** No new features, bug fixes are
  best-effort, and there's no public timeline for how long the service stays up.
- **Your cost data will become inaccessible.** Helicone hasn't announced a data
  export tool or retention commitment post-acquisition.
- **The integration model is identical.** Both Helicone and NullSpend use the
  same pattern: change your base URL. Migration is literally swapping one URL
  for another.

## The migration: one line change

### Helicone (before)

```bash
OPENAI_BASE_URL=https://oai.helicone.ai/v1
HELICONE_API_KEY=sk-helicone-...
```

### NullSpend (after)

```bash
OPENAI_BASE_URL=https://proxy.nullspend.com/v1
# Remove HELICONE_API_KEY — not needed
```

Add the NullSpend authentication header to your client:

```typescript
const client = new OpenAI({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: {
    "X-NullSpend-Key": process.env.NULLSPEND_API_KEY,
  },
});
```

That's it. Your existing code, streaming, error handling — everything else stays
the same.

### For Anthropic

```bash
# Helicone (before)
ANTHROPIC_BASE_URL=https://anthropic.helicone.ai

# NullSpend (after)
ANTHROPIC_BASE_URL=https://proxy.nullspend.com
```

## Feature mapping

| Helicone feature | NullSpend equivalent | Notes |
|---|---|---|
| Request logging | Cost event logging | Automatic — every request is logged |
| Cost tracking | Cost tracking | Per-request, per-model, per-key breakdown |
| Dashboard | Analytics dashboard | Daily spend, model breakdown, provider breakdown |
| Custom properties | [Tags](../api-reference/custom-headers.md#x-nullspend-tags) | `X-NullSpend-Tags` header — JSON object with up to 10 key-value pairs for cost attribution |
| Alerts | Budget enforcement + [Webhooks](../webhooks/overview.md) | Hard stops (not just alerts) + 15 webhook event types with HMAC-SHA256 signing |
| Caching | — | Not yet supported (roadmap) |
| Rate limiting | Budget enforcement + Velocity limits | Budget ceilings + velocity limits detect runaway loops by spend rate |
| User tracking | API key tracking + Session limits | Track costs per key, plus per-conversation spend caps via `X-NullSpend-Session` |
| Prompt templates | — | Not supported (out of scope) |
| — | W3C traceparent | Automatic `traceparent`/`tracestate` propagation to upstream providers |

## What NullSpend adds that Helicone didn't have

### Hard budget enforcement

Helicone could alert you when costs got high. NullSpend **blocks requests** when
a budget ceiling is hit. No soft limits — the proxy returns `429` and the
request never reaches the LLM provider. You don't get charged for blocked
requests.

### Multi-provider unified dashboard

See OpenAI and Anthropic costs in a single dashboard with provider breakdown
charts. Helicone supported multiple providers but the experience was fragmented.

### Identity-based enforcement

Budget enforcement is tied to API keys on the server side. There's no
client-side header that can be spoofed or omitted to bypass the budget. This
was a known issue with Helicone's header-based approach.

### Tags for cost attribution

Helicone had custom properties. NullSpend has [tags](../features/tags.md) — attach a JSON object via the `X-NullSpend-Tags` header to attribute costs to teams, environments, features, or anything else. Up to 10 keys per request, queryable in the dashboard.

### Velocity limits

Detect runaway agent loops automatically. Set a spend-rate threshold (e.g., "$5 in 60 seconds") and the proxy blocks requests when the rate is exceeded. Helicone had no equivalent.

### Session limits

Cap spend per conversation with the `X-NullSpend-Session` header. When a session's cumulative spend hits the limit, the proxy returns `429`. Useful for chat-based agents where each conversation should have a budget.

### Webhooks

15 event types with HMAC-SHA256 signed payloads: cost events, budget exceeded, threshold crossings, velocity alerts, budget resets, blocked requests, and more. Supports both full and thin (Stripe v2 pattern) payload modes. See [Webhooks](../webhooks/overview.md). Helicone had no webhook support.

### W3C traceparent propagation

The proxy automatically forwards `traceparent` and `tracestate` headers to upstream providers, integrating with your existing distributed tracing infrastructure.

## Step-by-step migration

### 1. Sign up for NullSpend

Go to [nullspend.com/signup](https://nullspend.com/signup) and create an
account.

### 2. Create an API key

In the dashboard, go to **Settings** → **Create API Key**. Copy the API
key (starts with `ns_live_sk_`).

### 3. Set a budget (optional but recommended)

Go to **Budgets** → **Create Budget**. Set a spending ceiling based on your
current Helicone cost data. Start generous — you can tighten later.

### 4. Update your environment variables

Replace your Helicone base URL with the NullSpend proxy URL. Add the
`X-NullSpend-Key` header to your client configuration.

### 5. Remove Helicone dependencies

```bash
# Remove the Helicone package if you installed it
npm uninstall @helicone/helicone
# or
pip uninstall helicone
```

Remove any Helicone-specific headers from your code:

```typescript
// Remove these:
// "Helicone-Auth": "Bearer sk-helicone-..."
// "Helicone-Property-*": ...
// "Helicone-Cache-Enabled": ...
```

### 6. Deploy and verify

Deploy your updated application. Check the NullSpend dashboard — cost events
should appear within seconds of your first LLM call.

## Pricing comparison

| Feature | Helicone (was) | NullSpend |
|---|---|---|
| Free tier | 10K logs, 3-day retention | $5K/mo proxied spend, 3 budgets, 3 team members, 30-day retention |
| Budget enforcement | Alerts only | Hard enforcement (all tiers) |
| Starting paid tier | $20/mo (limited) | $49/mo (unlimited budgets/keys/members, 90-day retention) |

## FAQ

**Will my Helicone data be lost?**
Helicone hasn't announced data export or retention guarantees. We recommend
exporting any critical data from Helicone's dashboard before migrating.

**Can I run both Helicone and NullSpend simultaneously?**
Not easily — both require changing the base URL, and you can only point at one
proxy at a time. However, you could route some traffic to each by using
different API keys with different base URLs.

**How long does migration take?**
The actual code change takes under 5 minutes. It's one environment variable
swap and adding the auth header.

**What if I was using Helicone's caching?**
NullSpend doesn't support response caching yet. If you relied on Helicone's
cache, you'll need to implement caching at the application level or wait for
our caching feature (on the roadmap).

**What about Helicone's custom properties?**
Use [tags](../features/tags.md) — add the `X-NullSpend-Tags` header with a JSON object to attribute costs by team, environment, feature, or any custom dimension. You can also segment by API key (one key per agent or user).
