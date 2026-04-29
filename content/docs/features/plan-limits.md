---
title: "Plan Limits & Governed Requests"
description: "Each NullSpend tier ships with a monthly governed-request allowance. Once you hit it, Free tier hard-blocks and paid tiers bill the overage."
---

Every NullSpend tier ships with a monthly **governed-request** allowance. Free tier hard-blocks once the cap is hit; paid tiers (Pro, Scale) keep serving traffic and bill overage to Stripe at a published per-request rate.

Unlike org-configured budgets (`budget_exceeded`), plan limits come from the NullSpend pricing tier itself. They reset at the start of each Stripe billing period.

## What counts as a governed request

Any LLM request that flows through the NullSpend proxy or SDK and is tracked against a budget. The counter is per-request, not per-token — a 1-token call and a 100K-token call both count once. Cost is still calculated and stored normally.

Non-counting events:

- Health checks (`/api/health`)
- Dashboard reads (analytics, list endpoints)
- Webhook deliveries
- Failed requests that never reached the provider (e.g., `unauthorized`, `bad_request`)

Counting events:

- `POST /v1/chat/completions` (OpenAI)
- `POST /v1/messages` (Anthropic)
- `POST /v1beta/models/{model}:generateContent` (Gemini)
- `POST /api/cost-events` and `POST /api/cost-events/batch` (SDK reporting)
- MCP proxy + MCP server tool invocations that produce a cost event

## Per-tier caps

| Tier | Monthly cap | Behavior at cap | Overage rate |
|---|---|---|---|
| Free | 100,000 | Hard block (`429 plan_limit_exceeded`) | None — wait for reset or upgrade |
| Pro | 500,000 | Continue serving, bill overage | $0.010 / request |
| Scale | 2,000,000 | Continue serving, bill overage | $0.005 / request |
| Enterprise | Unlimited | Continue serving | Custom |

Full plan comparison and FAQ live on the [pricing page](https://nullspend.dev/pricing).

## What happens when Free tier hits the cap

The proxy returns `429` with `error.code: "plan_limit_exceeded"`. The body includes the upgrade URL so callers can render an in-product CTA:

```json
{
  "error": {
    "code": "plan_limit_exceeded",
    "message": "Plan limit reached: 100000 of 100000 governed requests on free plan. Upgrade or wait for period reset.",
    "upgrade_url": "https://nullspend.dev/pricing",
    "self_host_url": "https://nullspend.dev/docs",
    "details": { "current_count": 100000, "block_at": 100000, "tier": "free" },
    "recovery": { "retryable": false, "owner_action_required": true, "retry_after_seconds": null, "docs": null }
  }
}
```

The response includes `X-NullSpend-Denied: 1` so callers can short-circuit handling without parsing the body.

A `plan_limit.exceeded` webhook fires for every block so out-of-band consumers (Slack, PagerDuty, ops dashboards) see the event in real time. See [Event Types](../webhooks/event-types.md#plan_limitexceeded) for the payload.

## How counters reset

The counter resets at the start of each Stripe billing period — typically the 1st of the month for monthly subscriptions. The reset is atomic: at the period boundary, the counter snaps to zero and traffic resumes immediately.

For self-hosted deployments, plan limits do not apply — the proxy you run is unlimited.

## SDK handling

Both SDKs surface plan-limit denials as a typed error:

```typescript
import { PlanLimitExceededError } from "@nullspend/sdk";

try {
  await openai.chat.completions.create({ /* ... */ });
} catch (err) {
  if (err instanceof PlanLimitExceededError) {
    console.log(`Plan cap hit (${err.count}/${err.blockAt}). Upgrade: ${err.upgradeUrl}`);
  }
}
```

```python
from nullspend import PlanLimitExceededError

try:
    openai.chat.completions.create(model="gpt-4o", messages=[...])
except PlanLimitExceededError as err:
    print(f"Plan cap hit ({err.count}/{err.block_at}). Upgrade: {err.upgrade_url}")
```

## Self-host alternative

The NullSpend proxy, SDK, MCP server, and Claude Agent adapter are open source. Running the proxy yourself removes the governed-request cap entirely — you only pay your own infrastructure costs. See the [GitHub repo](https://github.com/NullSpend/nullspend) for the deployment guide.

## Related

- [Errors — `plan_limit_exceeded`](../api-reference/errors.md#budget-enforcement) — full HTTP response shape
- [Event Types — `plan_limit.exceeded`](../webhooks/event-types.md#plan_limitexceeded) — webhook payload
- [Organizations](organizations.md) — per-tier feature breakdown
- [Cost Tracking](cost-tracking.md) — how each request becomes a cost event
- Pricing page — [nullspend.dev/pricing](https://nullspend.dev/pricing)
