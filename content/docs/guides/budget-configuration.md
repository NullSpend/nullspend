---
title: "Budget Configuration"
description: "Step-by-step guide to creating budgets, configuring velocity / session / loop limits, choosing enforcement policies, and using finalization reserves."
---

Set spending ceilings to prevent cost overruns from runaway agents.

## How budgets work

A budget is a spending ceiling attached to a user account, an API key, a tag, or a customer. When the estimated cost of a request would push cumulative spend over the budget, the proxy blocks the request with a `429 Too Many Requests` response.

Budget enforcement happens **before** the request is forwarded to the LLM provider. The proxy checks the current spend against the budget atomically using a Cloudflare Durable Object (single-threaded, no race conditions even under concurrent load).

## Creating a budget

1. Open the [NullSpend dashboard](https://nullspend.dev/app/budgets)
2. Click **Set Budget**
3. Configure:
   - **Budget for** — your account, a specific API key, a tag (e.g., `env=production`), or a customer
   - **Budget limit** — spending ceiling in dollars (e.g., $50.00)
   - **Reset interval** — none (manual reset), daily, weekly, or monthly
4. Optionally configure advanced guardrails (expand each section):
   - **Velocity limit** — max spend per sliding time window (triggers a cooldown if exceeded)
   - **Alert thresholds** — custom percentage thresholds for webhook alerts
   - **Session limit** — max spend per agent session (see below)
5. Click **Set Budget**

The budget takes effect immediately. No proxy restart or redeployment needed.

## Budget enforcement behavior

When a request would exceed the budget:

1. The proxy returns HTTP `429 Too Many Requests`
2. The response body includes a machine-readable error:
   ```json
   {
     "error": {
       "code": "budget_exceeded",
       "message": "Request blocked: estimated cost exceeds remaining budget",
       "details": null
     }
   }
   ```
3. Your application receives a standard HTTP error — handle it like any other
   rate limit or quota error

The blocked request is **never forwarded** to the LLM provider. You are not
charged by OpenAI/Anthropic for blocked requests.

## Enforcement policies

Each budget has a `policy` that controls what happens when the limit is exceeded:

| Policy | Behavior | Use case |
|---|---|---|
| `strict_block` (default) | Blocks the request with `429`. The request is never forwarded to the provider. | Production cost control. You want hard spending caps. |
| `soft_block` | Logs the overage and allows the request through. Cost events are tagged with budget status `denied`. | Transitioning from tracking to enforcement. See the impact before turning on hard blocks. |
| `warn` | Tracks spend but never blocks. Threshold webhooks still fire. | Monitoring-only mode. Useful during initial rollout or for customer budgets where you want visibility without disruption. |

Set the policy when creating a budget via the dashboard or API:

```json
{
  "entityType": "customer",
  "entityId": "acme-corp",
  "maxBudgetMicrodollars": 50000000,
  "policy": "warn"
}
```

Session limits and velocity limits always enforce `strict_block` regardless of the budget policy.

## Velocity limits

Velocity limits catch runaway loops — an agent stuck in a retry cycle can burn through a budget in seconds. When spend in a sliding window exceeds the velocity limit, the proxy trips a circuit breaker and blocks requests for a cooldown period.

Configure in the budget dialog:
- **Velocity limit** — dollar amount per window (e.g., $10)
- **Window** — sliding window in seconds (10-3600, default 60)
- **Cooldown** — block duration after tripping (10-3600, default 60)

Velocity denial returns `429` with `"code": "velocity_exceeded"` and a `Retry-After` header indicating when the cooldown expires.

## Session limits

Session limits cap how much a single agent session can spend, regardless of the overall budget. This prevents a single long-running agent task from consuming the entire budget.

### How it works

1. Your agent sets a session ID via the `X-NullSpend-Session` header on each request
2. The proxy tracks cumulative spend per session ID per budget entity
3. When a session's spend would exceed the session limit, the request is blocked

### Configuration

Set the session limit in the budget dialog under "Session limit (optional)". Enter a dollar amount (e.g., $5.00).

### Session denial response

```json
{
  "error": {
    "code": "session_limit_exceeded",
    "message": "Request blocked: session spend exceeds session limit. Start a new session.",
    "details": {
      "session_id": "agent-task-abc",
      "session_spend_microdollars": 4800000,
      "session_limit_microdollars": 5000000
    }
  }
}
```

No `Retry-After` header is sent — the session is done. The agent should start a new session (new session ID) to continue.

### Key behaviors

- **No session header = no enforcement.** Session limits only apply when the `X-NullSpend-Session` header is present.
- **Sessions are client-defined.** The proxy does not manage session lifecycle. Your agent decides when to start a new session by sending a new session ID.
- **Independent of budget resets.** Session spend does NOT reset when the budget period resets. A session spans calendar boundaries.
- **Always strict block.** Session limits are hard caps regardless of the budget policy (strict_block, soft_block, or warn).
- **24-hour cleanup.** Stale session data is automatically cleaned up after 24 hours of inactivity.

### Webhook event

When a session limit is exceeded, a `session.limit_exceeded` webhook event is dispatched (if webhooks are configured).

## Finalization reserve

When agents hit their budget limit, they're hard-killed mid-task. Finalization reserve holds back a portion of the budget so agents can finish gracefully.

### Setting a finalization reserve

1. In the budget dialog, expand **Finalization reserve**
2. Enter the dollar amount to hold back (e.g., $5.00 on a $100 budget)
3. Click **Set Budget**

The budget list shows a two-zone progress bar when a reserve is active: green for the normal zone, amber for the reserve.

### How agents use it

Your agent watches the response headers:

```typescript
const response = await trackedFetch("https://proxy.nullspend.dev/v1/chat/completions", {
  // ... normal request
});

const remaining = parseInt(response.headers.get("X-NullSpend-Budget-Effective-Remaining") || "0");
const reserve = parseInt(response.headers.get("X-NullSpend-Budget-Finalization-Reserve") || "0");

if (remaining <= 0 && reserve > 0) {
  // We're in the reserve zone — do one final cleanup request
  const finalResponse = await trackedFetch("https://proxy.nullspend.dev/v1/chat/completions", {
    headers: { "X-NullSpend-Finalize": "1" },
    // ... cleanup request
  });
}
```

Or with the SDK:

```typescript
const finalFetch = ns.createTrackedFetch("openai", { finalize: true });
```

### Recommendations

- Set the reserve to the cost of 1-2 cleanup requests (typically $1-5 for GPT-4o)
- Use the `X-NullSpend-Budget-Requests-Remaining` header to know when to start winding down
- The reserve only unlocks when the agent has actually reached the zone — setting `finalize` early has no effect

## Budget tracking

Budget spend is tracked in **real-time** using a Cloudflare Durable Object with embedded SQLite. This means:

- Spend updates are instantaneous (no batch processing delay)
- Concurrent requests are serialized (no race conditions)
- Budget state is durable and survives proxy restarts

## Viewing budget status

In the dashboard, go to **Budgets** to see:

- Current spend vs. ceiling for each budget
- Percentage utilization with color-coded health indicators
- Velocity limit indicator (lightning icon) and session limit indicator (clock icon)
- Reset interval and days remaining

## Best practices

### Start generous, tighten later

Set your initial budget higher than you expect to need. Once you have a few days
of cost data in the analytics dashboard, you can tighten the budget to a
reasonable ceiling with confidence.

### One budget per concern

Create separate API keys (and budgets) for different agents, environments, or
teams:

- `production-agent-alpha` — $200/month
- `production-agent-beta` — $100/month
- `staging` — $20/month
- `development` — $5/month

### Use session limits for long-running agents

If your agents run multi-step tasks (research, code generation, data analysis),
set a session limit to cap each task's cost. This prevents a single stuck agent
from consuming the entire budget while still allowing other tasks to proceed.

### Monitor before enforcing

Use the analytics dashboard to understand your spending patterns before setting
tight budgets. The daily spend chart and model breakdown help you identify which
models and agents drive the most cost.

## Pricing tier limits

| Tier | Budgets | API Keys | Webhooks | Team Members |
|---|---|---|---|---|
| Free | 3 | 10 | 2 | Unlimited |
| Pro ($49/mo) | Unlimited | Unlimited | Unlimited | Unlimited |
| Scale ($199/mo) | Unlimited | Unlimited | Unlimited | Unlimited |
| Enterprise | Unlimited | Unlimited | Unlimited | Unlimited |

The Free tier includes 3 budgets — enough for separate production, staging, and development keys.
Upgrade to Pro for unlimited budgets across all keys and environments. See the [pricing page](https://nullspend.dev/pricing) for the full feature matrix.

## API Reference

For programmatic budget management, see the [Budgets API](../api-reference/budgets-api.md). Key endpoints:

- [`POST /api/budgets`](../api-reference/budgets-api.md#create-or-update-budget) — create or update a budget (session auth)
- [`GET /api/budgets/status`](../api-reference/budgets-api.md#get-budget-status) — check remaining balance (API key auth)
