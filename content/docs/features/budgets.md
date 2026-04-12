---
title: "Budgets"
description: "Budgets are spending ceilings enforced by the proxy. When the estimated cost of a request would push spend over the limit, the proxy returns `429` — the reque"
---

Budgets are spending ceilings enforced by the proxy. When the estimated cost of a request would push spend over the limit, the proxy returns `429` — the request never reaches the provider and you are not charged.

For a task-oriented setup guide, see [Budget Configuration](../guides/budget-configuration.md).

## How Budgets Work

```
Request arrives
    │
    ├─ 1. Estimate cost (input tokens + max output tokens × 1.1 safety margin)
    │
    ├─ 2. Period reset ────────── due? ───────► Reset spend to 0, start new period
    │
    ├─ 3. Session limit check ─── exceeds? ──► 429 session_limit_exceeded
    │
    ├─ 4. Velocity check ──────── tripped? ───► 429 velocity_exceeded + Retry-After
    │
    ├─ 5. Budget check ────────── exceeds? ──► 429 budget_exceeded
    │
    ├─ 6. Reserve estimated cost (30s TTL)
    │
    ├─ 7. Forward request to provider
    │
    ├─ 8. Receive response, calculate actual cost
    │
    └─ 9. Reconcile: apply actual cost, release reservation
```

Budget enforcement uses a Cloudflare Durable Object with embedded SQLite. All checks and mutations are serialized — no race conditions, even under concurrent load.

## Budget Entity Types

| Entity Type | What It Scopes To |
|---|---|
| `user` | All requests from a user account (across all their API keys) |
| `api_key` | Requests from a specific API key |
| `tag` | Requests carrying a specific tag key-value pair (e.g., `env=production`) |
| `customer` | Requests carrying a specific customer ID via `X-NullSpend-Customer` header |

A single request can match multiple budgets (e.g., a user budget + an API key budget + a customer budget). All matching budgets must have sufficient remaining balance for the request to proceed.

## Configuration

| Field | Type | Description |
|---|---|---|
| `maxBudgetMicrodollars` | integer (required) | Spending ceiling in microdollars. $50 = 50,000,000 |
| `policy` | string | Enforcement policy: `"strict_block"` (default, denies requests), `"soft_block"` (logs but allows), or `"warn"` (tracks only). |
| `resetInterval` | string or null | `"daily"`, `"weekly"`, `"monthly"`, or `null` (no reset — manual only) |
| `thresholdPercentages` | integer[] | Webhook alert thresholds. Default: `[50, 80, 90, 95]`. Max 10 values, must be ascending, each 1–100. |
| `velocityLimitMicrodollars` | integer or null | Max spend per velocity window. See [Velocity Limits](#velocity-limits). |
| `velocityWindowSeconds` | integer | Sliding window size. Range: 10–3600. Default: 60. |
| `velocityCooldownSeconds` | integer | Block duration after velocity trip. Range: 10–3600. Default: 60. |
| `sessionLimitMicrodollars` | integer or null | Per-session spending cap. See [Session Limits](#session-limits). |

## Enforcement Lifecycle

The proxy checks budgets in this exact order. A denial at any step stops the pipeline — later steps are not evaluated.

### 1. Period Reset

If the budget has a `resetInterval` and the current period has elapsed, spend resets to 0 and a new period starts. A `budget.reset` webhook fires.

### 2. Session Limit Check

If the budget has a `sessionLimitMicrodollars` and the request includes an `X-NullSpend-Session` header, the proxy checks cumulative spend for that session. If `currentSessionSpend + estimatedCost > sessionLimit`, the request is denied.

### 3. Velocity Check (Circuit Breaker)

If the budget has a `velocityLimitMicrodollars`, the proxy checks spend within the sliding window. The velocity check uses a circuit breaker pattern:

- **Closed** (normal): requests pass through, velocity spend is tracked
- **Open** (tripped): all requests are denied until the cooldown expires
- **Recovery**: after cooldown, the breaker resets and a `velocity.recovered` webhook fires

If `estimatedSpend + estimate > velocityLimit`, the breaker trips.

### 4. Budget Exhaustion Check

If `currentSpend + reservations + estimatedCost > maxBudget`, the request is denied. Only budgets with `strict_block` policy deny requests (this is the default).

### 5. Reservation

The estimated cost is reserved for 30 seconds. Reservations prevent concurrent requests from collectively exceeding the budget. If a reservation expires (upstream timeout, crash), it is automatically cleaned up.

### 6. Reconciliation

After the provider responds, the proxy calculates the actual cost and reconciles:
- Adds actual cost to cumulative spend
- Removes the reservation
- Adjusts session spend by `actualCost - estimatedCost`

## 429 Response Bodies

### Budget Exceeded

```json
{
  "error": {
    "code": "budget_exceeded",
    "message": "Request blocked: estimated cost exceeds remaining budget",
    "details": null
  }
}
```

### Velocity Exceeded

```json
{
  "error": {
    "code": "velocity_exceeded",
    "message": "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.",
    "details": {
      "limitMicrodollars": 10000000,
      "windowSeconds": 60,
      "currentMicrodollars": 9500000
    }
  }
}
```

The response includes a `Retry-After` header with the cooldown duration in seconds.

### Session Limit Exceeded

```json
{
  "error": {
    "code": "session_limit_exceeded",
    "message": "Request blocked: session spend exceeds session limit. Start a new session.",
    "details": {
      "session_id": "conv_abc123",
      "session_spend_microdollars": 4800000,
      "session_limit_microdollars": 5000000
    }
  }
}
```

No `Retry-After` header — the session is done. Start a new session (new `X-NullSpend-Session` value) to continue.

### Tag Budget Exceeded

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

## Velocity Limits

Velocity limits catch runaway loops — an agent stuck in a retry cycle can burn through a budget in seconds.

**How it works:**

1. The proxy tracks spend within a sliding window (e.g., $10 in 60 seconds)
2. When spend exceeds the limit, a circuit breaker trips
3. All requests are blocked for the cooldown period
4. After cooldown, the breaker resets and requests resume
5. A `velocity.recovered` webhook fires on recovery

**Configuration:**

| Field | Range | Default | Description |
|---|---|---|---|
| `velocityLimitMicrodollars` | > 0 | null (disabled) | Max spend per window |
| `velocityWindowSeconds` | 10–3600 | 60 | Sliding window size |
| `velocityCooldownSeconds` | 10–3600 | 60 | Block duration after trip |

**Example:** $10 velocity limit with 60s window and 60s cooldown means: if your agents spend more than $10 within any 60-second sliding window, all requests are blocked for 60 seconds.

For the full reference — sliding window algorithm, circuit breaker states, and webhook payloads — see [Velocity Limits](velocity-limits.md).

## Session Limits

Session limits cap how much a single agent conversation can spend, regardless of the overall budget.

**How it works:**

1. Your agent sets `X-NullSpend-Session: conv_abc123` on each request
2. The proxy tracks cumulative spend per session ID
3. When a session's spend exceeds the limit, the request is blocked
4. The agent should start a new session (new ID) to continue

**Key behaviors:**

- **No header = no enforcement.** Session limits only apply when `X-NullSpend-Session` is present.
- **Client-defined sessions.** The proxy does not manage session lifecycle — your agent decides when to start a new session.
- **Independent of budget resets.** Session spend does NOT reset when the budget period resets.
- **Always strict.** Session limits are hard caps regardless of the budget policy.
- **24-hour cleanup.** Stale session data is automatically cleaned up after 24 hours of inactivity.

For the full reference — session tracking internals, header usage, and webhook payloads — see [Session Limits](session-limits.md).

## Threshold Alerts

When spend crosses a threshold percentage, a webhook fires:

- Thresholds **≥ 90%** fire as `budget.threshold.critical`
- Thresholds **< 90%** fire as `budget.threshold.warning`

Default thresholds are `[50, 80, 90, 95]`. Customize per budget with up to 10 values (ascending, each 1–100).

See [Webhook Event Types](../webhooks/event-types.md#budgetthresholdwarning) for payload details.

## Creating a Budget

### Dashboard

1. Go to **Budgets** → **Set Budget**
2. Choose entity (your account or a specific API key)
3. Set the spending ceiling
4. Optionally configure reset interval, velocity limits, session limits, and alert thresholds
5. Click **Set Budget** — takes effect immediately

### API

Budget creation and management uses session authentication (dashboard). See the [Budgets API](../api-reference/budgets-api.md) for full endpoint documentation.

```bash
# Requires dashboard session cookie
curl -X POST "https://nullspend.dev/api/budgets" \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{
    "entityType": "api_key",
    "entityId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "maxBudgetMicrodollars": 50000000,
    "resetInterval": "monthly",
    "velocityLimitMicrodollars": 10000000,
    "velocityWindowSeconds": 60,
    "velocityCooldownSeconds": 60,
    "sessionLimitMicrodollars": 5000000
  }'
```

To check budget status programmatically (with an API key), use [`GET /api/budgets/status`](../api-reference/budgets-api.md#get-budget-status).

## Best Practices

- **Start generous, tighten later.** Set initial budgets higher than expected. Once you have cost data, tighten with confidence.
- **One budget per concern.** Separate API keys (and budgets) for different agents, environments, or teams.
- **Use session limits for multi-step agents.** Cap each task's cost so a single stuck agent can't consume the entire budget.
- **Monitor before enforcing.** Use the analytics dashboard to understand spending patterns before setting tight ceilings.
- **Combine velocity + session limits.** Velocity catches sudden spikes; session limits catch slow accumulation over a long conversation.

## Related

- [Budget Configuration Guide](../guides/budget-configuration.md) — step-by-step setup walkthrough
- [Cost Tracking](cost-tracking.md) — how costs are calculated and recorded
- [Tags](tags.md) — tag-based cost attribution and tag budgets
- [Webhook Event Types](../webhooks/event-types.md) — budget.exceeded, velocity.exceeded, threshold alerts
- [Error Reference](../api-reference/errors.md) — all 429 error codes and response shapes
