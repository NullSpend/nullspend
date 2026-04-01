---
title: "Velocity Limits"
description: "Velocity limits catch runaway loops — an agent stuck in a retry cycle can burn through a budget in seconds before a human can react. They add a spending-rate "
---

Velocity limits catch runaway loops — an agent stuck in a retry cycle can burn through a budget in seconds before a human can react. They add a spending-rate constraint on top of the budget ceiling.

See [Budgets](budgets.md) for overall budget configuration.

## How It Works

```
Request arrives
  │
  ▼
Circuit breaker tripped?
  ├─ YES → 429 + Retry-After header
  │
  ▼
Estimate spend in sliding window
  │
  ▼
Under velocity limit?
  ├─ YES → Continue to budget check
  │
  ▼
Trip circuit breaker → 429 + Retry-After
  │
  ... cooldown expires ...
  │
  ▼
Reset counters → velocity.recovered webhook
  │
  ▼
First post-recovery request always passes
```

## Configuration

Set these fields when creating or updating a budget via the API:

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `velocityLimitMicrodollars` | integer or null | > 0 | null (disabled) | Max spend per window |
| `velocityWindowSeconds` | integer | 10–3600 | 60 | Sliding window size |
| `velocityCooldownSeconds` | integer | 10–3600 | 60 | Block duration after trip |

Setting `velocityLimitMicrodollars` to `null` disables velocity enforcement for that budget entity.

## Sliding Window Algorithm

The proxy uses a two-window counter to smooth spend estimation:

1. **Current window** tracks active spend since the last window boundary
2. **Previous window** fades out via linear decay as the current window progresses

**Formula:**

```
weight = max(0, (windowMs − elapsed) / windowMs)
estimatedSpend = prevSpend × weight + currSpend
```

**Window rotation:**

- When `now ≥ windowStart + windowMs`, shift: `prev ← curr`, reset `curr` to 0
- If more than one full window has elapsed since the last rotation, **both** prev and curr reset to 0 (the gap was too large for prev to be meaningful)

**Trip condition:**

```
estimatedSpend + requestEstimate > velocityLimit
```

## Circuit Breaker

The circuit breaker has three states:

**Closed** (normal operation)
- `tripped_at` is null
- Every request runs the sliding window check
- If the check fails, the breaker transitions to Open

**Open** (tripped)
- `tripped_at` is set and cooldown has not elapsed
- All requests are fast-denied with `429` and a `Retry-After` header
- No sliding window computation — instant rejection

**Recovery** (cooldown expired)
- `tripped_at` is set but cooldown has elapsed
- All counters are reset to zero
- A `velocity.recovered` webhook fires
- The first post-recovery request **always passes** (fresh window)

## 429 Response

When a request is denied by velocity enforcement, the proxy returns a `429` with the provider-appropriate error shape.

Both OpenAI and Anthropic routes return the same NullSpend error shape:

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

**Headers:**

| Header | Value |
|---|---|
| `Retry-After` | Seconds remaining in cooldown |
| `X-NullSpend-Trace-Id` | Trace ID for this request |

## Webhooks

Two webhook events are associated with velocity limits:

### `velocity.exceeded`

Fires when the circuit breaker trips. Key fields in `data.object`:

| Field | Description |
|---|---|
| `budget_entity_type` | `user` or `api_key` |
| `budget_entity_id` | The entity that tripped |
| `velocity_limit_microdollars` | Configured limit |
| `velocity_window_seconds` | Configured window |
| `velocity_current_microdollars` | Estimated spend at trip time |
| `cooldown_seconds` | How long the breaker stays open |
| `model` | Model of the request that tripped it |
| `provider` | `openai` or `anthropic` |
| `blocked_at` | ISO 8601 timestamp |

### `velocity.recovered`

Fires when the cooldown expires and the breaker resets. Key fields in `data.object`:

| Field | Description |
|---|---|
| `budget_entity_type` | `user` or `api_key` |
| `budget_entity_id` | The entity that recovered |
| `velocity_limit_microdollars` | Configured limit |
| `velocity_window_seconds` | Configured window |
| `velocity_cooldown_seconds` | Configured cooldown |
| `recovered_at` | ISO 8601 timestamp |

See [Event Types](../webhooks/event-types.md) for full JSON examples of both events.

## Enforcement Order

When a request arrives, the proxy checks limits in this order:

1. **Period reset** — if the budget period has elapsed, reset spend before any checks run
2. **Session limit** — checked before velocity to avoid affecting velocity counters on denied requests
3. **Velocity limit** — sliding window + circuit breaker
4. **Budget exhaustion** — is there enough budget remaining?
5. **Reservation** — reserve estimated cost for the request

Velocity increments are **deferred** until after the budget check passes. This prevents budget-denied requests from inflating velocity counters.

## Example

**Scenario:** $10 velocity limit, 60-second window, 60-second cooldown.

1. Agent starts a batch job, making rapid API calls
2. After 45 seconds, cumulative spend in the window hits $10.50
3. The next request's estimate would push the window past $10 → **circuit breaker trips**
4. `velocity.exceeded` webhook fires with `cooldown_seconds: 60`
5. All requests for the next 60 seconds get `429` with `Retry-After: 60`
6. At second 105 (45 + 60), cooldown expires
7. Counters reset to zero, `velocity.recovered` webhook fires
8. The next request passes — the agent resumes normally

## Related

- [Budgets](budgets.md) — overall budget configuration and enforcement
- [Session Limits](session-limits.md) — per-conversation spend caps
- [Event Types](../webhooks/event-types.md) — full webhook payload examples
- [Errors](../api-reference/errors.md) — all error codes and response shapes
