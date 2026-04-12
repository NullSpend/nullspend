# Webhook Event Types

NullSpend emits 18 event types. Each event is delivered as an HTTP POST with a JSON body.

## Event Envelope

### Full Event

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": { }
  }
}
```

### Thin Event

Used for `cost_event.created` on endpoints with `payloadMode: "thin"`. All other event types always use the full envelope.

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "related_object": {
    "id": "req_xyz",
    "type": "cost_event",
    "url": "/api/cost-events?requestId=req_xyz&provider=openai"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique event ID (`evt_` + UUID). Use for deduplication. |
| `type` | string | One of the 18 event types below. |
| `api_version` | string | API version (`"2026-04-01"`). |
| `created_at` | integer | Unix timestamp in seconds. |
| `data.object` | object | Event-specific payload (full mode). |
| `related_object` | object | Reference to fetchable object (thin mode). |

---

## Cost Events

### `cost_event.created`

Fires when a cost event is recorded — once per proxied request.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Unique request identifier |
| `event_type` | string | Request type: `"llm"` (LLM API call), `"tool"` (MCP tool invocation), or `"custom"` (SDK-reported) |
| `provider` | string | `"openai"` or `"anthropic"` |
| `model` | string | Model name (e.g., `gpt-4o`) |
| `input_tokens` | integer | Total input tokens |
| `output_tokens` | integer | Output tokens |
| `cached_input_tokens` | integer | Cached input tokens |
| `cost_microdollars` | integer | Total cost in microdollars |
| `duration_ms` | integer | Request duration in milliseconds |
| `upstream_duration_ms` | integer or null | Time spent waiting for the LLM provider |
| `session_id` | string or null | Session ID if set |
| `trace_id` | string or null | Trace ID |
| `tool_name` | string or null | MCP tool name |
| `tool_server` | string or null | MCP tool server |
| `tool_calls_requested` | array or null | Array of `{name, id}` objects representing tool calls, or `null` |
| `tool_definition_tokens` | integer | Token count for tool definitions (defaults to 0) |
| `api_key_id` | string | API key that made the request |
| `source` | string | `"proxy"`, `"api"`, or `"mcp"` |
| `tags` | object | Key-value pairs from `X-NullSpend-Tags` |
| `created_at` | string | ISO 8601 timestamp |

**Example:**

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "request_id": "chatcmpl-abc123",
      "event_type": "llm",
      "provider": "openai",
      "model": "gpt-4o",
      "input_tokens": 1000,
      "output_tokens": 500,
      "cached_input_tokens": 200,
      "cost_microdollars": 7,
      "duration_ms": 1234,
      "upstream_duration_ms": 1180,
      "session_id": null,
      "trace_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
      "tool_name": null,
      "tool_server": null,
      "tool_calls_requested": null,
      "tool_definition_tokens": 0,
      "api_key_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "source": "proxy",
      "tags": { "team": "billing", "env": "production" },
      "created_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

---

## Budget Events

### `budget.threshold.warning`

Fires when spend crosses a threshold percentage **below 90%** (e.g., 50%, 80%).

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | `"user"`, `"api_key"`, or `"tag"` |
| `budget_entity_id` | string | Entity identifier |
| `threshold_percent` | integer | Threshold crossed (e.g., 80) |
| `budget_spend_microdollars` | integer | Current spend |
| `budget_limit_microdollars` | integer | Budget ceiling |
| `budget_remaining_microdollars` | integer | Remaining budget (limit minus spend) |
| `triggered_by_request_id` | string | Request that triggered the crossing |

**Example:**

```json
{
  "id": "evt_b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "type": "budget.threshold.warning",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "api_key",
      "budget_entity_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "threshold_percent": 80,
      "budget_spend_microdollars": 40500000,
      "budget_limit_microdollars": 50000000,
      "budget_remaining_microdollars": 9500000,
      "triggered_by_request_id": "chatcmpl-abc123"
    }
  }
}
```

### `budget.threshold.critical`

Same structure as `budget.threshold.warning`. Fires when spend crosses a threshold **≥ 90%**.

```json
{
  "id": "evt_c3d4e5f6-a7b8-9012-cdef-123456789012",
  "type": "budget.threshold.critical",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "user",
      "budget_entity_id": "user_12345",
      "threshold_percent": 95,
      "budget_spend_microdollars": 47800000,
      "budget_limit_microdollars": 50000000,
      "budget_remaining_microdollars": 2200000,
      "triggered_by_request_id": "chatcmpl-def456"
    }
  }
}
```

### `budget.exceeded`

Fires when a request is blocked because the budget ceiling was hit.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `budget_limit_microdollars` | integer | Budget ceiling |
| `budget_spend_microdollars` | integer | Current spend |
| `estimated_request_cost_microdollars` | integer | Estimated cost of the blocked request |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `blocked_at` | string | ISO 8601 timestamp when blocked |

**Example:**

```json
{
  "id": "evt_d4e5f6a7-b8c9-0123-defa-234567890123",
  "type": "budget.exceeded",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "api_key",
      "budget_entity_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "budget_limit_microdollars": 50000000,
      "budget_spend_microdollars": 49800000,
      "estimated_request_cost_microdollars": 500000,
      "model": "gpt-4o",
      "provider": "openai",
      "blocked_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

### `budget.increased`

Fires when a budget limit is increased via a HITL budget increase approval.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `previous_limit_microdollars` | integer | Budget limit before the increase |
| `new_limit_microdollars` | integer | Budget limit after the increase |
| `increased_by_microdollars` | integer | Amount of the increase |
| `approved_by` | string | User who approved the increase |
| `action_id` | string | HITL action ID that triggered the increase |

**Example:**

```json
{
  "id": "evt_d4e5f6a7-b8c9-0123-defa-234567890124",
  "type": "budget.increased",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "user",
      "budget_entity_id": "user_12345",
      "previous_limit_microdollars": 50000000,
      "new_limit_microdollars": 100000000,
      "increased_by_microdollars": 50000000,
      "approved_by": "admin_user",
      "action_id": "ns_act_550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

### `budget.reset`

Fires when a budget period resets (daily, weekly, or monthly).

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `budget_limit_microdollars` | integer | Budget ceiling |
| `previous_spend_microdollars` | integer | Spend in the period that just ended |
| `new_period_start` | string | ISO 8601 timestamp of the new period |
| `reset_interval` | string | `"daily"`, `"weekly"`, or `"monthly"` |

**Example:**

```json
{
  "id": "evt_e5f6a7b8-c9d0-1234-efab-345678901234",
  "type": "budget.reset",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "user",
      "budget_entity_id": "user_12345",
      "budget_limit_microdollars": 50000000,
      "previous_spend_microdollars": 42000000,
      "new_period_start": "2026-04-01T00:00:00.000Z",
      "reset_interval": "monthly"
    }
  }
}
```

---

## Enforcement Events

### `request.blocked`

Fires when a request is blocked for any reason.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `reason` | string | `"budget"`, `"rate_limit"`, or `"policy"` |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `api_key_id` | string | API key |
| `details` | object or null | Additional context (varies by reason) |

**Example:**

```json
{
  "id": "evt_f6a7b8c9-d0e1-2345-fabc-456789012345",
  "type": "request.blocked",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "reason": "budget",
      "model": "gpt-4o",
      "provider": "openai",
      "api_key_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "details": null
    }
  }
}
```

### `velocity.exceeded`

Fires when a velocity limit trips the circuit breaker.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `velocity_limit_microdollars` | integer | Configured velocity limit |
| `velocity_window_seconds` | integer | Sliding window size |
| `velocity_current_microdollars` | integer | Spend in the current window |
| `cooldown_seconds` | integer | How long requests will be blocked |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `blocked_at` | string | ISO 8601 timestamp when blocked |

**Example:**

```json
{
  "id": "evt_a7b8c9d0-e1f2-3456-abcd-567890123456",
  "type": "velocity.exceeded",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "api_key",
      "budget_entity_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "velocity_limit_microdollars": 10000000,
      "velocity_window_seconds": 60,
      "velocity_current_microdollars": 10500000,
      "cooldown_seconds": 60,
      "model": "gpt-4o",
      "provider": "openai",
      "blocked_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

### `velocity.recovered`

Fires when the velocity circuit breaker closes after cooldown.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `velocity_limit_microdollars` | integer | Configured velocity limit |
| `velocity_window_seconds` | integer | Sliding window size |
| `velocity_cooldown_seconds` | integer | Cooldown duration that just ended |
| `recovered_at` | string | ISO 8601 timestamp when recovered |

**Example:**

```json
{
  "id": "evt_b8c9d0e1-f2a3-4567-bcde-678901234567",
  "type": "velocity.recovered",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "api_key",
      "budget_entity_id": "ns_key_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "velocity_limit_microdollars": 10000000,
      "velocity_window_seconds": 60,
      "velocity_cooldown_seconds": 60,
      "recovered_at": "2026-03-21T12:01:00.000Z"
    }
  }
}
```

### `session.limit_exceeded`

Fires when a session's cumulative spend exceeds the session limit.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | Entity type |
| `budget_entity_id` | string | Entity identifier |
| `session_id` | string | The session that was capped |
| `session_spend_microdollars` | integer | Cumulative session spend |
| `session_limit_microdollars` | integer | Configured session limit |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `blocked_at` | string | ISO 8601 timestamp when blocked |

**Example:**

```json
{
  "id": "evt_c9d0e1f2-a3b4-5678-cdef-789012345678",
  "type": "session.limit_exceeded",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "user",
      "budget_entity_id": "user_12345",
      "session_id": "conv_abc123",
      "session_spend_microdollars": 4800000,
      "session_limit_microdollars": 5000000,
      "model": "gpt-4o",
      "provider": "openai",
      "blocked_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

### `tag_budget.exceeded`

Fires when a tag-level budget is exceeded.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | `"tag"` |
| `budget_entity_id` | string | Tag entity ID (`key=value`) |
| `tag_key` | string | Tag key |
| `tag_value` | string | Tag value |
| `budget_limit_microdollars` | integer | Tag budget ceiling |
| `budget_spend_microdollars` | integer | Current tag spend |
| `estimated_request_cost_microdollars` | integer | Estimated cost of the blocked request |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `blocked_at` | string | ISO 8601 timestamp when blocked |

**Example:**

```json
{
  "id": "evt_d0e1f2a3-b4c5-6789-defa-890123456789",
  "type": "tag_budget.exceeded",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "tag",
      "budget_entity_id": "team=billing",
      "tag_key": "team",
      "tag_value": "billing",
      "budget_limit_microdollars": 50000000,
      "budget_spend_microdollars": 49500000,
      "estimated_request_cost_microdollars": 500000,
      "model": "gpt-4o",
      "provider": "openai",
      "blocked_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

---

## Margin Events

### `margin.threshold_crossed`

Fires when a customer's margin crosses into a **worse** health tier (e.g., moderate to at-risk). Improving margins do not trigger this event.

Dispatched from the dashboard during revenue sync, not from the proxy.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `customer.stripeId` | string | Stripe customer ID |
| `customer.name` | string or null | Customer display name |
| `customer.tagValue` | string | Cost event tag value |
| `margin.previous` | number | Previous margin as a decimal (e.g., 0.25 = 25%) |
| `margin.current` | number | Current margin as a decimal |
| `margin.previousTier` | string | `"healthy"`, `"moderate"`, `"at_risk"`, or `"critical"` |
| `margin.currentTier` | string | New (worse) tier |
| `revenue_microdollars` | integer | Current period revenue |
| `cost_microdollars` | integer | Current period AI cost |
| `period` | string | Calendar month (`YYYY-MM`) |

**Example:**

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "margin.threshold_crossed",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "customer": {
        "stripeId": "cus_abc123",
        "name": "Acme Corp",
        "tagValue": "acme-corp"
      },
      "margin": {
        "previous": 0.25,
        "current": -0.05,
        "previousTier": "moderate",
        "currentTier": "critical"
      },
      "revenue_microdollars": 50000000,
      "cost_microdollars": 52500000,
      "period": "2026-04"
    }
  }
}
```

### `customer_budget.exceeded`

Fires when a customer-scoped budget is exceeded. Similar to `budget.exceeded` but specific to per-customer budgets identified by the `X-NullSpend-Customer` header.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `budget_entity_type` | string | `"customer"` |
| `budget_entity_id` | string | Customer identifier |
| `customer_id` | string | Customer identifier (same as `budget_entity_id`) |
| `budget_limit_microdollars` | integer | Customer budget ceiling |
| `budget_spend_microdollars` | integer | Current customer spend |
| `estimated_request_cost_microdollars` | integer | Estimated cost of the blocked request |
| `model` | string | Requested model |
| `provider` | string | Provider name |
| `blocked_at` | string | ISO 8601 timestamp when blocked |

**Example:**

```json
{
  "id": "evt_e1f2a3b4-c5d6-7890-efab-901234567891",
  "type": "customer_budget.exceeded",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "budget_entity_type": "customer",
      "budget_entity_id": "acme-corp",
      "customer_id": "acme-corp",
      "budget_limit_microdollars": 25000000,
      "budget_spend_microdollars": 24800000,
      "estimated_request_cost_microdollars": 500000,
      "model": "gpt-4o",
      "provider": "openai",
      "blocked_at": "2026-03-21T12:00:00.000Z"
    }
  }
}
```

---

## HITL Action Events

See [Human-in-the-Loop](../features/human-in-the-loop.md) for the full approval workflow, state machine, and SDK integration.

### `action.created`

Fires when a human-in-the-loop approval action is created.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `action_id` | string | Action identifier (`ns_act_` + UUID) |
| `action_type` | string | Type of action |
| `agent_id` | string | Agent that created the action |
| `status` | string | `"pending"` |
| `payload` | object | Action payload (the data submitted for approval) |
| `created_at` | string | ISO 8601 timestamp |
| `expires_at` | string or null | When the action expires if not acted on |

**Example:**

```json
{
  "id": "evt_e1f2a3b4-c5d6-7890-efab-901234567890",
  "type": "action.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "action_id": "ns_act_550e8400-e29b-41d4-a716-446655440000",
      "action_type": "http_post",
      "agent_id": "my-agent",
      "status": "pending",
      "payload": { "amount": 500, "description": "Large purchase" },
      "created_at": "2026-03-21T12:00:00.000Z",
      "expires_at": "2026-03-21T13:00:00.000Z"
    }
  }
}
```

### `action.approved`

Fires when an action is approved.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `action_id` | string | Action identifier |
| `action_type` | string | Type of action |
| `agent_id` | string | Agent that created the action |
| `status` | string | `"approved"` |
| `approved_by` | string or null | User who approved |
| `approved_at` | string or null | ISO 8601 timestamp of approval |

### `action.rejected`

Fires when an action is rejected.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `action_id` | string | Action identifier |
| `action_type` | string | Type of action |
| `agent_id` | string | Agent that created the action |
| `status` | string | `"rejected"` |
| `rejected_by` | string or null | User who rejected |
| `rejected_at` | string or null | ISO 8601 timestamp of rejection |
| `reason` | string or null | Rejection reason |

### `action.expired`

Fires when an action's TTL expires.

**`data.object` fields:**

| Field | Type | Description |
|---|---|---|
| `action_id` | string | Action identifier |
| `action_type` | string | Type of action |
| `agent_id` | string | Agent that created the action |
| `status` | string | `"expired"` |
| `expired_at` | string or null | ISO 8601 timestamp of expiry |

---

## Test Events

### `test.ping`

Sent when you click "Test" in the dashboard. Use it to verify your endpoint is reachable and signature verification works.

**Example:**

```json
{
  "id": "evt_f2a3b4c5-d6e7-8901-fabc-012345678901",
  "type": "test.ping",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "message": "This is a test webhook event from NullSpend."
    }
  }
}
```

---

## Related

- [Webhooks Overview](overview.md) — setup, payload modes, transport
- [Webhook Security](security.md) — HMAC signature verification
- [Budgets](../features/budgets.md) — budget enforcement that triggers these events
- [Velocity Limits](../features/velocity-limits.md) — sliding window algorithm and circuit breaker
- [Session Limits](../features/session-limits.md) — per-conversation spend caps
- [Tags](../features/tags.md) — tags included in cost event payloads
- [Margins](../features/margins.md) — customer profitability tracking that triggers margin events
