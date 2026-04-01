# Python SDK

Python client for the NullSpend API.

## Installation

```bash
pip install nullspend
```

## Quick Start

```python
from nullspend import NullSpend, CostEventInput

ns = NullSpend(
    base_url="https://nullspend.com",
    api_key="ns_live_sk_...",
)

# Report a cost event
ns.report_cost(CostEventInput(
    provider="openai",
    model="gpt-4o",
    input_tokens=500,
    output_tokens=150,
    cost_microdollars=4625,
))
```

## Configuration

The `NullSpend` constructor accepts keyword arguments or a `NullSpendConfig` dataclass:

| Option | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | **required** | NullSpend dashboard URL (e.g. `https://nullspend.com`) |
| `api_key` | `str` | **required** | API key (`ns_live_sk_...`) |
| `api_version` | `str` | `"2026-04-01"` | API version sent via `NullSpend-Version` header |
| `request_timeout_s` | `float` | `30.0` | Per-request timeout in seconds |
| `max_retries` | `int` | `2` | Max retries on transient failures. Clamped to `[0, 10]` |
| `retry_base_delay_s` | `float` | `0.5` | Base delay between retries in seconds |

```python
# Using keyword arguments
ns = NullSpend(base_url="https://nullspend.com", api_key="ns_live_sk_...")

# Using config dataclass
from nullspend import NullSpendConfig

config = NullSpendConfig(
    base_url="https://nullspend.com",
    api_key="ns_live_sk_...",
    max_retries=3,
    request_timeout_s=60.0,
)
ns = NullSpend(config=config)
```

The client supports context manager usage for automatic cleanup:

```python
with NullSpend(base_url="...", api_key="...") as ns:
    ns.report_cost(...)
# HTTP client is automatically closed
```

## Actions (Human-in-the-Loop)

The SDK provides methods for the full [HITL approval workflow](../features/human-in-the-loop.md).

### `create_action(input)`

Create a new action for human approval.

```python
from nullspend import CreateActionInput

response = ns.create_action(CreateActionInput(
    agent_id="support-agent",
    action_type="send_email",
    payload={"to": "user@example.com", "subject": "Refund"},
    metadata={"ticket_id": "T-1234"},
    expires_in_seconds=1800,
))
print(response.id, response.status)  # "ns_act_..." "pending"
```

### `get_action(id)`

Fetch the current state of an action.

```python
action = ns.get_action("ns_act_550e8400-...")
print(action.status)  # "pending" | "approved" | "rejected" | ...
```

### `mark_result(id, input)`

Report execution status back to NullSpend.

```python
from nullspend import MarkResultInput

# Start executing
ns.mark_result(action_id, MarkResultInput(status="executing"))

# Report success
ns.mark_result(action_id, MarkResultInput(
    status="executed",
    result={"rows_deleted": 42},
))

# Or report failure
ns.mark_result(action_id, MarkResultInput(
    status="failed",
    error_message="Connection timeout",
))
```

### `wait_for_decision(id, **options)`

Poll until the action leaves `pending` status or the timeout elapses.

```python
decision = ns.wait_for_decision(
    action_id,
    poll_interval_s=2.0,   # default: 2.0
    timeout_s=300.0,        # default: 300.0 (5 min)
    on_poll=lambda action: print(action.status),
)
```

Throws `PollTimeoutError` if the timeout elapses while still `pending`.

### `propose_and_wait(options)`

High-level orchestrator that combines create, poll, execute, and report:

```python
from nullspend import ProposeAndWaitOptions

def execute(context):
    # Runs only after human approval.
    # context["action_id"] can be sent as X-NullSpend-Action-Id to correlate costs.
    return delete_old_logs()

result = ns.propose_and_wait(ProposeAndWaitOptions(
    agent_id="data-agent",
    action_type="db_write",
    payload={"query": "DELETE FROM logs WHERE age > 90"},
    execute=execute,
    expires_in_seconds=3600,
    poll_interval_s=2.0,   # default: 2.0
    timeout_s=300.0,        # default: 300.0 (5 min)
))
```

- On approval: marks `executing`, calls `execute(context)`, marks `executed` with result
- On rejection/expiry: raises `RejectedError`
- On execute failure: marks `failed`, re-raises the original error
- Handles `409` conflicts from concurrent writes gracefully

## Cost Reporting

### `report_cost(event)` — Single Event

```python
from nullspend import CostEventInput

result = ns.report_cost(CostEventInput(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    input_tokens=1000,
    output_tokens=500,
    cost_microdollars=6750,
    # Optional fields:
    cached_input_tokens=200,
    reasoning_tokens=0,
    duration_ms=1200,
    session_id="session-123",
    trace_id="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    event_type="llm",        # "llm" | "tool" | "custom"
    tool_name="search",
    tool_server="rag-server",
    tags={"team": "backend"},
))
```

### `report_cost_batch(events)` — Batch

```python
result = ns.report_cost_batch([
    CostEventInput(provider="openai", model="gpt-4o", input_tokens=500, output_tokens=150, cost_microdollars=4625),
    CostEventInput(provider="openai", model="gpt-4o-mini", input_tokens=1000, output_tokens=300, cost_microdollars=225),
])
print(result["inserted"])  # 2
```

## Budget Status

```python
status = ns.check_budget()

for entity in status.entities:
    print(
        f"{entity.entity_type}/{entity.entity_id}: "
        f"${entity.spend_microdollars / 1_000_000:.2f} / ${entity.limit_microdollars / 1_000_000:.2f}"
    )
```

### `list_budgets()`

Fetch all budgets for the authenticated org.

```python
result = ns.list_budgets()

for budget in result.data:
    spent = budget.spend_microdollars / 1_000_000
    limit = budget.max_budget_microdollars / 1_000_000
    print(f"{budget.entity_type}/{budget.entity_id}: ${spent:.2f} / ${limit:.2f}")
```

## Cost Awareness (Read APIs)

Query your spend data programmatically.

### `get_cost_summary(period?)`

Get aggregated spend data for a time period.

```python
summary = ns.get_cost_summary("30d")  # "7d" | "30d" | "90d"

print(f"Total spend: ${summary.totals['totalCostMicrodollars'] / 1_000_000:.2f}")
print(f"Total requests: {summary.totals['totalRequests']}")
```

### `list_cost_events(options?)`

Fetch recent cost events with pagination.

```python
from nullspend import ListCostEventsOptions

# Get the last 10 cost events
result = ns.list_cost_events(ListCostEventsOptions(limit=10))

for event in result.data:
    print(f"{event.model}: {event.input_tokens} in / {event.output_tokens} out — ${event.cost_microdollars / 1_000_000:.4f}")

# Paginate with cursor
if result.cursor:
    import json
    next_page = ns.list_cost_events(ListCostEventsOptions(limit=10, cursor=json.dumps(result.cursor)))
```

## Retry Behavior

The SDK automatically retries on transient failures:

**Retryable:** `429`, `500`, `502`, `503`, `504`, network errors (`httpx.TransportError`)

**Not retryable:** `4xx` errors other than `429`

**Backoff:** Full-jitter exponential — `max(0.001, random() * min(base * 2^attempt, 5s))`

**Idempotency:** Mutating requests (`POST`) include an `Idempotency-Key` header generated once and reused across retries.

## Error Handling

Three error classes, all extending `Exception`:

### `NullSpendError`

Base error for all SDK errors. Properties:

| Property | Type | Description |
|---|---|---|
| `status_code` | `int \| None` | HTTP status code (if from an API response) |
| `code` | `str \| None` | Machine-readable error code from the API |

```python
from nullspend import NullSpendError

try:
    ns.create_action(...)
except NullSpendError as err:
    print(err.status_code)  # 409
    print(err.code)          # "invalid_action_transition"
```

### `PollTimeoutError`

Thrown by `wait_for_decision` when the timeout elapses. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `action_id` | `str` | The action that timed out |
| `timeout_ms` | `int` | The timeout in milliseconds |

### `RejectedError`

Thrown by `propose_and_wait` when the action is rejected or expired. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `action_id` | `str` | The action that was rejected |
| `action_status` | `str` | The terminal status (`"rejected"` or `"expired"`) |

```python
from nullspend import RejectedError

try:
    ns.propose_and_wait(...)
except RejectedError as err:
    print(f"{err.action_id} was {err.action_status}")
```

## Types

All types are dataclasses exported from the package:

```python
from nullspend import (
    # Configuration
    NullSpendConfig,

    # Actions
    CreateActionInput,
    CreateActionResponse,
    ActionRecord,
    MarkResultInput,
    ProposeAndWaitOptions,

    # Cost reporting
    CostEventInput,
    CostEventRecord,

    # Budgets
    BudgetStatus,
    BudgetEntity,
    BudgetRecord,
    ListBudgetsResponse,

    # Cost awareness (read)
    ListCostEventsResponse,
    ListCostEventsOptions,
    CostSummaryResponse,

    # Errors
    NullSpendError,
    PollTimeoutError,
    RejectedError,
)
```

**Constants:**

```python
from nullspend.types import (
    ACTION_TYPES,        # tuple of valid action types
    TERMINAL_STATUSES,   # frozenset of terminal statuses
)
```

## Differences from the JavaScript SDK

| Feature | JavaScript SDK | Python SDK |
|---|---|---|
| HTTP client | `fetch` (configurable) | `httpx` (synchronous) |
| Async support | Native (`async/await`) | Synchronous only (v0.1.0) |
| Client-side batching | `queueCost()` / `flush()` / `shutdown()` | Not yet available |
| `onRetry` callback | Supported | Not yet available |
| Wall-time retry cap | `maxRetryTimeMs` | Not yet available |
| Timeout error class | `TimeoutError` | `PollTimeoutError` (avoids shadowing Python builtin) |

## Related

- [Human-in-the-Loop](../features/human-in-the-loop.md) — approval workflow concepts and best practices
- [Cost Tracking](../features/cost-tracking.md) — how cost events are recorded
- [Actions API](../api-reference/actions-api.md) — raw HTTP endpoint reference
- [Budgets API](../api-reference/budgets-api.md) — budget management endpoints
- [JavaScript SDK](javascript.md) — TypeScript/JavaScript client
- [Claude Agent Adapter](claude-agent.md) — adapter for the Claude Agent SDK
