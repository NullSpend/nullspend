# Python SDK

Python client for the NullSpend API — cost tracking, budget enforcement, and human-in-the-loop approval for AI agents.

## Installation

```bash
pip install nullspend
```

Requires Python 3.9+. The only runtime dependency is [httpx](https://www.python-httpx.org/).

## Quick Start

```python
from nullspend import NullSpend

# Reads NULLSPEND_API_KEY from environment
ns = NullSpend()

# Or provide explicitly
ns = NullSpend(api_key="ns_live_sk_...")
```

### Tracked Client (recommended)

Wrap your OpenAI or Anthropic SDK to automatically track costs — no manual `report_cost` calls needed:

```python
from openai import OpenAI
from nullspend import NullSpend

ns = NullSpend(
    api_key="ns_live_sk_...",
    cost_reporting={}  # enable batching
)

# Wrap OpenAI — costs are tracked automatically
openai = OpenAI(http_client=ns.openai)

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
# Cost event is calculated locally and reported in the background
```

### Manual Reporting

```python
from nullspend import NullSpend, CostEventInput

ns = NullSpend(api_key="ns_live_sk_...")

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
| `api_key` | `str` | `NULLSPEND_API_KEY` env var | API key (`ns_live_sk_...`) |
| `base_url` | `str` | `https://nullspend.dev` | NullSpend dashboard URL |
| `proxy_url` | `str` | `https://proxy.nullspend.dev` | NullSpend proxy URL (for proxy mode detection) |
| `api_version` | `str` | `"2026-04-01"` | API version sent via `NullSpend-Version` header |
| `request_timeout_s` | `float` | `30.0` | Per-request timeout in seconds |
| `max_retries` | `int` | `2` | Max retries on transient failures. Clamped to `[0, 10]` |
| `retry_base_delay_s` | `float` | `0.5` | Base delay between retries in seconds |
| `cost_reporting` | `CostReportingConfig` | — | Enable client-side cost event batching (see below) |

```python
# Minimal — reads API key from NULLSPEND_API_KEY env var
ns = NullSpend()

# Explicit keyword arguments
ns = NullSpend(
    api_key="ns_live_sk_...",
    max_retries=3,
    request_timeout_s=60.0,
)

# Using config dataclass
from nullspend import NullSpendConfig

config = NullSpendConfig(
    api_key="ns_live_sk_...",
    max_retries=3,
)
ns = NullSpend(config=config)
```

The client supports context manager usage for automatic cleanup:

```python
with NullSpend(api_key="...") as ns:
    ns.report_cost(...)
# HTTP client is automatically closed, cost reporter is flushed
```

## Tracked Client (Provider Wrappers)

Wrap your LLM provider's HTTP client to automatically track costs and enforce policies client-side.

### Basic Setup

```python
from openai import OpenAI
from anthropic import Anthropic
from nullspend import NullSpend

ns = NullSpend(
    api_key="ns_live_sk_...",
    cost_reporting={},  # required for tracked clients
)

# Shorthand properties — pre-configured httpx.Client for each provider
openai = OpenAI(http_client=ns.openai)
anthropic = Anthropic(http_client=ns.anthropic)
```

Cost events are calculated locally using the built-in pricing engine (56 models) and reported asynchronously in batches. Your requests go directly to the provider — no proxy required.

### `create_tracked_client(provider, **options)`

For full control over tracked client options:

```python
tracked = ns.create_tracked_client(
    "openai",
    customer="acme-corp",
    session_id="task-042",
    tags={"team": "backend"},
    enforcement=True,
    session_limit_microdollars=5_000_000,  # $5 per session
)

openai = OpenAI(http_client=tracked)
```

| Option | Type | Default | Description |
|---|---|---|---|
| `customer` | `str` | — | Customer ID for per-customer cost attribution |
| `enforcement` | `bool` | `False` | Enable budget, mandate, and session limit checks |
| `session_id` | `str` | — | Session identifier for cost correlation and session limits |
| `session_limit_microdollars` | `int` | — | Manual per-session spend cap |
| `tags` | `dict[str, str]` | — | Tags attached to every cost event |
| `trace_id` | `str` | — | Distributed trace ID |
| `action_id` | `str` | — | HITL action ID for cost correlation |
| `on_denied` | `Callable` | — | Called before raising enforcement errors |
| `on_cost_error` | `Callable` | — | Called on non-fatal cost tracking errors |

### Enforcement Mode

Enable `enforcement=True` to check budgets, model mandates, and session limits before each request:

```python
tracked = ns.create_tracked_client("openai", enforcement=True)

openai = OpenAI(http_client=tracked)

try:
    openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
except BudgetExceededError as e:
    print(f"Budget: ${e.remaining_microdollars / 1_000_000:.2f} remaining")
except MandateViolationError as e:
    print(f"Mandate: {e.mandate} blocks {e.requested}")
except SessionLimitExceededError as e:
    print(f"Session: ${e.session_spend_microdollars / 1_000_000:.2f} of ${e.session_limit_microdollars / 1_000_000:.2f}")
```

When `enforcement=True`, each request goes through:

1. **Mandate check** — is this model/provider allowed by key policy?
2. **Budget check** — does estimated cost fit within remaining budget?
3. **Session limit check** — does `session_spend + estimate` exceed the session limit?

If any check fails, the SDK raises the corresponding error **before** calling the provider. If the policy endpoint is unreachable, the SDK fails open (requests proceed) and calls `on_cost_error`.

### Proxy Mode vs Direct Mode

The SDK detects whether requests go through the NullSpend proxy (by comparing the request URL origin against `proxy_url`) or directly to the provider:

- **Proxy mode:** The proxy handles cost tracking and enforcement server-side. The SDK intercepts proxy 429 responses with `X-NullSpend-Denied: 1` and raises the corresponding error.
- **Direct mode:** The SDK tracks costs client-side using the built-in pricing engine and enforces locally via the policy cache.

### Streaming Support

Tracked clients handle streaming responses transparently via `TeeByteStream` — chunks are yielded to the caller while SSE data is accumulated for cost extraction. Cost events are queued when the stream completes.

### Finalization Reserve

When near the budget limit:

```python
final_client = ns.create_tracked_client("openai", finalize=True)

# This request can use the finalization reserve
response = final_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize and save results"}],
)
```

The `BudgetExceededError` includes reserve fields:

```python
from nullspend.errors import BudgetExceededError

try:
    response = tracked.chat.completions.create(...)
except BudgetExceededError as e:
    print(e.finalization_reserve_microdollars)   # Reserve amount, or None
    print(e.finalization_remaining_microdollars)  # Remaining after reserve, or None
```

## Customer Sessions

Scope cost tracking and enforcement to a specific customer for per-customer profitability tracking.

```python
session = ns.customer("acme-corp")

# Pre-configured httpx.Client for each provider, attributed to the customer
openai = OpenAI(http_client=session.openai)
anthropic = Anthropic(http_client=session.anthropic)
```

Customer IDs are validated (trimmed, max 256 chars, alphanumeric + `._:-`). All cost events from the session's tracked clients are tagged with the customer ID.

```python
# With enforcement and session limits
session = ns.customer(
    "acme-corp",
    enforcement=True,
    session_id="task-042",
    session_limit_microdollars=5_000_000,
    tags={"team": "backend"},
    on_denied=lambda reason: print(reason),
)
```

## Async Client

The `AsyncNullSpend` client mirrors every method from the sync client using `httpx.AsyncClient`:

```python
from nullspend import AsyncNullSpend

async with AsyncNullSpend(api_key="ns_live_sk_...") as ns:
    summary = await ns.get_cost_summary("30d")
    print(f"Total: ${summary.totals['totalCostMicrodollars'] / 1_000_000:.2f}")

    action = await ns.create_action(CreateActionInput(
        agent_id="support-agent",
        action_type="send_email",
        payload={"to": "user@example.com"},
    ))

    decision = await ns.wait_for_decision(action.id, timeout_s=300.0)
```

All methods have the same signatures as the sync client, but return coroutines.

## Cost Reporting

### Client-Side Batching (recommended)

Enable `cost_reporting` to batch cost events in the background:

```python
ns = NullSpend(
    api_key="ns_live_sk_...",
    cost_reporting=CostReportingConfig(
        batch_size=10,           # flush every 10 events
        flush_interval_ms=5000,  # or every 5 seconds
        max_queue_size=1000,     # drop events if queue is full
    ),
)

# Queue events — sent in batches automatically
ns.queue_cost(CostEventInput(
    provider="openai", model="gpt-4o",
    input_tokens=500, output_tokens=150, cost_microdollars=4625,
))

# Explicit flush and shutdown
ns.flush()     # drain queue immediately
ns.shutdown()  # flush + stop background thread
```

The background thread flushes on exit via `atexit`. Use `with NullSpend(...) as ns:` for automatic `shutdown()` on context exit.

### `report_cost(event)` — Single Event

```python
from nullspend import CostEventInput

result = ns.report_cost(CostEventInput(
    provider="anthropic",
    model="claude-sonnet-4-6",
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
    customer="acme-corp",
))
```

### `report_cost_batch(events)` — Batch

```python
result = ns.report_cost_batch([
    CostEventInput(provider="openai", model="gpt-4o", input_tokens=500, output_tokens=150, cost_microdollars=4625),
    CostEventInput(provider="openai", model="gpt-4o-mini", input_tokens=1000, output_tokens=300, cost_microdollars=225),
])
```

## Cost Calculation

The SDK includes a built-in pricing engine with 56 models (synced from `@nullspend/cost-engine`). Use it to compute cost events from API response usage:

```python
from nullspend import calculate_openai_cost_event, calculate_anthropic_cost_event

# From an OpenAI response
event = calculate_openai_cost_event(
    model="gpt-4o",
    usage={"prompt_tokens": 500, "completion_tokens": 150},
    duration_ms=1200,
)

# From an Anthropic response (with cache details)
event = calculate_anthropic_cost_event(
    model="claude-sonnet-4-6",
    usage={"input_tokens": 1000, "output_tokens": 500},
    cache_creation_detail={"cache_creation_tokens": 200},
    duration_ms=800,
)

# Check if a model is known
from nullspend import is_known_model, get_model_pricing
is_known_model("openai", "gpt-4o")  # True
get_model_pricing("openai", "gpt-4o")  # {"inputPerMTok": 2.5, "outputPerMTok": 10.0, ...}
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

Raises `PollTimeoutError` if the timeout elapses while still `pending`.

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
    poll_interval_s=2.0,
    timeout_s=300.0,
))
```

- On approval: marks `executing`, calls `execute(context)`, marks `executed` with result
- On rejection/expiry: raises `RejectedError`
- On execute failure: marks `failed`, re-raises the original error

### `request_budget_increase(options)`

Request a budget increase via the HITL approval flow:

```python
from nullspend import RequestBudgetIncreaseOptions

result = ns.request_budget_increase(RequestBudgetIncreaseOptions(
    agent_id="data-agent",
    amount_microdollars=10_000_000,  # request $10 increase
    reason="Need more budget for batch processing",
    entity_type="user",
    entity_id="user-123",
    poll_interval_s=2.0,
    timeout_s=600.0,
))
print(result.action_id, result.requested_amount_microdollars)
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

# Paginate with cursor — pass it straight back, no json.dumps needed
if result.cursor:
    next_page = ns.list_cost_events(ListCostEventsOptions(limit=10, cursor=result.cursor))
```

## Retry Behavior

The SDK automatically retries on transient failures:

**Retryable:** `429`, `500`, `502`, `503`, `504`, network errors (`httpx.TransportError`)

**Not retryable:** `4xx` errors other than `429`

**Backoff:** Full-jitter exponential — `max(0.001, random() * min(base * 2^attempt, 5s))`

**Idempotency:** Mutating requests (`POST`) include an `Idempotency-Key` header generated once and reused across retries.

## Error Handling

Eleven error classes, all extending `Exception`:

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

Raised by `wait_for_decision` when the timeout elapses. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `action_id` | `str` | The action that timed out |
| `timeout_ms` | `int` | The timeout in milliseconds |

### `RejectedError`

Raised by `propose_and_wait` when the action is rejected or expired. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `action_id` | `str` | The action that was rejected |
| `action_status` | `str` | The terminal status (`"rejected"` or `"expired"`) |

### `BudgetExceededError`

Raised when enforcement is enabled and estimated cost exceeds remaining budget, or when the proxy returns a `budget_exceeded` 429.

| Property | Type | Description |
|---|---|---|
| `remaining_microdollars` | `int` | Budget remaining when denial occurred |
| `entity_type` | `str \| None` | Budget entity type (`"api_key"`, `"customer"`, `"tag"`) |
| `entity_id` | `str \| None` | Entity identifier |
| `limit_microdollars` | `int \| None` | Budget ceiling |
| `spend_microdollars` | `int \| None` | Current spend |
| `upgrade_url` | `str \| None` | URL to upgrade (if configured) |

### `MandateViolationError`

Raised when the requested model/provider is not allowed by key policy.

| Property | Type | Description |
|---|---|---|
| `mandate` | `str` | The mandate that was violated (e.g. `"providers"`) |
| `requested` | `str` | What was requested |
| `allowed` | `list[str]` | What is allowed |

### `SessionLimitExceededError`

Raised when session spend would exceed the session limit.

| Property | Type | Description |
|---|---|---|
| `session_spend_microdollars` | `int` | Current session spend |
| `session_limit_microdollars` | `int` | Session limit |

### `VelocityExceededError`

Raised when request rate exceeds the velocity limit.

| Property | Type | Description |
|---|---|---|
| `retry_after_seconds` | `float \| None` | Seconds until the limit resets |
| `limit_microdollars` | `int \| None` | Velocity limit |
| `window_seconds` | `int \| None` | Velocity window |
| `current_microdollars` | `int \| None` | Current spend in window |

### `TagBudgetExceededError`

Raised when a tag-scoped budget is exceeded.

| Property | Type | Description |
|---|---|---|
| `tag_key` | `str \| None` | Tag key |
| `tag_value` | `str \| None` | Tag value |
| `remaining_microdollars` | `int \| None` | Budget remaining |
| `limit_microdollars` | `int \| None` | Budget ceiling |
| `spend_microdollars` | `int \| None` | Current spend |

### `LoopDetectedError`

Raised when repeated identical calls exceed the loop detection threshold (proxy or client-side detection).

| Property | Type | Description |
|---|---|---|
| `model` | `str` | Model the loop was detected against |
| `call_count` | `int` | Repeated-call count observed in the window |
| `window_seconds` | `int` | Sliding window size in seconds |
| `max_calls` | `int` | Configured ceiling that was exceeded |
| `detection_type` | `str` | `"per_key"` or other detector mode |

### `PlanLimitExceededError`

Raised when an org exceeds its NullSpend plan-tier governed-request cap. Distinct from `BudgetExceededError`, which is for org-configured budgets. The error carries the upgrade URL so callers can surface a CTA.

| Property | Type | Description |
|---|---|---|
| `count` | `int` | Governed requests used in the current period |
| `block_at` | `int` | Cap that triggered the block |
| `tier` | `str` | Current tier (e.g., `"free"`) |
| `upgrade_url` | `str \| None` | URL to upgrade the plan |
| `self_host_url` | `str \| None` | URL with self-host instructions |

```python
from nullspend import (
    BudgetExceededError,
    MandateViolationError,
    LoopDetectedError,
    PlanLimitExceededError,
)

try:
    openai.chat.completions.create(model="gpt-4o", messages=[...])
except BudgetExceededError as err:
    print(f"${err.remaining_microdollars / 1_000_000:.2f} remaining")
    if err.upgrade_url:
        print(f"Upgrade at: {err.upgrade_url}")
except MandateViolationError as err:
    print(f"{err.mandate}: {err.requested} not in {err.allowed}")
except LoopDetectedError as err:
    print(f"Loop blocked: {err.call_count}/{err.max_calls} calls in {err.window_seconds}s")
except PlanLimitExceededError as err:
    print(f"Plan cap hit ({err.count}/{err.block_at}). Upgrade: {err.upgrade_url}")
```

## Types

All types are dataclasses exported from the package:

```python
from nullspend import (
    # Clients
    NullSpend,
    AsyncNullSpend,
    CostReporter,

    # Configuration
    NullSpendConfig,
    CostReportingConfig,

    # Actions
    CreateActionInput,
    CreateActionResponse,
    ActionRecord,
    MarkResultInput,
    MutateActionResponse,
    ProposeAndWaitOptions,
    RequestBudgetIncreaseOptions,
    BudgetIncreaseResult,

    # Cost reporting
    CostEventInput,
    CostEventRecord,
    CostBreakdown,

    # Cost calculation
    calculate_openai_cost_event,
    calculate_anthropic_cost_event,
    get_model_pricing,
    is_known_model,

    # Tracked clients
    create_tracked_client,
    CustomerSession,
    validate_customer_id,

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
    BudgetExceededError,
    MandateViolationError,
    SessionLimitExceededError,
    VelocityExceededError,
    TagBudgetExceededError,
)
```

## Differences from the JavaScript SDK

The Python SDK (v0.2.0) has full feature parity with the JavaScript SDK for all core functionality.

| Feature | JavaScript SDK | Python SDK |
|---|---|---|
| HITL actions | Full support | Full support |
| Cost reporting | `reportCost`, `reportCostBatch` | `report_cost`, `report_cost_batch` |
| Client-side batching | `queueCost()` / `flush()` / `shutdown()` | `queue_cost()` / `flush()` / `shutdown()` |
| Budget status | `checkBudget`, `listBudgets` | `check_budget`, `list_budgets` |
| Cost awareness | `getCostSummary`, `listCostEvents` | `get_cost_summary`, `list_cost_events` |
| Tracked fetch | `createTrackedFetch` | `create_tracked_client` (returns `httpx.Client`) |
| Provider shorthands | `ns.createTrackedFetch("openai")` | `ns.openai` / `ns.anthropic` properties |
| Customer sessions | `ns.customer()` | `ns.customer()` |
| Enforcement | `enforcement: true` | `enforcement=True` |
| Budget negotiation | `requestBudgetIncrease()` | `request_budget_increase()` |
| Cost calculation | via cost-engine package | Built-in (bundled pricing data) |
| Error classes | 10 classes | 11 classes (adds `PollTimeoutError` alias) |
| Async support | Native (`async/await`) | `AsyncNullSpend` (separate client) |
| HTTP client | `fetch` (configurable) | `httpx` |
| `onRetry` callback | Supported | Not yet available |
| Timeout error class | `TimeoutError` | `PollTimeoutError` (avoids shadowing Python builtin) |

## Related

- [Human-in-the-Loop](../features/human-in-the-loop.md) — approval workflow concepts and best practices
- [Cost Tracking](../features/cost-tracking.md) — how cost events are recorded
- [Actions API](../api-reference/actions-api.md) — raw HTTP endpoint reference
- [Budgets API](../api-reference/budgets-api.md) — budget management endpoints
- [JavaScript SDK](javascript.md) — TypeScript/JavaScript client
- [Claude Agent Adapter](claude-agent.md) — adapter for the Claude Agent SDK
