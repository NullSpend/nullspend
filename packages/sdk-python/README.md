# nullspend

Python SDK for [NullSpend](https://nullspend.dev) — FinOps for AI agents.

## Installation

```bash
pip install nullspend
```

## Quick Start

```python
from nullspend import NullSpend, CostEventInput

ns = NullSpend(
    base_url="https://nullspend.dev",
    api_key="ns_live_sk_...",
)

# Report a cost event
ns.report_cost(CostEventInput(
    provider="openai",
    model="gpt-4o",
    input_tokens=1200,
    output_tokens=350,
    cost_microdollars=5250,
    tags={"environment": "production", "agent": "support-bot"},
))

# Check budget status
status = ns.check_budget()
for entity in status.entities:
    print(f"{entity.entity_type}/{entity.entity_id}: "
          f"${entity.remaining_microdollars / 1_000_000:.2f} remaining")
```

## Features

- Cost event reporting (single and batch)
- `@track_tool` decorator + `track()` inline for tool cost tracking
- Budget status and listing
- Cost analytics summaries
- Human-in-the-loop action management (create, poll, mark result)
- `propose_and_wait()` high-level orchestrator
- Tracked httpx transport with automatic cost tracking and enforcement
- Loop detection for stuck agents (client-side and proxy-side)
- Automatic retries with exponential backoff
- Idempotency keys on mutating requests
- Type hints throughout (py.typed)

## Tool Cost Tracking

Track costs for non-LLM tools (API calls, web searches, database queries) with a decorator or inline:

```python
from nullspend import NullSpend

ns = NullSpend(api_key="ns_live_sk_...", base_url="https://nullspend.dev")

# Decorator — reports cost after function executes, measures duration
@ns.track_tool(cost=0.02, tool_name="web_search")
def search(query: str) -> str:
    return requests.get(f"https://api.example.com/search?q={query}").text

# Inline — reports cost and returns the result
result = ns.track(call_api(), cost=0.01, tool_name="search")
```

Costs are reported even if the tool raises an exception (the API call happened, tokens were consumed). Failed calls are tagged with `_ns_error: "true"` for dashboard filtering.

Optional fields: `provider`, `model`, `tool_server`, `tags`, `customer`. Uses batch reporter when configured, falls back to direct reporting.

## Loop Detection

Detects agents stuck in infinite loops — repeated identical calls that burn budget without progress.

**Proxy users:** Loop detection is on by default. If your agent calls the same model with identical content 50+ times in 60 seconds, the proxy returns a 429 with `code: "loop_detected"`. No configuration needed.

**SDK users:** Opt in with one line:

```python
from nullspend import create_tracked_client

client = create_tracked_client("openai", loop_detection=True)
```

Customize thresholds:

```python
from nullspend import create_tracked_client, LoopDetectionConfig

client = create_tracked_client("openai", loop_detection=LoopDetectionConfig(
    max_calls=100,       # higher for batch workloads
    window_seconds=120,  # wider window
))
```

**Disabling:** Set `loop_max_calls=0` on the budget entity via the API or dashboard.

**Error handling:**

| Error | Code | When |
|---|---|---|
| `LoopDetectedError` | `loop_detected` | Same model+content called 50+ times in 60s |
| `BudgetExceededError` | `budget_exceeded` | Budget exhausted |
| `VelocityExceededError` | `velocity_exceeded` | Spend rate exceeds velocity limit |
| `SessionLimitExceededError` | `session_limit_exceeded` | Session spend cap reached |
| `TagBudgetExceededError` | `tag_budget_exceeded` | Tag-level budget exhausted |
| `MandateViolationError` | `mandate_violation` | Model/provider not allowed |

```python
from nullspend import LoopDetectedError, BudgetExceededError

try:
    response = client.chat.completions.create(...)
except LoopDetectedError as e:
    print(f"Loop detected: {e.model} called {e.call_count} times")
    print(f"Detection type: {e.detection_type}")  # "per_key" or "aggregate"
except BudgetExceededError as e:
    print(f"Budget exceeded: {e.remaining_microdollars} microdollars remaining")
    if e.recovery:
        print(f"Retryable: {e.recovery['retryable']}")
        print(f"Owner action required: {e.recovery['owner_action_required']}")
```

Every denial error includes an optional `recovery` dict with machine-readable hints:

| Field | Type | Meaning |
|---|---|---|
| `retryable` | `bool` | Whether the request can succeed if retried later |
| `owner_action_required` | `bool` | Whether a human or config change is needed |
| `retry_after_seconds` | `int \| None` | Seconds to wait before retry (retryable denials only) |
| `docs` | `str \| None` | Documentation URL for this error type |

`recovery` is `None` when connecting to an older proxy that doesn't include it.

## Documentation

See the [NullSpend docs](https://nullspend.dev/docs) for full API reference.
