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
- Budget status and listing
- Cost analytics summaries
- Human-in-the-loop action management (create, poll, mark result)
- `propose_and_wait()` high-level orchestrator
- Automatic retries with exponential backoff
- Idempotency keys on mutating requests
- Type hints throughout (py.typed)

## Documentation

See the [NullSpend docs](https://nullspend.dev/docs) for full API reference.
