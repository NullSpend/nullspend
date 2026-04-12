from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

ACTION_TYPES = (
    "send_email",
    "http_post",
    "http_delete",
    "shell_command",
    "db_write",
    "file_write",
    "file_delete",
    "budget_increase",
)

ActionType = Literal[
    "send_email",
    "http_post",
    "http_delete",
    "shell_command",
    "db_write",
    "file_write",
    "file_delete",
    "budget_increase",
]

ActionStatus = Literal[
    "pending",
    "approved",
    "rejected",
    "expired",
    "executing",
    "executed",
    "failed",
]

TERMINAL_STATUSES = frozenset({"rejected", "expired", "executed", "failed"})


# ---- Customer ID validation ----

MAX_CUSTOMER_ID_LENGTH = 256
_CUSTOMER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._:\-]+$")


def validate_customer_id(value: Any) -> str:
    """Validate and return a trimmed customer ID. Raises NullSpendError on invalid input."""
    from nullspend.errors import NullSpendError

    if not isinstance(value, str):
        raise NullSpendError(f"customer_id must be a string (got {type(value).__name__})")
    trimmed = value.strip()
    if not trimmed:
        raise NullSpendError("customer_id must not be empty")
    if len(trimmed) > MAX_CUSTOMER_ID_LENGTH:
        raise NullSpendError(
            f"customer_id must be at most {MAX_CUSTOMER_ID_LENGTH} characters "
            f"(got {len(trimmed)})"
        )
    if not _CUSTOMER_ID_PATTERN.match(trimmed):
        raise NullSpendError(
            f"customer_id contains invalid characters: {trimmed!r}. "
            "Allowed: a-z, A-Z, 0-9, '.', '_', ':', '-'"
        )
    return trimmed


# ---- Config ----


@dataclass
class NullSpendConfig:
    base_url: str = "https://nullspend.dev"
    api_key: str = ""
    api_version: str = "2026-04-01"
    request_timeout_s: float = 30.0
    max_retries: int = 2
    retry_base_delay_s: float = 0.5


@dataclass
class CostReportingConfig:
    batch_size: int = 10
    flush_interval_ms: int = 5000
    max_queue_size: int = 1000
    on_dropped: Callable[[int], None] | None = None
    on_flush_error: Callable[[Exception, list[CostEventInput]], None] | None = None


# ---- Actions ----


@dataclass
class CreateActionInput:
    agent_id: str
    action_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] | None = None
    expires_in_seconds: int | None = None


@dataclass
class CreateActionResponse:
    id: str
    status: str
    expires_at: str | None


@dataclass
class MutateActionResponse:
    id: str
    status: str
    approved_at: str | None = None
    rejected_at: str | None = None
    executed_at: str | None = None
    budget_increase: dict[str, Any] | None = None


@dataclass
class ActionRecord:
    id: str
    agent_id: str
    action_type: str
    status: ActionStatus
    payload: dict[str, Any]
    metadata: dict[str, Any] | None
    created_at: str
    approved_at: str | None
    rejected_at: str | None
    executed_at: str | None
    expires_at: str | None
    expired_at: str | None
    approved_by: str | None
    rejected_by: str | None
    result: dict[str, Any] | None
    error_message: str | None
    environment: str | None
    source_framework: str | None


@dataclass
class MarkResultInput:
    status: Literal["executing", "executed", "failed"]
    result: dict[str, Any] | None = None
    error_message: str | None = None


# ---- Cost Events ----


@dataclass
class CostBreakdown:
    input: int = 0
    output: int = 0
    cached: int = 0
    reasoning: int | None = None


@dataclass
class CostEventInput:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_microdollars: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_breakdown: CostBreakdown | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    trace_id: str | None = None
    event_type: str | None = None
    tool_name: str | None = None
    tool_server: str | None = None
    tags: dict[str, str] | None = None
    customer: str | None = None


@dataclass
class CostEventRecord:
    id: str
    request_id: str
    api_key_id: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    cost_microdollars: int
    duration_ms: int | None
    session_id: str | None
    trace_id: str | None
    source: str
    tags: dict[str, str] | None
    key_name: str | None
    created_at: str
    customer_id: str | None = None
    cost_breakdown: dict[str, Any] | None = None
    event_type: str | None = None
    tool_name: str | None = None


@dataclass
class ListCostEventsOptions:
    limit: int | None = None
    cursor: str | dict[str, str] | None = None


@dataclass
class ListCostEventsResponse:
    data: list[CostEventRecord]
    cursor: dict[str, str] | None


@dataclass
class CostSummaryResponse:
    daily: list[dict[str, Any]]
    models: Any
    providers: Any
    totals: dict[str, Any]
    keys: Any = None
    tools: Any = None
    sources: Any = None
    traces: Any = None
    cost_breakdown: dict[str, Any] | None = None


# ---- Budgets ----


@dataclass
class BudgetEntity:
    entity_type: str
    entity_id: str
    limit_microdollars: int
    spend_microdollars: int
    remaining_microdollars: int
    policy: str
    reset_interval: str | None
    current_period_start: str | None


@dataclass
class BudgetStatus:
    entities: list[BudgetEntity]


@dataclass
class BudgetRecord:
    id: str
    entity_type: str
    entity_id: str
    max_budget_microdollars: int
    spend_microdollars: int
    policy: str
    reset_interval: str | None
    current_period_start: str | None
    threshold_percentages: list[int]
    velocity_limit_microdollars: int | None
    velocity_window_seconds: int | None
    velocity_cooldown_seconds: int | None
    session_limit_microdollars: int | None
    created_at: str
    updated_at: str


@dataclass
class ListBudgetsResponse:
    data: list[BudgetRecord]


# ---- Propose and Wait ----


@dataclass
class ProposeAndWaitOptions:
    agent_id: str
    action_type: str
    payload: dict[str, Any]
    execute: Callable[..., Any | Awaitable[Any]]
    metadata: dict[str, Any] | None = None
    expires_in_seconds: int | None = None
    poll_interval_s: float = 2.0
    timeout_s: float = 300.0
    on_poll: Callable[[ActionRecord], None] | None = None


@dataclass
class RequestBudgetIncreaseOptions:
    agent_id: str
    amount_microdollars: int
    reason: str
    entity_type: str = "api_key"
    entity_id: str = "unknown"
    current_limit_microdollars: int = 0
    current_spend_microdollars: int = 0
    metadata: dict[str, Any] | None = None
    poll_interval_s: float = 2.0
    timeout_s: float = 300.0
    on_poll: Callable[[ActionRecord], None] | None = None


@dataclass
class CustomerSession:
    """A customer-scoped session with pre-bound tracked clients."""
    openai: Any  # httpx.Client
    anthropic: Any  # httpx.Client
    customer_id: str


@dataclass
class BudgetIncreaseResult:
    action_id: str
    requested_amount_microdollars: int
