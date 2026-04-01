from nullspend.client import NullSpend
from nullspend.errors import NullSpendError, PollTimeoutError, TimeoutError, RejectedError
from nullspend.types import (
    ActionRecord,
    ActionStatus,
    ActionType,
    BudgetEntity,
    BudgetRecord,
    BudgetStatus,
    CostEventInput,
    CostEventRecord,
    CostSummaryResponse,
    CreateActionInput,
    CreateActionResponse,
    ListBudgetsResponse,
    ListCostEventsOptions,
    ListCostEventsResponse,
    MarkResultInput,
    NullSpendConfig,
    ProposeAndWaitOptions,
)

__all__ = [
    "NullSpend",
    "NullSpendError",
    "PollTimeoutError",
    "TimeoutError",  # backward-compatible alias for PollTimeoutError
    "RejectedError",
    "ActionRecord",
    "ActionStatus",
    "ActionType",
    "BudgetEntity",
    "BudgetRecord",
    "BudgetStatus",
    "CostEventInput",
    "CostEventRecord",
    "CostSummaryResponse",
    "CreateActionInput",
    "CreateActionResponse",
    "ListBudgetsResponse",
    "ListCostEventsOptions",
    "ListCostEventsResponse",
    "MarkResultInput",
    "NullSpendConfig",
    "ProposeAndWaitOptions",
]

__version__ = "0.1.0"
