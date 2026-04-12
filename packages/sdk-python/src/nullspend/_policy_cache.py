"""TTL-cached policy enforcement with fail-open semantics.

Fetches budget/mandate/session-limit policy from the dashboard API,
caches with configurable TTL, deduplicates in-flight requests, and
returns stale data on fetch failure (fail-open).

Sync version uses threading.Lock for dedup.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TTL_S = 60.0


@dataclass
class PolicyBudget:
    remaining_microdollars: int
    max_microdollars: int
    spend_microdollars: int
    period_end: str | None
    entity_type: str
    entity_id: str


@dataclass
class PolicyResponse:
    budget: PolicyBudget | None = None
    allowed_models: list[str] | None = None
    allowed_providers: list[str] | None = None
    restrictions_active: bool = False
    session_limit_microdollars: int | None = None


@dataclass
class MandateResult:
    allowed: bool
    mandate: str | None = None
    requested: str | None = None
    allowed_list: list[str] | None = None


@dataclass
class BudgetResult:
    allowed: bool
    remaining: int | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    limit: int | None = None
    spend: int | None = None


def _parse_policy_response(data: dict[str, Any]) -> PolicyResponse:
    """Parse a raw API response into a PolicyResponse."""
    budget_data = data.get("budget")
    budget = None
    if isinstance(budget_data, dict):
        budget = PolicyBudget(
            remaining_microdollars=int(budget_data.get("remaining_microdollars", 0)),
            max_microdollars=int(budget_data.get("max_microdollars", 0)),
            spend_microdollars=int(budget_data.get("spend_microdollars", 0)),
            period_end=budget_data.get("period_end"),
            entity_type=budget_data.get("entity_type", ""),
            entity_id=budget_data.get("entity_id", ""),
        )

    return PolicyResponse(
        budget=budget,
        allowed_models=data.get("allowed_models"),
        allowed_providers=data.get("allowed_providers"),
        restrictions_active=bool(data.get("restrictions_active", False)),
        session_limit_microdollars=data.get("session_limit_microdollars"),
    )


class PolicyCache:
    """TTL-cached policy with fail-open semantics.

    Thread-safe. Deduplicates concurrent fetches. Returns stale data on error.
    """

    def __init__(
        self,
        fetch_fn: Callable[[], dict[str, Any]],
        ttl_s: float = DEFAULT_TTL_S,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self._fetch_fn = fetch_fn
        self._ttl_s = ttl_s
        self._on_error = on_error

        self._cached: PolicyResponse | None = None
        self._cached_at: float = 0.0
        self._lock = threading.Lock()
        self._fetching = False

    def get_policy(self) -> PolicyResponse | None:
        """Fetch policy if stale, return cached otherwise. Fail-open."""
        now = time.monotonic()

        with self._lock:
            if self._cached is not None and (now - self._cached_at) < self._ttl_s:
                return self._cached
            if self._fetching:
                return self._cached  # Another thread is fetching, return stale

            self._fetching = True

        try:
            data = self._fetch_fn()
            policy = _parse_policy_response(data)
            with self._lock:
                self._cached = policy
                self._cached_at = time.monotonic()
            return policy
        except Exception as err:
            if self._on_error:
                try:
                    self._on_error(err)
                except Exception:
                    pass
            with self._lock:
                return self._cached  # Fail-open: return stale or None
        finally:
            with self._lock:
                self._fetching = False

    def check_mandate(self, provider: str, model: str) -> MandateResult:
        """Check if provider/model is allowed by the cached policy. Fail-open."""
        cached = self._cached
        if cached is None:
            return MandateResult(allowed=True)

        if cached.allowed_providers is not None:
            if provider not in cached.allowed_providers:
                return MandateResult(
                    allowed=False,
                    mandate="allowed_providers",
                    requested=provider,
                    allowed_list=cached.allowed_providers,
                )

        if cached.allowed_models is not None:
            if model not in cached.allowed_models:
                return MandateResult(
                    allowed=False,
                    mandate="allowed_models",
                    requested=model,
                    allowed_list=cached.allowed_models,
                )

        return MandateResult(allowed=True)

    def check_budget(self, estimate_microdollars: int) -> BudgetResult:
        """Check if estimated cost fits within budget. Fail-open."""
        cached = self._cached
        if cached is None or cached.budget is None:
            return BudgetResult(allowed=True)

        b = cached.budget
        if estimate_microdollars > b.remaining_microdollars:
            return BudgetResult(
                allowed=False,
                remaining=b.remaining_microdollars,
                entity_type=b.entity_type,
                entity_id=b.entity_id,
                limit=b.max_microdollars,
                spend=b.spend_microdollars,
            )

        return BudgetResult(allowed=True, remaining=b.remaining_microdollars)

    def get_session_limit(self) -> int | None:
        """Get the session limit from cached policy."""
        cached = self._cached
        if cached is None:
            return None
        return cached.session_limit_microdollars

    def invalidate(self) -> None:
        """Clear the cache, forcing a fresh fetch on next access."""
        with self._lock:
            self._cached = None
            self._cached_at = 0.0
