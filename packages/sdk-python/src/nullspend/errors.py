"""NullSpend SDK error classes.

Error message formula (Stripe pattern): what + why + fix + url.
"""
from __future__ import annotations


# Per-error-class default docs URLs (D-1). Used when the proxy doesn't
# supply a `recovery["docs"]` URL — gives every error a stable place to point
# developers for resolution guidance. Keyed by class name.
_DEFAULT_ERROR_DOCS_URL = {
    "NullSpendError": "https://nullspend.dev/docs/errors",
    "PollTimeoutError": "https://nullspend.dev/docs/errors/timeout",
    "RejectedError": "https://nullspend.dev/docs/errors/rejected",
    "BudgetExceededError": "https://nullspend.dev/docs/errors/budget-exceeded",
    "MandateViolationError": "https://nullspend.dev/docs/errors/mandate-violation",
    "SessionLimitExceededError": "https://nullspend.dev/docs/errors/session-limit",
    "VelocityExceededError": "https://nullspend.dev/docs/errors/velocity",
    "LoopDetectedError": "https://nullspend.dev/docs/errors/loop-detected",
    "TagBudgetExceededError": "https://nullspend.dev/docs/errors/tag-budget",
    "PlanLimitExceededError": "https://nullspend.dev/docs/errors/plan-limit",
}


class NullSpendError(Exception):
    """Base exception for NullSpend SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code

    @property
    def docs_url(self) -> str:
        """Documentation URL for this error type. (D-1)

        Subclasses with a `recovery` dict prefer `recovery["docs"]` if present
        (proxy-supplied, may be customer-customized) and fall back to a stable
        per-class default.
        """
        recovery = getattr(self, "recovery", None)
        if isinstance(recovery, dict):
            docs = recovery.get("docs")
            if isinstance(docs, str) and docs:
                return docs
        return _DEFAULT_ERROR_DOCS_URL.get(
            type(self).__name__,
            _DEFAULT_ERROR_DOCS_URL["NullSpendError"],
        )


class PollTimeoutError(NullSpendError):
    """Raised when polling for a decision exceeds the timeout."""

    def __init__(self, action_id: str, timeout_ms: int):
        super().__init__(
            f"Timed out waiting for decision on action {action_id} after {timeout_ms}ms"
        )
        self.action_id = action_id
        self.timeout_ms = timeout_ms


# Backward-compatible alias — prefer PollTimeoutError to avoid shadowing builtins
TimeoutError = PollTimeoutError


class RejectedError(NullSpendError):
    """Raised when an action is rejected or expired instead of approved."""

    def __init__(self, action_id: str, status: str):
        super().__init__(f"Action {action_id} was {status}")
        self.action_id = action_id
        self.action_status = status


# ---- Enforcement Errors ----


class BudgetExceededError(NullSpendError):
    """Raised when a request would exceed the budget."""

    def __init__(
        self,
        remaining_microdollars: int,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit_microdollars: int | None = None,
        spend_microdollars: int | None = None,
        upgrade_url: str | None = None,
        *,
        finalization_reserve_microdollars: int | None = None,
        finalization_remaining_microdollars: int | None = None,
        recovery: dict | None = None,
    ):
        remaining_dollars = remaining_microdollars / 1_000_000
        parts = [f"Budget exceeded. ${remaining_dollars:.2f} remaining"]
        if limit_microdollars is not None and spend_microdollars is not None:
            limit_d = limit_microdollars / 1_000_000
            spend_d = spend_microdollars / 1_000_000
            parts.append(f"(limit: ${limit_d:.2f}, spent: ${spend_d:.2f})")
        if entity_type and entity_id:
            parts.append(f"on {entity_type}/{entity_id}")
        fix = "Increase at https://nullspend.dev/app/budgets"
        if upgrade_url:
            fix = f"Upgrade at {upgrade_url}"
        parts.append(f". {fix}")
        super().__init__(" ".join(parts))
        self.remaining_microdollars = remaining_microdollars
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.limit_microdollars = limit_microdollars
        self.spend_microdollars = spend_microdollars
        self.upgrade_url = upgrade_url
        self.finalization_reserve_microdollars = finalization_reserve_microdollars
        self.finalization_remaining_microdollars = finalization_remaining_microdollars
        self.recovery = recovery


class MandateViolationError(NullSpendError):
    """Raised when a request violates an allowed-models or allowed-providers mandate."""

    def __init__(
        self,
        mandate: str,
        requested: str,
        allowed: list[str],
    ):
        allowed_str = ", ".join(allowed) if allowed else "none"
        super().__init__(
            f"Mandate violation: {mandate} '{requested}' is not allowed. "
            f"Allowed: [{allowed_str}]. "
            f"Update your policy at https://nullspend.dev/app/budgets"
        )
        self.mandate = mandate
        self.requested = requested
        self.allowed = allowed


class SessionLimitExceededError(NullSpendError):
    """Raised when per-session spend exceeds the configured limit."""

    def __init__(
        self,
        session_spend_microdollars: int,
        session_limit_microdollars: int,
        recovery: dict | None = None,
    ):
        spend_d = session_spend_microdollars / 1_000_000
        limit_d = session_limit_microdollars / 1_000_000
        super().__init__(
            f"Session limit exceeded. Spent ${spend_d:.2f} of ${limit_d:.2f} session limit. "
            f"Start a new session or increase the session limit."
        )
        self.session_spend_microdollars = session_spend_microdollars
        self.session_limit_microdollars = session_limit_microdollars
        self.recovery = recovery


class VelocityExceededError(NullSpendError):
    """Raised when the velocity (rate) limit is exceeded."""

    def __init__(
        self,
        retry_after_seconds: float | None = None,
        limit_microdollars: int | None = None,
        window_seconds: int | None = None,
        current_microdollars: int | None = None,
        recovery: dict | None = None,
    ):
        parts = ["Velocity limit exceeded."]
        if retry_after_seconds is not None:
            parts.append(f"Retry after {retry_after_seconds:.0f}s.")
        if limit_microdollars is not None and window_seconds is not None:
            limit_d = limit_microdollars / 1_000_000
            parts.append(f"Limit: ${limit_d:.2f} per {window_seconds}s window.")
        super().__init__(" ".join(parts))
        self.retry_after_seconds = retry_after_seconds
        self.limit_microdollars = limit_microdollars
        self.window_seconds = window_seconds
        self.current_microdollars = current_microdollars
        self.recovery = recovery


class LoopDetectedError(NullSpendError):
    """Raised when repeated identical calls exceed the loop detection threshold."""

    def __init__(
        self,
        model: str,
        call_count: int,
        window_seconds: int,
        max_calls: int,
        detection_type: str = "per_key",
        recovery: dict | None = None,
    ):
        self.model = model
        self.call_count = call_count
        self.window_seconds = window_seconds
        self.max_calls = max_calls
        self.detection_type = detection_type
        self.recovery = recovery
        super().__init__(
            f"Loop detected: {model} called {call_count} times with identical "
            f"content in {window_seconds}s (limit: {max_calls}). "
            f"Check for retry loops or stuck agent logic. "
            f"Adjust at https://nullspend.dev/app/budgets or set loop_max_calls=0 to disable.",
            status_code=429,
            code="loop_detected",
        )


class TagBudgetExceededError(NullSpendError):
    """Raised when a tag-level budget is exceeded."""

    def __init__(
        self,
        tag_key: str | None = None,
        tag_value: str | None = None,
        remaining_microdollars: int | None = None,
        limit_microdollars: int | None = None,
        spend_microdollars: int | None = None,
        recovery: dict | None = None,
    ):
        parts = ["Tag budget exceeded"]
        if tag_key and tag_value:
            parts.append(f"for {tag_key}={tag_value}")
        if remaining_microdollars is not None:
            remaining_d = remaining_microdollars / 1_000_000
            parts.append(f". ${remaining_d:.2f} remaining")
        if limit_microdollars is not None and spend_microdollars is not None:
            limit_d = limit_microdollars / 1_000_000
            spend_d = spend_microdollars / 1_000_000
            parts.append(f"(limit: ${limit_d:.2f}, spent: ${spend_d:.2f})")
        parts.append(". Update tag budgets at https://nullspend.dev/app/budgets")
        super().__init__(" ".join(parts))
        self.tag_key = tag_key
        self.tag_value = tag_value
        self.remaining_microdollars = remaining_microdollars
        self.limit_microdollars = limit_microdollars
        self.spend_microdollars = spend_microdollars
        self.recovery = recovery


class PlanLimitExceededError(NullSpendError):
    """Raised when an org exceeds its NullSpend plan-tier governed-request cap.

    Unlike BudgetExceededError (org-configured budget), plan-limit denials come
    from NullSpend's pricing tiers. ``upgrade_url`` points to the NullSpend
    pricing page, ``self_host_url`` to the alternative remediation path. Both
    are top-level ``error.*`` fields on the 429 response body (parsed by
    :func:`_dispatch_denial` in ``_tracked_client.py``).

    ``count``, ``block_at``, and ``tier`` carry the decision frozen at the
    ORIGINAL request time (PR-2c codex-round-3 C1: DO idempotency replay
    persists the original outcome; retries of the same request produce
    identical errors regardless of current counter state).
    """

    def __init__(
        self,
        count: int,
        block_at: int,
        tier: str,
        upgrade_url: str | None = None,
        self_host_url: str | None = None,
        recovery: dict | None = None,
    ):
        safe_count = count if isinstance(count, (int, float)) and count >= 0 else 0
        safe_block_at = block_at if isinstance(block_at, (int, float)) and block_at >= 0 else 0
        safe_tier = tier if isinstance(tier, str) and tier else "unknown"
        super().__init__(
            f"Plan limit reached: {safe_count} of {safe_block_at} governed requests "
            f"on {safe_tier} plan. Upgrade or wait for period reset.",
            status_code=429,
            code="plan_limit_exceeded",
        )
        self.count = safe_count
        self.block_at = safe_block_at
        self.tier = safe_tier
        self.upgrade_url = upgrade_url
        self.self_host_url = self_host_url
        self.recovery = recovery
