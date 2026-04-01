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
