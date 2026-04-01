from __future__ import annotations

import json
import random
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx

from nullspend.errors import NullSpendError, RejectedError, PollTimeoutError
from nullspend.types import (
    ActionRecord,
    BudgetStatus,
    CostEventInput,
    CostSummaryResponse,
    CreateActionInput,
    CreateActionResponse,
    ListBudgetsResponse,
    ListCostEventsOptions,
    ListCostEventsResponse,
    MarkResultInput,
    NullSpendConfig,
    ProposeAndWaitOptions,
    CostEventRecord,
    BudgetEntity,
    BudgetRecord,
)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRY_DELAY_S = 5.0
_API_KEY_HEADER = "x-nullspend-key"
_SAFE_PATH_SEGMENT_RE = __import__("re").compile(r"^[a-zA-Z0-9_\-.:]+$")


def _validate_path_segment(value: str, name: str) -> str:
    """Validate a value is safe to use in a URL path segment."""
    if not value or not _SAFE_PATH_SEGMENT_RE.match(value):
        raise NullSpendError(f"Invalid {name}: must be alphanumeric (got {value!r})")
    return value



def _retry_delay_s(attempt: int, base_delay_s: float) -> float:
    """Full-jitter exponential backoff."""
    ceiling = min(base_delay_s * (2**attempt), _MAX_RETRY_DELAY_S)
    return max(0.001, random.random() * ceiling)


def _parse_action_record(data: dict[str, Any]) -> ActionRecord:
    return ActionRecord(
        id=data["id"],
        agent_id=data.get("agentId", ""),
        action_type=data.get("actionType", ""),
        status=data.get("status", "pending"),
        payload=data.get("payload", {}),
        metadata=data.get("metadata"),
        created_at=data.get("createdAt", ""),
        approved_at=data.get("approvedAt"),
        rejected_at=data.get("rejectedAt"),
        executed_at=data.get("executedAt"),
        expires_at=data.get("expiresAt"),
        expired_at=data.get("expiredAt"),
        approved_by=data.get("approvedBy"),
        rejected_by=data.get("rejectedBy"),
        result=data.get("result"),
        error_message=data.get("errorMessage"),
        environment=data.get("environment"),
        source_framework=data.get("sourceFramework"),
    )


def _parse_cost_event(data: dict[str, Any]) -> CostEventRecord:
    return CostEventRecord(
        id=data["id"],
        request_id=data.get("requestId", ""),
        api_key_id=data.get("apiKeyId"),
        provider=data["provider"],
        model=data["model"],
        input_tokens=data.get("inputTokens", 0),
        output_tokens=data.get("outputTokens", 0),
        cached_input_tokens=data.get("cachedInputTokens", 0),
        reasoning_tokens=data.get("reasoningTokens", 0),
        cost_microdollars=data.get("costMicrodollars", 0),
        duration_ms=data.get("durationMs"),
        session_id=data.get("sessionId"),
        trace_id=data.get("traceId"),
        source=data.get("source", ""),
        tags=data.get("tags"),
        key_name=data.get("keyName"),
        created_at=data.get("createdAt", ""),
    )


def _parse_budget_entity(data: dict[str, Any]) -> BudgetEntity:
    return BudgetEntity(
        entity_type=data["entityType"],
        entity_id=data["entityId"],
        limit_microdollars=data["limitMicrodollars"],
        spend_microdollars=data["spendMicrodollars"],
        remaining_microdollars=data["remainingMicrodollars"],
        policy=data["policy"],
        reset_interval=data.get("resetInterval"),
        current_period_start=data.get("currentPeriodStart"),
    )


def _parse_budget_record(data: dict[str, Any]) -> BudgetRecord:
    return BudgetRecord(
        id=data["id"],
        entity_type=data["entityType"],
        entity_id=data["entityId"],
        max_budget_microdollars=data["maxBudgetMicrodollars"],
        spend_microdollars=data["spendMicrodollars"],
        policy=data["policy"],
        reset_interval=data.get("resetInterval"),
        current_period_start=data.get("currentPeriodStart"),
        threshold_percentages=data.get("thresholdPercentages", []),
        velocity_limit_microdollars=data.get("velocityLimitMicrodollars"),
        velocity_window_seconds=data.get("velocityWindowSeconds"),
        velocity_cooldown_seconds=data.get("velocityCooldownSeconds"),
        session_limit_microdollars=data.get("sessionLimitMicrodollars"),
        created_at=data.get("createdAt", ""),
        updated_at=data.get("updatedAt", ""),
    )


class NullSpend:
    """Python client for the NullSpend API.

    Usage::

        from nullspend import NullSpend

        ns = NullSpend(
            base_url="https://nullspend.com",
            api_key="ns_live_sk_...",
        )

        # Report a cost event
        ns.report_cost(CostEventInput(
            provider="openai",
            model="gpt-4o",
            input_tokens=1200,
            output_tokens=350,
            cost_microdollars=5250,
        ))

        # Check budget status
        status = ns.check_budget()
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        config: NullSpendConfig | None = None,
        api_version: str = "2026-04-01",
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_base_delay_s: float = 0.5,
    ):
        if config:
            self._base_url = config.base_url.rstrip("/")
            self._api_key = config.api_key
            self._api_version = config.api_version
            self._timeout_s = config.request_timeout_s
            self._max_retries = config.max_retries
            self._retry_base_delay_s = config.retry_base_delay_s
        else:
            if not base_url:
                raise NullSpendError("base_url is required")
            if not api_key:
                raise NullSpendError("api_key is required")
            self._base_url = base_url.rstrip("/")
            self._api_key = api_key
            self._api_version = api_version
            self._timeout_s = request_timeout_s
            self._max_retries = min(10, max(0, max_retries))
            self._retry_base_delay_s = max(0, retry_base_delay_s)

        self._client = httpx.Client(timeout=self._timeout_s)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> NullSpend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ---- Actions ----

    def create_action(self, input: CreateActionInput) -> CreateActionResponse:
        body = {
            "agentId": input.agent_id,
            "actionType": input.action_type,
            "payload": input.payload,
        }
        if input.metadata is not None:
            body["metadata"] = input.metadata
        if input.expires_in_seconds is not None:
            body["expiresInSeconds"] = input.expires_in_seconds

        data = self._request("POST", "/api/actions", body)
        return CreateActionResponse(
            id=data["id"],
            status=data["status"],
            expires_at=data.get("expiresAt"),
        )

    def get_action(self, action_id: str) -> ActionRecord:
        _validate_path_segment(action_id, "action_id")
        data = self._request("GET", f"/api/actions/{action_id}")
        return _parse_action_record(data.get("data", data))

    def mark_result(self, action_id: str, input: MarkResultInput) -> dict[str, Any]:
        _validate_path_segment(action_id, "action_id")
        body: dict[str, Any] = {"status": input.status}
        if input.result is not None:
            body["result"] = input.result
        if input.error_message is not None:
            body["errorMessage"] = input.error_message
        return self._request("POST", f"/api/actions/{action_id}/result", body)

    # ---- Cost Reporting ----

    def report_cost(self, event: CostEventInput) -> dict[str, Any]:
        body: dict[str, Any] = {
            "provider": event.provider,
            "model": event.model,
            "inputTokens": event.input_tokens,
            "outputTokens": event.output_tokens,
            "costMicrodollars": event.cost_microdollars,
        }
        if event.cached_input_tokens:
            body["cachedInputTokens"] = event.cached_input_tokens
        if event.reasoning_tokens:
            body["reasoningTokens"] = event.reasoning_tokens
        if event.duration_ms is not None:
            body["durationMs"] = event.duration_ms
        if event.session_id is not None:
            body["sessionId"] = event.session_id
        if event.trace_id is not None:
            body["traceId"] = event.trace_id
        if event.event_type is not None:
            body["eventType"] = event.event_type
        if event.tool_name is not None:
            body["toolName"] = event.tool_name
        if event.tool_server is not None:
            body["toolServer"] = event.tool_server
        if event.tags is not None:
            body["tags"] = event.tags
        return self._request("POST", "/api/cost-events", body)

    def report_cost_batch(self, events: list[CostEventInput]) -> dict[str, Any]:
        batch = []
        for event in events:
            item: dict[str, Any] = {
                "provider": event.provider,
                "model": event.model,
                "inputTokens": event.input_tokens,
                "outputTokens": event.output_tokens,
                "costMicrodollars": event.cost_microdollars,
            }
            if event.cached_input_tokens:
                item["cachedInputTokens"] = event.cached_input_tokens
            if event.reasoning_tokens:
                item["reasoningTokens"] = event.reasoning_tokens
            if event.duration_ms is not None:
                item["durationMs"] = event.duration_ms
            if event.session_id is not None:
                item["sessionId"] = event.session_id
            if event.trace_id is not None:
                item["traceId"] = event.trace_id
            if event.event_type is not None:
                item["eventType"] = event.event_type
            if event.tags is not None:
                item["tags"] = event.tags
            batch.append(item)
        return self._request("POST", "/api/cost-events/batch", {"events": batch})

    # ---- Budget Status ----

    def check_budget(self) -> BudgetStatus:
        data = self._request("GET", "/api/budgets/status")
        return BudgetStatus(
            entities=[_parse_budget_entity(e) for e in data.get("entities", [])],
        )

    def list_budgets(self) -> ListBudgetsResponse:
        data = self._request("GET", "/api/budgets")
        return ListBudgetsResponse(
            data=[_parse_budget_record(b) for b in data.get("data", [])],
        )

    # ---- Cost Events (Read) ----

    def list_cost_events(
        self, options: ListCostEventsOptions | None = None,
    ) -> ListCostEventsResponse:
        params: dict[str, str] = {}
        if options:
            if options.limit is not None:
                params["limit"] = str(options.limit)
            if options.cursor is not None:
                params["cursor"] = options.cursor
        qs = urlencode(params) if params else ""
        path = f"/api/cost-events?{qs}" if qs else "/api/cost-events"
        data = self._request("GET", path)
        return ListCostEventsResponse(
            data=[_parse_cost_event(e) for e in data.get("data", [])],
            cursor=data.get("cursor"),
        )

    def get_cost_summary(
        self, period: str = "30d",
    ) -> CostSummaryResponse:
        if period not in ("7d", "30d", "90d"):
            raise NullSpendError(f"Invalid period: must be '7d', '30d', or '90d' (got {period!r})")
        data = self._request("GET", f"/api/cost-events/summary?period={period}")
        inner = data.get("data", data)
        return CostSummaryResponse(
            daily=inner.get("daily", []),
            models=inner.get("models", {}),
            providers=inner.get("providers", {}),
            totals=inner.get("totals", {}),
        )

    # ---- Polling ----

    def wait_for_decision(
        self,
        action_id: str,
        *,
        poll_interval_s: float = 2.0,
        timeout_s: float = 300.0,
        on_poll: Any | None = None,
    ) -> ActionRecord:
        import time

        timeout_ms = int(timeout_s * 1000)
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            action = self.get_action(action_id)
            if on_poll:
                on_poll(action)

            if action.status != "pending":
                return action

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_s, remaining))

        raise PollTimeoutError(action_id, timeout_ms)

    # ---- High-level Orchestrator ----

    def propose_and_wait(self, options: ProposeAndWaitOptions) -> Any:
        response = self.create_action(CreateActionInput(
            agent_id=options.agent_id,
            action_type=options.action_type,
            payload=options.payload,
            metadata=options.metadata,
            expires_in_seconds=options.expires_in_seconds,
        ))

        decision = self.wait_for_decision(
            response.id,
            poll_interval_s=options.poll_interval_s,
            timeout_s=options.timeout_s,
            on_poll=options.on_poll,
        )

        if decision.status != "approved":
            raise RejectedError(response.id, decision.status)

        # Mark executing
        try:
            self.mark_result(response.id, MarkResultInput(status="executing"))
        except NullSpendError as e:
            if e.status_code == 409:
                current = self.get_action(response.id)
                if current.status != "executing":
                    raise
            else:
                raise

        # Execute — pass context dict matching JS SDK's { actionId: id } pattern
        try:
            result = options.execute({"action_id": response.id})
        except Exception as err:
            try:
                self.mark_result(
                    response.id,
                    MarkResultInput(status="failed", error_message=str(err)),
                )
            except Exception:
                pass
            raise

        # Mark executed
        serializable = (
            result if isinstance(result, dict) else {"value": result}
        )
        try:
            self.mark_result(
                response.id,
                MarkResultInput(status="executed", result=serializable),
            )
        except NullSpendError as e:
            if e.status_code == 409:
                current = self.get_action(response.id)
                if current.status != "executed":
                    raise
            else:
                raise

        return result

    # ---- HTTP ----

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"

        headers = {
            _API_KEY_HEADER: self._api_key,
            "NullSpend-Version": self._api_version,
            "Accept": "application/json",
        }

        if body is not None:
            headers["Content-Type"] = "application/json"

        if method != "GET":
            headers["Idempotency-Key"] = f"ns_{uuid.uuid4()}"

        last_error: NullSpendError | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                import time
                delay = _retry_delay_s(attempt - 1, self._retry_base_delay_s)
                time.sleep(delay)

            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    content=json.dumps(body) if body is not None else None,
                )
            except httpx.TransportError as err:
                last_error = NullSpendError(
                    f"{method} {path} network error: {err}"
                )
                if attempt < self._max_retries:
                    continue
                raise last_error from err

            if response.is_success:
                data = response.json()
                if not isinstance(data, dict):
                    raise NullSpendError(
                        f"{method} {path} returned unexpected JSON type: {type(data).__name__}",
                        response.status_code,
                    )
                return data

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                last_error = NullSpendError(
                    f"{method} {path} failed: HTTP {response.status_code}",
                    response.status_code,
                )
                continue

            # Non-retryable error
            detail = response.reason_phrase or f"HTTP {response.status_code}"
            code: str | None = None
            try:
                data = response.json()
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    code = err_obj.get("code")
                    detail = err_obj.get("message", detail)
            except Exception:
                pass

            raise NullSpendError(
                f"{method} {path} failed: {detail}",
                response.status_code,
                code,
            )

        raise last_error  # type: ignore[misc]
