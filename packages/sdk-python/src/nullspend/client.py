from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx

from nullspend._retry import (
    RETRYABLE_STATUS_CODES,
    calculate_retry_delay_s,
    parse_retry_after_s,
)
from nullspend.errors import NullSpendError, RejectedError, PollTimeoutError
from nullspend.types import (
    ActionRecord,
    BudgetIncreaseResult,
    BudgetStatus,
    CostBreakdown,
    CostEventInput,
    CostReportingConfig,
    CostSummaryResponse,
    CreateActionInput,
    CreateActionResponse,
    CustomerSession,
    ListBudgetsResponse,
    ListCostEventsOptions,
    ListCostEventsResponse,
    MarkResultInput,
    MutateActionResponse,
    NullSpendConfig,
    ProposeAndWaitOptions,
    RequestBudgetIncreaseOptions,
    CostEventRecord,
    BudgetEntity,
    BudgetRecord,
)

logger = logging.getLogger("nullspend")

_DEFAULT_BASE_URL = "https://nullspend.dev"
_API_KEY_HEADER = "x-nullspend-key"
_SDK_VERSION = "0.2.0"
_SAFE_PATH_SEGMENT_RE = __import__("re").compile(r"^[a-zA-Z0-9_\-.:]+$")


def _validate_path_segment(value: str, name: str) -> str:
    if not value or not _SAFE_PATH_SEGMENT_RE.match(value):
        raise NullSpendError(f"Invalid {name}: must be alphanumeric (got {value!r})")
    return value


def _safe_parse(parser_name: str, data: dict[str, Any], builder: Any) -> Any:
    """Wrap a parser function to convert KeyError into NullSpendError."""
    try:
        return builder(data)
    except KeyError as e:
        raise NullSpendError(
            f"Unexpected response format in {parser_name}: missing field {e}"
        ) from e


def _parse_action_record(data: dict[str, Any]) -> ActionRecord:
    return _safe_parse("action", data, lambda d: ActionRecord(
        id=d["id"],
        agent_id=d.get("agentId", ""),
        action_type=d.get("actionType", ""),
        status=d.get("status", "pending"),
        payload=d.get("payload", {}),
        metadata=d.get("metadata"),
        created_at=d.get("createdAt", ""),
        approved_at=d.get("approvedAt"),
        rejected_at=d.get("rejectedAt"),
        executed_at=d.get("executedAt"),
        expires_at=d.get("expiresAt"),
        expired_at=d.get("expiredAt"),
        approved_by=d.get("approvedBy"),
        rejected_by=d.get("rejectedBy"),
        result=d.get("result"),
        error_message=d.get("errorMessage"),
        environment=d.get("environment"),
        source_framework=d.get("sourceFramework"),
    ))


def _parse_cost_event(data: dict[str, Any]) -> CostEventRecord:
    return _safe_parse("cost_event", data, lambda d: CostEventRecord(
        id=d["id"],
        request_id=d.get("requestId", ""),
        api_key_id=d.get("apiKeyId"),
        provider=d["provider"],
        model=d["model"],
        input_tokens=d.get("inputTokens", 0),
        output_tokens=d.get("outputTokens", 0),
        cached_input_tokens=d.get("cachedInputTokens", 0),
        reasoning_tokens=d.get("reasoningTokens", 0),
        cost_microdollars=d.get("costMicrodollars", 0),
        duration_ms=d.get("durationMs"),
        session_id=d.get("sessionId"),
        trace_id=d.get("traceId"),
        source=d.get("source", ""),
        tags=d.get("tags"),
        key_name=d.get("keyName"),
        created_at=d.get("createdAt", ""),
        customer_id=d.get("customerId"),
        cost_breakdown=d.get("costBreakdown"),
        event_type=d.get("eventType"),
        tool_name=d.get("toolName"),
    ))


def _parse_budget_entity(data: dict[str, Any]) -> BudgetEntity:
    return _safe_parse("budget_entity", data, lambda d: BudgetEntity(
        entity_type=d["entityType"],
        entity_id=d["entityId"],
        limit_microdollars=d["limitMicrodollars"],
        spend_microdollars=d["spendMicrodollars"],
        remaining_microdollars=d["remainingMicrodollars"],
        policy=d["policy"],
        reset_interval=d.get("resetInterval"),
        current_period_start=d.get("currentPeriodStart"),
    ))


def _parse_budget_record(data: dict[str, Any]) -> BudgetRecord:
    return _safe_parse("budget_record", data, lambda d: BudgetRecord(
        id=d["id"],
        entity_type=d["entityType"],
        entity_id=d["entityId"],
        max_budget_microdollars=d["maxBudgetMicrodollars"],
        spend_microdollars=d["spendMicrodollars"],
        policy=d["policy"],
        reset_interval=d.get("resetInterval"),
        current_period_start=d.get("currentPeriodStart"),
        threshold_percentages=d.get("thresholdPercentages", []),
        velocity_limit_microdollars=d.get("velocityLimitMicrodollars"),
        velocity_window_seconds=d.get("velocityWindowSeconds"),
        velocity_cooldown_seconds=d.get("velocityCooldownSeconds"),
        session_limit_microdollars=d.get("sessionLimitMicrodollars"),
        created_at=d.get("createdAt", ""),
        updated_at=d.get("updatedAt", ""),
    ))


def _serialize_cost_event(event: CostEventInput) -> dict[str, Any]:
    """Serialize a CostEventInput to a camelCase dict for the API."""
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
    if event.cost_breakdown is not None:
        body["costBreakdown"] = {
            "input": event.cost_breakdown.input,
            "output": event.cost_breakdown.output,
            "cached": event.cost_breakdown.cached,
        }
        if event.cost_breakdown.reasoning is not None:
            body["costBreakdown"]["reasoning"] = event.cost_breakdown.reasoning
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
    if event.customer is not None:
        body["customer"] = event.customer
    return body


class NullSpend:
    """Python client for the NullSpend API.

    Usage::

        from nullspend import NullSpend

        # Reads NULLSPEND_API_KEY from environment
        ns = NullSpend()

        # Or provide explicitly
        ns = NullSpend(api_key="ns_live_sk_...")

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
        proxy_url: str | None = None,
        api_version: str = "2026-04-01",
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_base_delay_s: float = 0.5,
        cost_reporting: CostReportingConfig | None = None,
    ):
        if config:
            resolved_base = config.base_url
            resolved_key = config.api_key
            if not resolved_key:
                raise NullSpendError(
                    "API key is required in NullSpendConfig. "
                    "Get a key at https://nullspend.dev/app/keys"
                )
            self._base_url = resolved_base.rstrip("/")
            self._api_key = resolved_key
            self._api_version = config.api_version
            self._timeout_s = config.request_timeout_s
            self._max_retries = config.max_retries
            self._retry_base_delay_s = config.retry_base_delay_s
        else:
            resolved_base = (
                base_url
                or os.environ.get("NULLSPEND_BASE_URL")
                or _DEFAULT_BASE_URL
            )
            resolved_key = (
                api_key
                or os.environ.get("NULLSPEND_API_KEY")
            )
            if not resolved_key:
                raise NullSpendError(
                    "API key is required. Pass api_key= or set NULLSPEND_API_KEY "
                    "environment variable. Get a key at https://nullspend.dev/app/keys"
                )
            self._base_url = resolved_base.rstrip("/")
            self._api_key = resolved_key
            self._api_version = api_version
            self._timeout_s = request_timeout_s
            self._max_retries = min(10, max(0, max_retries))
            self._retry_base_delay_s = max(0, retry_base_delay_s)

        # Validate proxy_url at construction time (fail-fast)
        resolved_proxy = proxy_url or os.environ.get("NULLSPEND_PROXY_URL")
        if resolved_proxy:
            from urllib.parse import urlparse as _urlparse
            parsed = _urlparse(resolved_proxy.rstrip("/"))
            if parsed.scheme not in ("http", "https"):
                raise NullSpendError(
                    f"proxy_url must use http or https (got {parsed.scheme!r}). "
                    f"Example: https://proxy.nullspend.dev"
                )
            if not parsed.hostname:
                raise NullSpendError(
                    f"proxy_url must be a valid absolute URL, got: {resolved_proxy!r}"
                )
            self._proxy_url: str | None = resolved_proxy.rstrip("/")
        else:
            self._proxy_url = None

        self._client = httpx.Client(timeout=self._timeout_s)
        self._policy_caches: list[Any] = []

        # Wire CostReporter if config provided
        if cost_reporting is not None:
            from nullspend._cost_reporter import CostReporter as _CostReporterImpl
            self._cost_reporter: CostReporter | None = _CostReporterImpl(
                cost_reporting,
                lambda batch: self.report_cost_batch(batch),
            )
        else:
            self._cost_reporter = None

    def close(self) -> None:
        """Close the underlying HTTP client, tracked clients, and flush pending cost events."""
        if self._cost_reporter is not None:
            self._cost_reporter.shutdown()
        # Close cached tracked clients
        for attr in ("_openai_client", "_anthropic_client"):
            client = getattr(self, attr, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
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
        inner = data.get("data", data)
        return CreateActionResponse(
            id=inner["id"],
            status=inner["status"],
            expires_at=inner.get("expiresAt"),
        )

    def get_action(self, action_id: str) -> ActionRecord:
        _validate_path_segment(action_id, "action_id")
        data = self._request("GET", f"/api/actions/{action_id}")
        return _parse_action_record(data.get("data", data))

    def mark_result(self, action_id: str, input: MarkResultInput) -> MutateActionResponse:
        _validate_path_segment(action_id, "action_id")
        body: dict[str, Any] = {"status": input.status}
        if input.result is not None:
            body["result"] = input.result
        if input.error_message is not None:
            body["errorMessage"] = input.error_message
        data = self._request("POST", f"/api/actions/{action_id}/result", body)
        inner = data.get("data", data)
        return MutateActionResponse(
            id=inner.get("id", action_id),
            status=inner.get("status", input.status),
            approved_at=inner.get("approvedAt"),
            rejected_at=inner.get("rejectedAt"),
            executed_at=inner.get("executedAt"),
            budget_increase=inner.get("budgetIncrease"),
        )

    # ---- Cost Reporting ----

    def report_cost(self, event: CostEventInput) -> dict[str, Any]:
        return self._request("POST", "/api/cost-events", _serialize_cost_event(event))

    def report_cost_batch(self, events: list[CostEventInput]) -> dict[str, Any]:
        batch = [_serialize_cost_event(e) for e in events]
        return self._request("POST", "/api/cost-events/batch", {"events": batch})

    def queue_cost(self, event: CostEventInput) -> None:
        """Enqueue a cost event for batched reporting. Requires cost_reporting config."""
        if self._cost_reporter is None:
            raise NullSpendError(
                "Cost reporter not configured. Pass cost_reporting=CostReportingConfig() "
                "to NullSpend() to enable batched cost reporting."
            )
        self._cost_reporter.enqueue(event)

    def flush(self) -> None:
        """Flush any pending batched cost events immediately."""
        if self._cost_reporter is not None:
            self._cost_reporter.flush()

    def shutdown(self) -> None:
        """Gracefully shut down the cost reporter, flushing all pending events."""
        if self._cost_reporter is not None:
            self._cost_reporter.shutdown()

    # ---- Tracked Clients ----

    def create_tracked_client(
        self,
        provider: str,
        *,
        customer: str | None = None,
        session_id: str | None = None,
        tags: dict[str, str] | None = None,
        trace_id: str | None = None,
        action_id: str | None = None,
        enforcement: bool = False,
        session_limit_microdollars: int | None = None,
        on_cost_error: Any | None = None,
        on_denied: Any | None = None,
    ) -> httpx.Client:
        """Create an httpx.Client with automatic cost tracking for a provider.

        Use as http_client for OpenAI/Anthropic SDKs:
            client = OpenAI(http_client=ns.create_tracked_client("openai"))
        """
        from nullspend._tracked_client import create_tracked_client as _create
        from nullspend._policy_cache import PolicyCache

        policy_cache = None
        if enforcement:
            policy_cache = PolicyCache(
                fetch_fn=lambda: self._request("GET", "/api/policy"),
                on_error=on_cost_error,
            )
            self._policy_caches.append(policy_cache)

        queue_fn = self._cost_reporter.enqueue if self._cost_reporter else self._queue_cost_direct

        return _create(
            provider,
            proxy_url=self._proxy_url,
            api_key=self._api_key,
            api_version=self._api_version,
            customer=customer,
            session_id=session_id,
            tags=tags,
            trace_id=trace_id,
            action_id=action_id,
            enforcement=enforcement,
            session_limit_microdollars=session_limit_microdollars,
            policy_cache=policy_cache,
            queue_cost=queue_fn,
            on_cost_error=on_cost_error,
            on_denied=on_denied,
            timeout=self._timeout_s,
        )

    _direct_cost_error_logged = False

    def _queue_cost_direct(self, event: CostEventInput) -> None:
        """Fallback: report cost immediately when no CostReporter is configured."""
        try:
            self.report_cost(event)
        except Exception as err:
            if not NullSpend._direct_cost_error_logged:
                NullSpend._direct_cost_error_logged = True
                logger.warning(
                    "nullspend: Failed to report cost event (%s). "
                    "Subsequent errors will be silent.",
                    err,
                )

    @property
    def openai(self) -> httpx.Client:
        """Tracked httpx.Client for OpenAI. Simplest integration path.

        Usage:
            from openai import OpenAI
            client = OpenAI(http_client=ns.openai)
        """
        if not hasattr(self, "_openai_client") or self._openai_client is None:
            self._openai_client = self.create_tracked_client("openai")
        return self._openai_client

    @property
    def anthropic(self) -> httpx.Client:
        """Tracked httpx.Client for Anthropic. Simplest integration path.

        Usage:
            from anthropic import Anthropic
            client = Anthropic(http_client=ns.anthropic)
        """
        if not hasattr(self, "_anthropic_client") or self._anthropic_client is None:
            self._anthropic_client = self.create_tracked_client("anthropic")
        return self._anthropic_client

    def customer(
        self,
        customer_id: str,
        *,
        tags: dict[str, str] | None = None,
        session_id: str | None = None,
        session_limit_microdollars: int | None = None,
        enforcement: bool = False,
        on_cost_error: Any | None = None,
        on_denied: Any | None = None,
    ) -> CustomerSession:
        """Create a customer-scoped session with tracked clients.

        Usage:
            session = ns.customer("acme-corp")
            openai_client = OpenAI(http_client=session.openai)
        """
        from nullspend.types import validate_customer_id
        validated_id = validate_customer_id(customer_id)

        cache: dict[str, httpx.Client] = {}

        def get_client(provider: str) -> httpx.Client:
            if provider not in cache:
                cache[provider] = self.create_tracked_client(
                    provider,
                    customer=validated_id,
                    tags=tags,
                    session_id=session_id,
                    session_limit_microdollars=session_limit_microdollars,
                    enforcement=enforcement,
                    on_cost_error=on_cost_error,
                    on_denied=on_denied,
                )
            return cache[provider]

        return CustomerSession(
            openai=get_client("openai"),
            anthropic=get_client("anthropic"),
            customer_id=validated_id,
        )

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
                params["cursor"] = (
                    json.dumps(options.cursor)
                    if isinstance(options.cursor, dict)
                    else options.cursor
                )
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
            keys=inner.get("keys"),
            tools=inner.get("tools"),
            sources=inner.get("sources"),
            traces=inner.get("traces"),
            cost_breakdown=inner.get("costBreakdown"),
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

        # Execute
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

    # ---- Budget Negotiation ----

    def request_budget_increase(
        self, options: RequestBudgetIncreaseOptions,
    ) -> BudgetIncreaseResult:
        """Request a budget increase via the HITL approval flow.

        Creates a budget_increase action, waits for human approval, and returns
        the result. On approval, the budget is increased server-side automatically.

        The payload shape must match the server's budgetIncreasePayloadSchema:
        entityType, entityId, requestedAmountMicrodollars, currentLimitMicrodollars,
        currentSpendMicrodollars, reason.
        """
        # Match server's budgetIncreasePayloadSchema exactly
        payload: dict[str, Any] = {
            "entityType": options.entity_type,
            "entityId": options.entity_id,
            "requestedAmountMicrodollars": options.amount_microdollars,
            "currentLimitMicrodollars": options.current_limit_microdollars,
            "currentSpendMicrodollars": options.current_spend_microdollars,
            "reason": options.reason,
        }

        def _execute(ctx: dict[str, Any]) -> dict[str, Any]:
            # On approval, the server-side budget increase has already run.
            # Return the action ID and requested amount (matching TS SDK pattern).
            return {
                "actionId": ctx.get("action_id", "unknown"),
                "requestedAmountMicrodollars": options.amount_microdollars,
            }

        result = self.propose_and_wait(ProposeAndWaitOptions(
            agent_id=options.agent_id,
            action_type="budget_increase",
            payload=payload,
            execute=_execute,
            metadata=options.metadata,
            poll_interval_s=options.poll_interval_s,
            timeout_s=options.timeout_s,
            on_poll=options.on_poll,
        ))

        return BudgetIncreaseResult(
            action_id=result.get("actionId", "unknown"),
            requested_amount_microdollars=options.amount_microdollars,
        )

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
            "X-NullSpend-SDK": f"python/{_SDK_VERSION}",
            "Accept": "application/json",
        }

        if body is not None:
            headers["Content-Type"] = "application/json"

        if method != "GET":
            headers["Idempotency-Key"] = f"ns_{uuid.uuid4()}"

        last_error: NullSpendError | None = None
        # Retry-After from the previous response, used as the delay for the
        # next attempt. Set to None to fall back to jitter backoff.
        _pending_retry_after: float | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                if _pending_retry_after is not None:
                    # Server told us exactly how long to wait
                    time.sleep(_pending_retry_after)
                    _pending_retry_after = None
                else:
                    delay = calculate_retry_delay_s(attempt - 1, self._retry_base_delay_s)
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
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                # Capture Retry-After for the NEXT iteration's sleep
                _pending_retry_after = parse_retry_after_s(
                    response.headers.get("retry-after"),
                    max_s=self._retry_base_delay_s * (2 ** (attempt + 1)),
                )
                last_error = NullSpendError(
                    f"{method} {path} failed: HTTP {response.status_code}",
                    response.status_code,
                )
                continue

            # Non-retryable error — build actionable message
            status = response.status_code
            detail = response.reason_phrase or f"HTTP {status}"
            code: str | None = None
            try:
                data = response.json()
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    code = err_obj.get("code")
                    detail = err_obj.get("message", detail)
            except Exception:
                pass

            # Append actionable guidance for common failures
            if status == 401:
                detail = (
                    f"{detail}. Check your NULLSPEND_API_KEY environment "
                    f"variable or the api_key constructor argument. "
                    f"Get a key at https://nullspend.dev/app/keys"
                )

            raise NullSpendError(
                f"{method} {path} failed: {detail}",
                status,
                code,
            )

        raise last_error  # type: ignore[misc]
