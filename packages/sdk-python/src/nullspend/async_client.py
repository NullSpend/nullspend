"""Async NullSpend client using httpx.AsyncClient.

Mirrors every method from the sync NullSpend client.
Uses asyncio for polling and lifecycle management.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx

from nullspend._retry import (
    RETRYABLE_STATUS_CODES,
    calculate_retry_delay_s,
    parse_retry_after_s,
)
from nullspend.client import (
    _DEFAULT_BASE_URL,
    _API_KEY_HEADER,
    _SDK_VERSION,
    _validate_path_segment,
    _parse_action_record,
    _parse_cost_event,
    _parse_budget_entity,
    _parse_budget_record,
    _serialize_cost_event,
)
from nullspend.errors import NullSpendError, RejectedError, PollTimeoutError
from nullspend.types import (
    ActionRecord,
    BudgetIncreaseResult,
    BudgetStatus,
    CostEventInput,
    CostSummaryResponse,
    CreateActionInput,
    CreateActionResponse,
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


class AsyncNullSpend:
    """Async Python client for the NullSpend API.

    Usage::

        import asyncio
        from nullspend import AsyncNullSpend

        async def main():
            async with AsyncNullSpend() as ns:
                status = await ns.check_budget()
                print(status)

        asyncio.run(main())
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
            if not config.api_key:
                raise NullSpendError(
                    "API key is required in NullSpendConfig. "
                    "Get a key at https://nullspend.dev/app/keys"
                )
            self._base_url = config.base_url.rstrip("/")
            self._api_key = config.api_key
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

        self._client = httpx.AsyncClient(timeout=self._timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncNullSpend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ---- Actions ----

    async def create_action(self, input: CreateActionInput) -> CreateActionResponse:
        body = {
            "agentId": input.agent_id,
            "actionType": input.action_type,
            "payload": input.payload,
        }
        if input.metadata is not None:
            body["metadata"] = input.metadata
        if input.expires_in_seconds is not None:
            body["expiresInSeconds"] = input.expires_in_seconds

        data = await self._request("POST", "/api/actions", body)
        return CreateActionResponse(
            id=data["id"],
            status=data["status"],
            expires_at=data.get("expiresAt"),
        )

    async def get_action(self, action_id: str) -> ActionRecord:
        _validate_path_segment(action_id, "action_id")
        data = await self._request("GET", f"/api/actions/{action_id}")
        return _parse_action_record(data.get("data", data))

    async def mark_result(self, action_id: str, input: MarkResultInput) -> MutateActionResponse:
        _validate_path_segment(action_id, "action_id")
        body: dict[str, Any] = {"status": input.status}
        if input.result is not None:
            body["result"] = input.result
        if input.error_message is not None:
            body["errorMessage"] = input.error_message
        data = await self._request("POST", f"/api/actions/{action_id}/result", body)
        return MutateActionResponse(
            id=data.get("id", action_id),
            status=data.get("status", input.status),
            approved_at=data.get("approvedAt"),
            rejected_at=data.get("rejectedAt"),
            executed_at=data.get("executedAt"),
            budget_increase=data.get("budgetIncrease"),
        )

    # ---- Cost Reporting ----

    async def report_cost(self, event: CostEventInput) -> dict[str, Any]:
        return await self._request("POST", "/api/cost-events", _serialize_cost_event(event))

    async def report_cost_batch(self, events: list[CostEventInput]) -> dict[str, Any]:
        batch = [_serialize_cost_event(e) for e in events]
        return await self._request("POST", "/api/cost-events/batch", {"events": batch})

    # ---- Budget Status ----

    async def check_budget(self) -> BudgetStatus:
        data = await self._request("GET", "/api/budgets/status")
        return BudgetStatus(
            entities=[_parse_budget_entity(e) for e in data.get("entities", [])],
        )

    async def list_budgets(self) -> ListBudgetsResponse:
        data = await self._request("GET", "/api/budgets")
        return ListBudgetsResponse(
            data=[_parse_budget_record(b) for b in data.get("data", [])],
        )

    # ---- Cost Events (Read) ----

    async def list_cost_events(
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
        data = await self._request("GET", path)
        return ListCostEventsResponse(
            data=[_parse_cost_event(e) for e in data.get("data", [])],
            cursor=data.get("cursor"),
        )

    async def get_cost_summary(
        self, period: str = "30d",
    ) -> CostSummaryResponse:
        if period not in ("7d", "30d", "90d"):
            raise NullSpendError(f"Invalid period: must be '7d', '30d', or '90d' (got {period!r})")
        data = await self._request("GET", f"/api/cost-events/summary?period={period}")
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

    async def wait_for_decision(
        self,
        action_id: str,
        *,
        poll_interval_s: float = 2.0,
        timeout_s: float = 300.0,
        on_poll: Any | None = None,
    ) -> ActionRecord:
        import time as _time

        timeout_ms = int(timeout_s * 1000)
        deadline = _time.monotonic() + timeout_s

        while _time.monotonic() < deadline:
            action = await self.get_action(action_id)
            if on_poll:
                result = on_poll(action)
                if asyncio.iscoroutine(result):
                    await result

            if action.status != "pending":
                return action

            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval_s, remaining))

        raise PollTimeoutError(action_id, timeout_ms)

    # ---- High-level Orchestrator ----

    async def propose_and_wait(self, options: ProposeAndWaitOptions) -> Any:
        response = await self.create_action(CreateActionInput(
            agent_id=options.agent_id,
            action_type=options.action_type,
            payload=options.payload,
            metadata=options.metadata,
            expires_in_seconds=options.expires_in_seconds,
        ))

        decision = await self.wait_for_decision(
            response.id,
            poll_interval_s=options.poll_interval_s,
            timeout_s=options.timeout_s,
            on_poll=options.on_poll,
        )

        if decision.status != "approved":
            raise RejectedError(response.id, decision.status)

        # Mark executing
        try:
            await self.mark_result(response.id, MarkResultInput(status="executing"))
        except NullSpendError as e:
            if e.status_code == 409:
                current = await self.get_action(response.id)
                if current.status != "executing":
                    raise
            else:
                raise

        # Execute — properly await async executors
        try:
            result = options.execute({"action_id": response.id})
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                result = await result
        except Exception as err:
            try:
                await self.mark_result(
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
            await self.mark_result(
                response.id,
                MarkResultInput(status="executed", result=serializable),
            )
        except NullSpendError as e:
            if e.status_code == 409:
                current = await self.get_action(response.id)
                if current.status != "executed":
                    raise
            else:
                raise

        return result

    # ---- Budget Negotiation ----

    async def request_budget_increase(
        self, options: RequestBudgetIncreaseOptions,
    ) -> BudgetIncreaseResult:
        """Request a budget increase via the HITL approval flow.

        The payload shape must match the server's budgetIncreasePayloadSchema.
        """
        payload: dict[str, Any] = {
            "entityType": options.entity_type,
            "entityId": options.entity_id,
            "requestedAmountMicrodollars": options.amount_microdollars,
            "currentLimitMicrodollars": options.current_limit_microdollars,
            "currentSpendMicrodollars": options.current_spend_microdollars,
            "reason": options.reason,
        }

        async def _execute(ctx: dict[str, Any]) -> dict[str, Any]:
            return {
                "actionId": ctx.get("action_id", "unknown"),
                "requestedAmountMicrodollars": options.amount_microdollars,
            }

        result = await self.propose_and_wait(ProposeAndWaitOptions(
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

    async def _request(
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
        _pending_retry_after: float | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                if _pending_retry_after is not None:
                    await asyncio.sleep(_pending_retry_after)
                    _pending_retry_after = None
                else:
                    delay = calculate_retry_delay_s(attempt - 1, self._retry_base_delay_s)
                    await asyncio.sleep(delay)

            try:
                response = await self._client.request(
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
                _pending_retry_after = parse_retry_after_s(
                    response.headers.get("retry-after"),
                    max_s=self._retry_base_delay_s * (2 ** (attempt + 1)),
                )
                last_error = NullSpendError(
                    f"{method} {path} failed: HTTP {response.status_code}",
                    response.status_code,
                )
                continue

            # Non-retryable error
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
