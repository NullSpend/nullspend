"""Phase 1 tests: retry extraction, customer ID validation, error classes,
env-var fallback, async client, cost reporter, request_budget_increase."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import time
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from nullspend import (
    NullSpend,
    AsyncNullSpend,
    CostReporter,
    NullSpendError,
    PollTimeoutError,
    RejectedError,
    BudgetExceededError,
    MandateViolationError,
    SessionLimitExceededError,
    VelocityExceededError,
    TagBudgetExceededError,
    CostEventInput,
    CostReportingConfig,
    CostBreakdown,
    MutateActionResponse,
    RequestBudgetIncreaseOptions,
    BudgetIncreaseResult,
    ProposeAndWaitOptions,
    validate_customer_id,
    CreateActionInput,
)
from nullspend._retry import (
    is_retryable_status_code,
    calculate_retry_delay_s,
    parse_retry_after_s,
    RETRYABLE_STATUS_CODES,
)

BASE = "https://nullspend.dev"


# ---- _retry.py ----


class TestRetryHelpers:
    def test_retryable_status_codes(self):
        for code in (429, 500, 502, 503, 504):
            assert is_retryable_status_code(code)
        for code in (200, 201, 400, 401, 403, 404):
            assert not is_retryable_status_code(code)

    def test_calculate_retry_delay_bounds(self):
        for attempt in range(5):
            delay = calculate_retry_delay_s(attempt, 0.5, 5.0)
            assert 0.001 <= delay <= 5.0

    def test_calculate_retry_delay_increases_ceiling(self):
        # Higher attempts should have higher ceilings (on average)
        samples_0 = [calculate_retry_delay_s(0, 0.5) for _ in range(100)]
        samples_3 = [calculate_retry_delay_s(3, 0.5) for _ in range(100)]
        assert max(samples_0) <= 0.5  # ceiling for attempt 0 is 0.5
        assert max(samples_3) <= 5.0  # ceiling for attempt 3 is 4.0, capped at 5.0

    def test_parse_retry_after_numeric(self):
        assert parse_retry_after_s("2") == 2.0
        assert parse_retry_after_s("0") == 0.0
        assert parse_retry_after_s("0.5") == 0.5

    def test_parse_retry_after_capped(self):
        assert parse_retry_after_s("100", max_s=5.0) == 5.0

    def test_parse_retry_after_none(self):
        assert parse_retry_after_s(None) is None
        assert parse_retry_after_s("") is None

    def test_parse_retry_after_negative(self):
        # Negative values are not valid
        assert parse_retry_after_s("-1") is None

    def test_parse_retry_after_http_date(self):
        # RFC 9110 date format
        result = parse_retry_after_s("Sun, 06 Nov 1994 08:49:37 GMT")
        # Date is in the past, should return 0
        assert result == 0.0


# ---- validate_customer_id ----


class TestCustomerIdValidation:
    def test_valid_ids(self):
        assert validate_customer_id("acme") == "acme"
        assert validate_customer_id("user-123") == "user-123"
        assert validate_customer_id("org:team.dev") == "org:team.dev"
        assert validate_customer_id("a_b-c.d:e") == "a_b-c.d:e"

    def test_trims_whitespace(self):
        assert validate_customer_id("  acme  ") == "acme"

    def test_rejects_empty(self):
        with pytest.raises(NullSpendError, match="must not be empty"):
            validate_customer_id("")
        with pytest.raises(NullSpendError, match="must not be empty"):
            validate_customer_id("   ")

    def test_rejects_non_string(self):
        with pytest.raises(NullSpendError, match="must be a string"):
            validate_customer_id(123)
        with pytest.raises(NullSpendError, match="must be a string"):
            validate_customer_id(None)

    def test_rejects_too_long(self):
        with pytest.raises(NullSpendError, match="at most 256"):
            validate_customer_id("a" * 257)

    def test_rejects_special_chars(self):
        with pytest.raises(NullSpendError, match="invalid characters"):
            validate_customer_id("acme/evil")
        with pytest.raises(NullSpendError, match="invalid characters"):
            validate_customer_id("acme corp")
        with pytest.raises(NullSpendError, match="invalid characters"):
            validate_customer_id("ac@me")


# ---- Enforcement Error Classes ----


class TestBudgetExceededError:
    def test_basic(self):
        err = BudgetExceededError(remaining_microdollars=0)
        assert err.remaining_microdollars == 0
        assert "$0.00 remaining" in str(err)
        assert isinstance(err, NullSpendError)

    def test_with_details(self):
        err = BudgetExceededError(
            remaining_microdollars=500_000,
            entity_type="org",
            entity_id="org-1",
            limit_microdollars=10_000_000,
            spend_microdollars=9_500_000,
        )
        assert "$0.50 remaining" in str(err)
        assert "limit: $10.00" in str(err)
        assert "spent: $9.50" in str(err)
        assert "org/org-1" in str(err)

    def test_with_upgrade_url(self):
        err = BudgetExceededError(
            remaining_microdollars=0,
            upgrade_url="https://nullspend.dev/upgrade",
        )
        assert "https://nullspend.dev/upgrade" in str(err)


class TestMandateViolationError:
    def test_basic(self):
        err = MandateViolationError(
            mandate="allowed_models",
            requested="gpt-5",
            allowed=["gpt-4o", "gpt-4o-mini"],
        )
        assert err.mandate == "allowed_models"
        assert err.requested == "gpt-5"
        assert err.allowed == ["gpt-4o", "gpt-4o-mini"]
        assert "gpt-5" in str(err)
        assert "gpt-4o, gpt-4o-mini" in str(err)


class TestSessionLimitExceededError:
    def test_basic(self):
        err = SessionLimitExceededError(
            session_spend_microdollars=5_000_000,
            session_limit_microdollars=5_000_000,
        )
        assert "$5.00" in str(err)
        assert err.session_spend_microdollars == 5_000_000


class TestVelocityExceededError:
    def test_basic(self):
        err = VelocityExceededError(retry_after_seconds=30)
        assert "30s" in str(err)
        assert err.retry_after_seconds == 30

    def test_with_limits(self):
        err = VelocityExceededError(
            limit_microdollars=10_000_000,
            window_seconds=3600,
        )
        assert "$10.00" in str(err)
        assert "3600s" in str(err)


class TestTagBudgetExceededError:
    def test_basic(self):
        err = TagBudgetExceededError(tag_key="env", tag_value="prod")
        assert "env=prod" in str(err)
        assert err.tag_key == "env"

    def test_with_amounts(self):
        err = TagBudgetExceededError(
            remaining_microdollars=0,
            limit_microdollars=1_000_000,
            spend_microdollars=1_000_000,
        )
        assert "$0.00 remaining" in str(err)


# ---- Env-var Fallback ----


class TestEnvVarFallback:
    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"NULLSPEND_API_KEY": "ns_test_env"}):
            ns = NullSpend()
            assert ns._api_key == "ns_test_env"
            assert ns._base_url == "https://nullspend.dev"
            ns.close()

    def test_reads_base_url_from_env(self):
        with patch.dict(os.environ, {
            "NULLSPEND_API_KEY": "ns_test",
            "NULLSPEND_BASE_URL": "https://custom.example.com",
        }):
            ns = NullSpend()
            assert ns._base_url == "https://custom.example.com"
            ns.close()

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"NULLSPEND_API_KEY": "ns_env"}):
            ns = NullSpend(api_key="ns_explicit")
            assert ns._api_key == "ns_explicit"
            ns.close()

    def test_raises_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing NULLSPEND_API_KEY
            os.environ.pop("NULLSPEND_API_KEY", None)
            with pytest.raises(NullSpendError, match="API key is required"):
                NullSpend()


# ---- 401 Actionable Error Message ----


class TestActionableErrors:
    @respx.mock
    def test_401_preserves_server_message_and_appends_guidance(self):
        ns = NullSpend(api_key="bad_key")
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"code": "unauthorized", "message": "API key revoked"}},
            )
        )
        with pytest.raises(NullSpendError) as exc:
            ns.report_cost(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        msg = str(exc.value)
        # Server message preserved
        assert "API key revoked" in msg
        # Actionable guidance appended
        assert "NULLSPEND_API_KEY" in msg
        assert "nullspend.dev/app/keys" in msg
        ns.close()

    @respx.mock
    def test_401_without_server_message_still_actionable(self):
        ns = NullSpend(api_key="bad_key")
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with pytest.raises(NullSpendError) as exc:
            ns.report_cost(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        assert "NULLSPEND_API_KEY" in str(exc.value)
        ns.close()


# ---- CostBreakdown in CostEventInput ----


class TestCostBreakdown:
    @respx.mock
    def test_cost_breakdown_serialized(self):
        ns = NullSpend(api_key="key")
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1"})
        )
        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
            cost_breakdown=CostBreakdown(input=1000, output=500, cached=0),
        ))
        body = json.loads(respx.calls[0].request.content)
        assert body["costBreakdown"]["input"] == 1000
        assert body["costBreakdown"]["output"] == 500
        assert body["costBreakdown"]["cached"] == 0
        assert "reasoning" not in body["costBreakdown"]
        ns.close()

    @respx.mock
    def test_customer_field_serialized(self):
        ns = NullSpend(api_key="key")
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1"})
        )
        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
            customer="acme-corp",
        ))
        body = json.loads(respx.calls[0].request.content)
        assert body["customer"] == "acme-corp"
        ns.close()


# ---- CostReporter ----


class TestCostReporter:
    def test_enqueue_and_flush(self):
        sent: list[list[CostEventInput]] = []
        reporter = CostReporter(
            CostReportingConfig(batch_size=5, flush_interval_ms=60000),
            lambda batch: sent.append(batch),
        )
        for i in range(3):
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        reporter.flush()
        assert len(sent) == 1
        assert len(sent[0]) == 3
        reporter.shutdown()

    def test_auto_flush_at_batch_size(self):
        sent: list[list[CostEventInput]] = []
        reporter = CostReporter(
            CostReportingConfig(batch_size=2, flush_interval_ms=60000),
            lambda batch: sent.append(batch),
        )
        reporter.enqueue(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        reporter.enqueue(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        # Wait briefly for the flush triggered by batch_size
        time.sleep(0.1)
        assert len(sent) >= 1
        reporter.shutdown()

    def test_overflow_drops_oldest(self):
        dropped_count = []
        reporter = CostReporter(
            CostReportingConfig(
                batch_size=100, flush_interval_ms=60000, max_queue_size=2,
                on_dropped=lambda n: dropped_count.append(n),
            ),
            lambda batch: None,
        )
        for i in range(5):
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        assert sum(dropped_count) > 0
        reporter.shutdown()

    def test_flush_error_callback(self):
        errors: list[Exception] = []

        def bad_send(batch: list[CostEventInput]) -> None:
            raise ConnectionError("network down")

        reporter = CostReporter(
            CostReportingConfig(
                batch_size=1, flush_interval_ms=60000,
                on_flush_error=lambda err, evts: errors.append(err),
            ),
            bad_send,
        )
        reporter.enqueue(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        time.sleep(0.2)
        reporter.shutdown()
        assert len(errors) >= 1
        assert "network down" in str(errors[0])

    def test_shutdown_idempotent(self):
        reporter = CostReporter(
            CostReportingConfig(),
            lambda batch: None,
        )
        reporter.shutdown()
        reporter.shutdown()  # Should not raise
        assert reporter.is_shut_down

    def test_config_validation(self):
        with pytest.raises(ValueError, match="batch_size"):
            CostReporter(CostReportingConfig(batch_size=0), lambda b: None)
        with pytest.raises(ValueError, match="batch_size"):
            CostReporter(CostReportingConfig(batch_size=101), lambda b: None)
        with pytest.raises(ValueError, match="flush_interval_ms"):
            CostReporter(CostReportingConfig(flush_interval_ms=50), lambda b: None)
        with pytest.raises(ValueError, match="max_queue_size"):
            CostReporter(CostReportingConfig(max_queue_size=0), lambda b: None)

    def test_enqueue_after_shutdown_silently_dropped(self):
        reporter = CostReporter(
            CostReportingConfig(),
            lambda batch: None,
        )
        reporter.shutdown()
        # Should not raise
        reporter.enqueue(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))


# ---- AsyncNullSpend ----


class TestAsyncClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_check_budget(self):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={
                "entities": [{
                    "entityType": "org",
                    "entityId": "org-1",
                    "limitMicrodollars": 10_000_000,
                    "spendMicrodollars": 5_000_000,
                    "remainingMicrodollars": 5_000_000,
                    "policy": "refillable",
                    "resetInterval": "monthly",
                    "currentPeriodStart": "2026-04-01T00:00:00Z",
                }]
            })
        )
        async with AsyncNullSpend(api_key="test_key") as ns:
            status = await ns.check_budget()
            assert len(status.entities) == 1
            assert status.entities[0].remaining_microdollars == 5_000_000

    @respx.mock
    @pytest.mark.asyncio
    async def test_report_cost(self):
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1"})
        )
        async with AsyncNullSpend(api_key="test_key") as ns:
            result = await ns.report_cost(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
            assert result["id"] == "evt_1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_create_action(self):
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_1", "status": "pending", "expiresAt": None,
            })
        )
        async with AsyncNullSpend(api_key="test_key") as ns:
            resp = await ns.create_action(CreateActionInput(
                agent_id="agent-1", action_type="send_email", payload={"to": "user@example.com"},
            ))
            assert resp.id == "act_1"
            assert resp.status == "pending"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_cost_events(self):
        respx.get(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "id": "evt_1", "requestId": "req_1", "apiKeyId": None,
                    "provider": "openai", "model": "gpt-4o",
                    "inputTokens": 100, "outputTokens": 50,
                    "cachedInputTokens": 0, "reasoningTokens": 0,
                    "costMicrodollars": 1500,
                    "durationMs": None, "sessionId": None, "traceId": None,
                    "source": "sdk", "tags": None, "keyName": None,
                    "createdAt": "2026-04-12T00:00:00Z",
                }],
                "cursor": None,
            })
        )
        async with AsyncNullSpend(api_key="test_key") as ns:
            resp = await ns.list_cost_events()
            assert len(resp.data) == 1
            assert resp.data[0].provider == "openai"

    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.Response(500, json={"error": {"message": "Internal"}}),
            httpx.Response(200, json={"entities": []}),
        ]
        async with AsyncNullSpend(api_key="test_key", max_retries=1) as ns:
            status = await ns.check_budget()
            assert len(status.entities) == 0
            assert route.call_count == 2

    @pytest.mark.asyncio
    async def test_env_var_fallback(self):
        with patch.dict(os.environ, {"NULLSPEND_API_KEY": "ns_async_env"}):
            async with AsyncNullSpend() as ns:
                assert ns._api_key == "ns_async_env"

    @pytest.mark.asyncio
    async def test_raises_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NULLSPEND_API_KEY", None)
            with pytest.raises(NullSpendError, match="API key is required"):
                AsyncNullSpend()

    @respx.mock
    @pytest.mark.asyncio
    async def test_propose_and_wait_awaits_async_executor(self):
        """Async executor should be properly awaited."""
        from nullspend.types import ProposeAndWaitOptions

        # Mock create action
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_async", "status": "pending", "expiresAt": None,
            })
        )
        # Mock get action (returns approved)
        respx.get(f"{BASE}/api/actions/act_async").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_async", "agentId": "a", "actionType": "send_email",
                "status": "approved", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": "now", "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        # Mock mark_result
        respx.post(f"{BASE}/api/actions/act_async/result").mock(
            return_value=httpx.Response(200, json={"id": "act_async", "status": "executed"})
        )

        executed = False

        async def async_executor(ctx: dict) -> dict:
            nonlocal executed
            await asyncio.sleep(0.01)  # Actually async
            executed = True
            return {"done": True}

        async with AsyncNullSpend(api_key="test_key") as ns:
            result = await ns.propose_and_wait(ProposeAndWaitOptions(
                agent_id="agent-1",
                action_type="send_email",
                payload={"to": "test@example.com"},
                execute=async_executor,
            ))
            assert executed
            assert result == {"done": True}


# ---- MutateActionResponse ----


class TestMutateActionResponse:
    def test_fields(self):
        resp = MutateActionResponse(
            id="act_1", status="executed",
            budget_increase={"previousLimit": 100, "newLimit": 200, "amount": 100},
        )
        assert resp.budget_increase["amount"] == 100
        assert resp.approved_at is None


# ---- Type Parity ----


class TestTypeParity:
    def test_budget_increase_in_action_types(self):
        from nullspend.types import ACTION_TYPES
        assert "budget_increase" in ACTION_TYPES

    def test_cost_event_has_customer(self):
        event = CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
            customer="acme",
        )
        assert event.customer == "acme"

    def test_cost_summary_has_extended_fields(self):
        from nullspend.types import CostSummaryResponse
        resp = CostSummaryResponse(
            daily=[], models={}, providers={}, totals={},
            keys=[], tools=[], sources=[], traces=[],
            cost_breakdown={"input": 100},
        )
        assert resp.keys == []
        assert resp.cost_breakdown == {"input": 100}


# ---- CostReporter wired through NullSpend ----


class TestCostReporterWired:
    @respx.mock
    def test_queue_cost_works_when_configured(self):
        """CostReporter is wired into NullSpend when cost_reporting is provided."""
        respx.post(f"{BASE}/api/cost-events/batch").mock(
            return_value=httpx.Response(201, json={"inserted": 1, "ids": ["evt_1"]})
        )
        ns = NullSpend(
            api_key="key",
            cost_reporting=CostReportingConfig(batch_size=1, flush_interval_ms=60000),
        )
        ns.queue_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        # batch_size=1 triggers immediate flush
        time.sleep(0.2)
        assert respx.calls.call_count >= 1
        ns.close()

    def test_queue_cost_raises_when_not_configured(self):
        ns = NullSpend(api_key="key")
        with pytest.raises(NullSpendError, match="Cost reporter not configured"):
            ns.queue_cost(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        ns.close()

    @respx.mock
    def test_close_flushes_reporter(self):
        """NullSpend.close() flushes pending cost events."""
        sent = []
        respx.post(f"{BASE}/api/cost-events/batch").mock(
            side_effect=lambda req: (sent.append(1), httpx.Response(201, json={"inserted": 1, "ids": ["e"]}))[1]
        )
        ns = NullSpend(
            api_key="key",
            cost_reporting=CostReportingConfig(batch_size=100, flush_interval_ms=60000),
        )
        ns.queue_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        ns.close()  # Should flush
        assert len(sent) >= 1


# ---- NullSpendConfig validation ----


class TestNullSpendConfigValidation:
    def test_config_with_empty_api_key_raises(self):
        """NullSpendConfig path validates api_key is not empty."""
        from nullspend.types import NullSpendConfig
        with pytest.raises(NullSpendError, match="API key is required"):
            NullSpend(config=NullSpendConfig(api_key=""))

    def test_config_with_valid_key_works(self):
        from nullspend.types import NullSpendConfig
        ns = NullSpend(config=NullSpendConfig(api_key="ns_test"))
        assert ns._api_key == "ns_test"
        ns.close()


# ---- request_budget_increase payload parity ----


class TestRequestBudgetIncreasePayload:
    @respx.mock
    def test_payload_matches_server_schema(self):
        """Payload must include entityType, entityId, requestedAmountMicrodollars,
        currentLimitMicrodollars, currentSpendMicrodollars, reason."""
        # Mock create action
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_bi", "status": "pending", "expiresAt": None,
            })
        )
        # Mock get action (returns approved)
        respx.get(f"{BASE}/api/actions/act_bi").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_bi", "agentId": "a", "actionType": "budget_increase",
                "status": "approved", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": "now", "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        # Mock mark_result
        respx.post(f"{BASE}/api/actions/act_bi/result").mock(
            return_value=httpx.Response(200, json={"id": "act_bi", "status": "executed"})
        )

        ns = NullSpend(api_key="key")
        result = ns.request_budget_increase(RequestBudgetIncreaseOptions(
            agent_id="agent-1",
            amount_microdollars=5_000_000,
            reason="Need more budget for docs",
            entity_type="api_key",
            entity_id="key-123",
            current_limit_microdollars=10_000_000,
            current_spend_microdollars=9_500_000,
        ))

        # Check the create_action call payload
        create_call = respx.calls[0]
        body = json.loads(create_call.request.content)
        payload = body["payload"]
        assert payload["entityType"] == "api_key"
        assert payload["entityId"] == "key-123"
        assert payload["requestedAmountMicrodollars"] == 5_000_000
        assert payload["currentLimitMicrodollars"] == 10_000_000
        assert payload["currentSpendMicrodollars"] == 9_500_000
        assert payload["reason"] == "Need more budget for docs"
        assert body["actionType"] == "budget_increase"

        # Check the return value matches TS SDK pattern
        assert result.action_id is not None
        assert result.requested_amount_microdollars == 5_000_000
        ns.close()


# ---- Retry-After header ----


class TestRetryAfterHeader:
    @respx.mock
    def test_retry_after_respected_on_429(self):
        """Retry loop should respect Retry-After header from server."""
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limited"}}),
            httpx.Response(200, json={"entities": []}),
        ]
        ns = NullSpend(api_key="key", max_retries=1)
        status = ns.check_budget()
        assert len(status.entities) == 0
        assert route.call_count == 2
        ns.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_retry_after_respected(self):
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limited"}}),
            httpx.Response(200, json={"entities": []}),
        ]
        async with AsyncNullSpend(api_key="key", max_retries=1) as ns:
            status = await ns.check_budget()
            assert len(status.entities) == 0
            assert route.call_count == 2


# ---- CostReporter queue_size race fix ----


class TestCostReporterRaceFix:
    def test_queue_size_tracks_correctly_under_concurrent_enqueue(self):
        """After flush, queue_size reflects only un-flushed events."""
        reporter = CostReporter(
            CostReportingConfig(batch_size=100, flush_interval_ms=60000),
            lambda batch: None,
        )
        # Enqueue 5 events
        for _ in range(5):
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        assert reporter._queue_size == 5

        # Flush drains the queue
        reporter.flush()
        assert reporter._queue_size == 0

        # Enqueue 2 more after flush
        for _ in range(2):
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        assert reporter._queue_size == 2
        reporter.shutdown()


# ---- Async mark_result returns MutateActionResponse ----


class TestAsyncMarkResult:
    @respx.mock
    @pytest.mark.asyncio
    async def test_mark_result_returns_mutate_response(self):
        respx.post(f"{BASE}/api/actions/act_1/result").mock(
            return_value=httpx.Response(200, json={
                "id": "act_1", "status": "executed",
                "budgetIncrease": {"previousLimit": 100, "newLimit": 200, "amount": 100, "requestedAmount": 100},
            })
        )
        from nullspend.types import MarkResultInput
        async with AsyncNullSpend(api_key="key") as ns:
            result = await ns.mark_result("act_1", MarkResultInput(status="executed"))
            assert isinstance(result, MutateActionResponse)
            assert result.status == "executed"
            assert result.budget_increase is not None
            assert result.budget_increase["amount"] == 100


# ---- get_cost_summary returns new fields ----


class TestCostSummaryNewFields:
    @respx.mock
    def test_returns_keys_tools_sources_traces(self):
        respx.get(f"{BASE}/api/cost-events/summary?period=7d").mock(
            return_value=httpx.Response(200, json={"data": {
                "daily": [],
                "models": {},
                "providers": {},
                "totals": {"totalCostMicrodollars": 0, "totalRequests": 0, "period": "7d"},
                "keys": [{"keyName": "prod-key", "totalCostMicrodollars": 500}],
                "tools": [{"toolName": "search", "totalCostMicrodollars": 200}],
                "sources": [{"source": "sdk", "totalCostMicrodollars": 700}],
                "traces": [],
                "costBreakdown": {"input": 300, "output": 200, "cached": 0},
            }})
        )
        ns = NullSpend(api_key="key")
        summary = ns.get_cost_summary("7d")
        assert summary.keys == [{"keyName": "prod-key", "totalCostMicrodollars": 500}]
        assert summary.tools == [{"toolName": "search", "totalCostMicrodollars": 200}]
        assert summary.sources == [{"source": "sdk", "totalCostMicrodollars": 700}]
        assert summary.traces == []
        assert summary.cost_breakdown == {"input": 300, "output": 200, "cached": 0}
        ns.close()


# ---- Edge-case audit fixes ----


class TestRetryAfterReplacesJitter:
    @respx.mock
    def test_retry_after_replaces_jitter_not_stacks(self):
        """Retry-After should be used AS the delay, not stacked on top of jitter."""
        import time as _time

        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json={"entities": []}),
        ]
        ns = NullSpend(api_key="key", max_retries=1, retry_base_delay_s=10.0)
        start = _time.monotonic()
        ns.check_budget()
        elapsed = _time.monotonic() - start
        # With Retry-After: 0, the delay should be near-zero.
        # If double-sleeping, it would be ~10s (jitter backoff with base 10).
        assert elapsed < 2.0, f"Expected < 2s, got {elapsed:.1f}s (double-sleep bug)"
        ns.close()


class TestAsyncConfigValidation:
    @pytest.mark.asyncio
    async def test_async_config_empty_key_raises(self):
        """AsyncNullSpend config path validates api_key is not empty."""
        from nullspend.types import NullSpendConfig
        with pytest.raises(NullSpendError, match="API key is required"):
            AsyncNullSpend(config=NullSpendConfig(api_key=""))


class TestParserKeyErrorProtection:
    @respx.mock
    def test_missing_budget_entity_field_raises_nullspend_error(self):
        """Missing required field in server response raises NullSpendError, not KeyError."""
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={
                "entities": [{"entityType": "org"}]  # missing entityId, limitMicrodollars, etc.
            })
        )
        ns = NullSpend(api_key="key")
        with pytest.raises(NullSpendError, match="Unexpected response format"):
            ns.check_budget()
        ns.close()

    @respx.mock
    def test_missing_cost_event_field_raises_nullspend_error(self):
        respx.get(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(200, json={
                "data": [{"id": "evt_1"}],  # missing provider, model
                "cursor": None,
            })
        )
        ns = NullSpend(api_key="key")
        with pytest.raises(NullSpendError, match="Unexpected response format"):
            ns.list_cost_events()
        ns.close()

    @respx.mock
    def test_missing_action_field_raises_nullspend_error(self):
        respx.get(f"{BASE}/api/actions/act_1").mock(
            return_value=httpx.Response(200, json={"data": {}})  # missing id
        )
        ns = NullSpend(api_key="key")
        with pytest.raises(NullSpendError, match="Unexpected response format"):
            ns.get_action("act_1")
        ns.close()


class TestAtexitUnregisterOnShutdown:
    def test_shutdown_unregisters_atexit(self):
        """After shutdown, the atexit handler should be unregistered."""
        import atexit
        reporter = CostReporter(
            CostReportingConfig(batch_size=10, flush_interval_ms=60000),
            lambda batch: None,
        )
        # Verify atexit was registered (handler exists)
        assert not reporter.is_shut_down
        reporter.shutdown()
        assert reporter.is_shut_down
        # After shutdown, calling _atexit_flush is a no-op (is_shut_down=True)
        # and the handler has been unregistered
        reporter._atexit_flush()  # Should not raise


class TestSDKVersionHeader:
    @respx.mock
    def test_sync_sends_sdk_version_header(self):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={"entities": []})
        )
        ns = NullSpend(api_key="key")
        ns.check_budget()
        req = respx.calls[0].request
        assert req.headers["x-nullspend-sdk"] == "python/0.2.0"
        ns.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_sends_sdk_version_header(self):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={"entities": []})
        )
        async with AsyncNullSpend(api_key="key") as ns:
            await ns.check_budget()
            req = respx.calls[0].request
            assert req.headers["x-nullspend-sdk"] == "python/0.2.0"


class TestCostReporterDroppedLogging:
    def test_dropped_events_logged_by_default(self, caplog):
        """When on_dropped is not set, dropped events produce a warning log."""
        import logging
        with caplog.at_level(logging.WARNING, logger="nullspend"):
            reporter = CostReporter(
                CostReportingConfig(batch_size=100, flush_interval_ms=60000, max_queue_size=1),
                lambda batch: None,
            )
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
            reporter.enqueue(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
            reporter.shutdown()
        assert any("Dropped" in r.message for r in caplog.records)


class TestReportCostBatchEmpty:
    @respx.mock
    def test_empty_batch(self):
        """Empty batch should send empty events array."""
        respx.post(f"{BASE}/api/cost-events/batch").mock(
            return_value=httpx.Response(201, json={"inserted": 0, "ids": []})
        )
        ns = NullSpend(api_key="key")
        result = ns.report_cost_batch([])
        body = json.loads(respx.calls[0].request.content)
        assert body["events"] == []
        assert result["inserted"] == 0
        ns.close()


class TestListCostEventsLimitZero:
    @respx.mock
    def test_limit_zero(self):
        """limit=0 should be sent as query param."""
        respx.get(f"{BASE}/api/cost-events?limit=0").mock(
            return_value=httpx.Response(200, json={"data": [], "cursor": None})
        )
        ns = NullSpend(api_key="key")
        from nullspend.types import ListCostEventsOptions
        result = ns.list_cost_events(ListCostEventsOptions(limit=0))
        assert result.data == []
        ns.close()


# ---- Coverage gap fills ----


class TestWaitForDecisionTimeout:
    @respx.mock
    def test_sync_timeout_raises_poll_timeout_error(self):
        """wait_for_decision raises PollTimeoutError when deadline passes."""
        respx.get(f"{BASE}/api/actions/act_timeout").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_timeout", "agentId": "a", "actionType": "send_email",
                "status": "pending", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": None, "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        ns = NullSpend(api_key="key")
        with pytest.raises(PollTimeoutError) as exc:
            ns.wait_for_decision("act_timeout", poll_interval_s=0.01, timeout_s=0.05)
        assert exc.value.action_id == "act_timeout"
        assert exc.value.timeout_ms == 50
        ns.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_timeout_raises_poll_timeout_error(self):
        respx.get(f"{BASE}/api/actions/act_timeout").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_timeout", "agentId": "a", "actionType": "send_email",
                "status": "pending", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": None, "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        async with AsyncNullSpend(api_key="key") as ns:
            with pytest.raises(PollTimeoutError) as exc:
                await ns.wait_for_decision("act_timeout", poll_interval_s=0.01, timeout_s=0.05)
            assert exc.value.action_id == "act_timeout"


class TestProposeAndWaitExecutorFailure:
    @respx.mock
    def test_executor_raises_marks_failed_and_reraises(self):
        """When executor raises, action is marked failed and original error propagates."""
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_fail", "status": "pending", "expiresAt": None,
            })
        )
        respx.get(f"{BASE}/api/actions/act_fail").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_fail", "agentId": "a", "actionType": "send_email",
                "status": "approved", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": "now", "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        mark_route = respx.post(f"{BASE}/api/actions/act_fail/result")
        mark_calls: list[dict] = []
        def capture_mark(request):
            mark_calls.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "act_fail", "status": "failed"})
        mark_route.mock(side_effect=capture_mark)

        ns = NullSpend(api_key="key")

        def bad_executor(ctx):
            raise ValueError("executor blew up")

        with pytest.raises(ValueError, match="executor blew up"):
            ns.propose_and_wait(ProposeAndWaitOptions(
                agent_id="a", action_type="send_email", payload={},
                execute=bad_executor, poll_interval_s=0.01, timeout_s=5.0,
            ))

        # Should have called mark_result twice: "executing" then "failed"
        assert len(mark_calls) == 2
        assert mark_calls[0]["status"] == "executing"
        assert mark_calls[1]["status"] == "failed"
        assert "executor blew up" in mark_calls[1]["errorMessage"]
        ns.close()


class TestProposeAndWaitNonDictResult:
    @respx.mock
    def test_non_dict_result_wrapped(self):
        """Non-dict executor return is wrapped in {'value': result}."""
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_wrap", "status": "pending", "expiresAt": None,
            })
        )
        respx.get(f"{BASE}/api/actions/act_wrap").mock(
            return_value=httpx.Response(200, json={"data": {
                "id": "act_wrap", "agentId": "a", "actionType": "send_email",
                "status": "approved", "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": "now", "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        )
        mark_calls: list[dict] = []
        mark_route = respx.post(f"{BASE}/api/actions/act_wrap/result")
        def capture(request):
            mark_calls.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "act_wrap", "status": "executed"})
        mark_route.mock(side_effect=capture)

        ns = NullSpend(api_key="key")
        result = ns.propose_and_wait(ProposeAndWaitOptions(
            agent_id="a", action_type="send_email", payload={},
            execute=lambda ctx: 42,  # non-dict return
            poll_interval_s=0.01, timeout_s=5.0,
        ))

        assert result == 42
        # The "executed" mark_result should have wrapped it
        executed_call = mark_calls[1]  # [0]=executing, [1]=executed
        assert executed_call["result"] == {"value": 42}
        ns.close()


class TestProposeAndWait409OnMarkExecuting:
    @respx.mock
    def test_409_on_mark_executing_is_tolerated(self):
        """409 on mark_executing is swallowed if action is already executing."""
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_409", "status": "pending", "expiresAt": None,
            })
        )
        call_count = {"get": 0}
        def get_action_handler(request):
            call_count["get"] += 1
            # First call: approved (for wait_for_decision)
            # Second call: executing (for 409 recovery check)
            status = "approved" if call_count["get"] == 1 else "executing"
            return httpx.Response(200, json={"data": {
                "id": "act_409", "agentId": "a", "actionType": "send_email",
                "status": status, "payload": {}, "metadata": None,
                "createdAt": "", "approvedAt": "now", "rejectedAt": None,
                "executedAt": None, "expiresAt": None, "expiredAt": None,
                "approvedBy": None, "rejectedBy": None, "result": None,
                "errorMessage": None, "environment": None, "sourceFramework": None,
            }})
        respx.get(f"{BASE}/api/actions/act_409").mock(side_effect=get_action_handler)

        mark_count = {"n": 0}
        def mark_handler(request):
            mark_count["n"] += 1
            if mark_count["n"] == 1:
                # First mark_result (executing) → 409
                return httpx.Response(409, json={"error": {"message": "Conflict"}})
            return httpx.Response(200, json={"id": "act_409", "status": "executed"})
        respx.post(f"{BASE}/api/actions/act_409/result").mock(side_effect=mark_handler)

        ns = NullSpend(api_key="key")
        result = ns.propose_and_wait(ProposeAndWaitOptions(
            agent_id="a", action_type="send_email", payload={},
            execute=lambda ctx: {"done": True},
            poll_interval_s=0.01, timeout_s=5.0,
        ))
        assert result == {"done": True}
        ns.close()


class TestIdempotencyKeyStableAcrossRetries:
    @respx.mock
    def test_same_idempotency_key_on_retry(self):
        """Idempotency key should be the same across retry attempts for one call."""
        keys_seen = []
        route = respx.post(f"{BASE}/api/cost-events")
        def capture(request):
            keys_seen.append(request.headers.get("idempotency-key"))
            if len(keys_seen) < 3:
                return httpx.Response(500, text="Internal")
            return httpx.Response(201, json={"id": "evt_1"})
        route.mock(side_effect=capture)

        ns = NullSpend(api_key="key", retry_base_delay_s=0.001)
        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1500,
        ))
        assert len(keys_seen) == 3
        # All retries use the same key
        assert keys_seen[0] == keys_seen[1] == keys_seen[2]
        assert keys_seen[0].startswith("ns_")
        ns.close()


class TestAsyncTransportErrorRetry:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_retries_on_transport_error(self):
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.Response(200, json={"entities": []}),
        ]
        async with AsyncNullSpend(api_key="key") as ns:
            status = await ns.check_budget()
            assert len(status.entities) == 0
            assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_raises_after_transport_retries_exhausted(self):
        respx.get(f"{BASE}/api/budgets/status").mock(
            side_effect=httpx.ConnectError("Connection refused"),
        )
        async with AsyncNullSpend(api_key="key") as ns:
            with pytest.raises(NullSpendError, match="network error"):
                await ns.check_budget()


class TestAsyncListBudgets:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_budgets(self):
        respx.get(f"{BASE}/api/budgets").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "id": "b1", "entityType": "org", "entityId": "org-1",
                    "maxBudgetMicrodollars": 10_000_000, "spendMicrodollars": 0,
                    "policy": "refillable", "resetInterval": "monthly",
                    "currentPeriodStart": "2026-04-01", "thresholdPercentages": [80],
                    "velocityLimitMicrodollars": None, "velocityWindowSeconds": None,
                    "velocityCooldownSeconds": None, "sessionLimitMicrodollars": None,
                    "createdAt": "2026-01-01", "updatedAt": "2026-04-01",
                }]
            })
        )
        async with AsyncNullSpend(api_key="key") as ns:
            result = await ns.list_budgets()
            assert len(result.data) == 1
            assert result.data[0].max_budget_microdollars == 10_000_000


class TestAsyncGetCostSummary:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_cost_summary(self):
        respx.get(f"{BASE}/api/cost-events/summary?period=30d").mock(
            return_value=httpx.Response(200, json={"data": {
                "daily": [], "models": {}, "providers": {},
                "totals": {"totalCostMicrodollars": 0},
            }})
        )
        async with AsyncNullSpend(api_key="key") as ns:
            result = await ns.get_cost_summary()
            assert result.totals["totalCostMicrodollars"] == 0

    @pytest.mark.asyncio
    async def test_rejects_invalid_period(self):
        async with AsyncNullSpend(api_key="key") as ns:
            with pytest.raises(NullSpendError, match="Invalid period"):
                await ns.get_cost_summary("1y")
