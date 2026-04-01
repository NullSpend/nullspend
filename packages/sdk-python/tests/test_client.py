import json

import httpx
import pytest
import respx

from nullspend import NullSpend, NullSpendError, PollTimeoutError, TimeoutError, RejectedError
from nullspend.types import (
    CostEventInput,
    CreateActionInput,
    ListCostEventsOptions,
    MarkResultInput,
    NullSpendConfig,
    ProposeAndWaitOptions,
)

BASE = "https://nullspend.test"


@pytest.fixture
def ns():
    client = NullSpend(base_url=BASE, api_key="ns_test_key")
    yield client
    client.close()


class TestConfig:
    def test_requires_base_url(self):
        with pytest.raises(NullSpendError, match="base_url is required"):
            NullSpend(base_url="", api_key="key")

    def test_requires_api_key(self):
        with pytest.raises(NullSpendError, match="api_key is required"):
            NullSpend(base_url="https://example.com", api_key="")

    def test_strips_trailing_slash(self):
        client = NullSpend(base_url="https://example.com///", api_key="key")
        assert client._base_url == "https://example.com"
        client.close()


class TestReportCost:
    @respx.mock
    def test_report_cost_success(self, ns):
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(
                201,
                json={"id": "ns_evt_123", "createdAt": "2026-03-27T00:00:00Z"},
            )
        )

        result = ns.report_cost(CostEventInput(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_microdollars=1500,
        ))

        assert result["id"] == "ns_evt_123"
        req = respx.calls[0].request
        assert req.headers["x-nullspend-key"] == "ns_test_key"
        assert req.headers["nullspend-version"] == "2026-04-01"
        body = json.loads(req.content)
        assert body["provider"] == "openai"
        assert body["inputTokens"] == 100

    @respx.mock
    def test_report_cost_with_session_and_tags(self, ns):
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1", "createdAt": "2026-03-27T00:00:00Z"})
        )

        ns.report_cost(CostEventInput(
            provider="anthropic",
            model="claude-sonnet-4-5",
            input_tokens=500,
            output_tokens=200,
            cost_microdollars=3000,
            session_id="session-abc",
            tags={"env": "prod", "team": "ml"},
        ))

        body = json.loads(respx.calls[0].request.content)
        assert body["sessionId"] == "session-abc"
        assert body["tags"] == {"env": "prod", "team": "ml"}

    @respx.mock
    def test_report_cost_batch(self, ns):
        respx.post(f"{BASE}/api/cost-events/batch").mock(
            return_value=httpx.Response(201, json={"inserted": 2, "ids": ["a", "b"]})
        )

        result = ns.report_cost_batch([
            CostEventInput(provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50, cost_microdollars=1000),
            CostEventInput(provider="openai", model="gpt-4o-mini", input_tokens=200, output_tokens=80, cost_microdollars=500),
        ])

        assert result["inserted"] == 2
        body = json.loads(respx.calls[0].request.content)
        assert len(body["events"]) == 2


class TestActions:
    @respx.mock
    def test_create_action(self, ns):
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(
                201,
                json={"id": "act_1", "status": "pending", "expiresAt": "2026-03-27T01:00:00Z"},
            )
        )

        result = ns.create_action(CreateActionInput(
            agent_id="agent-1",
            action_type="send_email",
            payload={"to": "user@example.com", "subject": "Hello"},
        ))

        assert result.id == "act_1"
        assert result.status == "pending"

    @respx.mock
    def test_get_action(self, ns):
        respx.get(f"{BASE}/api/actions/act_1").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "id": "act_1",
                    "agentId": "agent-1",
                    "actionType": "send_email",
                    "status": "approved",
                    "payload": {},
                    "metadata": None,
                    "createdAt": "2026-03-27T00:00:00Z",
                    "approvedAt": "2026-03-27T00:01:00Z",
                    "rejectedAt": None,
                    "executedAt": None,
                    "expiresAt": None,
                    "expiredAt": None,
                    "approvedBy": "user-1",
                    "rejectedBy": None,
                    "result": None,
                    "errorMessage": None,
                    "environment": None,
                    "sourceFramework": None,
                }
            })
        )

        action = ns.get_action("act_1")
        assert action.status == "approved"
        assert action.approved_by == "user-1"

    @respx.mock
    def test_mark_result(self, ns):
        respx.post(f"{BASE}/api/actions/act_1/result").mock(
            return_value=httpx.Response(200, json={"id": "act_1", "status": "executed"})
        )

        result = ns.mark_result("act_1", MarkResultInput(
            status="executed",
            result={"ok": True},
        ))

        assert result["status"] == "executed"


class TestBudgets:
    @respx.mock
    def test_check_budget(self, ns):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={
                "entities": [{
                    "entityType": "api_key",
                    "entityId": "key_1",
                    "limitMicrodollars": 5000000,
                    "spendMicrodollars": 2100000,
                    "remainingMicrodollars": 2900000,
                    "policy": "strict_block",
                    "resetInterval": "monthly",
                    "currentPeriodStart": "2026-03-01T00:00:00Z",
                }]
            })
        )

        status = ns.check_budget()
        assert len(status.entities) == 1
        assert status.entities[0].remaining_microdollars == 2900000

    @respx.mock
    def test_list_budgets(self, ns):
        respx.get(f"{BASE}/api/budgets").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "id": "budget_1",
                    "entityType": "api_key",
                    "entityId": "key_1",
                    "maxBudgetMicrodollars": 5000000,
                    "spendMicrodollars": 0,
                    "policy": "strict_block",
                    "resetInterval": "monthly",
                    "currentPeriodStart": "2026-03-01T00:00:00Z",
                    "thresholdPercentages": [50, 80, 95],
                    "velocityLimitMicrodollars": None,
                    "velocityWindowSeconds": None,
                    "velocityCooldownSeconds": None,
                    "sessionLimitMicrodollars": None,
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-03-01T00:00:00Z",
                }]
            })
        )

        result = ns.list_budgets()
        assert len(result.data) == 1
        assert result.data[0].threshold_percentages == [50, 80, 95]


class TestCostEvents:
    @respx.mock
    def test_list_cost_events(self, ns):
        respx.get(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "id": "evt_1",
                    "requestId": "req-001",
                    "apiKeyId": "ns_key_abc",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cachedInputTokens": 20,
                    "reasoningTokens": 0,
                    "costMicrodollars": 1500,
                    "durationMs": 320,
                    "sessionId": "session-abc",
                    "traceId": None,
                    "source": "proxy",
                    "tags": {"env": "prod"},
                    "keyName": "prod-key",
                    "createdAt": "2026-03-27T00:00:00Z",
                }],
                "cursor": None,
            })
        )

        result = ns.list_cost_events()
        assert len(result.data) == 1
        evt = result.data[0]
        assert evt.session_id == "session-abc"
        assert evt.tags == {"env": "prod"}
        assert evt.request_id == "req-001"
        assert evt.api_key_id == "ns_key_abc"
        assert evt.cached_input_tokens == 20
        assert evt.reasoning_tokens == 0
        assert evt.source == "proxy"
        assert evt.key_name == "prod-key"

    @respx.mock
    def test_list_cost_events_with_pagination(self, ns):
        respx.get(f"{BASE}/api/cost-events?limit=5").mock(
            return_value=httpx.Response(200, json={
                "data": [],
                "cursor": {"createdAt": "2026-03-27T00:00:00Z", "id": "evt_1"},
            })
        )

        result = ns.list_cost_events(ListCostEventsOptions(limit=5))
        assert result.cursor is not None
        assert result.cursor["id"] == "evt_1"


class TestRetries:
    @respx.mock
    def test_retries_on_500(self, ns):
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(200, json={"entities": []}),
        ]

        status = ns.check_budget()
        assert len(status.entities) == 0
        assert route.call_count == 3

    @respx.mock
    def test_raises_after_max_retries(self, ns):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(NullSpendError, match="Internal Server Error"):
            ns.check_budget()

    @respx.mock
    def test_no_retry_on_400(self, ns):
        route = respx.post(f"{BASE}/api/cost-events")
        route.mock(return_value=httpx.Response(
            400,
            json={"error": {"code": "validation_error", "message": "Invalid input"}},
        ))

        with pytest.raises(NullSpendError, match="Invalid input"):
            ns.report_cost(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=0, output_tokens=0, cost_microdollars=0,
            ))

        assert route.call_count == 1

    @respx.mock
    def test_idempotency_key_sent_on_post(self, ns):
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1", "createdAt": "2026-03-27T00:00:00Z"})
        )

        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1000,
        ))

        req = respx.calls[0].request
        assert "idempotency-key" in req.headers
        assert req.headers["idempotency-key"].startswith("ns_")

    @respx.mock
    def test_no_idempotency_key_on_get(self, ns):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={"entities": []})
        )

        ns.check_budget()

        req = respx.calls[0].request
        assert "idempotency-key" not in req.headers


class TestTimeoutNaming:
    def test_poll_timeout_does_not_shadow_builtin(self):
        """PollTimeoutError is distinct from Python's builtin TimeoutError."""
        import builtins
        assert PollTimeoutError is not builtins.TimeoutError
        # The alias still works for backward compat
        assert TimeoutError is PollTimeoutError

    def test_poll_timeout_is_nullspend_error(self):
        err = PollTimeoutError("act_1", 5000)
        assert isinstance(err, NullSpendError)
        assert err.action_id == "act_1"
        assert err.timeout_ms == 5000


class TestCachedTokens:
    @respx.mock
    def test_zero_cached_tokens_not_sent(self, ns):
        """cached_input_tokens=0 (default) should not appear in the body."""
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1", "createdAt": "2026-03-27T00:00:00Z"})
        )

        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1000,
        ))

        body = json.loads(respx.calls[0].request.content)
        assert "cachedInputTokens" not in body
        assert "reasoningTokens" not in body

    @respx.mock
    def test_nonzero_cached_tokens_sent(self, ns):
        """cached_input_tokens=50 should be included in the body."""
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1", "createdAt": "2026-03-27T00:00:00Z"})
        )

        ns.report_cost(CostEventInput(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_microdollars=1000,
            cached_input_tokens=50,
            reasoning_tokens=10,
        ))

        body = json.loads(respx.calls[0].request.content)
        assert body["cachedInputTokens"] == 50
        assert body["reasoningTokens"] == 10


class TestContextManager:
    @respx.mock
    def test_context_manager_closes_client(self):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json={"entities": []})
        )

        with NullSpend(base_url=BASE, api_key="ns_test_key") as ns:
            status = ns.check_budget()
            assert len(status.entities) == 0
        # Client should be closed after exiting context — no assertion needed,
        # just verify no exception is thrown


class TestErrorParsing:
    @respx.mock
    def test_parses_error_response(self, ns):
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"code": "authentication_required", "message": "Missing API key", "details": None}},
            )
        )

        with pytest.raises(NullSpendError) as exc_info:
            ns.create_action(CreateActionInput(
                agent_id="agent-1", action_type="send_email", payload={},
            ))

        assert exc_info.value.status_code == 401
        assert exc_info.value.code == "authentication_required"
        assert "Missing API key" in str(exc_info.value)


class TestPathValidation:
    def test_get_action_rejects_path_traversal(self, ns):
        with pytest.raises(NullSpendError, match="Invalid action_id"):
            ns.get_action("../../admin/users")

    def test_get_action_rejects_slash(self, ns):
        with pytest.raises(NullSpendError, match="Invalid action_id"):
            ns.get_action("act/123")

    def test_get_action_rejects_empty(self, ns):
        with pytest.raises(NullSpendError, match="Invalid action_id"):
            ns.get_action("")

    def test_mark_result_rejects_path_traversal(self, ns):
        with pytest.raises(NullSpendError, match="Invalid action_id"):
            ns.mark_result("../evil", MarkResultInput(status="executed"))

    @respx.mock
    def test_get_action_accepts_valid_ids(self, ns):
        respx.get(f"{BASE}/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "id": "act_1", "agentId": "a", "actionType": "send_email",
                    "status": "pending", "payload": {}, "metadata": None,
                    "createdAt": "2026-03-27T00:00:00Z",
                    "approvedAt": None, "rejectedAt": None, "executedAt": None,
                    "expiresAt": None, "expiredAt": None, "approvedBy": None,
                    "rejectedBy": None, "result": None, "errorMessage": None,
                    "environment": None, "sourceFramework": None,
                }
            })
        )
        action = ns.get_action("ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert action.id == "act_1"


class TestCostSummary:
    @respx.mock
    def test_get_cost_summary(self, ns):
        respx.get(f"{BASE}/api/cost-events/summary?period=7d").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "daily": [{"date": "2026-03-27", "totalCostMicrodollars": 50000}],
                    "models": {"gpt-4o": 50000},
                    "providers": {"openai": 50000},
                    "totals": {"totalCostMicrodollars": 50000, "totalRequests": 10},
                }
            })
        )

        result = ns.get_cost_summary("7d")
        assert len(result.daily) == 1
        assert result.totals["totalRequests"] == 10

    def test_rejects_invalid_period(self, ns):
        with pytest.raises(NullSpendError, match="Invalid period"):
            ns.get_cost_summary("1y")

    def test_rejects_injection_period(self, ns):
        with pytest.raises(NullSpendError, match="Invalid period"):
            ns.get_cost_summary("7d&evil=true")


class TestNullSpendConfig:
    @respx.mock
    def test_config_object_path(self):
        respx.get("https://config-test.com/api/budgets/status").mock(
            return_value=httpx.Response(200, json={"entities": []})
        )

        config = NullSpendConfig(
            base_url="https://config-test.com",
            api_key="ns_config_key",
            api_version="2026-04-01",
            request_timeout_s=10.0,
            max_retries=1,
            retry_base_delay_s=0.1,
        )
        client = NullSpend(config=config)
        status = client.check_budget()
        assert len(status.entities) == 0

        req = respx.calls[0].request
        assert req.headers["x-nullspend-key"] == "ns_config_key"
        client.close()


class TestResponseTypeCheck:
    @respx.mock
    def test_raises_on_non_dict_json_response(self, ns):
        """Server returning a JSON array should raise NullSpendError, not AttributeError."""
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json=[])
        )

        with pytest.raises(NullSpendError, match="unexpected JSON type"):
            ns.check_budget()

    @respx.mock
    def test_raises_on_scalar_json_response(self, ns):
        respx.get(f"{BASE}/api/budgets/status").mock(
            return_value=httpx.Response(200, json="ok")
        )

        with pytest.raises(NullSpendError, match="unexpected JSON type"):
            ns.check_budget()


class TestReportCostAllFields:
    @respx.mock
    def test_all_optional_fields_sent(self, ns):
        """Verify every optional field appears in the request body when set."""
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(201, json={"id": "evt_1", "createdAt": "2026-03-27T00:00:00Z"})
        )

        ns.report_cost(CostEventInput(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_microdollars=1500,
            cached_input_tokens=20,
            reasoning_tokens=5,
            duration_ms=320,
            session_id="session-xyz",
            trace_id="a1b2c3d4e5f67890a1b2c3d4e5f67890",
            event_type="llm",
            tool_name="search",
            tool_server="rag-server",
            tags={"env": "prod"},
        ))

        body = json.loads(respx.calls[0].request.content)
        assert body["cachedInputTokens"] == 20
        assert body["reasoningTokens"] == 5
        assert body["durationMs"] == 320
        assert body["sessionId"] == "session-xyz"
        assert body["traceId"] == "a1b2c3d4e5f67890a1b2c3d4e5f67890"
        assert body["eventType"] == "llm"
        assert body["toolName"] == "search"
        assert body["toolServer"] == "rag-server"
        assert body["tags"] == {"env": "prod"}


class TestNetworkErrors:
    @respx.mock
    def test_retries_on_transport_error(self, ns):
        route = respx.get(f"{BASE}/api/budgets/status")
        route.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.Response(200, json={"entities": []}),
        ]

        status = ns.check_budget()
        assert len(status.entities) == 0
        assert route.call_count == 2

    @respx.mock
    def test_raises_after_transport_error_retries_exhausted(self, ns):
        respx.get(f"{BASE}/api/budgets/status").mock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with pytest.raises(NullSpendError, match="network error"):
            ns.check_budget()


class TestProposeAndWait:
    @respx.mock
    def test_happy_path(self, ns):
        # create_action
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_1", "status": "pending", "expiresAt": None,
            })
        )
        # get_action (poll) — return approved immediately
        respx.get(f"{BASE}/api/actions/act_1").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "id": "act_1", "agentId": "agent-1", "actionType": "send_email",
                    "status": "approved", "payload": {}, "metadata": None,
                    "createdAt": "2026-03-27T00:00:00Z",
                    "approvedAt": "2026-03-27T00:01:00Z",
                    "rejectedAt": None, "executedAt": None,
                    "expiresAt": None, "expiredAt": None,
                    "approvedBy": "user-1", "rejectedBy": None,
                    "result": None, "errorMessage": None,
                    "environment": None, "sourceFramework": None,
                }
            })
        )
        # mark_result (executing)
        result_route = respx.post(f"{BASE}/api/actions/act_1/result")
        result_route.side_effect = [
            httpx.Response(200, json={"id": "act_1", "status": "executing"}),
            httpx.Response(200, json={"id": "act_1", "status": "executed"}),
        ]

        executed = False

        def execute(context):
            nonlocal executed
            assert isinstance(context, dict)
            assert context["action_id"] == "act_1"
            executed = True
            return {"output": "done"}

        result = ns.propose_and_wait(ProposeAndWaitOptions(
            agent_id="agent-1",
            action_type="send_email",
            payload={"to": "test@example.com"},
            execute=execute,
            poll_interval_s=0.01,
            timeout_s=5.0,
        ))

        assert executed
        assert result == {"output": "done"}

    @respx.mock
    def test_rejected_raises(self, ns):
        respx.post(f"{BASE}/api/actions").mock(
            return_value=httpx.Response(201, json={
                "id": "act_1", "status": "pending", "expiresAt": None,
            })
        )
        respx.get(f"{BASE}/api/actions/act_1").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "id": "act_1", "agentId": "a", "actionType": "send_email",
                    "status": "rejected", "payload": {}, "metadata": None,
                    "createdAt": "2026-03-27T00:00:00Z",
                    "approvedAt": None, "rejectedAt": "2026-03-27T00:01:00Z",
                    "executedAt": None, "expiresAt": None, "expiredAt": None,
                    "approvedBy": None, "rejectedBy": "user-1",
                    "result": None, "errorMessage": None,
                    "environment": None, "sourceFramework": None,
                }
            })
        )

        with pytest.raises(RejectedError, match="rejected"):
            ns.propose_and_wait(ProposeAndWaitOptions(
                agent_id="agent-1",
                action_type="send_email",
                payload={},
                execute=lambda ctx: None,
                poll_interval_s=0.01,
                timeout_s=5.0,
            ))
