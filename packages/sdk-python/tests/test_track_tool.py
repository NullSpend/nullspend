"""Tests for track_tool decorator and track inline method."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import respx
from httpx import Response

from nullspend import NullSpend
from nullspend.types import CostEventInput


@pytest.fixture
def ns():
    """NullSpend client with mocked report_cost."""
    client = NullSpend(api_key="ns_test_sk_xxx", base_url="https://test.nullspend.dev")
    client.report_cost = MagicMock(return_value={"id": "evt_123"})
    return client


@pytest.fixture
def ns_batched():
    """NullSpend client with cost reporter (batched mode)."""
    from nullspend.types import CostReportingConfig
    with respx.mock:
        respx.post("https://test.nullspend.dev/api/cost-events/batch").mock(
            return_value=Response(200, json={"data": []})
        )
        client = NullSpend(
            api_key="ns_test_sk_xxx",
            base_url="https://test.nullspend.dev",
            cost_reporting=CostReportingConfig(batch_size=100, flush_interval_ms=60000),
        )
        yield client
        client.shutdown()


# ── track_tool decorator ──────────────────────────────────────────


class TestTrackToolDecorator:

    def test_reports_cost_event_after_function(self, ns):
        @ns.track_tool(cost=0.02, tool_name="web_search")
        def search(query: str) -> str:
            return f"results for {query}"

        result = search("python")
        assert result == "results for python"
        ns.report_cost.assert_called_once()
        event = ns.report_cost.call_args[0][0]
        assert isinstance(event, CostEventInput)
        assert event.cost_microdollars == 20_000
        assert event.tool_name == "web_search"
        assert event.event_type == "tool"

    def test_default_provider_and_model(self, ns):
        @ns.track_tool(cost=0.01, tool_name="calculator")
        def calc():
            return 42

        calc()
        event = ns.report_cost.call_args[0][0]
        assert event.provider == "tool"
        assert event.model == "custom"

    def test_custom_provider_and_model(self, ns):
        @ns.track_tool(cost=0.05, tool_name="search", provider="serper", model="google")
        def search():
            return "results"

        search()
        event = ns.report_cost.call_args[0][0]
        assert event.provider == "serper"
        assert event.model == "google"

    def test_passes_tags_and_customer(self, ns):
        @ns.track_tool(cost=0.01, tool_name="db_query", tags={"env": "prod"}, customer="cust_123")
        def query():
            return []

        query()
        event = ns.report_cost.call_args[0][0]
        assert event.tags == {"env": "prod"}
        assert event.customer == "cust_123"

    def test_passes_tool_server(self, ns):
        @ns.track_tool(cost=0.01, tool_name="mcp_tool", tool_server="github-mcp")
        def tool():
            return "ok"

        tool()
        event = ns.report_cost.call_args[0][0]
        assert event.tool_server == "github-mcp"

    def test_measures_duration(self, ns):
        @ns.track_tool(cost=0.01, tool_name="slow_tool")
        def slow():
            time.sleep(0.05)
            return "done"

        slow()
        event = ns.report_cost.call_args[0][0]
        assert event.duration_ms is not None
        assert event.duration_ms >= 40  # at least ~50ms

    def test_preserves_function_name(self, ns):
        @ns.track_tool(cost=0.01, tool_name="search")
        def my_search_function():
            return "ok"

        assert my_search_function.__name__ == "my_search_function"

    def test_preserves_args_and_kwargs(self, ns):
        @ns.track_tool(cost=0.01, tool_name="calc")
        def add(a, b, *, extra=0):
            return a + b + extra

        result = add(3, 4, extra=10)
        assert result == 17

    def test_cost_report_failure_does_not_crash(self, ns):
        ns.report_cost.side_effect = Exception("network error")

        @ns.track_tool(cost=0.01, tool_name="fragile")
        def fragile():
            return "ok"

        result = fragile()
        assert result == "ok"  # Function still returns, error swallowed

    def test_cost_conversion_precision(self, ns):
        @ns.track_tool(cost=0.001, tool_name="cheap")
        def cheap():
            return "ok"

        cheap()
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == 1000  # $0.001 = 1000 microdollars

    def test_uses_batch_reporter_when_available(self, ns_batched):
        enqueue_spy = MagicMock(wraps=ns_batched._cost_reporter.enqueue)
        ns_batched._cost_reporter.enqueue = enqueue_spy

        @ns_batched.track_tool(cost=0.02, tool_name="batch_tool")
        def tool():
            return "ok"

        tool()
        enqueue_spy.assert_called_once()
        event = enqueue_spy.call_args[0][0]
        assert event.tool_name == "batch_tool"

    def test_exception_still_reports_cost(self, ns):
        """Tool that raises still gets its cost reported."""
        @ns.track_tool(cost=0.05, tool_name="failing_tool")
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

        # Cost was still reported despite the exception
        ns.report_cost.assert_called_once()
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == 50_000
        assert event.tool_name == "failing_tool"
        assert event.tags is not None
        assert event.tags.get("_ns_error") == "true"

    def test_exception_still_reports_with_duration(self, ns):
        """Duration is measured even when function raises."""
        @ns.track_tool(cost=0.01, tool_name="slow_fail")
        def slow_fail():
            time.sleep(0.05)
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            slow_fail()

        event = ns.report_cost.call_args[0][0]
        assert event.duration_ms is not None
        assert event.duration_ms >= 40

    def test_zero_cost_tool(self, ns):
        """Zero-cost tools (free API calls tracked for observability)."""
        @ns.track_tool(cost=0, tool_name="free_api")
        def free_call():
            return "data"

        result = free_call()
        assert result == "data"
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == 0
        assert event.tool_name == "free_api"

    def test_async_function_raises_type_error(self, ns):
        """Wrapping an async function raises TypeError immediately."""
        with pytest.raises(TypeError, match="cannot wrap async function"):
            @ns.track_tool(cost=0.01, tool_name="async_tool")
            async def async_fn():
                return "ok"

    def test_error_tag_not_set_on_success(self, ns):
        """Successful calls don't get the _ns_error tag."""
        @ns.track_tool(cost=0.01, tool_name="ok_tool")
        def ok():
            return "fine"

        ok()
        event = ns.report_cost.call_args[0][0]
        assert event.tags is None or "_ns_error" not in (event.tags or {})

    def test_error_tag_merges_with_existing_tags(self, ns):
        """On failure, _ns_error tag merges with user-provided tags."""
        @ns.track_tool(cost=0.01, tool_name="tagged_fail", tags={"env": "prod"})
        def fail():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            fail()

        event = ns.report_cost.call_args[0][0]
        assert event.tags["env"] == "prod"
        assert event.tags["_ns_error"] == "true"


# ── track inline ──────────────────────────────────────────────────


class TestTrackInline:

    def test_reports_and_returns_result(self, ns):
        result = ns.track("search results", cost=0.02, tool_name="web_search")
        assert result == "search results"
        ns.report_cost.assert_called_once()
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == 20_000
        assert event.tool_name == "web_search"
        assert event.event_type == "tool"

    def test_returns_any_type(self, ns):
        assert ns.track(42, cost=0.01, tool_name="calc") == 42
        assert ns.track(None, cost=0.01, tool_name="void") is None
        assert ns.track([1, 2, 3], cost=0.01, tool_name="list") == [1, 2, 3]
        assert ns.track({"key": "val"}, cost=0.01, tool_name="dict") == {"key": "val"}

    def test_passes_optional_fields(self, ns):
        ns.track("ok", cost=0.05, tool_name="api", tool_server="external",
                 provider="http", model="rest", tags={"env": "test"}, customer="c1")
        event = ns.report_cost.call_args[0][0]
        assert event.tool_server == "external"
        assert event.provider == "http"
        assert event.model == "rest"
        assert event.tags == {"env": "test"}
        assert event.customer == "c1"

    def test_cost_report_failure_does_not_crash(self, ns):
        ns.report_cost.side_effect = Exception("fail")
        result = ns.track("ok", cost=0.01, tool_name="fragile")
        assert result == "ok"

    def test_uses_batch_reporter_when_available(self, ns_batched):
        enqueue_spy = MagicMock(wraps=ns_batched._cost_reporter.enqueue)
        ns_batched._cost_reporter.enqueue = enqueue_spy

        ns_batched.track("ok", cost=0.01, tool_name="batch_inline")
        enqueue_spy.assert_called_once()


# ── Edge cases ────────────────────────────────────────────────────


class TestTrackToolEdgeCases:

    def test_negative_cost_produces_negative_microdollars(self, ns):
        """Negative cost is allowed (refunds/credits). Passes to API for validation."""
        @ns.track_tool(cost=-0.01, tool_name="refund")
        def refund():
            return "refunded"

        refund()
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == -10_000

    def test_very_large_cost(self, ns):
        """Large cost values don't overflow."""
        @ns.track_tool(cost=10_000.0, tool_name="expensive")
        def expensive():
            return "done"

        expensive()
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == 10_000_000_000  # $10K in microdollars

    def test_track_inline_negative_cost(self, ns):
        ns.track("ok", cost=-0.05, tool_name="credit")
        event = ns.report_cost.call_args[0][0]
        assert event.cost_microdollars == -50_000

    def test_fractional_cost_precision(self, ns):
        """$0.029 → 29000 microdollars (float precision edge)."""
        @ns.track_tool(cost=0.029, tool_name="precise")
        def precise():
            return "ok"

        precise()
        event = ns.report_cost.call_args[0][0]
        # int(0.029 * 1_000_000) may be 28999 due to float representation
        assert event.cost_microdollars in (28999, 29000)
