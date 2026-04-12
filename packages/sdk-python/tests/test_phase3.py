"""Phase 3 tests: policy cache, tracked client, customer sessions, enforcement.

Covers policy cache TTL/dedup/fail-open, tracked transport with provider parsers,
429 denial interception, TeeByteStream, proxy detection, customer sessions,
ns.openai/ns.anthropic shorthands.
"""
from __future__ import annotations

import json
import time
import pytest
import httpx
import respx

from nullspend import (
    NullSpend,
    NullSpendError,
    BudgetExceededError,
    MandateViolationError,
    SessionLimitExceededError,
    VelocityExceededError,
    TagBudgetExceededError,
    CostEventInput,
    CostReportingConfig,
    CustomerSession,
    create_tracked_client,
)
from nullspend._policy_cache import (
    PolicyCache,
    PolicyResponse,
    PolicyBudget,
    MandateResult,
    BudgetResult,
    _parse_policy_response,
)
from nullspend._tracked_client import (
    TrackedTransport,
    TeeByteStream,
    _is_tracked_route,
    _is_proxied,
    _parse_denial_payload,
    _dispatch_denial,
    _estimate_cost_microdollars,
    _extract_model_from_body,
    _is_streaming_request,
    _is_streaming_response,
    _extract_openai_usage,
    _extract_anthropic_usage,
    _safe_denied,
)

BASE = "https://nullspend.dev"


# ---- Provider Parsers ----


class TestProviderParsers:
    def test_openai_tracked_routes(self):
        assert _is_tracked_route("openai", "https://api.openai.com/v1/chat/completions", "POST")
        assert _is_tracked_route("openai", "https://api.openai.com/v1/completions", "POST")
        assert _is_tracked_route("openai", "https://api.openai.com/v1/embeddings", "POST")
        assert not _is_tracked_route("openai", "https://api.openai.com/v1/models", "GET")
        assert not _is_tracked_route("openai", "https://api.openai.com/v1/chat/completions", "GET")

    def test_anthropic_tracked_routes(self):
        assert _is_tracked_route("anthropic", "https://api.anthropic.com/v1/messages", "POST")
        assert not _is_tracked_route("anthropic", "https://api.anthropic.com/v1/messages", "GET")
        assert not _is_tracked_route("anthropic", "https://api.anthropic.com/v1/models", "POST")

    def test_extract_model_from_body(self):
        assert _extract_model_from_body(b'{"model": "gpt-4o"}') == "gpt-4o"
        assert _extract_model_from_body(b'{"model": "claude-sonnet-4-5"}') == "claude-sonnet-4-5"
        assert _extract_model_from_body(b'{}') is None
        assert _extract_model_from_body(None) is None
        assert _extract_model_from_body(b'invalid json') is None

    def test_is_streaming_request(self):
        assert _is_streaming_request(b'{"stream": true}')
        assert not _is_streaming_request(b'{"stream": false}')
        assert not _is_streaming_request(b'{}')
        assert not _is_streaming_request(None)

    def test_extract_openai_usage(self):
        assert _extract_openai_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 50}}) is not None
        assert _extract_openai_usage({"usage": {}}) is None
        assert _extract_openai_usage({}) is None

    def test_extract_anthropic_usage(self):
        usage, detail = _extract_anthropic_usage({"usage": {"input_tokens": 100, "output_tokens": 50}})
        assert usage is not None
        assert detail is None
        usage2, detail2 = _extract_anthropic_usage({"usage": {
            "input_tokens": 100, "output_tokens": 50,
            "cache_creation": {"ephemeral_5m_input_tokens": 200},
        }})
        assert usage2 is not None
        assert detail2 is not None


# ---- Policy Cache ----


class TestPolicyCache:
    def test_basic_fetch_and_cache(self):
        call_count = [0]
        def fetch():
            call_count[0] += 1
            return {"budget": None, "restrictions_active": False}

        pc = PolicyCache(fetch_fn=fetch, ttl_s=60.0)
        p1 = pc.get_policy()
        p2 = pc.get_policy()
        assert p1 is not None
        assert p1 is p2  # Same cached object
        assert call_count[0] == 1  # Only fetched once

    def test_ttl_expiry(self):
        call_count = [0]
        def fetch():
            call_count[0] += 1
            return {"budget": None, "restrictions_active": False}

        pc = PolicyCache(fetch_fn=fetch, ttl_s=0.01)
        pc.get_policy()
        time.sleep(0.02)
        pc.get_policy()
        assert call_count[0] == 2

    def test_fail_open_returns_stale(self):
        call_count = [0]
        def fetch():
            call_count[0] += 1
            if call_count[0] > 1:
                raise ConnectionError("network down")
            return {"budget": None, "restrictions_active": False}

        pc = PolicyCache(fetch_fn=fetch, ttl_s=0.01)
        p1 = pc.get_policy()
        assert p1 is not None
        time.sleep(0.02)
        p2 = pc.get_policy()
        assert p2 is p1  # Stale cache returned

    def test_fail_open_returns_none_if_no_cache(self):
        def fetch():
            raise ConnectionError("never works")

        errors = []
        pc = PolicyCache(fetch_fn=fetch, on_error=lambda e: errors.append(e))
        result = pc.get_policy()
        assert result is None
        assert len(errors) == 1

    def test_invalidate_clears_cache(self):
        call_count = [0]
        def fetch():
            call_count[0] += 1
            return {"budget": None, "restrictions_active": False}

        pc = PolicyCache(fetch_fn=fetch, ttl_s=60.0)
        pc.get_policy()
        assert call_count[0] == 1
        pc.invalidate()
        pc.get_policy()
        assert call_count[0] == 2

    def test_check_mandate_no_policy(self):
        pc = PolicyCache(fetch_fn=lambda: {"budget": None}, ttl_s=60.0)
        result = pc.check_mandate("openai", "gpt-4o")
        assert result.allowed is True

    def test_check_mandate_allowed(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "allowed_providers": ["openai", "anthropic"],
            "allowed_models": ["gpt-4o", "claude-sonnet-4-5"],
        }, ttl_s=60.0)
        pc.get_policy()
        assert pc.check_mandate("openai", "gpt-4o").allowed is True

    def test_check_mandate_provider_denied(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "allowed_providers": ["openai"],
        }, ttl_s=60.0)
        pc.get_policy()
        result = pc.check_mandate("anthropic", "claude-sonnet-4-5")
        assert result.allowed is False
        assert result.mandate == "allowed_providers"
        assert result.requested == "anthropic"

    def test_check_mandate_model_denied(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "allowed_models": ["gpt-4o-mini"],
        }, ttl_s=60.0)
        pc.get_policy()
        result = pc.check_mandate("openai", "gpt-5")
        assert result.allowed is False
        assert result.mandate == "allowed_models"

    def test_check_budget_allowed(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "budget": {
                "remaining_microdollars": 5_000_000,
                "max_microdollars": 10_000_000,
                "spend_microdollars": 5_000_000,
                "period_end": None,
                "entity_type": "org",
                "entity_id": "org-1",
            },
        }, ttl_s=60.0)
        pc.get_policy()
        result = pc.check_budget(1_000_000)
        assert result.allowed is True
        assert result.remaining == 5_000_000

    def test_check_budget_denied(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "budget": {
                "remaining_microdollars": 100,
                "max_microdollars": 10_000_000,
                "spend_microdollars": 9_999_900,
                "period_end": None,
                "entity_type": "org",
                "entity_id": "org-1",
            },
        }, ttl_s=60.0)
        pc.get_policy()
        result = pc.check_budget(1_000)
        assert result.allowed is False
        assert result.remaining == 100
        assert result.entity_type == "org"

    def test_get_session_limit(self):
        pc = PolicyCache(fetch_fn=lambda: {
            "session_limit_microdollars": 5_000_000,
        }, ttl_s=60.0)
        pc.get_policy()
        assert pc.get_session_limit() == 5_000_000

    def test_get_session_limit_none(self):
        pc = PolicyCache(fetch_fn=lambda: {}, ttl_s=60.0)
        pc.get_policy()
        assert pc.get_session_limit() is None


# ---- Proxy Detection ----


class TestProxyDetection:
    def test_url_origin_match(self):
        assert _is_proxied("https://proxy.nullspend.dev/v1/chat/completions", "https://proxy.nullspend.dev", None)

    def test_url_origin_mismatch(self):
        assert not _is_proxied("https://api.openai.com/v1/chat/completions", "https://proxy.nullspend.dev", None)

    def test_port_strict(self):
        assert not _is_proxied("https://proxy.nullspend.dev:8443/v1/chat", "https://proxy.nullspend.dev", None)

    def test_no_proxy_url(self):
        assert not _is_proxied("https://api.openai.com/v1/chat", None, None)

    def test_header_fallback(self):
        assert _is_proxied("https://custom.proxy.com/v1/chat", "https://proxy.nullspend.dev",
                          {"x-nullspend-key": "ns_live_sk_..."})


# ---- Denial Parsing ----


class TestDenialParsing:
    def test_parses_budget_exceeded(self):
        response = httpx.Response(
            429,
            headers={"x-nullspend-denied": "1"},
            json={"error": {
                "code": "budget_exceeded",
                "message": "Budget exceeded",
                "details": {"entity_type": "org", "entity_id": "org-1",
                           "remaining_microdollars": 0,
                           "budget_limit_microdollars": 10_000_000,
                           "budget_spend_microdollars": 10_000_000},
                "upgrade_url": "https://nullspend.dev/upgrade",
            }},
        )
        parsed = _parse_denial_payload(response)
        assert parsed is not None
        assert parsed["code"] == "budget_exceeded"
        assert parsed["upgrade_url"] == "https://nullspend.dev/upgrade"

    def test_returns_none_without_denied_header(self):
        response = httpx.Response(429, json={"error": {"code": "rate_limited"}})
        assert _parse_denial_payload(response) is None

    def test_returns_none_for_malformed_body(self):
        response = httpx.Response(429, headers={"x-nullspend-denied": "1"}, text="not json")
        assert _parse_denial_payload(response) is None

    def test_dispatch_budget_exceeded(self):
        with pytest.raises(BudgetExceededError):
            _dispatch_denial({
                "code": "budget_exceeded",
                "details": {"remaining_microdollars": 0},
                "upgrade_url": "https://nullspend.dev/upgrade",
                "retry_after_seconds": None,
            }, None, None)

    def test_dispatch_velocity_exceeded(self):
        with pytest.raises(VelocityExceededError):
            _dispatch_denial({
                "code": "velocity_exceeded",
                "details": {"limitMicrodollars": 1000, "windowSeconds": 60},
                "upgrade_url": None,
                "retry_after_seconds": 30,
            }, None, None)

    def test_dispatch_session_limit(self):
        with pytest.raises(SessionLimitExceededError):
            _dispatch_denial({
                "code": "session_limit_exceeded",
                "details": {"session_spend_microdollars": 5000, "session_limit_microdollars": 5000},
                "upgrade_url": None,
                "retry_after_seconds": None,
            }, None, None)

    def test_dispatch_tag_budget(self):
        with pytest.raises(TagBudgetExceededError):
            _dispatch_denial({
                "code": "tag_budget_exceeded",
                "details": {"tag_key": "env", "tag_value": "prod"},
                "upgrade_url": None,
                "retry_after_seconds": None,
            }, None, None)

    def test_dispatch_unknown_code_surfaces_drift(self):
        errors = []
        _dispatch_denial({
            "code": "new_unknown_code",
            "details": {},
            "upgrade_url": None,
            "retry_after_seconds": None,
        }, None, lambda e: errors.append(e))
        assert len(errors) == 1
        assert "Unknown denial code" in str(errors[0])


# ---- Safe Denied ----


class TestSafeDenied:
    def test_calls_callback(self):
        reasons = []
        _safe_denied(lambda r: reasons.append(r), {"type": "budget"}, None)
        assert len(reasons) == 1

    def test_swallows_callback_error(self):
        def bad_callback(r):
            raise ValueError("boom")
        errors = []
        _safe_denied(bad_callback, {"type": "budget"}, lambda e: errors.append(e))
        assert len(errors) == 1

    def test_none_callback_is_noop(self):
        _safe_denied(None, {"type": "budget"}, None)  # Should not raise


# ---- Cost Estimation ----


class TestCostEstimation:
    def test_known_model(self):
        est = _estimate_cost_microdollars("openai", "gpt-4o", b'{"max_tokens": 1000}')
        assert est > 0

    def test_unknown_model(self):
        assert _estimate_cost_microdollars("openai", "unknown", None) == 0

    def test_no_model(self):
        assert _estimate_cost_microdollars("openai", None, None) == 0

    def test_default_max_tokens(self):
        est = _estimate_cost_microdollars("openai", "gpt-4o", b'{}')
        assert est > 0  # Uses default 4096


# ---- Customer Session ----


class TestCustomerSession:
    def test_creates_session(self):
        ns = NullSpend(api_key="test")
        session = ns.customer("acme")
        assert isinstance(session, CustomerSession)
        assert session.customer_id == "acme"
        assert session.openai is not None
        assert session.anthropic is not None
        ns.close()

    def test_validates_customer_id(self):
        ns = NullSpend(api_key="test")
        with pytest.raises(NullSpendError, match="invalid characters"):
            ns.customer("acme/evil")
        ns.close()

    def test_caches_per_provider(self):
        ns = NullSpend(api_key="test")
        session = ns.customer("acme")
        assert session.openai is session.openai  # Same object (property is cached on session)
        ns.close()


# ---- ns.openai / ns.anthropic shorthands ----


class TestClientShorthands:
    def test_openai_shorthand(self):
        ns = NullSpend(api_key="test")
        client = ns.openai
        assert isinstance(client, httpx.Client)
        assert ns.openai is client  # Cached
        ns.close()

    def test_anthropic_shorthand(self):
        ns = NullSpend(api_key="test")
        client = ns.anthropic
        assert isinstance(client, httpx.Client)
        assert ns.anthropic is client
        ns.close()


# ---- Proxy URL Validation ----


class TestProxyUrlValidation:
    def test_valid_https(self):
        ns = NullSpend(api_key="test", proxy_url="https://proxy.nullspend.dev")
        assert ns._proxy_url == "https://proxy.nullspend.dev"
        ns.close()

    def test_valid_http(self):
        ns = NullSpend(api_key="test", proxy_url="http://localhost:8080")
        assert ns._proxy_url == "http://localhost:8080"
        ns.close()

    def test_rejects_ftp(self):
        with pytest.raises(NullSpendError, match="http or https"):
            NullSpend(api_key="test", proxy_url="ftp://bad.com")

    def test_strips_trailing_slash(self):
        ns = NullSpend(api_key="test", proxy_url="https://proxy.nullspend.dev/")
        assert ns._proxy_url == "https://proxy.nullspend.dev"
        ns.close()

    def test_env_var_fallback(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"NULLSPEND_PROXY_URL": "https://env-proxy.example.com"}):
            ns = NullSpend(api_key="test")
            assert ns._proxy_url == "https://env-proxy.example.com"
            ns.close()


# ---- Parse Policy Response ----


class TestParsePolicyResponse:
    def test_full_response(self):
        policy = _parse_policy_response({
            "budget": {
                "remaining_microdollars": 5_000_000,
                "max_microdollars": 10_000_000,
                "spend_microdollars": 5_000_000,
                "period_end": "2026-05-01",
                "entity_type": "org",
                "entity_id": "org-1",
            },
            "allowed_models": ["gpt-4o"],
            "allowed_providers": ["openai"],
            "restrictions_active": True,
            "session_limit_microdollars": 1_000_000,
        })
        assert policy.budget is not None
        assert policy.budget.remaining_microdollars == 5_000_000
        assert policy.allowed_models == ["gpt-4o"]
        assert policy.restrictions_active is True
        assert policy.session_limit_microdollars == 1_000_000

    def test_empty_response(self):
        policy = _parse_policy_response({})
        assert policy.budget is None
        assert policy.allowed_models is None
        assert policy.restrictions_active is False


# ---- TrackedTransport end-to-end ----


class TestTrackedTransportE2E:
    def _make_transport(self, handler, provider="openai", **kwargs):
        """Helper to create a TrackedTransport with a mock inner transport."""
        mock_transport = httpx.MockTransport(handler)
        return TrackedTransport(
            transport=mock_transport,
            provider=provider,
            **kwargs,
        )

    def test_header_injection(self):
        """Customer, tags, traceId, actionId headers are injected."""
        captured = {}
        def handler(request):
            captured.update(dict(request.headers))
            return httpx.Response(200, json={"id": "msg_1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        transport = self._make_transport(
            handler,
            customer="acme",
            tags={"env": "prod"},
            trace_id="trace-123",
            action_id="act-456",
        )
        client = httpx.Client(transport=transport)
        client.post("https://api.openai.com/v1/chat/completions",
                     json={"model": "gpt-4o", "messages": []})

        assert captured["x-nullspend-customer"] == "acme"
        assert json.loads(captured["x-nullspend-tags"]) == {"env": "prod"}
        assert captured["x-nullspend-traceid"] == "trace-123"
        assert captured["x-nullspend-actionid"] == "act-456"
        client.close()

    def test_non_tracked_route_passes_through(self):
        """Non-tracked routes (GET, /models) are passed through without modification."""
        def handler(request):
            return httpx.Response(200, json={"models": []})

        transport = self._make_transport(handler, customer="acme")
        client = httpx.Client(transport=transport)
        resp = client.get("https://api.openai.com/v1/models")
        assert resp.status_code == 200
        # Customer header should still be injected (headers are added for ALL requests)
        client.close()

    def test_non_streaming_cost_extraction(self):
        """Non-streaming response extracts usage and queues cost event."""
        queued = []
        def handler(request):
            return httpx.Response(200, json={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            })

        transport = self._make_transport(
            handler,
            queue_cost=lambda evt: queued.append(evt),
        )
        client = httpx.Client(transport=transport)
        resp = client.post("https://api.openai.com/v1/chat/completions",
                          json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]})
        assert resp.status_code == 200

        assert len(queued) == 1
        assert queued[0].provider == "openai"
        assert queued[0].model == "gpt-4o"
        assert queued[0].input_tokens == 100
        assert queued[0].output_tokens == 50
        assert queued[0].cost_microdollars > 0
        client.close()

    def test_streaming_cost_extraction_via_tee(self):
        """Streaming response wraps stream with TeeByteStream for cost extraction."""
        queued = []

        class SSEStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hi"}}]}\n\n'
                yield b'data: {"usage": {"prompt_tokens": 50, "completion_tokens": 20}}\n\n'
                yield b'data: [DONE]\n\n'

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=SSEStream(),
            )

        transport = self._make_transport(
            handler,
            queue_cost=lambda evt: queued.append(evt),
        )
        client = httpx.Client(transport=transport)
        resp = client.post("https://api.openai.com/v1/chat/completions",
                          json={"model": "gpt-4o", "messages": [], "stream": True})

        # Consumer must iterate the stream to trigger cost extraction
        chunks = list(resp.iter_bytes())
        assert len(chunks) >= 1  # May be combined into fewer chunks
        # Cost event should be queued after stream completes
        assert len(queued) == 1
        assert queued[0].input_tokens == 50
        assert queued[0].output_tokens == 20
        client.close()

    def test_proxy_429_interception(self):
        """Proxy 429 with X-NullSpend-Denied header raises BudgetExceededError."""
        def handler(request):
            return httpx.Response(429, headers={"x-nullspend-denied": "1"}, json={
                "error": {
                    "code": "budget_exceeded",
                    "message": "Budget exceeded",
                    "details": {"remaining_microdollars": 0},
                },
            })

        transport = self._make_transport(
            handler,
            proxy_url="https://api.openai.com",
            enforcement=True,
        )
        client = httpx.Client(transport=transport)
        with pytest.raises(BudgetExceededError):
            client.post("https://api.openai.com/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": []})
        client.close()

    def test_upstream_429_not_intercepted(self):
        """Upstream 429 (no X-NullSpend-Denied) passes through as-is."""
        def handler(request):
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        transport = self._make_transport(
            handler,
            proxy_url="https://api.openai.com",
            enforcement=True,
        )
        client = httpx.Client(transport=transport)
        resp = client.post("https://api.openai.com/v1/chat/completions",
                          json={"model": "gpt-4o", "messages": []})
        assert resp.status_code == 429  # Passed through, no exception
        client.close()

    def test_stream_injection_adds_include_usage(self):
        """OpenAI streaming requests get stream_options.include_usage injected."""
        captured_body = {}
        def handler(request):
            captured_body["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "x", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        transport = self._make_transport(handler)
        client = httpx.Client(transport=transport)
        client.post("https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [], "stream": True})

        assert captured_body["body"]["stream_options"]["include_usage"] is True
        client.close()

    def test_session_spend_accumulation(self):
        """Session spend accumulates across multiple requests."""
        queued = []
        call_count = [0]
        def handler(request):
            call_count[0] += 1
            return httpx.Response(200, json={
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            })

        transport = self._make_transport(
            handler,
            session_id="sess-1",
            queue_cost=lambda evt: queued.append(evt),
        )
        client = httpx.Client(transport=transport)

        for _ in range(3):
            client.post("https://api.openai.com/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": []})

        assert len(queued) == 3
        assert transport._session_spend > 0
        # Each request costs the same, so spend should be 3x a single event
        assert transport._session_spend == queued[0].cost_microdollars * 3
        client.close()

    def test_enforcement_mandate_block(self):
        """Mandate violation blocks the request before it reaches the transport."""
        from nullspend._policy_cache import PolicyCache

        pc = PolicyCache(fetch_fn=lambda: {
            "allowed_models": ["gpt-4o-mini"],
        }, ttl_s=60.0)
        pc.get_policy()

        def handler(request):
            pytest.fail("Transport should not be called when mandate blocks")

        transport = self._make_transport(
            handler,
            enforcement=True,
            policy_cache=pc,
        )
        client = httpx.Client(transport=transport)
        with pytest.raises(MandateViolationError):
            client.post("https://api.openai.com/v1/chat/completions",
                       json={"model": "gpt-5", "messages": []})
        client.close()

    def test_enforcement_budget_block(self):
        """Budget exceeded blocks the request before it reaches the transport."""
        from nullspend._policy_cache import PolicyCache

        pc = PolicyCache(fetch_fn=lambda: {
            "budget": {
                "remaining_microdollars": 1,  # Almost nothing left
                "max_microdollars": 100,
                "spend_microdollars": 99,
                "period_end": None,
                "entity_type": "org",
                "entity_id": "org-1",
            },
        }, ttl_s=60.0)
        pc.get_policy()

        def handler(request):
            pytest.fail("Transport should not be called when budget blocks")

        transport = self._make_transport(
            handler,
            enforcement=True,
            policy_cache=pc,
        )
        client = httpx.Client(transport=transport)
        with pytest.raises(BudgetExceededError):
            client.post("https://api.openai.com/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": []})
        client.close()

    def test_error_response_not_tracked(self):
        """Non-2xx responses are not cost-tracked."""
        queued = []
        def handler(request):
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        transport = self._make_transport(
            handler,
            queue_cost=lambda evt: queued.append(evt),
        )
        client = httpx.Client(transport=transport)
        resp = client.post("https://api.openai.com/v1/chat/completions",
                          json={"model": "gpt-4o", "messages": []})
        assert resp.status_code == 400
        assert len(queued) == 0  # No cost event for errors
        client.close()

    def test_anthropic_non_streaming_extraction(self):
        """Anthropic non-streaming response extracts usage correctly."""
        queued = []
        def handler(request):
            return httpx.Response(200, json={
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "usage": {"input_tokens": 200, "output_tokens": 100},
            })

        transport = self._make_transport(
            handler,
            provider="anthropic",
            queue_cost=lambda evt: queued.append(evt),
        )
        client = httpx.Client(transport=transport)
        client.post("https://api.anthropic.com/v1/messages",
                    json={"model": "claude-sonnet-4-5", "messages": []})

        assert len(queued) == 1
        assert queued[0].provider == "anthropic"
        assert queued[0].input_tokens == 200
        client.close()


# ---- NullSpend.close() cleanup ----


class TestCloseCleanup:
    def test_close_closes_tracked_clients(self):
        ns = NullSpend(api_key="test")
        openai_client = ns.openai
        anthropic_client = ns.anthropic
        ns.close()
        # After close, the tracked clients should be closed
        # httpx.Client raises RuntimeError if used after close
        with pytest.raises(RuntimeError):
            openai_client.get("https://example.com")


# ---- _queue_cost_direct observability ----


class TestQueueCostDirectLogging:
    @respx.mock
    def test_logs_first_error(self, caplog):
        import logging
        respx.post(f"{BASE}/api/cost-events").mock(
            return_value=httpx.Response(500, json={"error": {"message": "Internal"}})
        )
        ns = NullSpend(api_key="key")
        NullSpend._direct_cost_error_logged = False  # Reset class-level flag
        with caplog.at_level(logging.WARNING, logger="nullspend"):
            ns._queue_cost_direct(CostEventInput(
                provider="openai", model="gpt-4o",
                input_tokens=100, output_tokens=50, cost_microdollars=1500,
            ))
        assert any("Failed to report cost event" in r.message for r in caplog.records)
        ns.close()
        NullSpend._direct_cost_error_logged = False  # Clean up
