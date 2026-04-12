"""Tracked httpx transport for automatic cost tracking and enforcement.

Custom httpx.BaseTransport that wraps the real transport to:
1. Inject NullSpend headers (customer, tags, traceId, actionId)
2. Run client-side enforcement (mandate, budget, session limit)
3. Intercept proxy 429 denials (X-NullSpend-Denied header)
4. Extract usage from responses (JSON or SSE) and queue cost events
5. Track per-session spend for session limit enforcement

Provider parsers are merged here as private functions per eng review.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from nullspend._cost_calculator import (
    calculate_openai_cost_event,
    calculate_anthropic_cost_event,
    get_model_pricing,
)
from nullspend._sse_parser import SSEAccumulator
from nullspend.errors import (
    NullSpendError,
    BudgetExceededError,
    MandateViolationError,
    SessionLimitExceededError,
    VelocityExceededError,
    TagBudgetExceededError,
)
from nullspend.types import CostEventInput

logger = logging.getLogger("nullspend")

TrackedProvider = str  # "openai" | "anthropic"

# Default cost estimate input tokens when body can't be parsed
_DEFAULT_ESTIMATE_INPUT_TOKENS = 1000
_DEFAULT_MAX_TOKENS = 4096


# ---- Provider parsers (merged per eng review) ----


def _is_tracked_route(provider: str, url: str, method: str) -> bool:
    """Check if a request URL/method is a tracked API route."""
    if method.upper() != "POST":
        return False
    path = urlparse(url).path
    if provider == "openai":
        return path.endswith(("/chat/completions", "/completions", "/embeddings"))
    elif provider == "anthropic":
        return path.endswith("/messages")
    return False


def _extract_model_from_body(body: bytes | str | None) -> str | None:
    if not body:
        return None
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        data = json.loads(text)
        if isinstance(data, dict):
            model = data.get("model")
            return str(model) if model else None
    except Exception:
        pass
    return None


def _is_streaming_request(body: bytes | str | None) -> bool:
    if not body:
        return False
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        data = json.loads(text)
        return isinstance(data, dict) and data.get("stream") is True
    except Exception:
        return False


def _is_streaming_response(response: httpx.Response) -> bool:
    ct = response.headers.get("content-type", "")
    return "text/event-stream" in ct


def _extract_openai_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usage")
    if isinstance(usage, dict) and "prompt_tokens" in usage:
        return usage
    return None


def _extract_anthropic_usage(data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    usage = data.get("usage")
    cache_detail = None
    if isinstance(usage, dict) and "input_tokens" in usage:
        cache_creation = usage.get("cache_creation")
        if isinstance(cache_creation, dict):
            cache_detail = cache_creation
        return usage, cache_detail
    return None, None


# ---- Denial parsing ----


def _to_finite_number(value: Any) -> int | None:
    """Safely coerce a value to a finite integer."""
    if isinstance(value, (int, float)):
        import math
        if math.isfinite(value):
            return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
            import math
            if math.isfinite(parsed):
                return int(parsed)
        except (ValueError, TypeError):
            pass
    return None


def _parse_denial_payload(response: httpx.Response) -> dict[str, Any] | None:
    """Parse a NullSpend denial from a 429 response.

    Returns None if the response is not a NullSpend denial (upstream provider 429).
    Gated on X-NullSpend-Denied: 1 header.
    """
    if response.headers.get("x-nullspend-denied") != "1":
        return None

    try:
        data = response.json()
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    error = data.get("error")
    if not isinstance(error, dict):
        return None

    code = error.get("code")
    if not isinstance(code, str):
        return None

    details = error.get("details")
    if details is not None and not isinstance(details, dict):
        details = None

    upgrade_url = error.get("upgrade_url")
    if not isinstance(upgrade_url, str):
        upgrade_url = None

    retry_after = None
    ra_header = response.headers.get("retry-after")
    if ra_header:
        try:
            val = int(ra_header)
            if val >= 0:
                retry_after = val
        except (ValueError, TypeError):
            pass

    return {
        "code": code,
        "details": details,
        "retry_after_seconds": retry_after,
        "upgrade_url": upgrade_url,
    }


def _dispatch_denial(
    parsed: dict[str, Any],
    on_denied: Callable | None,
    on_cost_error: Callable | None,
) -> None:
    """Dispatch a parsed denial to the appropriate typed error."""
    code = parsed["code"]
    details = parsed.get("details") or {}
    upgrade_url = parsed.get("upgrade_url")
    retry_after = parsed.get("retry_after_seconds")

    if code == "budget_exceeded":
        remaining = _to_finite_number(details.get("remaining_microdollars")) or 0
        _safe_denied(on_denied, {
            "type": "budget",
            "remaining": remaining,
            "entity_type": details.get("entity_type"),
            "entity_id": details.get("entity_id"),
            "limit": _to_finite_number(details.get("budget_limit_microdollars")),
            "spend": _to_finite_number(details.get("budget_spend_microdollars")),
        }, on_cost_error)
        raise BudgetExceededError(
            remaining_microdollars=remaining,
            entity_type=details.get("entity_type"),
            entity_id=details.get("entity_id"),
            limit_microdollars=_to_finite_number(details.get("budget_limit_microdollars")),
            spend_microdollars=_to_finite_number(details.get("budget_spend_microdollars")),
            upgrade_url=upgrade_url,
        )

    elif code == "customer_budget_exceeded":
        remaining = _to_finite_number(details.get("remaining_microdollars")) or 0
        _safe_denied(on_denied, {
            "type": "budget",
            "remaining": remaining,
            "entity_type": "customer",
            "entity_id": details.get("customer_id"),
            "limit": _to_finite_number(details.get("budget_limit_microdollars")),
            "spend": _to_finite_number(details.get("budget_spend_microdollars")),
        }, on_cost_error)
        raise BudgetExceededError(
            remaining_microdollars=remaining,
            entity_type="customer",
            entity_id=details.get("customer_id"),
            limit_microdollars=_to_finite_number(details.get("budget_limit_microdollars")),
            spend_microdollars=_to_finite_number(details.get("budget_spend_microdollars")),
            upgrade_url=upgrade_url,
        )

    elif code == "velocity_exceeded":
        _safe_denied(on_denied, {
            "type": "velocity",
            "retry_after_seconds": retry_after,
            "limit": _to_finite_number(details.get("limitMicrodollars")),
            "window": _to_finite_number(details.get("windowSeconds")),
            "current": _to_finite_number(details.get("currentMicrodollars")),
        }, on_cost_error)
        raise VelocityExceededError(
            retry_after_seconds=retry_after,
            limit_microdollars=_to_finite_number(details.get("limitMicrodollars")),
            window_seconds=_to_finite_number(details.get("windowSeconds")),
            current_microdollars=_to_finite_number(details.get("currentMicrodollars")),
        )

    elif code == "session_limit_exceeded":
        spend = _to_finite_number(details.get("session_spend_microdollars")) or 0
        limit = _to_finite_number(details.get("session_limit_microdollars")) or 0
        _safe_denied(on_denied, {
            "type": "session_limit",
            "session_spend": spend,
            "session_limit": limit,
        }, on_cost_error)
        raise SessionLimitExceededError(
            session_spend_microdollars=spend,
            session_limit_microdollars=limit,
        )

    elif code == "tag_budget_exceeded":
        remaining = _to_finite_number(details.get("remaining_microdollars")) or 0
        _safe_denied(on_denied, {
            "type": "tag_budget",
            "tag_key": details.get("tag_key"),
            "tag_value": details.get("tag_value"),
            "remaining": remaining,
            "limit": _to_finite_number(details.get("budget_limit_microdollars")),
            "spend": _to_finite_number(details.get("budget_spend_microdollars")),
        }, on_cost_error)
        raise TagBudgetExceededError(
            tag_key=details.get("tag_key"),
            tag_value=details.get("tag_value"),
            remaining_microdollars=remaining,
            limit_microdollars=_to_finite_number(details.get("budget_limit_microdollars")),
            spend_microdollars=_to_finite_number(details.get("budget_spend_microdollars")),
        )

    else:
        # Unknown code: surface as drift signal via on_cost_error
        if on_cost_error:
            try:
                on_cost_error(NullSpendError(
                    f"Unknown denial code from proxy: {code!r}. "
                    "SDK may need updating."
                ))
            except Exception:
                pass


def _safe_denied(
    on_denied: Callable | None,
    reason: dict[str, Any],
    on_cost_error: Callable | None,
) -> None:
    """Defensively invoke the on_denied callback."""
    if on_denied is None:
        return
    try:
        on_denied(reason)
    except Exception as err:
        if on_cost_error:
            try:
                on_cost_error(err)
            except Exception:
                pass


# ---- Cost estimation ----


def _estimate_cost_microdollars(
    provider: str,
    model: str | None,
    body: bytes | str | None,
) -> int:
    """Estimate the cost of a request for budget pre-check."""
    if not model:
        return 0
    pricing = get_model_pricing(provider, model)
    if not pricing:
        return 0

    max_tokens = _DEFAULT_MAX_TOKENS
    if body:
        try:
            text = body.decode("utf-8") if isinstance(body, bytes) else body
            data = json.loads(text)
            if isinstance(data, dict):
                mt = data.get("max_tokens") or data.get("max_completion_tokens")
                if isinstance(mt, (int, float)) and mt > 0:
                    max_tokens = int(mt)
        except Exception:
            pass

    input_est = _DEFAULT_ESTIMATE_INPUT_TOKENS * pricing["inputPerMTok"]
    output_est = max_tokens * pricing["outputPerMTok"]
    return round(input_est + output_est)


# ---- Proxy detection ----


def _is_proxied(url: str, proxy_url: str | None, headers: dict[str, str] | None) -> bool:
    """Check if a request is going through the NullSpend proxy."""
    if not proxy_url:
        return False

    try:
        req_parsed = urlparse(url)
        proxy_parsed = urlparse(proxy_url)
        # Strict origin match (scheme + host + port)
        if (
            req_parsed.scheme == proxy_parsed.scheme
            and req_parsed.hostname == proxy_parsed.hostname
            and (req_parsed.port or (443 if req_parsed.scheme == "https" else 80))
            == (proxy_parsed.port or (443 if proxy_parsed.scheme == "https" else 80))
        ):
            return True
    except Exception:
        pass

    # Header fallback (only when proxyUrl is configured)
    if headers and "x-nullspend-key" in {k.lower() for k in headers}:
        return True

    return False


# ---- TeeByteStream ----


class TeeByteStream(httpx.SyncByteStream):
    """Wraps a byte stream, yielding chunks to caller while
    accumulating SSE data for cost extraction."""

    def __init__(
        self,
        stream: httpx.SyncByteStream,
        accumulator: SSEAccumulator,
        on_complete: Callable[[SSEAccumulator], None],
    ):
        self._stream = stream
        self._accumulator = accumulator
        self._on_complete = on_complete
        self._completed = False

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._accumulator.feed(chunk)
                yield chunk
        finally:
            self._finish()

    def close(self):
        self._finish()
        self._stream.close()

    def _finish(self):
        if not self._completed:
            self._completed = True
            try:
                self._accumulator.finalize()
                self._on_complete(self._accumulator)
            except Exception:
                pass


# ---- Tracked Transport ----


class TrackedTransport(httpx.BaseTransport):
    """Custom httpx transport that wraps the real transport for cost tracking."""

    def __init__(
        self,
        transport: httpx.BaseTransport,
        provider: str,
        proxy_url: str | None = None,
        api_key: str | None = None,
        api_version: str = "2026-04-01",
        customer: str | None = None,
        session_id: str | None = None,
        tags: dict[str, str] | None = None,
        trace_id: str | None = None,
        action_id: str | None = None,
        enforcement: bool = False,
        session_limit_microdollars: int | None = None,
        policy_cache: Any | None = None,
        queue_cost: Callable[[CostEventInput], None] | None = None,
        on_cost_error: Callable[[Exception], None] | None = None,
        on_denied: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._transport = transport
        self._provider = provider
        self._proxy_url = proxy_url
        self._api_key = api_key
        self._api_version = api_version
        self._customer = customer
        self._session_id = session_id
        self._tags = tags
        self._trace_id = trace_id
        self._action_id = action_id
        self._enforcement = enforcement
        self._session_limit = session_limit_microdollars
        self._policy_cache = policy_cache
        self._queue_cost = queue_cost
        self._on_cost_error = on_cost_error or self._default_cost_error
        self._on_denied = on_denied
        self._session_spend = 0
        self._first_error_logged = False

    def _default_cost_error(self, err: Exception) -> None:
        if not self._first_error_logged:
            self._first_error_logged = True
            logger.warning(
                "nullspend: Cost tracking error (%s). "
                "Set on_cost_error callback to customize. "
                "Subsequent errors will be silent.",
                err,
            )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        method = request.method
        body = request.content

        # Phase 1: Header injection
        if self._customer:
            request.headers["x-nullspend-customer"] = self._customer
        if self._tags:
            request.headers["x-nullspend-tags"] = json.dumps(self._tags)
        if self._trace_id:
            request.headers["x-nullspend-traceid"] = self._trace_id
        if self._action_id:
            request.headers["x-nullspend-actionid"] = self._action_id

        # Check if this is a tracked route
        if not _is_tracked_route(self._provider, url, method):
            return self._transport.handle_request(request)

        model = _extract_model_from_body(body)
        # Use post-injection headers for proxy detection (header fallback needs
        # to see injected x-nullspend-key if present)
        is_proxied = _is_proxied(url, self._proxy_url, dict(request.headers))

        # Phase 2: Proxy mode — skip client-side tracking, intercept 429
        if is_proxied:
            response = self._transport.handle_request(request)
            if response.status_code == 429 and self._enforcement:
                response.read()  # Read body for denial parsing
                parsed = _parse_denial_payload(response)
                if parsed:
                    _dispatch_denial(parsed, self._on_denied, self._on_cost_error)
            return response

        # Phase 3: Client-side enforcement (direct mode)
        if self._enforcement and self._policy_cache:
            try:
                self._policy_cache.get_policy()

                # Mandate check
                mandate_result = self._policy_cache.check_mandate(self._provider, model or "")
                if not mandate_result.allowed:
                    _safe_denied(self._on_denied, {
                        "type": "mandate",
                        "mandate": mandate_result.mandate,
                        "requested": mandate_result.requested,
                        "allowed": mandate_result.allowed_list,
                    }, self._on_cost_error)
                    raise MandateViolationError(
                        mandate=mandate_result.mandate or "",
                        requested=mandate_result.requested or "",
                        allowed=mandate_result.allowed_list or [],
                    )

                # Budget estimate check
                estimate = _estimate_cost_microdollars(self._provider, model, body)
                budget_result = self._policy_cache.check_budget(estimate)
                if not budget_result.allowed:
                    _safe_denied(self._on_denied, {
                        "type": "budget",
                        "remaining": budget_result.remaining,
                        "entity_type": budget_result.entity_type,
                        "entity_id": budget_result.entity_id,
                        "limit": budget_result.limit,
                        "spend": budget_result.spend,
                    }, self._on_cost_error)
                    raise BudgetExceededError(
                        remaining_microdollars=budget_result.remaining or 0,
                        entity_type=budget_result.entity_type,
                        entity_id=budget_result.entity_id,
                        limit_microdollars=budget_result.limit,
                        spend_microdollars=budget_result.spend,
                    )

                # Session limit check
                if self._session_id:
                    session_limit = self._session_limit or self._policy_cache.get_session_limit()
                    if session_limit is not None and (self._session_spend + estimate) > session_limit:
                        _safe_denied(self._on_denied, {
                            "type": "session_limit",
                            "session_spend": self._session_spend,
                            "session_limit": session_limit,
                        }, self._on_cost_error)
                        raise SessionLimitExceededError(
                            session_spend_microdollars=self._session_spend,
                            session_limit_microdollars=session_limit,
                        )

            except (MandateViolationError, BudgetExceededError, SessionLimitExceededError):
                raise
            except Exception as err:
                # Policy fetch failed — fail-open, but still enforce manual session limit
                try:
                    self._on_cost_error(err)
                except Exception:
                    pass

        # Phase 4: Stream injection for OpenAI
        if (
            self._provider == "openai"
            and _is_streaming_request(body)
            and body
        ):
            try:
                text = body.decode("utf-8") if isinstance(body, bytes) else body
                parsed_body = json.loads(text)
                stream_options = parsed_body.get("stream_options") or {}
                stream_options["include_usage"] = True
                parsed_body["stream_options"] = stream_options
                new_body = json.dumps(parsed_body).encode("utf-8")
                request = httpx.Request(
                    method=method,
                    url=request.url,
                    headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"},
                    content=new_body,
                )
            except Exception:
                pass  # Invalid JSON — pass through unchanged

        # Phase 5: Execute request
        start_time = time.monotonic()
        response = self._transport.handle_request(request)

        if not response.is_success:
            return response

        # Phase 6: Cost tracking
        duration_ms = int((time.monotonic() - start_time) * 1000)
        metadata = {
            "session_id": self._session_id,
            "trace_id": self._trace_id,
            "tags": self._tags,
            "customer": self._customer,
        }

        if _is_streaming_response(response) and response.stream:
            # Wrap stream with TeeByteStream for cost extraction.
            # Construct a new Response with the tee stream (documented httpx pattern)
            # instead of mutating response.stream.
            accumulator = SSEAccumulator(self._provider)

            def on_stream_complete(acc: SSEAccumulator) -> None:
                result = acc.finalize() if not acc.finalized else acc.finalize_partial()
                if result.usage:
                    self._queue_cost_from_usage(
                        model or result.model or "unknown",
                        result.usage,
                        getattr(result, "cache_creation_detail", None),
                        duration_ms,
                        metadata,
                    )

            tee = TeeByteStream(response.stream, accumulator, on_stream_complete)
            response = httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=tee,
            )
        else:
            # Non-streaming: read response and extract usage
            try:
                response.read()
                data = response.json()
                if isinstance(data, dict):
                    self._handle_non_streaming_usage(
                        model or "unknown", data, duration_ms, metadata,
                    )
            except Exception as err:
                try:
                    self._on_cost_error(err)
                except Exception:
                    pass

        return response

    def _queue_cost_from_usage(
        self,
        model: str,
        usage: dict[str, Any],
        cache_detail: dict[str, Any] | None,
        duration_ms: int,
        metadata: dict[str, Any],
    ) -> None:
        try:
            if self._provider == "openai":
                event = calculate_openai_cost_event(model, usage, duration_ms, metadata)
            elif self._provider == "anthropic":
                event = calculate_anthropic_cost_event(model, usage, cache_detail, duration_ms, metadata)
            else:
                return

            self._session_spend += event.cost_microdollars
            if self._queue_cost:
                self._queue_cost(event)
        except Exception as err:
            try:
                self._on_cost_error(err)
            except Exception:
                pass

    def _handle_non_streaming_usage(
        self,
        model: str,
        data: dict[str, Any],
        duration_ms: int,
        metadata: dict[str, Any],
    ) -> None:
        try:
            if self._provider == "openai":
                usage = _extract_openai_usage(data)
                if usage:
                    self._queue_cost_from_usage(model, usage, None, duration_ms, metadata)
            elif self._provider == "anthropic":
                usage, cache_detail = _extract_anthropic_usage(data)
                if usage:
                    self._queue_cost_from_usage(model, usage, cache_detail, duration_ms, metadata)
        except Exception as err:
            try:
                self._on_cost_error(err)
            except Exception:
                pass

    def close(self) -> None:
        self._transport.close()


# ---- Factory functions ----


def create_tracked_client(
    provider: str,
    *,
    proxy_url: str | None = None,
    api_key: str | None = None,
    api_version: str = "2026-04-01",
    customer: str | None = None,
    session_id: str | None = None,
    tags: dict[str, str] | None = None,
    trace_id: str | None = None,
    action_id: str | None = None,
    enforcement: bool = False,
    session_limit_microdollars: int | None = None,
    policy_cache: Any | None = None,
    queue_cost: Callable[[CostEventInput], None] | None = None,
    on_cost_error: Callable[[Exception], None] | None = None,
    on_denied: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = 30.0,
) -> httpx.Client:
    """Create an httpx.Client with NullSpend cost tracking transport.

    Use this as `http_client` for OpenAI/Anthropic SDKs:

        from openai import OpenAI
        tracked = create_tracked_client("openai", api_key="ns_live_sk_...")
        client = OpenAI(http_client=tracked)
    """
    real_transport = httpx.HTTPTransport()
    tracked = TrackedTransport(
        transport=real_transport,
        provider=provider,
        proxy_url=proxy_url,
        api_key=api_key,
        api_version=api_version,
        customer=customer,
        session_id=session_id,
        tags=tags,
        trace_id=trace_id,
        action_id=action_id,
        enforcement=enforcement,
        session_limit_microdollars=session_limit_microdollars,
        policy_cache=policy_cache,
        queue_cost=queue_cost,
        on_cost_error=on_cost_error,
        on_denied=on_denied,
    )
    return httpx.Client(transport=tracked, timeout=timeout)
