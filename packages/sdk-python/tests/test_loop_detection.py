"""Loop detection tests for the Python SDK.

Tests the LoopDetector class, LoopDetectedError, denial parsing,
TrackedTransport integration, and content hash computation.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

import httpx
import pytest
import respx

from nullspend._loop_detector import LoopCheck, LoopDetector
from nullspend._tracked_client import TrackedTransport, _dispatch_denial, create_tracked_client
from nullspend.errors import LoopDetectedError, NullSpendError
from nullspend.types import LoopDetectionConfig


# ── LoopDetector class tests ─────────────────────────────────────


class TestLoopDetectorBasic:
    """Basic per-key detection."""

    def test_threshold_hit_with_same_key_hash(self):
        d = LoopDetector(max_calls=5, window_seconds=60.0)
        for _ in range(4):
            result = d.check("openai:gpt-4o", "abc123")
            assert not result.is_loop
        result = d.check("openai:gpt-4o", "abc123")
        assert result.is_loop
        assert result.call_count == 5

    def test_different_content_no_trigger(self):
        d = LoopDetector(max_calls=5, window_seconds=60.0)
        for i in range(10):
            result = d.check("openai:gpt-4o", f"hash_{i}")
            assert not result.is_loop

    def test_different_models_independent(self):
        d = LoopDetector(max_calls=3, window_seconds=60.0)
        for _ in range(2):
            d.check("openai:gpt-4o", "abc")
            d.check("openai:gpt-4o-mini", "abc")
        # Neither should be at threshold yet (2 each, threshold is 3)
        r1 = d.check("openai:gpt-4o", "abc")
        assert r1.is_loop  # 3rd call for gpt-4o
        r2 = d.check("openai:gpt-4o-mini", "abc")
        assert r2.is_loop  # 3rd call for gpt-4o-mini

    def test_near_threshold_allowed(self):
        d = LoopDetector(max_calls=50, window_seconds=60.0)
        for _ in range(49):
            result = d.check("openai:gpt-4o", "abc")
            assert not result.is_loop
        result = d.check("openai:gpt-4o", "abc")
        assert result.is_loop
        assert result.call_count == 50

    def test_exactly_at_threshold(self):
        d = LoopDetector(max_calls=1, window_seconds=60.0)
        result = d.check("openai:gpt-4o", "abc")
        assert result.is_loop
        assert result.call_count == 1


class TestLoopDetectorWindow:
    """Window expiry behavior."""

    def test_entries_expire_after_window(self):
        d = LoopDetector(max_calls=3, window_seconds=0.1)
        d.check("openai:gpt-4o", "abc")
        d.check("openai:gpt-4o", "abc")
        time.sleep(0.15)
        # Entries expired — counter resets
        result = d.check("openai:gpt-4o", "abc")
        assert not result.is_loop
        assert result.call_count == 1

    def test_memory_bounded_by_window(self):
        d = LoopDetector(max_calls=100, window_seconds=0.05)
        for _ in range(50):
            d.check("openai:gpt-4o", "abc")
        time.sleep(0.1)
        # After window, old entries pruned
        d.check("openai:gpt-4o", "abc")
        assert len(d._call_log["openai:gpt-4o"]) == 1

    # SEC-2: per-key entry cap bounds memory under unique-hash burst.
    def test_per_key_entry_cap_bounds_memory(self):
        d = LoopDetector(max_calls=10, window_seconds=60.0)
        for i in range(500):
            d.check("openai:gpt-4o", f"hash-{i}")
        # Cap is 10 * 10 = 100; expect entries length to never exceed cap.
        assert len(d._call_log["openai:gpt-4o"]) == 100

    def test_per_key_cap_preserves_loop_detection(self):
        d = LoopDetector(max_calls=10, window_seconds=60.0)
        # Burst of unique hashes — would grow unboundedly without cap.
        for i in range(200):
            d.check("openai:gpt-4o", f"hash-{i}")
        # Now hammer with same hash — loop should still trigger because
        # eviction is FIFO (oldest unique-hash entries drop, recent loop-hash
        # entries are preserved).
        result = None
        for _ in range(50):
            result = d.check("openai:gpt-4o", "loop-hash")
        assert result is not None
        assert result.is_loop


class TestLoopDetectorAggregate:
    """Aggregate multi-model loop detection."""

    def test_aggregate_triggers_at_threshold(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=3)
        # 3 distinct keys, each with 3+ same-content repeats
        for key in ["openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-20250514"]:
            for _ in range(3):
                d.check(key, "same_hash")
        # Next call on any key should trigger aggregate
        result = d.check("openai:gpt-4o", "same_hash")
        assert result.is_loop

    def test_aggregate_no_false_positive_diverse_content(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=3)
        # 5 distinct keys but each with unique content (no 3+ repeats)
        for i, key in enumerate(["a:m1", "a:m2", "a:m3", "a:m4", "a:m5"]):
            for j in range(5):
                d.check(key, f"unique_{i}_{j}")
        result = d.check("a:m1", "final_unique")
        assert not result.is_loop

    def test_aggregate_requires_min_repeats(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=3,
                          aggregate_min_repeats=3)
        # 5 keys, each with only 2 same-content repeats (below min_repeats=3)
        for key in ["a:m1", "a:m2", "a:m3", "a:m4", "a:m5"]:
            d.check(key, "same")
            d.check(key, "same")
        result = d.check("a:m1", "same")
        # Now a:m1 has 3 repeats, but only 1 key qualifies (needs 3)
        assert not result.is_loop


class TestLoopDetectorWarning:
    """Warning fires at 80% threshold."""

    def test_warning_at_80_percent(self):
        d = LoopDetector(max_calls=10, window_seconds=60.0, warning_ratio=0.8)
        for _ in range(7):
            result = d.check("openai:gpt-4o", "abc")
            assert not result.is_warning
        result = d.check("openai:gpt-4o", "abc")  # 8th call = 80%
        assert result.is_warning
        assert not result.is_loop

    def test_warning_fires_once_per_composite_key(self):
        d = LoopDetector(max_calls=10, window_seconds=60.0, warning_ratio=0.8)
        for _ in range(8):
            d.check("openai:gpt-4o", "abc")
        # Warning already fired on 8th call, 9th should not fire again
        result = d.check("openai:gpt-4o", "abc")
        assert not result.is_warning  # Already fired

    def test_warning_resets_after_window_expiry(self):
        d = LoopDetector(max_calls=10, window_seconds=0.1, warning_ratio=0.8)
        for _ in range(8):
            d.check("openai:gpt-4o", "abc")
        time.sleep(0.15)
        # Window expired — entries pruned, warning cleared
        # Re-populate to 80% and verify warning fires again
        warned = False
        for _ in range(8):
            result = d.check("openai:gpt-4o", "abc")
            if result.is_warning:
                warned = True
        assert warned, "Warning should fire again after window expiry"


class TestLoopDetectorReset:
    """Reset call history."""

    def test_reset_all(self):
        d = LoopDetector(max_calls=5, window_seconds=60.0)
        for _ in range(4):
            d.check("openai:gpt-4o", "abc")
        d.reset()
        result = d.check("openai:gpt-4o", "abc")
        assert not result.is_loop
        assert result.call_count == 1

    def test_reset_specific_key(self):
        d = LoopDetector(max_calls=5, window_seconds=60.0)
        for _ in range(4):
            d.check("openai:gpt-4o", "abc")
            d.check("openai:gpt-4o-mini", "abc")
        d.reset("openai:gpt-4o")
        # gpt-4o reset, gpt-4o-mini still at 4
        r1 = d.check("openai:gpt-4o", "abc")
        assert r1.call_count == 1
        r2 = d.check("openai:gpt-4o-mini", "abc")
        assert r2.is_loop  # 5th call


class TestLoopDetectorThreadSafety:
    """Thread safety verification."""

    def test_concurrent_threads_no_crash(self):
        d = LoopDetector(max_calls=1000, window_seconds=60.0)
        errors: list[Exception] = []

        def worker(key: str):
            try:
                for _ in range(100):
                    d.check(key, "abc")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"t{i}:model",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_counts_accurate(self):
        d = LoopDetector(max_calls=1000, window_seconds=60.0)
        def worker():
            for _ in range(50):
                d.check("openai:gpt-4o", "abc")
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 10 threads × 50 = 500 calls
        result = d.check("openai:gpt-4o", "abc")
        assert result.call_count == 501  # 500 + this one


# ── Content hash tests ───────────────────────────────────────────


class TestContentHash:
    """Content hash computation."""

    def test_deterministic(self):
        h1 = LoopDetector.content_hash(b'{"model":"gpt-4o"}')
        h2 = LoopDetector.content_hash(b'{"model":"gpt-4o"}')
        assert h1 == h2

    def test_8_hex_chars(self):
        h = LoopDetector.content_hash(b"hello world")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_body_returns_empty(self):
        assert LoopDetector.content_hash(None) == "empty"
        assert LoopDetector.content_hash(b"") == "empty"
        assert LoopDetector.content_hash("") == "empty"

    def test_bytes_vs_str_parity(self):
        body = '{"model":"gpt-4o"}'
        h1 = LoopDetector.content_hash(body)
        h2 = LoopDetector.content_hash(body.encode("utf-8"))
        assert h1 == h2

    def test_8kb_cap(self):
        prefix = "A" * 8192
        h1 = LoopDetector.content_hash(prefix + "XXX")
        h2 = LoopDetector.content_hash(prefix + "YYY")
        assert h1 == h2  # Same because only first 8KB is hashed

    # SDK-T-2 / proxy audit T-7: cross-SDK parity. The TS SDK locks the same
    # fixtures in packages/sdk/src/loop-detector.test.ts under the
    # "cross-SDK content hash parity" describe block. If you change the hash
    # algorithm or truncation here you MUST update both fixtures, or runtime
    # SDKs will diverge from the proxy and from each other.
    def test_cross_sdk_fixture_hello(self):
        # SHA-256("hello").hexdigest()[:8]
        assert LoopDetector.content_hash(b"hello") == "2cf24dba"

    def test_cross_sdk_fixture_openai_body(self):
        body = b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
        # SHA-256(body).hexdigest()[:8]
        assert LoopDetector.content_hash(body) == "2c8329de"

    def test_different_user_messages_different_hash(self):
        """RAG agent: same system prompt, different user messages → different hashes."""
        system = "You are a helpful assistant."
        body1 = json.dumps({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is React?"},
        ]})
        body2 = json.dumps({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is Vue.js?"},
        ]})
        assert LoopDetector.content_hash(body1) != LoopDetector.content_hash(body2)


# ── LoopDetectedError tests ──────────────────────────────────────


class TestLoopDetectedError:
    """Error class fields and message format."""

    def test_fields_populated(self):
        err = LoopDetectedError(
            model="gpt-4o", call_count=50, window_seconds=60, max_calls=50,
        )
        assert err.model == "gpt-4o"
        assert err.call_count == 50
        assert err.window_seconds == 60
        assert err.max_calls == 50
        assert err.status_code == 429
        assert err.code == "loop_detected"

    def test_message_format(self):
        err = LoopDetectedError(
            model="gpt-4o", call_count=50, window_seconds=60, max_calls=50,
        )
        assert "gpt-4o" in str(err)
        assert "50 times" in str(err)
        assert "60s" in str(err)
        assert "nullspend.dev" in str(err)

    def test_inherits_from_nullspend_error(self):
        err = LoopDetectedError(
            model="gpt-4o", call_count=50, window_seconds=60, max_calls=50,
        )
        assert isinstance(err, NullSpendError)


# ── LoopDetectionConfig tests ────────────────────────────────────


class TestLoopDetectionConfig:
    """Config dataclass."""

    def test_default_values(self):
        cfg = LoopDetectionConfig()
        assert cfg.max_calls == 50
        assert cfg.window_seconds == 60.0
        assert cfg.aggregate_max_keys == 5

    def test_custom_values(self):
        cfg = LoopDetectionConfig(max_calls=100, window_seconds=120, aggregate_max_keys=10)
        assert cfg.max_calls == 100
        assert cfg.window_seconds == 120
        assert cfg.aggregate_max_keys == 10


# ── Denial parsing tests ─────────────────────────────────────────


class TestDenialParsing:
    """Proxy 429 with loop_detected code is parsed correctly."""

    def test_loop_detected_code_dispatches(self):
        denied_reasons: list[dict] = []
        parsed = {
            "code": "loop_detected",
            "details": {
                "type": "per_key",
                "model": "gpt-4o",
                "callCount": 50,
                "windowSeconds": 60,
                "maxCalls": 50,
            },
        }
        with pytest.raises(LoopDetectedError) as exc_info:
            _dispatch_denial(parsed, lambda r: denied_reasons.append(r), None)
        assert exc_info.value.model == "gpt-4o"
        assert exc_info.value.call_count == 50
        assert len(denied_reasons) == 1
        assert denied_reasons[0]["type"] == "loop"
        assert denied_reasons[0]["detection_type"] == "per_key"

    def test_aggregate_denial_parsing(self):
        parsed = {
            "code": "loop_detected",
            "details": {
                "type": "aggregate",
                "model": "aggregate",
                "callCount": 5,
                "windowSeconds": 60,
                "maxCalls": 5,
            },
        }
        with pytest.raises(LoopDetectedError) as exc_info:
            _dispatch_denial(parsed, None, None)
        assert exc_info.value.model == "aggregate"
        assert exc_info.value.call_count == 5

    def test_missing_details_uses_defaults(self):
        parsed = {"code": "loop_detected", "details": {}}
        with pytest.raises(LoopDetectedError) as exc_info:
            _dispatch_denial(parsed, None, None)
        assert exc_info.value.model == "unknown"
        assert exc_info.value.call_count == 0
        assert exc_info.value.window_seconds == 60
        assert exc_info.value.max_calls == 50


# ── TrackedTransport integration tests ───────────────────────────


class TestTrackedTransportLoopDetection:
    """Integration of loop detection into TrackedTransport."""

    def _make_transport(self, loop_detection=None, **kwargs):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        return TrackedTransport(
            transport=mock_transport,
            provider="openai",
            loop_detection=loop_detection,
            **kwargs,
        )

    def test_no_loop_detection_by_default(self):
        t = self._make_transport()
        assert t._loop_detector is None

    def test_loop_detection_true_creates_detector(self):
        t = self._make_transport(loop_detection=True)
        assert t._loop_detector is not None
        assert t._loop_detector._max_calls == 50

    def test_loop_detection_config_creates_detector(self):
        cfg = LoopDetectionConfig(max_calls=100, window_seconds=120)
        t = self._make_transport(loop_detection=cfg)
        assert t._loop_detector is not None
        assert t._loop_detector._max_calls == 100
        assert t._loop_detector._window == 120

    def test_loop_detection_none_no_detector(self):
        t = self._make_transport(loop_detection=None)
        assert t._loop_detector is None

    def test_raises_loop_detected_error_at_threshold(self):
        t = self._make_transport(loop_detection=LoopDetectionConfig(max_calls=3, window_seconds=60))
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        for _ in range(2):
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
            t.handle_request(req)
        with pytest.raises(LoopDetectedError) as exc_info:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
            t.handle_request(req)
        assert exc_info.value.call_count == 3
        assert exc_info.value.model == "gpt-4o"

    def test_different_bodies_no_false_positive(self):
        t = self._make_transport(loop_detection=LoopDetectionConfig(max_calls=3, window_seconds=60))
        for i in range(5):
            body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": f"msg {i}"}]})
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
            t.handle_request(req)  # Should not raise

    def test_loop_check_fires_before_proxy_call(self):
        """Loop check blocks before any upstream call is made."""
        call_count = 0

        def counting_handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={"choices": []})

        mock_transport = httpx.MockTransport(counting_handler)
        t = TrackedTransport(
            transport=mock_transport,
            provider="openai",
            loop_detection=LoopDetectionConfig(max_calls=2, window_seconds=60),
        )
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        req1 = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
        t.handle_request(req1)
        assert call_count == 1
        with pytest.raises(LoopDetectedError):
            req2 = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
            t.handle_request(req2)
        assert call_count == 1  # No second upstream call

    def test_non_tracked_routes_bypass_loop_check(self):
        t = self._make_transport(loop_detection=LoopDetectionConfig(max_calls=1, window_seconds=60))
        # GET request to a non-tracked route
        req = httpx.Request("GET", "https://api.openai.com/v1/models")
        t.handle_request(req)  # Should not raise even with max_calls=1


class TestCreateTrackedClientLoopDetection:
    """create_tracked_client passes loop_detection through."""

    def test_creates_client_with_loop_detection_true(self):
        client = create_tracked_client("openai", loop_detection=True)
        transport = client._transport
        assert isinstance(transport, TrackedTransport)
        assert transport._loop_detector is not None

    def test_creates_client_with_loop_detection_config(self):
        cfg = LoopDetectionConfig(max_calls=25)
        client = create_tracked_client("openai", loop_detection=cfg)
        transport = client._transport
        assert isinstance(transport, TrackedTransport)
        assert transport._loop_detector._max_calls == 25

    def test_creates_client_without_loop_detection(self):
        client = create_tracked_client("openai")
        transport = client._transport
        assert isinstance(transport, TrackedTransport)
        assert transport._loop_detector is None


# ── BUG-1 fix: stale aggregate entries ────────────────────────────


class TestAggregateStaleEntryPruning:
    """Aggregate detection prunes stale entries from non-current keys."""

    def test_stale_keys_dont_count_toward_aggregate(self):
        """Keys with only expired entries should not qualify as repeating."""
        d = LoopDetector(max_calls=100, window_seconds=0.1, aggregate_max_keys=3)
        # Populate 3 keys with 3+ same-content repeats
        for key in ["a:m1", "a:m2", "a:m3"]:
            for _ in range(3):
                d.check(key, "same")
        # Wait for window to expire
        time.sleep(0.15)
        # Now only check one key — stale keys should be pruned during aggregate scan
        result = d.check("a:m1", "same")
        # Only a:m1 has a fresh entry (count=1), others expired
        assert not result.is_loop

    def test_expired_keys_cleaned_from_call_log(self):
        """Fully expired keys should be removed from _call_log during aggregate scan."""
        d = LoopDetector(max_calls=100, window_seconds=0.1, aggregate_max_keys=5)
        d.check("a:old1", "hash1")
        d.check("a:old2", "hash2")
        time.sleep(0.15)
        # This check should prune old1 and old2 during aggregate scan
        d.check("a:fresh", "hash3")
        assert "a:old1" not in d._call_log
        assert "a:old2" not in d._call_log
        assert "a:fresh" in d._call_log


# ── Config validation ─────────────────────────────────────────────


class TestLoopDetectorValidation:
    """LoopDetector validates constructor arguments."""

    def test_negative_max_calls_raises(self):
        with pytest.raises(ValueError, match="max_calls must be >= 0"):
            LoopDetector(max_calls=-1)

    def test_zero_max_calls_blocks_every_call(self):
        """max_calls=0 blocks every single call (count 1 >= 0)."""
        d = LoopDetector(max_calls=0)
        # Every call is a "loop" since count >= 0 is always true
        for i in range(10):
            result = d.check("a:m1", f"hash_{i}")
            assert result.is_loop, f"Call {i} should be blocked with max_calls=0"
            assert result.detection_type == "per_key"

    def test_zero_max_calls_warning_fires(self):
        """With max_calls=0, warning threshold is int(0 * 0.8) = 0, so warning fires on first call too."""
        d = LoopDetector(max_calls=0)
        result = d.check("a:m1", "abc")
        assert result.is_loop
        assert result.is_warning  # warning_count=0, count=1 >= 0

    def test_zero_window_seconds_raises(self):
        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            LoopDetector(window_seconds=0)

    def test_negative_window_seconds_raises(self):
        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            LoopDetector(window_seconds=-1)

    def test_negative_aggregate_max_keys_raises(self):
        with pytest.raises(ValueError, match="aggregate_max_keys must be >= 0"):
            LoopDetector(aggregate_max_keys=-1)


# ── Proxy 429 loop_detected via TrackedTransport ──────────────────


class TestTrackedTransportProxy429LoopDenial:
    """Full integration: proxy returns 429 loop_detected, SDK parses it."""

    def test_proxy_429_loop_detected_raises_error(self):
        """Proxy mode: 429 with loop_detected code raises LoopDetectedError."""
        denial_body = json.dumps({
            "error": {
                "code": "loop_detected",
                "message": "Loop detected: gpt-4o called 50 times...",
                "details": {
                    "type": "per_key",
                    "model": "gpt-4o",
                    "provider": "openai",
                    "callCount": 50,
                    "windowSeconds": 60,
                    "maxCalls": 50,
                },
            },
        })
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(
                429,
                content=denial_body.encode(),
                headers={
                    "x-nullspend-denied": "1",
                    "retry-after": "5",
                    "content-type": "application/json",
                },
            )
        )
        t = TrackedTransport(
            transport=mock_transport,
            provider="openai",
            proxy_url="https://proxy.nullspend.dev",
            enforcement=True,
        )
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        req = httpx.Request(
            "POST", "https://proxy.nullspend.dev/v1/chat/completions",
            content=body.encode(),
        )
        with pytest.raises(LoopDetectedError) as exc_info:
            t.handle_request(req)
        assert exc_info.value.model == "gpt-4o"
        assert exc_info.value.call_count == 50
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "loop_detected"

    def test_proxy_429_loop_detected_calls_on_denied(self):
        """The on_denied callback receives loop denial reason."""
        denial_body = json.dumps({
            "error": {
                "code": "loop_detected",
                "details": {
                    "type": "aggregate",
                    "model": "aggregate",
                    "callCount": 5,
                    "windowSeconds": 60,
                    "maxCalls": 5,
                },
            },
        })
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(
                429,
                content=denial_body.encode(),
                headers={
                    "x-nullspend-denied": "1",
                    "content-type": "application/json",
                },
            )
        )
        denied_reasons: list[dict] = []
        t = TrackedTransport(
            transport=mock_transport,
            provider="openai",
            proxy_url="https://proxy.nullspend.dev",
            enforcement=True,
            on_denied=lambda r: denied_reasons.append(r),
        )
        body = json.dumps({"model": "gpt-4o", "messages": []})
        req = httpx.Request(
            "POST", "https://proxy.nullspend.dev/v1/chat/completions",
            content=body.encode(),
        )
        with pytest.raises(LoopDetectedError):
            t.handle_request(req)
        assert len(denied_reasons) == 1
        assert denied_reasons[0]["type"] == "loop"
        assert denied_reasons[0]["detection_type"] == "aggregate"
        assert denied_reasons[0]["model"] == "aggregate"
        assert denied_reasons[0]["call_count"] == 5
        assert denied_reasons[0]["window_seconds"] == 60
        assert denied_reasons[0]["max_calls"] == 5

    def test_proxy_429_without_enforcement_passes_through(self):
        """Without enforcement=True, proxy 429 is returned as-is (no raise)."""
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(
                429,
                content=b'{"error":{"code":"loop_detected","details":{}}}',
                headers={"x-nullspend-denied": "1", "content-type": "application/json"},
            )
        )
        t = TrackedTransport(
            transport=mock_transport,
            provider="openai",
            proxy_url="https://proxy.nullspend.dev",
            enforcement=False,  # not enforcing
        )
        body = json.dumps({"model": "gpt-4o", "messages": []})
        req = httpx.Request(
            "POST", "https://proxy.nullspend.dev/v1/chat/completions",
            content=body.encode(),
        )
        response = t.handle_request(req)
        assert response.status_code == 429  # Passed through, not raised


# ── Logger warning output ─────────────────────────────────────────


class TestLoopDetectorWarningLogger:
    """Warning logger output at 80% threshold."""

    def test_warning_logged_at_80_percent(self, caplog):
        """logger.warning fires when loop count reaches 80% of threshold."""
        import logging
        t = TrackedTransport(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []})),
            provider="openai",
            loop_detection=LoopDetectionConfig(max_calls=10, window_seconds=60),
        )
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        with caplog.at_level(logging.WARNING, logger="nullspend"):
            for i in range(9):
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
                t.handle_request(req)
        # 8th call is 80% of 10 — should have triggered warning
        assert any("approaching loop threshold" in r.message for r in caplog.records)

    def test_no_warning_below_80_percent(self, caplog):
        """No warning before reaching 80% threshold."""
        import logging
        t = TrackedTransport(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []})),
            provider="openai",
            loop_detection=LoopDetectionConfig(max_calls=10, window_seconds=60),
        )
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        with caplog.at_level(logging.WARNING, logger="nullspend"):
            for i in range(7):  # 70% — below threshold
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
                t.handle_request(req)
        assert not any("approaching loop threshold" in r.message for r in caplog.records)


# ── detection_type field ──────────────────────────────────────────


class TestDetectionTypeField:
    """LoopCheck and LoopDetectedError carry detection_type."""

    def test_per_key_detection_type(self):
        d = LoopDetector(max_calls=3, window_seconds=60.0)
        for _ in range(3):
            result = d.check("openai:gpt-4o", "abc")
        assert result.detection_type == "per_key"

    def test_aggregate_detection_type(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=3)
        for key in ["a:m1", "a:m2", "a:m3"]:
            for _ in range(3):
                d.check(key, "same")
        result = d.check("a:m1", "same")
        assert result.is_loop
        assert result.detection_type == "aggregate"

    def test_non_loop_returns_per_key_default(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0)
        result = d.check("openai:gpt-4o", "abc")
        assert not result.is_loop
        assert result.detection_type == "per_key"

    def test_error_class_carries_detection_type(self):
        err = LoopDetectedError(
            model="gpt-4o", call_count=50, window_seconds=60,
            max_calls=50, detection_type="aggregate",
        )
        assert err.detection_type == "aggregate"

    def test_error_class_default_detection_type(self):
        err = LoopDetectedError(model="gpt-4o", call_count=50, window_seconds=60, max_calls=50)
        assert err.detection_type == "per_key"

    def test_proxy_denial_carries_detection_type(self):
        parsed = {
            "code": "loop_detected",
            "details": {
                "type": "aggregate",
                "model": "aggregate",
                "callCount": 5,
                "windowSeconds": 60,
                "maxCalls": 5,
            },
        }
        with pytest.raises(LoopDetectedError) as exc_info:
            _dispatch_denial(parsed, None, None)
        assert exc_info.value.detection_type == "aggregate"

    def test_transport_raises_with_aggregate_type(self):
        """SDK-side aggregate detection passes type through to error."""
        t = TrackedTransport(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []})),
            provider="openai",
            loop_detection=LoopDetectionConfig(max_calls=100, window_seconds=60, aggregate_max_keys=3),
        )
        # Populate 2 keys with 3+ repeats (not yet at aggregate threshold of 3 keys)
        for model in ["m1", "m2"]:
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": "same"}]})
            for _ in range(3):
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body.encode())
                t.handle_request(req)
        # 3rd key at 2 repeats (still safe)
        body3 = json.dumps({"model": "m3", "messages": [{"role": "user", "content": "same"}]})
        for _ in range(2):
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body3.encode())
            t.handle_request(req)
        # 3rd call on m3 gives 3rd qualifying key → aggregate triggers
        with pytest.raises(LoopDetectedError) as exc_info:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=body3.encode())
            t.handle_request(req)
        assert exc_info.value.detection_type == "aggregate"


# ── reset then re-trigger warning ─────────────────────────────────


class TestResetRetriggerWarning:
    """After reset, warnings should fire again."""

    def test_warning_fires_again_after_reset(self):
        d = LoopDetector(max_calls=10, window_seconds=60.0, warning_ratio=0.8)
        # Reach 80% → warning fires
        for _ in range(8):
            d.check("openai:gpt-4o", "abc")
        # Reset
        d.reset()
        # Re-populate to 80% → warning should fire again
        warned = False
        for _ in range(8):
            result = d.check("openai:gpt-4o", "abc")
            if result.is_warning:
                warned = True
        assert warned
