"""Loop detection stress tests for the Python SDK.

These exercise real threading, real timing, memory bounds, and edge cases
that unit tests with mocks can't catch.

Run: cd packages/sdk-python && python -m pytest tests/test_loop_detection_stress.py -v
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import threading
import time

import pytest

from nullspend._loop_detector import LoopCheck, LoopDetector
from nullspend.errors import LoopDetectedError
from nullspend.types import LoopDetectionConfig


# ── Concurrency stress ────────────────────────────────────────────


class TestThreadingStress:
    """Real multi-threaded stress tests."""

    def test_10_threads_100_calls_each_same_key(self):
        """10 threads hammering the same key. Exactly 1000 calls should be recorded."""
        d = LoopDetector(max_calls=2000, window_seconds=60.0)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    d.check("openai:gpt-4o", "samehash")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        result = d.check("openai:gpt-4o", "samehash")
        assert result.call_count == 1001  # 1000 + this check

    def test_10_threads_independent_keys(self):
        """Each thread uses its own key. Verify no cross-contamination."""
        # High aggregate threshold (20) to avoid triggering with 10 keys
        d = LoopDetector(max_calls=200, window_seconds=60.0, aggregate_max_keys=20)

        def worker(thread_id: int):
            key = f"thread-{thread_id}:model"
            for _ in range(50):
                d.check(key, "hash")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads done. Check each key's count (50 prior + this check = 51).
        for i in range(10):
            key = f"thread-{i}:model"
            r = d.check(key, "hash")
            assert r.call_count == 51, f"{key} has count {r.call_count}, expected 51"

    def test_concurrent_loop_detection_fires_correctly(self):
        """Multiple threads, threshold=50. Some threads should hit the threshold."""
        d = LoopDetector(max_calls=50, window_seconds=60.0)
        loop_detected_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal loop_detected_count
            for _ in range(20):
                r = d.check("openai:gpt-4o", "same")
                if r.is_loop:
                    with lock:
                        loop_detected_count += 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 5 threads x 20 calls = 100 total. First 49 pass, calls 50-100 are loops.
        # So 51 calls should be detected as loops.
        assert loop_detected_count == 51

    def test_thread_a_loops_thread_b_independent(self):
        """Thread A: loop on model-1. Thread B: different model, should not be affected."""
        d = LoopDetector(max_calls=10, window_seconds=60.0)
        b_results: list[bool] = []

        def thread_a():
            for _ in range(20):
                d.check("openai:model-1", "hash")

        def thread_b():
            for _ in range(5):
                r = d.check("openai:model-2", "hash")
                b_results.append(r.is_loop)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        # Thread B made only 5 calls (threshold 10), none should be loops
        assert all(not x for x in b_results)


# ── Timing precision ─────────────────────────────────────────────


class TestTimingPrecision:
    """Real time-based tests for window boundaries."""

    def test_window_expiry_resets_counter(self):
        """Populate to threshold-1, wait for window, verify counter resets."""
        d = LoopDetector(max_calls=5, window_seconds=0.2)
        for _ in range(4):
            d.check("openai:gpt-4o", "hash")
        time.sleep(0.3)
        # All 4 entries should have expired
        result = d.check("openai:gpt-4o", "hash")
        assert result.call_count == 1
        assert not result.is_loop

    def test_window_boundary_precision(self):
        """Entries just inside the window are counted, just outside are not."""
        d = LoopDetector(max_calls=10, window_seconds=0.15)
        # Make 5 calls
        for _ in range(5):
            d.check("openai:gpt-4o", "hash")
        # Wait 100ms (inside 150ms window)
        time.sleep(0.1)
        # Make 4 more calls
        for _ in range(4):
            d.check("openai:gpt-4o", "hash")
        # Total should be 9 (5 old + 4 new, all within window)
        result = d.check("openai:gpt-4o", "hash")
        assert result.call_count == 10  # 5+4+1
        assert result.is_loop  # 10 >= 10

    def test_stale_entries_pruned_during_aggregate(self):
        """Aggregate detection ignores entries outside the window."""
        d = LoopDetector(max_calls=100, window_seconds=0.1, aggregate_max_keys=3)
        # Build 3 qualifying keys
        for model in ["m1", "m2", "m3"]:
            for _ in range(3):
                d.check(f"openai:{model}", "same")
        # Wait for all to expire
        time.sleep(0.15)
        # Now only check m1 — m2 and m3 should be pruned, not qualifying
        result = d.check("openai:m1", "same")
        assert not result.is_loop
        assert result.call_count == 1


# ── Aggregate edge cases ─────────────────────────────────────────


class TestAggregateEdgeCases:
    """Precise aggregate threshold boundary tests."""

    def test_4_keys_at_3_repeats_below_threshold_5(self):
        """4 qualifying keys (need 5). Should NOT trigger."""
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=5)
        for model in ["m1", "m2", "m3", "m4"]:
            for _ in range(3):
                d.check(f"openai:{model}", "same")
        result = d.check("openai:m1", "same")
        assert not result.is_loop

    def test_5_keys_at_2_repeats_below_min_repeats(self):
        """5 keys but each only has 2 repeats (min_repeats=3). Should NOT trigger."""
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=5)
        for model in ["m1", "m2", "m3", "m4", "m5"]:
            for _ in range(2):
                d.check(f"openai:{model}", "same")
        result = d.check("openai:m1", "same")
        # m1 now has 3 repeats, but only 1 key qualifies
        assert not result.is_loop

    def test_mixed_content_some_keys_qualify(self):
        """Some keys have repeating content, some don't. Only qualifying keys count."""
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=3)
        # 2 keys with 3+ same-content repeats
        for model in ["m1", "m2"]:
            for _ in range(5):
                d.check(f"openai:{model}", "same")
        # 3 keys with all unique content (don't qualify)
        for model in ["m3", "m4", "m5"]:
            for i in range(5):
                d.check(f"openai:{model}", f"unique_{model}_{i}")
        result = d.check("openai:m1", "same")
        # Only 2 qualifying keys, need 3
        assert not result.is_loop


# ── Memory bounds ─────────────────────────────────────────────────


class TestMemoryBounds:
    """Verify memory doesn't grow unboundedly."""

    def test_10000_distinct_keys_memory_bounded(self):
        """10K unique model+hash combinations. After window expires, memory should be freed."""
        d = LoopDetector(max_calls=100, window_seconds=0.1)
        for i in range(10_000):
            d.check(f"provider:model-{i}", f"hash-{i}")

        # At this point, _call_log has up to 10K entries
        assert len(d._call_log) <= 10_000

        # Wait for window expiry
        time.sleep(0.15)

        # One more check should prune during aggregate scan
        d.check("provider:model-0", "hash-new")

        # Most keys should have been cleaned up (only model-0 remains fresh)
        assert len(d._call_log) < 100, f"Expected pruning, got {len(d._call_log)} keys"

    def test_warnings_fired_set_bounded_by_recheck(self):
        """_warnings_fired entries are discarded when the same composite is rechecked
        and count has dropped below threshold (e.g., after window expiry)."""
        d = LoopDetector(max_calls=10, window_seconds=0.1, warning_ratio=0.5)
        # Fire warning for one composite
        for _ in range(6):  # 60% > 50% warning ratio
            d.check("openai:model-0", "hash-0")
        assert len(d._warnings_fired) == 1

        # Wait for window expiry
        time.sleep(0.15)

        # Re-check the SAME composite — count drops to 1, warning is discarded
        d.check("openai:model-0", "hash-0")
        assert len(d._warnings_fired) == 0


# ── Content hash stress ───────────────────────────────────────────


class TestContentHashStress:
    """Content hash collision and edge case stress tests."""

    def test_collision_rate_100k_unique_bodies(self):
        """Hash 100K unique bodies, measure collision rate. Should be < 0.1%."""
        hashes: set[str] = set()
        collisions = 0
        for i in range(100_000):
            body = json.dumps({"model": "gpt-4o", "messages": [{"content": f"unique-{i}"}]})
            h = LoopDetector.content_hash(body)
            if h in hashes:
                collisions += 1
            hashes.add(h)

        collision_rate = collisions / 100_000
        # SHA-256 truncated to 32 bits: birthday paradox predicts ~1 collision at 65K
        # At 100K we expect a few. Under 0.5% is fine.
        assert collision_rate < 0.005, f"Collision rate {collision_rate:.4f} too high"

    def test_bodies_sharing_8kb_prefix_same_hash(self):
        """Bodies that differ only after 8KB produce the same hash."""
        prefix = "A" * 8192
        h1 = LoopDetector.content_hash(prefix + "XXXXXXX")
        h2 = LoopDetector.content_hash(prefix + "YYYYYYY")
        assert h1 == h2

    def test_bodies_differing_within_8kb_different_hash(self):
        """Bodies that differ within the first 8KB produce different hashes."""
        prefix = "A" * 8000
        h1 = LoopDetector.content_hash(prefix + "XXX")
        h2 = LoopDetector.content_hash(prefix + "YYY")
        assert h1 != h2

    def test_unicode_bodies_deterministic(self):
        """Unicode content produces consistent, deterministic hashes."""
        body = json.dumps({"messages": [{"content": "你好世界 こんにちは 🌍"}]})
        h1 = LoopDetector.content_hash(body)
        h2 = LoopDetector.content_hash(body)
        assert h1 == h2
        assert len(h1) == 8

    def test_empty_and_none_sentinel(self):
        """Empty/None bodies return 'empty' sentinel, not a real hash."""
        assert LoopDetector.content_hash(None) == "empty"
        assert LoopDetector.content_hash("") == "empty"
        assert LoopDetector.content_hash(b"") == "empty"
        # "empty" is NOT a valid hex hash
        assert LoopDetector.content_hash("anything") != "empty"


# ── Detection type field ──────────────────────────────────────────


class TestDetectionTypeStress:
    """Verify detection_type is correct under various conditions."""

    def test_per_key_detection_type_at_threshold(self):
        d = LoopDetector(max_calls=5, window_seconds=60.0)
        for _ in range(5):
            result = d.check("openai:gpt-4o", "hash")
        assert result.detection_type == "per_key"

    def test_aggregate_detection_type(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0, aggregate_max_keys=2)
        for model in ["m1", "m2"]:
            for _ in range(3):
                d.check(f"openai:{model}", "same")
        result = d.check("openai:m1", "same")
        assert result.is_loop
        assert result.detection_type == "aggregate"

    def test_non_loop_always_per_key(self):
        d = LoopDetector(max_calls=100, window_seconds=60.0)
        result = d.check("openai:gpt-4o", "hash")
        assert not result.is_loop
        assert result.detection_type == "per_key"


# ── Reset under concurrent access ────────────────────────────────


class TestResetStress:
    """Reset while threads are actively checking."""

    def test_reset_during_concurrent_checks(self):
        """Reset mid-stream doesn't crash."""
        d = LoopDetector(max_calls=1000, window_seconds=60.0)
        errors: list[Exception] = []
        stop = threading.Event()

        def checker():
            try:
                while not stop.is_set():
                    d.check("openai:gpt-4o", "hash")
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(50):
                    time.sleep(0.001)
                    d.reset()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=checker) for _ in range(5)]
        reset_thread = threading.Thread(target=resetter)
        for t in threads:
            t.start()
        reset_thread.start()
        reset_thread.join()
        stop.set()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent reset: {errors}"
