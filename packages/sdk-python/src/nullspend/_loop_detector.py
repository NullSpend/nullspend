"""Sliding-window loop detection for repeated LLM calls."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from typing import NamedTuple


class LoopCheck(NamedTuple):
    is_loop: bool
    is_warning: bool
    call_count: int
    detection_type: str = "per_key"  # "per_key" or "aggregate"


class LoopDetector:
    """Detects repeated calls to the same model+content within a time window.

    Thread-safe. One instance per tracked client.
    Time-pruned lists, bounded by window expiry.
    """

    def __init__(
        self,
        max_calls: int = 50,
        window_seconds: float = 60.0,
        aggregate_max_keys: int = 5,
        aggregate_min_repeats: int = 3,
        warning_ratio: float = 0.8,
    ):
        if max_calls < 0:
            raise ValueError(f"max_calls must be >= 0 (got {max_calls})")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0 (got {window_seconds})")
        if aggregate_max_keys < 0:
            raise ValueError(f"aggregate_max_keys must be >= 0 (got {aggregate_max_keys})")
        self._max_calls = max_calls
        self._window = window_seconds
        self._aggregate_max_keys = aggregate_max_keys
        self._aggregate_min_repeats = aggregate_min_repeats
        self._warning_ratio = warning_ratio
        # key -> list of (timestamp, content_hash) — time-pruned on each check
        self._call_log: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self._warnings_fired: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def content_hash(body: bytes | str | None) -> str:
        """Hash full request body to 8 hex chars. Cap at 8KB for large payloads."""
        if not body:
            return "empty"
        if isinstance(body, str):
            raw = body[:8192].encode("utf-8")
        else:
            raw = body[:8192]
        return hashlib.sha256(raw).hexdigest()[:8]

    def check(self, key: str, content_hash: str) -> LoopCheck:
        """Record a call and check for loops.

        Returns LoopCheck(is_loop, is_warning, call_count).
        """
        now = time.monotonic()
        cutoff = now - self._window
        warning_count = int(self._max_calls * self._warning_ratio)

        with self._lock:
            # Per-key check: prune old entries, append new, then cap to bound
            # memory under burst load with many distinct content hashes (SEC-2).
            # Cap at 10× max_calls — enough headroom for legitimate detection
            # (counting by hash) while preventing unbounded growth.
            entries = self._call_log[key]
            self._call_log[key] = [(t, h) for t, h in entries if t > cutoff]
            entries = self._call_log[key]
            entries.append((now, content_hash))
            entry_cap = max(self._max_calls * 10, self._max_calls)
            if len(entries) > entry_cap:
                del entries[0:len(entries) - entry_cap]

            # Count matching content hash
            count = sum(1 for _, h in entries if h == content_hash)

            # Per-key loop
            is_loop = count >= self._max_calls
            detection_type = "per_key"

            # Aggregate: count distinct keys with 3+ same-content repeats.
            # Prune all keys lazily here to avoid counting stale entries
            # from keys that haven't been checked recently (BUG-1 fix).
            if not is_loop:
                qualifying_keys = 0
                stale_keys: list[str] = []
                for k, ents in list(self._call_log.items()):
                    if k != key:  # Current key already pruned above
                        ents = [(t, h) for t, h in ents if t > cutoff]
                        self._call_log[k] = ents
                    if not ents:
                        stale_keys.append(k)
                        continue
                    hash_counts: dict[str, int] = {}
                    for _, h in ents:
                        hash_counts[h] = hash_counts.get(h, 0) + 1
                    if any(c >= self._aggregate_min_repeats for c in hash_counts.values()):
                        qualifying_keys += 1
                # Clean up fully-expired keys to prevent dict growth
                for k in stale_keys:
                    del self._call_log[k]
                if qualifying_keys >= self._aggregate_max_keys:
                    is_loop = True
                    count = qualifying_keys
                    detection_type = "aggregate"

            # Warning (fires once per window per composite key, uses >= not ==)
            composite = f"{key}:{content_hash}"
            is_warning = False
            if count >= warning_count and composite not in self._warnings_fired:
                is_warning = True
                self._warnings_fired.add(composite)
            # Reset warning if dropped below threshold
            if count < warning_count:
                self._warnings_fired.discard(composite)

            return LoopCheck(is_loop=is_loop, is_warning=is_warning, call_count=count, detection_type=detection_type)

    def reset(self, key: str | None = None) -> None:
        """Reset call history."""
        with self._lock:
            if key is None:
                self._call_log.clear()
                self._warnings_fired.clear()
            else:
                self._call_log.pop(key, None)
                self._warnings_fired = {
                    k for k in self._warnings_fired if not k.startswith(f"{key}:")
                }
