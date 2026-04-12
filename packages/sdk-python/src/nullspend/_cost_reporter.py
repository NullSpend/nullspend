"""Background cost event batching with flush/shutdown lifecycle.

Sync version uses queue.Queue + daemon threading.Thread.
atexit handler for normal process exit. Context manager __exit__ calls shutdown().
"""
from __future__ import annotations

import atexit
import logging
import queue
import threading
from typing import Any, Callable

from nullspend.types import CostEventInput, CostReportingConfig

logger = logging.getLogger("nullspend")

_MAX_DRAIN_ITERATIONS = 16


class CostReporter:
    """Batches cost events and sends them periodically or when batch_size is reached.

    Thread-safe. Uses a daemon thread for background flushing so it won't
    prevent process exit. An atexit handler flushes remaining events on
    normal process exit.
    """

    def __init__(
        self,
        config: CostReportingConfig,
        send_batch: Callable[[list[CostEventInput]], None],
    ):
        # Validate config
        if not 1 <= config.batch_size <= 100:
            raise ValueError(f"batch_size must be 1-100 (got {config.batch_size})")
        if config.flush_interval_ms < 100:
            raise ValueError(f"flush_interval_ms must be >= 100 (got {config.flush_interval_ms})")
        if config.max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1 (got {config.max_queue_size})")

        self._batch_size = config.batch_size
        self._flush_interval_s = config.flush_interval_ms / 1000.0
        self._max_queue_size = config.max_queue_size
        self._on_dropped = config.on_dropped
        self._on_flush_error = config.on_flush_error
        self._send_batch = send_batch

        self._queue: queue.Queue[CostEventInput] = queue.Queue(maxsize=0)
        self._queue_size = 0  # Tracked separately for overflow detection
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._is_shut_down = False

        # Start daemon thread for periodic flushing
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="nullspend-cost-reporter",
        )
        self._thread.start()

        # Register atexit for normal process exit
        atexit.register(self._atexit_flush)

    @property
    def is_shut_down(self) -> bool:
        return self._is_shut_down

    def enqueue(self, event: CostEventInput) -> None:
        """Add a cost event to the queue. Thread-safe."""
        if self._is_shut_down:
            return

        with self._lock:
            if self._queue_size >= self._max_queue_size:
                # Drop oldest events
                dropped = 0
                while self._queue_size >= self._max_queue_size:
                    try:
                        self._queue.get_nowait()
                        self._queue_size -= 1
                        dropped += 1
                    except queue.Empty:
                        break
                if dropped:
                    if self._on_dropped:
                        try:
                            self._on_dropped(dropped)
                        except Exception:
                            pass
                    else:
                        logger.warning(
                            "nullspend: Dropped %d cost event(s) (queue full, max_queue_size=%d). "
                            "Set on_dropped callback to customize.",
                            dropped, self._max_queue_size,
                        )

            self._queue.put_nowait(event)
            self._queue_size += 1
            should_flush = self._queue_size >= self._batch_size

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Drain the queue and send events in batches. Thread-safe, deduped."""
        if not self._flush_lock.acquire(blocking=False):
            return  # Another flush is running

        try:
            events: list[CostEventInput] = []
            while True:
                try:
                    events.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            with self._lock:
                # Subtract drained count instead of resetting to 0.
                # This prevents a race where enqueue() increments _queue_size
                # between the drain and the decrement.
                self._queue_size = max(0, self._queue_size - len(events))

            if not events:
                return

            # Send in chunks
            for i in range(0, len(events), self._batch_size):
                chunk = events[i : i + self._batch_size]
                try:
                    self._send_batch(chunk)
                except Exception as err:
                    if self._on_flush_error:
                        try:
                            self._on_flush_error(err, chunk)
                        except Exception:
                            pass
                    else:
                        logger.warning(
                            "nullspend: Failed to send cost event batch (%s). "
                            "Set on_flush_error callback to customize.",
                            err,
                        )
        finally:
            self._flush_lock.release()

    def shutdown(self) -> None:
        """Gracefully shut down: stop the background thread and drain all events."""
        if self._is_shut_down:
            return
        self._is_shut_down = True
        self._stop_event.set()

        # Unregister atexit to break the reference chain (prevents GC leak
        # when multiple NullSpend instances are created and discarded)
        atexit.unregister(self._atexit_flush)

        # Drain loop with guard against pathological producers
        for _ in range(_MAX_DRAIN_ITERATIONS):
            self.flush()
            if self._queue.empty():
                break

        self._thread.join(timeout=5.0)

    def _flush_loop(self) -> None:
        """Background thread: flush periodically until stopped."""
        while not self._stop_event.wait(timeout=self._flush_interval_s):
            self.flush()

    def _atexit_flush(self) -> None:
        """Best-effort flush on normal process exit."""
        if not self._is_shut_down:
            try:
                self.flush()
            except Exception:
                pass
