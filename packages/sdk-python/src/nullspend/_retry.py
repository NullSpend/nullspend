"""Retry helpers shared between sync and async clients."""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY_S = 5.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5


def is_retryable_status_code(status: int) -> bool:
    return status in RETRYABLE_STATUS_CODES


def calculate_retry_delay_s(
    attempt: int,
    base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
    max_delay_s: float = MAX_RETRY_DELAY_S,
) -> float:
    """Full-jitter exponential backoff. Always >= 0.001s."""
    ceiling = min(base_delay_s * (2 ** attempt), max_delay_s)
    return max(0.001, random.random() * ceiling)


def parse_retry_after_s(value: str | None, max_s: float = MAX_RETRY_DELAY_S) -> float | None:
    """Parse Retry-After header value (seconds or HTTP date). Returns seconds or None."""
    if not value:
        return None
    # Try numeric seconds first
    try:
        seconds = float(value)
        if math.isfinite(seconds) and seconds >= 0:
            return min(seconds, max_s)
        return None
    except ValueError:
        pass
    # Try HTTP date (RFC 9110)
    try:
        dt = parsedate_to_datetime(value)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        if delta < 0:
            return 0.0
        return min(delta, max_s)
    except Exception:
        return None
