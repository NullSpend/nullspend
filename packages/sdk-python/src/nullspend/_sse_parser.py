"""SSE stream parsers for OpenAI and Anthropic streaming responses.

Yields raw bytes to the caller while accumulating usage data for cost
extraction. Designed for use with httpx streaming (sync iter_bytes /
async aiter_bytes).

64KB line length safety valve prevents unbounded memory on malformed streams.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, AsyncIterator

_MAX_LINE_LENGTH = 65_536  # 64KB safety valve


@dataclass
class OpenAISSEResult:
    """Accumulated usage from an OpenAI SSE stream."""
    model: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class AnthropicSSEResult:
    """Accumulated usage from an Anthropic SSE stream."""
    model: str | None = None
    usage: dict[str, Any] | None = None
    cache_creation_detail: dict[str, Any] | None = None


class SSEAccumulator:
    """Line-buffered SSE parser that accumulates usage data from byte chunks.

    Feed raw bytes via feed(). When the stream ends, call finalize() to get
    the parsed result. Call finalize_partial() for cancellation (partial data).
    """

    def __init__(self, provider: str):
        self._provider = provider
        self._buffer = ""
        self._model: str | None = None
        self._usage: dict[str, Any] | None = None
        self._cache_creation_detail: dict[str, Any] | None = None
        self._current_event: str | None = None
        self.finalized = False

    def feed(self, chunk: bytes) -> None:
        """Feed a chunk of bytes from the stream."""
        try:
            text = chunk.decode("utf-8", errors="replace")
        except Exception:
            return

        self._buffer += text

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")

            # Safety valve: skip oversized lines
            if len(line) > _MAX_LINE_LENGTH:
                continue

            self._process_line(line)

        # Safety valve: truncate buffer if it grows too large without newlines
        if len(self._buffer) > _MAX_LINE_LENGTH:
            self._buffer = ""

    def _process_line(self, line: str) -> None:
        if self._provider == "openai":
            self._process_openai_line(line)
        elif self._provider == "anthropic":
            self._process_anthropic_line(line)

    def _process_openai_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return

        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return

        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return

        if not isinstance(data, dict):
            return

        # Model: first one wins
        if self._model is None and "model" in data:
            self._model = data["model"]

        # Usage: last one wins (final chunk has usage)
        if "usage" in data and isinstance(data["usage"], dict):
            self._usage = data["usage"]

    def _process_anthropic_line(self, line: str) -> None:
        # Track event type
        if line.startswith("event:"):
            self._current_event = line[6:].strip()
            return

        if not line.startswith("data:"):
            return

        data_str = line[5:].strip()
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return

        if not isinstance(data, dict):
            return

        event = self._current_event

        if event == "message_start":
            message = data.get("message", {})
            if isinstance(message, dict):
                if "model" in message:
                    self._model = message["model"]
                if "usage" in message and isinstance(message["usage"], dict):
                    self._usage = message["usage"]
                    # Extract cache_creation detail if present
                    cache_creation = message["usage"].get("cache_creation")
                    if isinstance(cache_creation, dict):
                        self._cache_creation_detail = cache_creation

        elif event == "message_delta":
            # Update usage with delta values
            delta_usage = data.get("usage")
            if isinstance(delta_usage, dict) and self._usage is not None:
                for key, val in delta_usage.items():
                    if isinstance(val, (int, float)):
                        self._usage[key] = val

        # Reset event after processing data line
        self._current_event = None

    def finalize(self) -> OpenAISSEResult | AnthropicSSEResult:
        """Finalize parsing and return the accumulated result."""
        self.finalized = True
        # Process any remaining buffer content
        if self._buffer:
            remaining = self._buffer.rstrip("\r")
            if remaining and len(remaining) <= _MAX_LINE_LENGTH:
                self._process_line(remaining)
            self._buffer = ""

        if self._provider == "openai":
            return OpenAISSEResult(model=self._model, usage=self._usage)
        return AnthropicSSEResult(
            model=self._model,
            usage=self._usage,
            cache_creation_detail=self._cache_creation_detail,
        )

    def finalize_partial(self) -> OpenAISSEResult | AnthropicSSEResult:
        """Finalize with whatever data we have (for cancelled streams)."""
        return self.finalize()


def iter_sse_with_accumulator(
    byte_iterator: Iterator[bytes],
    provider: str,
) -> tuple[Iterator[bytes], SSEAccumulator]:
    """Wrap a sync byte iterator with an SSE accumulator.

    Returns a new iterator that yields the same bytes (passthrough)
    while feeding them to the accumulator for usage extraction.
    """
    accumulator = SSEAccumulator(provider)

    def _tee() -> Iterator[bytes]:
        for chunk in byte_iterator:
            accumulator.feed(chunk)
            yield chunk

    return _tee(), accumulator


async def aiter_sse_with_accumulator(
    byte_iterator: AsyncIterator[bytes],
    provider: str,
) -> tuple[AsyncIterator[bytes], SSEAccumulator]:
    """Wrap an async byte iterator with an SSE accumulator.

    Returns a new async iterator that yields the same bytes (passthrough)
    while feeding them to the accumulator for usage extraction.
    """
    accumulator = SSEAccumulator(provider)

    async def _tee() -> AsyncIterator[bytes]:
        async for chunk in byte_iterator:
            accumulator.feed(chunk)
            yield chunk

    return _tee(), accumulator
