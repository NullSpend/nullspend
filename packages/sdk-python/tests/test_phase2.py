"""Phase 2 tests: cost calculator, SSE parser, pricing data.

Covers all 47 models, edge cases, long-context multipliers, cache write TTLs,
rounding residual distribution, SSE parsing for OpenAI and Anthropic.
"""
from __future__ import annotations

import json
import pytest

from nullspend._cost_calculator import (
    PRICING_MAP,
    calculate_openai_cost_event,
    calculate_anthropic_cost_event,
    cost_component,
    get_model_pricing,
    is_known_model,
    _distribute_residual,
    _LONG_CONTEXT_THRESHOLD,
)
from nullspend._sse_parser import (
    SSEAccumulator,
    OpenAISSEResult,
    AnthropicSSEResult,
    iter_sse_with_accumulator,
    _MAX_LINE_LENGTH,
)
from nullspend.types import CostBreakdown


# ---- Pricing data validation ----


class TestPricingData:
    def test_pricing_data_loaded(self):
        assert len(PRICING_MAP) >= 47

    def test_every_entry_has_required_fields(self):
        for key, pricing in PRICING_MAP.items():
            assert "inputPerMTok" in pricing, f"{key} missing inputPerMTok"
            assert "cachedInputPerMTok" in pricing, f"{key} missing cachedInputPerMTok"
            assert "outputPerMTok" in pricing, f"{key} missing outputPerMTok"
            assert pricing["inputPerMTok"] >= 0, f"{key} negative inputPerMTok"
            assert pricing["outputPerMTok"] >= 0, f"{key} negative outputPerMTok"

    def test_anthropic_models_have_cache_write_rates(self):
        for key, pricing in PRICING_MAP.items():
            if key.startswith("anthropic/"):
                assert "cacheWrite5mPerMTok" in pricing, f"{key} missing cacheWrite5mPerMTok"

    def test_known_model_lookup(self):
        assert is_known_model("openai", "gpt-4o")
        assert is_known_model("anthropic", "claude-sonnet-4-5")
        assert not is_known_model("openai", "nonexistent")
        assert not is_known_model("fakeprovider", "gpt-4o")

    def test_get_model_pricing_returns_none_for_unknown(self):
        assert get_model_pricing("openai", "nonexistent") is None


# ---- cost_component ----


class TestCostComponent:
    def test_basic(self):
        # 1000 tokens at $2.50/MTok = 2500 microdollars (since 1000 * 2.5 = 2500)
        assert cost_component(1000, 2.5) == 2500.0

    def test_zero_tokens(self):
        assert cost_component(0, 2.5) == 0.0

    def test_negative_tokens(self):
        assert cost_component(-100, 2.5) == 0.0

    def test_zero_rate(self):
        assert cost_component(1000, 0) == 0.0

    def test_negative_rate(self):
        assert cost_component(1000, -1.0) == 0.0

    def test_large_values(self):
        # 1M tokens at $10/MTok = 10M microdollars = $10
        result = cost_component(1_000_000, 10.0)
        assert result == 10_000_000.0


# ---- Rounding residual ----


class TestDistributeResidual:
    def test_no_residual(self):
        # All round cleanly
        i, c, o = _distribute_residual(10, 5.0, 2.0, 3.0)
        assert i + c + o == 10

    def test_residual_to_output(self):
        # Output is largest
        i, c, o = _distribute_residual(10, 1.0, 1.0, 8.4)
        assert i + c + o == 10

    def test_residual_to_input(self):
        # Input is largest
        i, c, o = _distribute_residual(10, 8.4, 1.0, 1.0)
        assert i + c + o == 10


# ---- OpenAI cost calculation ----


class TestOpenAICostBasic:
    def test_gpt4o_basic(self):
        pricing = get_model_pricing("openai", "gpt-4o")
        result = calculate_openai_cost_event("gpt-4o", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.cost_microdollars > 0
        assert result.cost_breakdown is not None
        assert result.event_type == "llm"
        # Verify math: 1000 * 2.5 + 500 * 10.0 = 2500 + 5000 = 7500
        assert result.cost_microdollars == 7500

    def test_with_cached_tokens(self):
        result = calculate_openai_cost_event("gpt-4o", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 400},
        })
        assert result.cached_input_tokens == 400
        # Normal input: 600, cached: 400
        # 600 * 2.5 + 400 * 1.25 + 500 * 10.0 = 1500 + 500 + 5000 = 7000
        assert result.cost_microdollars == 7000

    def test_with_reasoning_tokens(self):
        result = calculate_openai_cost_event("o3", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "completion_tokens_details": {"reasoning_tokens": 200},
        })
        assert result.reasoning_tokens == 200
        assert result.cost_breakdown is not None
        assert result.cost_breakdown.reasoning is not None
        assert result.cost_breakdown.reasoning > 0

    def test_unknown_model(self):
        result = calculate_openai_cost_event("unknown-model", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        assert result.cost_microdollars == 0
        assert result.cost_breakdown is None

    def test_zero_tokens(self):
        result = calculate_openai_cost_event("gpt-4o", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })
        assert result.cost_microdollars == 0

    def test_negative_tokens_clamped(self):
        result = calculate_openai_cost_event("gpt-4o", {
            "prompt_tokens": -100,
            "completion_tokens": -50,
        })
        assert result.cost_microdollars == 0
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_metadata_passed_through(self):
        result = calculate_openai_cost_event("gpt-4o", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }, duration_ms=320, metadata={
            "sessionId": "sess-1",
            "traceId": "trace-1",
            "tags": {"env": "prod"},
            "customer": "acme",
        })
        assert result.duration_ms == 320
        assert result.session_id == "sess-1"
        assert result.trace_id == "trace-1"
        assert result.tags == {"env": "prod"}
        assert result.customer == "acme"

    def test_missing_usage_fields(self):
        """Handles missing fields gracefully (defaults to 0)."""
        result = calculate_openai_cost_event("gpt-4o", {})
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cost_microdollars == 0


# ---- Parameterized across all OpenAI models ----

_OPENAI_MODELS = [k.split("/")[1] for k in PRICING_MAP if k.startswith("openai/")]

class TestOpenAIAllModels:
    @pytest.mark.parametrize("model", _OPENAI_MODELS)
    def test_basic_cost_positive(self, model):
        result = calculate_openai_cost_event(model, {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        assert result.cost_microdollars > 0
        assert result.cost_breakdown is not None
        # Components should sum to total
        bd = result.cost_breakdown
        assert bd.input + bd.cached + bd.output == result.cost_microdollars

    @pytest.mark.parametrize("model", _OPENAI_MODELS)
    def test_cached_cost_less_than_uncached(self, model):
        """Cached tokens should be cheaper than full-price input."""
        pricing = get_model_pricing("openai", model)
        assert pricing["cachedInputPerMTok"] <= pricing["inputPerMTok"]


# ---- Anthropic cost calculation ----


class TestAnthropicCostBasic:
    def test_claude_sonnet_basic(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 1000,
            "output_tokens": 500,
        })
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-5"
        assert result.cost_microdollars > 0
        assert result.cost_breakdown is not None
        assert result.event_type == "llm"

    def test_with_cache_read(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 300,
        })
        assert result.cached_input_tokens == 300
        # Total input includes cache read
        assert result.input_tokens == 1300

    def test_with_cache_write(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 2000,
        })
        # Total input includes cache creation
        assert result.input_tokens == 3000

    def test_with_ephemeral_cache_detail(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 3000,
        }, cache_creation_detail={
            "ephemeral_5m_input_tokens": 2000,
            "ephemeral_1h_input_tokens": 1000,
        })
        assert result.cost_microdollars > 0
        # Cache cost uses split rates
        assert result.cost_breakdown is not None

    def test_long_context_multiplier(self):
        """Tokens >200K trigger 2x input and 1.5x output multipliers."""
        # Short context
        short = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 100_000,
            "output_tokens": 1000,
        })
        # Long context (>200K)
        long_ = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 200_001,
            "output_tokens": 1000,
        })
        # Long context should be more expensive per token
        # Input: 2x rate, Output: 1.5x rate
        assert long_.cost_microdollars > short.cost_microdollars

    def test_exactly_200k_not_long_context(self):
        """200,000 tokens is NOT long context (strict >)."""
        pricing = get_model_pricing("anthropic", "claude-sonnet-4-5")
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 200_000,
            "output_tokens": 100,
        })
        # Should use base rate: 200000 * input_rate + 100 * output_rate
        expected = round(200_000 * pricing["inputPerMTok"] + 100 * pricing["outputPerMTok"])
        assert result.cost_microdollars == expected

    def test_unknown_model(self):
        result = calculate_anthropic_cost_event("unknown-model", {
            "input_tokens": 1000,
            "output_tokens": 500,
        })
        assert result.cost_microdollars == 0
        assert result.cost_breakdown is None

    def test_zero_tokens(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 0,
            "output_tokens": 0,
        })
        assert result.cost_microdollars == 0

    def test_negative_tokens_clamped(self):
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": -100,
            "output_tokens": -50,
        })
        assert result.cost_microdollars == 0
        assert result.input_tokens == 0


# ---- Parameterized across all Anthropic models ----

_ANTHROPIC_MODELS = [k.split("/")[1] for k in PRICING_MAP if k.startswith("anthropic/")]

class TestAnthropicAllModels:
    @pytest.mark.parametrize("model", _ANTHROPIC_MODELS)
    def test_basic_cost_positive(self, model):
        result = calculate_anthropic_cost_event(model, {
            "input_tokens": 1000,
            "output_tokens": 500,
        })
        assert result.cost_microdollars > 0
        assert result.cost_breakdown is not None
        bd = result.cost_breakdown
        assert bd.input + bd.cached + bd.output == result.cost_microdollars

    @pytest.mark.parametrize("model", _ANTHROPIC_MODELS)
    def test_has_cache_write_rates(self, model):
        pricing = get_model_pricing("anthropic", model)
        assert pricing is not None
        assert pricing.get("cacheWrite5mPerMTok", 0) >= 0


# ---- OpenAI SSE Parser ----


class TestOpenAISSEParser:
    def test_basic_stream(self):
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hello"}}]}\n\n')
        acc.feed(b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n')
        acc.feed(b'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}\n\n')
        acc.feed(b'data: [DONE]\n\n')
        result = acc.finalize()
        assert isinstance(result, OpenAISSEResult)
        assert result.model == "gpt-4o"
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50

    def test_model_first_wins(self):
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"model": "gpt-4o"}\n\n')
        acc.feed(b'data: {"model": "gpt-4o-mini"}\n\n')
        result = acc.finalize()
        assert result.model == "gpt-4o"

    def test_usage_last_wins(self):
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}\n\n')
        acc.feed(b'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}\n\n')
        result = acc.finalize()
        assert result.usage["prompt_tokens"] == 100

    def test_empty_stream(self):
        acc = SSEAccumulator("openai")
        result = acc.finalize()
        assert result.model is None
        assert result.usage is None

    def test_malformed_json_skipped(self):
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {invalid json}\n\n')
        acc.feed(b'data: {"model": "gpt-4o"}\n\n')
        result = acc.finalize()
        assert result.model == "gpt-4o"

    def test_done_sentinel_skipped(self):
        acc = SSEAccumulator("openai")
        acc.feed(b'data: [DONE]\n\n')
        result = acc.finalize()
        assert result.model is None

    def test_multi_chunk_line_buffering(self):
        """Lines split across multiple chunks."""
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"mod')
        acc.feed(b'el": "gpt-4o"}\n\n')
        result = acc.finalize()
        assert result.model == "gpt-4o"

    def test_safety_valve_oversized_line(self):
        """Lines exceeding 64KB are dropped."""
        acc = SSEAccumulator("openai")
        huge_line = b'data: {"x": "' + b'A' * (_MAX_LINE_LENGTH + 100) + b'"}\n\n'
        acc.feed(huge_line)
        acc.feed(b'data: {"model": "gpt-4o"}\n\n')
        result = acc.finalize()
        assert result.model == "gpt-4o"  # Still works after oversized line

    def test_non_data_lines_ignored(self):
        acc = SSEAccumulator("openai")
        acc.feed(b': this is a comment\n')
        acc.feed(b'retry: 3000\n')
        acc.feed(b'data: {"model": "gpt-4o"}\n\n')
        result = acc.finalize()
        assert result.model == "gpt-4o"


# ---- Anthropic SSE Parser ----


class TestAnthropicSSEParser:
    def test_basic_stream(self):
        acc = SSEAccumulator("anthropic")
        acc.feed(b'event: message_start\n')
        acc.feed(b'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": {"input_tokens": 100, "output_tokens": 0}}}\n\n')
        acc.feed(b'event: content_block_delta\n')
        acc.feed(b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n')
        acc.feed(b'event: message_delta\n')
        acc.feed(b'data: {"type": "message_delta", "usage": {"output_tokens": 50}}\n\n')
        acc.feed(b'event: message_stop\n')
        acc.feed(b'data: {"type": "message_stop"}\n\n')
        result = acc.finalize()
        assert isinstance(result, AnthropicSSEResult)
        assert result.model == "claude-sonnet-4-5"
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 50  # Updated by delta

    def test_cache_creation_detail_extracted(self):
        acc = SSEAccumulator("anthropic")
        acc.feed(b'event: message_start\n')
        acc.feed(b'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": {"input_tokens": 100, "output_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 200}}}}\n\n')
        result = acc.finalize()
        assert result.cache_creation_detail is not None
        assert result.cache_creation_detail["ephemeral_5m_input_tokens"] == 500
        assert result.cache_creation_detail["ephemeral_1h_input_tokens"] == 200

    def test_empty_stream(self):
        acc = SSEAccumulator("anthropic")
        result = acc.finalize()
        assert result.model is None
        assert result.usage is None

    def test_malformed_json_skipped(self):
        acc = SSEAccumulator("anthropic")
        acc.feed(b'event: message_start\n')
        acc.feed(b'data: not valid json\n\n')
        acc.feed(b'event: message_start\n')
        acc.feed(b'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": {"input_tokens": 50, "output_tokens": 0}}}\n\n')
        result = acc.finalize()
        assert result.model == "claude-sonnet-4-5"


# ---- iter_sse_with_accumulator (sync tee) ----


class TestIterSSEWithAccumulator:
    def test_passthrough_integrity(self):
        """All bytes are yielded unchanged."""
        chunks = [
            b'data: {"model": "gpt-4o"}\n\n',
            b'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}\n\n',
            b'data: [DONE]\n\n',
        ]
        tee_iter, acc = iter_sse_with_accumulator(iter(chunks), "openai")
        collected = list(tee_iter)
        assert collected == chunks

        result = acc.finalize()
        assert result.model == "gpt-4o"
        assert result.usage["prompt_tokens"] == 100

    def test_empty_stream(self):
        tee_iter, acc = iter_sse_with_accumulator(iter([]), "openai")
        collected = list(tee_iter)
        assert collected == []
        result = acc.finalize()
        assert result.usage is None


# ---- Breakdown components sum to total ----


class TestBreakdownIntegrity:
    def test_openai_breakdown_sums(self):
        """For every known OpenAI model, breakdown components sum to total."""
        for model in _OPENAI_MODELS:
            result = calculate_openai_cost_event(model, {
                "prompt_tokens": 1234,
                "completion_tokens": 567,
                "prompt_tokens_details": {"cached_tokens": 200},
            })
            bd = result.cost_breakdown
            assert bd is not None
            non_reasoning = bd.input + bd.cached + bd.output
            assert non_reasoning == result.cost_microdollars, f"{model}: {non_reasoning} != {result.cost_microdollars}"

    def test_anthropic_breakdown_sums(self):
        for model in _ANTHROPIC_MODELS:
            result = calculate_anthropic_cost_event(model, {
                "input_tokens": 1234,
                "output_tokens": 567,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 100,
            })
            bd = result.cost_breakdown
            assert bd is not None
            total = bd.input + bd.cached + bd.output
            assert total == result.cost_microdollars, f"{model}: {total} != {result.cost_microdollars}"


# ---- Long context math verification ----


class TestLongContextMath:
    def test_anthropic_2x_input_multiplier(self):
        pricing = get_model_pricing("anthropic", "claude-sonnet-4-5")
        # Just above threshold
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 200_001,
            "output_tokens": 0,
        })
        # Expected: 200001 * inputPerMTok * 2
        expected = round(200_001 * pricing["inputPerMTok"] * 2.0)
        assert result.cost_microdollars == expected

    def test_anthropic_1_5x_output_multiplier(self):
        pricing = get_model_pricing("anthropic", "claude-sonnet-4-5")
        # Use large input to trigger long context, but measure output cost
        result = calculate_anthropic_cost_event("claude-sonnet-4-5", {
            "input_tokens": 200_001,
            "output_tokens": 10_000,
        })
        input_cost = round(200_001 * pricing["inputPerMTok"] * 2.0)
        output_cost = round(10_000 * pricing["outputPerMTok"] * 1.5)
        expected = input_cost + output_cost
        # Allow +-1 for rounding
        assert abs(result.cost_microdollars - expected) <= 1


# ---- Audit gap fills ----


class TestAsyncSSETee:
    @pytest.mark.asyncio
    async def test_async_passthrough_integrity(self):
        """Async tee yields all bytes unchanged while accumulating usage."""
        from nullspend._sse_parser import aiter_sse_with_accumulator

        chunks = [
            b'data: {"model": "gpt-4o"}\n\n',
            b'data: {"usage": {"prompt_tokens": 200, "completion_tokens": 80}}\n\n',
            b'data: [DONE]\n\n',
        ]

        async def async_chunks():
            for c in chunks:
                yield c

        tee_iter, acc = await aiter_sse_with_accumulator(async_chunks(), "openai")
        collected = [chunk async for chunk in tee_iter]
        assert collected == chunks

        result = acc.finalize()
        assert result.model == "gpt-4o"
        assert result.usage["prompt_tokens"] == 200
        assert result.usage["completion_tokens"] == 80

    @pytest.mark.asyncio
    async def test_async_empty_stream(self):
        from nullspend._sse_parser import aiter_sse_with_accumulator

        async def empty():
            return
            yield  # make it an async generator

        tee_iter, acc = await aiter_sse_with_accumulator(empty(), "openai")
        collected = [chunk async for chunk in tee_iter]
        assert collected == []
        result = acc.finalize()
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_async_anthropic_tee(self):
        from nullspend._sse_parser import aiter_sse_with_accumulator

        chunks = [
            b'event: message_start\n',
            b'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": {"input_tokens": 50, "output_tokens": 0}}}\n\n',
            b'event: message_delta\n',
            b'data: {"type": "message_delta", "usage": {"output_tokens": 25}}\n\n',
        ]

        async def async_chunks():
            for c in chunks:
                yield c

        tee_iter, acc = await aiter_sse_with_accumulator(async_chunks(), "anthropic")
        collected = [chunk async for chunk in tee_iter]
        assert collected == chunks

        result = acc.finalize()
        assert result.model == "claude-sonnet-4-5"
        assert result.usage["output_tokens"] == 25


class TestGoogleModelPricing:
    def test_gemini_pro_pricing_accessible(self):
        pricing = get_model_pricing("google", "gemini-2.5-pro")
        assert pricing is not None
        assert pricing["inputPerMTok"] > 0
        assert pricing["outputPerMTok"] > 0

    def test_gemini_flash_pricing_accessible(self):
        pricing = get_model_pricing("google", "gemini-2.5-flash")
        assert pricing is not None
        assert pricing["inputPerMTok"] > 0

    def test_google_is_known_model(self):
        assert is_known_model("google", "gemini-2.5-pro")
        assert is_known_model("google", "gemini-2.5-flash")
        assert not is_known_model("google", "gemini-1.0")


class TestFinalizePartialWithData:
    def test_partial_openai_stream_preserves_model(self):
        """Cancelled stream mid-read should preserve whatever was parsed."""
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hi"}}]}\n\n')
        # Stream cancelled before usage arrives
        result = acc.finalize_partial()
        assert result.model == "gpt-4o"
        assert result.usage is None  # No usage yet
        assert acc.finalized

    def test_partial_anthropic_stream_preserves_usage(self):
        acc = SSEAccumulator("anthropic")
        acc.feed(b'event: message_start\n')
        acc.feed(b'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-5", "usage": {"input_tokens": 100, "output_tokens": 0}}}\n\n')
        # Stream cancelled before message_delta or message_stop
        result = acc.finalize_partial()
        assert result.model == "claude-sonnet-4-5"
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 0  # No delta received

    def test_partial_with_buffer_residue(self):
        """Buffer has incomplete line when stream is cancelled."""
        acc = SSEAccumulator("openai")
        acc.feed(b'data: {"model": "gpt-4o"}\n\n')
        acc.feed(b'data: {"usage": {"prompt_tok')  # Incomplete line in buffer
        result = acc.finalize_partial()
        assert result.model == "gpt-4o"
        # The incomplete line couldn't be parsed, so no usage
        assert result.usage is None
