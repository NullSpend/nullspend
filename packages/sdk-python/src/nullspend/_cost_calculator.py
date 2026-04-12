"""Cost calculation for OpenAI and Anthropic API responses.

Ports the exact math from packages/sdk/src/cost-calculator.ts and
packages/cost-engine/src/pricing.ts. Pricing data loaded from
_pricing_data.json (bundled in package).

Key design decisions:
- cost_component() returns unrounded float microdollars
- Total cost is rounded ONCE at the end (avoids accumulating per-component errors)
- Rounding residual is distributed to the largest component
- Anthropic long-context: >200K total input tokens triggers rate multipliers
  (2x input/cached/cache-write, 1.5x output)
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, TypedDict

from nullspend.types import CostBreakdown, CostEventInput


# ---- Pricing data ----


class ModelPricing(TypedDict, total=False):
    inputPerMTok: float
    cachedInputPerMTok: float
    outputPerMTok: float
    cacheWrite5mPerMTok: float   # Anthropic only
    cacheWrite1hPerMTok: float   # Anthropic only


def _load_pricing_map() -> dict[str, ModelPricing]:
    """Load pricing-data.json from the package."""
    ref = resources.files("nullspend").joinpath("_pricing_data.json")
    data = ref.read_text(encoding="utf-8")
    return json.loads(data)


PRICING_MAP: dict[str, ModelPricing] = _load_pricing_map()

# Anthropic long-context threshold (strict >)
_LONG_CONTEXT_THRESHOLD = 200_000


def get_model_pricing(provider: str, model: str) -> ModelPricing | None:
    return PRICING_MAP.get(f"{provider}/{model}")


def is_known_model(provider: str, model: str) -> bool:
    return f"{provider}/{model}" in PRICING_MAP


# ---- Core math ----


def cost_component(tokens: int, rate_per_mtok: float) -> float:
    """Compute a single cost component in unrounded microdollars.

    Dimensional analysis: tokens x ($/MTok) = microdollars
    (because $/MTok = $ per 10^6 tokens, and 10^6 microdollars = $1,
     so tokens * rate gives microdollars directly).
    """
    if tokens <= 0 or rate_per_mtok <= 0:
        return 0.0
    return float(tokens) * rate_per_mtok


def _distribute_residual(
    total: int,
    input_raw: float,
    cached_raw: float,
    output_raw: float,
) -> tuple[int, int, int]:
    """Round each component and distribute rounding residual to the largest."""
    r_input = round(input_raw)
    r_cached = round(cached_raw)
    r_output = round(output_raw)
    residual = total - (r_input + r_cached + r_output)

    if residual != 0:
        if output_raw >= input_raw and output_raw >= cached_raw:
            r_output += residual
        elif input_raw >= cached_raw:
            r_input += residual
        else:
            r_cached += residual

    return r_input, r_cached, r_output


# ---- OpenAI ----


def calculate_openai_cost_event(
    model: str,
    usage: dict[str, Any],
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> CostEventInput:
    """Calculate cost for an OpenAI API response.

    Args:
        model: The model name (e.g., "gpt-4o")
        usage: The usage object from the API response. Expected shape:
            {
                "prompt_tokens": int,
                "completion_tokens": int,
                "prompt_tokens_details": {"cached_tokens": int},
                "completion_tokens_details": {"reasoning_tokens": int},
            }
        duration_ms: Request duration in milliseconds
        metadata: Optional dict with sessionId, traceId, tags, customer
    """
    meta = metadata or {}

    # Step 1: Extract and clamp token counts
    prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
    completion_tokens = max(0, int(usage.get("completion_tokens") or 0))

    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}

    cached_tokens = max(0, int(prompt_details.get("cached_tokens") or 0))
    reasoning_tokens = max(0, int(completion_details.get("reasoning_tokens") or 0))

    # Step 2: Normal (non-cached) input tokens
    normal_input_tokens = prompt_tokens - cached_tokens

    # Step 3: Look up pricing
    pricing = get_model_pricing("openai", model)

    # Step 4: Calculate cost
    cost_microdollars = 0
    breakdown: CostBreakdown | None = None

    if pricing:
        input_raw = cost_component(normal_input_tokens, pricing["inputPerMTok"])
        cached_raw = cost_component(cached_tokens, pricing["cachedInputPerMTok"])
        output_raw = cost_component(completion_tokens, pricing["outputPerMTok"])

        cost_microdollars = max(0, round(input_raw + cached_raw + output_raw))

        adj_input, adj_cached, adj_output = _distribute_residual(
            cost_microdollars, input_raw, cached_raw, output_raw,
        )

        reasoning_cost: int | None = None
        if reasoning_tokens > 0:
            reasoning_cost = round(cost_component(reasoning_tokens, pricing["outputPerMTok"]))

        breakdown = CostBreakdown(
            input=adj_input,
            output=adj_output,
            cached=adj_cached,
            reasoning=reasoning_cost,
        )

    return CostEventInput(
        provider="openai",
        model=model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_microdollars=cost_microdollars,
        cost_breakdown=breakdown,
        duration_ms=duration_ms,
        session_id=meta.get("sessionId") or meta.get("session_id"),
        trace_id=meta.get("traceId") or meta.get("trace_id"),
        tags=meta.get("tags"),
        customer=meta.get("customer"),
        event_type="llm",
    )


# ---- Anthropic ----


def calculate_anthropic_cost_event(
    model: str,
    usage: dict[str, Any],
    cache_creation_detail: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> CostEventInput:
    """Calculate cost for an Anthropic API response.

    Args:
        model: The model name (e.g., "claude-sonnet-4-5")
        usage: The usage object from the API response. Expected shape:
            {
                "input_tokens": int,
                "output_tokens": int,
                "cache_creation_input_tokens": int,
                "cache_read_input_tokens": int,
            }
        cache_creation_detail: Optional ephemeral cache split. Shape:
            {
                "ephemeral_5m_input_tokens": int,
                "ephemeral_1h_input_tokens": int,
            }
        duration_ms: Request duration in milliseconds
        metadata: Optional dict with sessionId, traceId, tags, customer
    """
    meta = metadata or {}

    # Step 1: Extract and clamp
    input_tokens = max(0, int(usage.get("input_tokens") or 0))
    output_tokens = max(0, int(usage.get("output_tokens") or 0))
    cache_creation_tokens = max(0, int(usage.get("cache_creation_input_tokens") or 0))
    cache_read_tokens = max(0, int(usage.get("cache_read_input_tokens") or 0))

    # Step 2: Total input for long-context detection
    total_input_tokens = input_tokens + cache_creation_tokens + cache_read_tokens

    # Step 3: Look up pricing
    pricing = get_model_pricing("anthropic", model)

    # Step 4: Long-context multipliers
    is_long_context = total_input_tokens > _LONG_CONTEXT_THRESHOLD

    if pricing:
        input_rate = pricing["inputPerMTok"] * (2.0 if is_long_context else 1.0)
        cache_read_rate = pricing["cachedInputPerMTok"] * (2.0 if is_long_context else 1.0)
        output_rate = pricing["outputPerMTok"] * (1.5 if is_long_context else 1.0)

        cw5m_base = pricing.get("cacheWrite5mPerMTok", 0) or 0
        cw1h_base = pricing.get("cacheWrite1hPerMTok", 0) or 0
        cache_write_5m_rate = cw5m_base * (2.0 if is_long_context else 1.0)
        cache_write_1h_rate = cw1h_base * (2.0 if is_long_context else 1.0)
    else:
        input_rate = cache_read_rate = output_rate = 0.0
        cache_write_5m_rate = cache_write_1h_rate = 0.0

    # Step 5: Cache write cost (may split by TTL)
    if cache_creation_detail and "ephemeral_5m_input_tokens" in cache_creation_detail:
        tokens_5m = max(0, int(cache_creation_detail.get("ephemeral_5m_input_tokens") or 0))
        tokens_1h = max(0, int(cache_creation_detail.get("ephemeral_1h_input_tokens") or 0))
        cache_write_cost = (
            cost_component(tokens_5m, cache_write_5m_rate) +
            cost_component(tokens_1h, cache_write_1h_rate)
        )
    else:
        # Fallback: all cache creation tokens at 5m rate
        cache_write_cost = cost_component(cache_creation_tokens, cache_write_5m_rate)

    # Step 6: Calculate cost
    cost_microdollars = 0
    breakdown: CostBreakdown | None = None

    if pricing:
        input_raw = cost_component(input_tokens, input_rate)
        cached_raw = cache_write_cost + cost_component(cache_read_tokens, cache_read_rate)
        output_raw = cost_component(output_tokens, output_rate)

        cost_microdollars = max(0, round(input_raw + cached_raw + output_raw))

        adj_input, adj_cached, adj_output = _distribute_residual(
            cost_microdollars, input_raw, cached_raw, output_raw,
        )

        breakdown = CostBreakdown(
            input=adj_input,
            output=adj_output,
            cached=adj_cached,
        )

    return CostEventInput(
        provider="anthropic",
        model=model,
        input_tokens=total_input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read_tokens,
        reasoning_tokens=0,
        cost_microdollars=cost_microdollars,
        cost_breakdown=breakdown,
        duration_ms=duration_ms,
        session_id=meta.get("sessionId") or meta.get("session_id"),
        trace_id=meta.get("traceId") or meta.get("trace_id"),
        tags=meta.get("tags"),
        customer=meta.get("customer"),
        event_type="llm",
    )
