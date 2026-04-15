---
title: "Supported Models"
description: "NullSpend supports 56 models across OpenAI, Anthropic, and Google Gemini with full proxy routing, cost tracking, and budget enforcement."
---

NullSpend supports 56 models across OpenAI (26), Anthropic (22), and Google Gemini (8) with full proxy routing, cost tracking, and budget enforcement.

## Cost Formula

```
cost_microdollars = Math.round(Σ(tokens × rate_per_million_tokens))
```

Rates are in **dollars per million tokens**. The result is in **microdollars** (1 microdollar = $0.000001).

For the full calculation logic including cached tokens, cache writes, and long context multipliers, see [Cost Tracking](../features/cost-tracking.md).

## OpenAI Models

26 models. Rates in $/MTok.

| Model | Input | Cached Input | Output |
|---|---|---|---|
| `gpt-4o` | 2.50 | 1.25 | 10.00 |
| `gpt-4o-mini` | 0.15 | 0.075 | 0.60 |
| `gpt-4.1` | 2.00 | 0.50 | 8.00 |
| `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 |
| `gpt-4.1-nano` | 0.10 | 0.025 | 0.40 |
| `o4-mini` | 1.10 | 0.275 | 4.40 |
| `o3` | 2.00 | 0.50 | 8.00 |
| `o3-mini` | 1.10 | 0.55 | 4.40 |
| `o3-pro` | 20.00 | 20.00 | 80.00 |
| `o1` | 15.00 | 7.50 | 60.00 |
| `o1-pro` | 150.00 | 150.00 | 600.00 |
| `o1-mini` | 1.10 | 0.55 | 4.40 |
| `gpt-5` | 1.25 | 0.125 | 10.00 |
| `gpt-5-mini` | 0.25 | 0.025 | 2.00 |
| `gpt-5-nano` | 0.05 | 0.005 | 0.40 |
| `gpt-5-pro` | 15.00 | 15.00 | 120.00 |
| `gpt-5.1` | 1.25 | 0.125 | 10.00 |
| `gpt-5.2` | 1.75 | 0.175 | 14.00 |
| `gpt-5.2-pro` | 21.00 | 21.00 | 168.00 |
| `gpt-5.4` | 2.50 | 0.25 | 15.00 |
| `gpt-5.4-mini` | 0.75 | 0.075 | 4.50 |
| `gpt-5.4-nano` | 0.20 | 0.02 | 1.25 |
| `gpt-5.4-pro` | 30.00 | 30.00 | 180.00 |
| `o3-deep-research` | 10.00 | 2.50 | 40.00 |
| `o4-mini-deep-research` | 2.00 | 0.50 | 8.00 |
| `computer-use-preview` | 3.00 | 3.00 | 12.00 |

OpenAI cost formula: `(prompt_tokens - cached_tokens) × input + cached_tokens × cached + completion_tokens × output`. Reasoning tokens are a subset of completion tokens — not double-counted.

## Anthropic Models

22 models (10 aliases + 12 dated variants). Rates in $/MTok.

### Aliases

| Model | Input | Cached Input | Cache Write (5m) | Cache Write (1h) | Output |
|---|---|---|---|---|---|
| `claude-opus-4-6` | 5.00 | 0.50 | 6.25 | 10.00 | 25.00 |
| `claude-opus-4-5` | 5.00 | 0.50 | 6.25 | 10.00 | 25.00 |
| `claude-opus-4-1` | 15.00 | 1.50 | 18.75 | 30.00 | 75.00 |
| `claude-opus-4` | 15.00 | 1.50 | 18.75 | 30.00 | 75.00 |
| `claude-sonnet-4-6` | 3.00 | 0.30 | 3.75 | 6.00 | 15.00 |
| `claude-sonnet-4-5` | 3.00 | 0.30 | 3.75 | 6.00 | 15.00 |
| `claude-sonnet-4` | 3.00 | 0.30 | 3.75 | 6.00 | 15.00 |
| `claude-haiku-4-5` | 1.00 | 0.10 | 1.25 | 2.00 | 5.00 |
| `claude-haiku-3.5` | 0.80 | 0.08 | 1.00 | 1.60 | 4.00 |
| `claude-haiku-3` | 0.25 | 0.03 | 0.30 | 0.50 | 1.25 |

### Dated Variants

Dated variants share the exact same rates as their alias:

| Model | Same Rates As |
|---|---|
| `claude-opus-4-6-20260205` | `claude-opus-4-6` |
| `claude-sonnet-4-6-20260217` | `claude-sonnet-4-6` |
| `claude-sonnet-4-5-20250929` | `claude-sonnet-4-5` |
| `claude-opus-4-5-20251101` | `claude-opus-4-5` |
| `claude-haiku-4-5-20251001` | `claude-haiku-4-5` |
| `claude-opus-4-1-20250805` | `claude-opus-4-1` |
| `claude-opus-4-20250514` | `claude-opus-4` |
| `claude-sonnet-4-20250514` | `claude-sonnet-4` |
| `claude-3-5-haiku-20241022` | `claude-haiku-3.5` |
| `claude-3-haiku-20240307` | `claude-haiku-3` |
| `claude-opus-4-0` | `claude-opus-4` |
| `claude-sonnet-4-0` | `claude-sonnet-4` |

### Long Context Pricing

When total input tokens (input + cache creation + cache read) exceed **200,000 tokens**, multipliers apply:

| Component | Multiplier |
|---|---|
| Input | 2× |
| Cached Input (read) | 2× |
| Cache Write (5m and 1h) | 2× |
| Output | 1.5× |

### Cache Write TTLs

Anthropic offers two cache write tiers:

| Tier | TTL | Rate Column |
|---|---|---|
| Ephemeral (5-minute) | 5 minutes | Cache Write (5m) |
| Extended (1-hour) | 1 hour | Cache Write (1h) |

If the response includes `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`, each is priced at its respective rate. Otherwise, all cache creation tokens use the 5-minute rate.

## Google Gemini Models

8 models. Rates in $/MTok. Proxy routes natively via `/v1beta/models/{model}:generateContent`.

| Model | Input | Cached Input | Output |
|---|---|---|---|
| `gemini-2.5-pro` | 1.25 | 0.125 | 10.00 |
| `gemini-2.5-flash` | 0.30 | 0.03 | 2.50 |
| `gemini-2.5-flash-lite` | 0.10 | 0.01 | 0.40 |
| `gemini-2.0-flash` | 0.10 | 0.025 | 0.40 |
| `gemini-2.0-flash-lite` | 0.075 | — | 0.30 |
| `gemini-3-flash-preview` | 0.50 | 0.05 | 3.00 |
| `gemini-3.1-pro-preview` | 2.00 | 0.20 | 12.00 |
| `gemini-3.1-flash-lite-preview` | 0.25 | 0.025 | 1.50 |

Gemini cost formula: `(promptTokenCount - cachedContentTokenCount) × input + cachedContentTokenCount × cached + candidatesTokenCount × output`. Thinking tokens (`thoughtsTokenCount`) are a subset of output, tracked as `_ns_thinking_tokens` tag.

**Model aliases:** Dated model names (e.g., `gemini-2.5-flash-preview-04-17`) are resolved to their base model for pricing via prefix matching.

**Tiered pricing:** `gemini-2.5-pro` and `gemini-3.1-pro-preview` charge 2× rates for prompts exceeding 200K tokens. NullSpend currently uses the ≤200K rate. Long-context cost underreporting for these two models is a known limitation.
