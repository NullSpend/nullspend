# Supported Models

NullSpend supports 45 models across OpenAI (23) and Anthropic (22) with full proxy routing, cost tracking, and budget enforcement.

## Cost Formula

```
cost_microdollars = Math.round(Σ(tokens × rate_per_million_tokens))
```

Rates are in **dollars per million tokens**. The result is in **microdollars** (1 microdollar = $0.000001).

For the full calculation logic including cached tokens, cache writes, and long context multipliers, see [Cost Tracking](../features/cost-tracking.md).

## OpenAI Models

23 models. Rates in $/MTok.

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
| `o1` | 15.00 | 7.50 | 60.00 |
| `gpt-5` | 1.25 | 0.125 | 10.00 |
| `gpt-5-mini` | 0.25 | 0.025 | 2.00 |
| `gpt-5-nano` | 0.05 | 0.005 | 0.40 |
| `gpt-5.1` | 1.25 | 0.125 | 10.00 |
| `gpt-5.2` | 1.75 | 0.175 | 14.00 |
| `gpt-5.3-chat-latest` | 1.75 | 0.175 | 14.00 |
| `gpt-5.3-codex` | 1.75 | 0.175 | 14.00 |
| `gpt-5.4` | 2.50 | 0.25 | 15.00 |
| `gpt-5.4-mini` | 0.75 | 0.075 | 4.50 |
| `gpt-5.4-nano` | 0.20 | 0.02 | 1.25 |
| `gpt-5.4-pro` | 30.00 | 30.00 | 180.00 |
| `o3-deep-research` | 5.00 | 5.00 | 20.00 |
| `o4-mini-deep-research` | 1.00 | 1.00 | 4.00 |
| `computer-use-preview` | 1.50 | 1.50 | 6.00 |

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

## Unknown Models

If a request uses a model not in the pricing catalog, the proxy returns `400` with error code `invalid_model`. See the [error reference](../api-reference/errors.md#request-validation) for details.

To request a new model, contact support.

## Related

- [Cost Tracking](../features/cost-tracking.md) — full cost calculation formulas for each provider
- [Error Reference](../api-reference/errors.md) — `invalid_model` error details
