# Cost Tracking

NullSpend calculates the cost of every LLM request automatically. No SDK or code changes beyond the initial proxy setup.

## How It Works

```
Your App ──► NullSpend Proxy ──► Provider (OpenAI / Anthropic)
                                        │
                ◄── response ───────────┘
                │
                ├─ Extract token counts from response
                ├─ Look up model pricing
                ├─ Calculate cost in microdollars
                └─ Log cost event (async, never blocks response)
```

The proxy extracts usage data from the provider's response — token counts for input, output, cached, and reasoning tokens — then calculates cost using the model's pricing rates. Cost is expressed in **microdollars** (1 microdollar = $0.000001, so $1.00 = 1,000,000 microdollars).

Cost logging is asynchronous. It never adds latency to your response.

## OpenAI Cost Formula

```
normalInputTokens = prompt_tokens - cached_tokens
cost = (normalInputTokens × inputRate)
     + (cachedTokens × cachedInputRate)
     + (completionTokens × outputRate)
```

- **Reasoning tokens** (from o3, o4-mini, o1) are a subset of `completion_tokens`. They are tracked separately for attribution but are **not** double-counted — the output cost already includes them.
- The proxy tries `model` from the request first, then falls back to the model in the response (handles aliases).
- Each component is rounded individually, then a residual correction is applied to the largest component to guarantee the parts sum to the total.

### Example

A `gpt-4o` request with 1,000 input tokens (200 cached) and 500 output tokens:

```
normalInput = 1000 - 200 = 800
inputCost   = 800 × 2.50 / 1,000,000 = 0.002 → 2 microdollars
cachedCost  = 200 × 1.25 / 1,000,000 = 0.00025 → 0 microdollars
outputCost  = 500 × 10.00 / 1,000,000 = 0.005 → 5 microdollars
total       = 7 microdollars ($0.000007)
```

## Anthropic Cost Formula

```
cost = (input_tokens × inputRate)
     + (cache_creation_tokens × cacheWriteRate)
     + (cache_read_tokens × cachedInputRate)
     + (output_tokens × outputRate)
```

Key differences from OpenAI:

- **`input_tokens` is already uncached.** No subtraction needed — Anthropic separates input and cache read tokens in the response.
- **Cache write has two TTL tiers.** If the response includes `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`, each is priced at its respective rate. Otherwise, all cache creation tokens use the 5-minute rate.
- **Long context pricing.** When `input_tokens + cache_creation_tokens + cache_read_tokens > 200,000`, multipliers apply:
  - Input, cached input, cache write: **2x**
  - Output: **1.5x**
- **Thinking tokens** (from Claude with extended thinking) are included in `output_tokens` and priced at the output rate.

### Example

A `claude-sonnet-4-5` request with 5,000 input tokens, 1,000 cache read tokens, and 2,000 output tokens:

```
inputCost  = 5000 × 3.00 / 1,000,000 = 0.015 → 15 microdollars
cachedCost = 1000 × 0.30 / 1,000,000 = 0.0003 → 0 microdollars
outputCost = 2000 × 15.00 / 1,000,000 = 0.03 → 30 microdollars
total      = 45 microdollars ($0.000045)
```

## Pre-Request Cost Estimation

Before forwarding a request, the proxy estimates the maximum possible cost. This estimate is used for [budget enforcement](budgets.md) — if the estimate exceeds the remaining budget, the request is blocked before reaching the provider.

**Formula:**

```
inputTokenEstimate  = ceil(JSON.stringify(requestBody).length / 4)
outputTokenEstimate = max_completion_tokens ?? max_tokens ?? modelCap ?? defaultCap

estimatedCost = round((inputCost + outputCost) × 1.1)
```

- The `÷ 4` ratio approximates 4 characters per token.
- The `1.1×` safety margin accounts for estimation imprecision.
- If the model is unknown, the fallback estimate is $1.00 (1,000,000 microdollars).

**Default output caps by model:**

| Models | Default Cap |
|---|---|
| o3, o3-mini, o4-mini, o1 | 100,000 tokens |
| claude-opus-4-6, claude-opus-4-5 | 128,000 tokens |
| claude-sonnet-4-6, claude-sonnet-4-5, claude-opus-4-1, claude-haiku-4-5 | 64,000 tokens |
| claude-haiku-3.5 | 8,000 tokens |
| claude-haiku-3 | 4,000 tokens |
| All other OpenAI models | 16,384 tokens |
| All other Anthropic models | 64,000 tokens |

## What's Recorded

Every request produces a **cost event** with these fields:

| Field | Type | Description |
|---|---|---|
| `requestId` | string | Unique request identifier |
| `provider` | string | `"openai"` or `"anthropic"` |
| `model` | string | Model used (e.g., `gpt-4o`, `claude-sonnet-4-5`) |
| `inputTokens` | integer | Total input tokens (OpenAI: prompt_tokens; Anthropic: input + cache creation + cache read) |
| `outputTokens` | integer | Output/completion tokens |
| `cachedInputTokens` | integer | Cached input tokens (OpenAI: cached_tokens; Anthropic: cache_read_tokens) |
| `reasoningTokens` | integer | Reasoning tokens (OpenAI only, subset of output) |
| `costMicrodollars` | integer | Total cost in microdollars |
| `costBreakdown` | object | Per-component costs: `{ input, cached, output }` |
| `durationMs` | integer | Total request duration in milliseconds |
| `sessionId` | string? | Session ID from `X-NullSpend-Session` header |
| `traceId` | string? | [Trace ID](tracing.md) (from `traceparent`, `X-NullSpend-Trace-Id`, or auto-generated) |
| `source` | string | `"proxy"`, `"api"`, or `"mcp"` |
| `tags` | object | Key-value pairs from `X-NullSpend-Tags` header |
| `apiKeyId` | string | API key that made the request |
| `createdAt` | timestamp | When the event was recorded |

Cost events are deduplicated by `(requestId, provider)` — reprocessing the same request is safe.

## Cost Sources

| Source | How It Works |
|---|---|
| **proxy** | Automatic. Every request through `proxy.nullspend.com` generates a cost event. |
| **api** | Manual. POST to `/api/cost-events` with token counts. Used by the SDK or direct HTTP. |
| **mcp** | MCP tool calls. Budget check and cost event ingestion via `/v1/mcp/budget/check` and `/v1/mcp/events`. |

## Cancelled Streams

When a streaming response is cancelled before completion (client disconnects), the proxy cannot extract final token counts from the response. Instead, it logs an **estimated** cost event with two system tags:

- `_ns_estimated: "true"` — cost is an estimate, not exact
- `_ns_cancelled: "true"` — the stream was cancelled

The estimate uses the pre-request estimation formula. These events are included in budget spend tracking to prevent cancelled streams from creating a gap between tracked and actual spend.

## Where to See Costs

- **Dashboard** — Daily spend chart, model breakdown, per-key breakdown at [nullspend.com/app/analytics](https://nullspend.com/app/analytics)
- **API** — [`GET /api/cost-events`](../api-reference/cost-events-api.md#list-cost-events) with filters for model, provider, API key, tags, trace ID, and more. Ingest custom events with [`POST /api/cost-events`](../api-reference/cost-events-api.md#ingest-single-event).
- **Webhooks** — `cost_event.created` fires for every cost event. Supports [full and thin payload modes](../webhooks/overview.md).

## Related

- [Supported Models](../reference/supported-models.md) — full pricing table for all 38 models
- [Tracing](tracing.md) — correlate requests across a multi-step agent run
- [Tags](tags.md) — attribute costs to teams, environments, and features
- [Budgets](budgets.md) — enforce spending ceilings based on cost tracking
- [Webhooks](../webhooks/overview.md) — get notified on every cost event
