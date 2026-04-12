# @nullspend/cost-engine

Model pricing catalog and cost calculation engine for OpenAI, Anthropic, and Google AI.

## Supported Models (47 total)

- **OpenAI** (23): gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro, gpt-5.3-chat-latest, gpt-5.3-codex, gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o4-mini, o3, o3-mini, o1, o3-deep-research, o4-mini-deep-research, computer-use-preview
- **Anthropic** (22): claude-sonnet-4-6, claude-opus-4-6, claude-sonnet-4-5, claude-opus-4-5, claude-opus-4-1, claude-opus-4, claude-sonnet-4, claude-haiku-4-5, claude-haiku-3.5, claude-haiku-3, claude-opus-4-0, claude-sonnet-4-0 (plus 10 dated variants)
- **Google** (2): gemini-2.5-pro, gemini-2.5-flash

## Install

```bash
npm install @nullspend/cost-engine
```

## Quick Start

```typescript
import { getModelPricing, costComponent, isKnownModel } from "@nullspend/cost-engine";

// Check if a model is in the catalog
isKnownModel("openai", "gpt-4o"); // true

// Get pricing rates ($/million tokens)
const pricing = getModelPricing("openai", "gpt-4o");
if (pricing) {
  // { inputPerMTok: 2.50, cachedInputPerMTok: 1.25, outputPerMTok: 10.00 }

  // Calculate cost for a token component (returns unrounded microdollars)
  const inputCost = costComponent(1500, pricing.inputPerMTok);
  const outputCost = costComponent(500, pricing.outputPerMTok);
  const totalMicrodollars = Math.round(inputCost + outputCost);
}
```

## API

### `getModelPricing(provider: string, model: string): ModelPricing | null`

Returns pricing rates for a model, or `null` if not in the catalog.

```typescript
getModelPricing("anthropic", "claude-sonnet-4-6");
// { inputPerMTok: 3.0, cachedInputPerMTok: 0.3, outputPerMTok: 15.0,
//   cacheWrite5mPerMTok: 3.75, cacheWrite1hPerMTok: 6.0 }
```

### `isKnownModel(provider: string, model: string): boolean`

Returns `true` if the model exists in the pricing catalog.

### `costComponent(tokens: number, ratePerMTok: number): number`

Calculates cost in microdollars (unrounded float) for a given token count and rate. Sum all components and call `Math.round()` once at the end to avoid accumulation drift.

### `getAllPricing(): Readonly<Record<string, ModelPricing>>`

Returns the complete frozen pricing catalog keyed by `"provider/model"`.

## Types

```typescript
type Provider = "openai" | "anthropic" | "google";

interface ModelPricing {
  inputPerMTok: number;
  cachedInputPerMTok: number;
  outputPerMTok: number;
  cacheWrite5mPerMTok?: number;   // Anthropic only — 5-min cache write (1.25x input)
  cacheWrite1hPerMTok?: number;   // Anthropic only — 1-hour cache write (2.0x input)
}

interface CostEvent {
  requestId: string;
  provider: Provider;
  model: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  costMicrodollars: number;
  durationMs?: number;
}
```

## Cost Units

All costs are in **microdollars** (1 microdollar = $0.000001, so $1 = 1,000,000 microdollars). Rates in `ModelPricing` are **$/million tokens**. This avoids floating-point precision issues common with fractional dollar amounts.

## License

Apache-2.0
