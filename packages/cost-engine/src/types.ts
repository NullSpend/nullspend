export type Provider = "openai" | "anthropic" | "google";

/**
 * Pricing rates for a model, stored as **integer microdollars per million
 * tokens** (µ$/MTok). Multiply the old $/MTok value by 1,000,000 to get
 * these values (e.g., $2.50/MTok → 2,500,000).
 *
 * Use {@link costComponent} to convert token counts + rates to microdollars.
 */
export interface ModelPricing {
  /** Input token rate in integer µ$/MTok. */
  inputPerMTok: number;
  /** Cached input token rate in integer µ$/MTok. */
  cachedInputPerMTok: number;
  /** Output token rate in integer µ$/MTok. */
  outputPerMTok: number;
  /** Anthropic 5-minute cache write rate (1.25x base input) in integer µ$/MTok. Only present for Anthropic models. */
  cacheWrite5mPerMTok?: number;
  /** Anthropic 1-hour extended cache write rate (2.0x base input) in integer µ$/MTok. Only present for Anthropic models. */
  cacheWrite1hPerMTok?: number;
}

export interface CostEvent {
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
