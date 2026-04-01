export type Provider = "openai" | "anthropic" | "google";

export interface ModelPricing {
  inputPerMTok: number;
  cachedInputPerMTok: number;
  outputPerMTok: number;
  /** Anthropic 5-minute cache write rate (1.25x base input). Only present for Anthropic models. */
  cacheWrite5mPerMTok?: number;
  /** Anthropic 1-hour extended cache write rate (2.0x base input). Only present for Anthropic models. */
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
