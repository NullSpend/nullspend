import type { ModelPricing } from "./types.js";
import pricingData from "./pricing-data.json";

const pricingMap = Object.freeze(pricingData as Record<string, ModelPricing>);

/**
 * Look up pricing for a model. Returns null if the model is not in the
 * curated pricing database.
 *
 * @param provider - e.g. "openai", "anthropic", "google"
 * @param model - e.g. "gpt-4o", "claude-sonnet-4-6", "gemini-2.5-flash"
 */
export function getModelPricing(
  provider: string,
  model: string,
): ModelPricing | null {
  return pricingMap[`${provider}/${model}`] ?? null;
}

/**
 * Check if a model has pricing data (i.e. is in the allowlist).
 * Unknown models have no cost tracking, so the proxy should reject them.
 */
export function isKnownModel(provider: string, model: string): boolean {
  return `${provider}/${model}` in pricingMap;
}

/**
 * Return the full pricing catalog as a record of "provider/model" → ModelPricing.
 * Used by the policy endpoint to find cheapest allowed models.
 */
export function getAllPricing(): Readonly<Record<string, ModelPricing>> {
  return pricingMap;
}

/**
 * Compute a single cost component in **unrounded microdollars** (float).
 *
 * Dimensional analysis:
 *   tokens × ($/MTok) = tokens × ($ / 10^6 tokens) × 10^6 µ$/$ = microdollars
 *
 * The caller sums all components and calls `Math.round()` once to get the
 * final integer microdollar value. Rounding once at the end avoids
 * accumulating per-component rounding errors.
 */
export function costComponent(tokens: number, ratePerMTok: number): number {
  if (tokens <= 0 || ratePerMTok <= 0) return 0;
  return tokens * ratePerMTok;
}
