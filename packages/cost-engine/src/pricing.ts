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
 * Scale factor for pricing rates stored in pricing-data.json.
 *
 * Rates are stored as **integer microdollars per million tokens** (µ$/MTok).
 * Example: $2.50/MTok → 2,500,000 µ$/MTok.
 *
 * This eliminates IEEE 754 float imprecision in the multiplication step:
 * `tokens * intRate` is exact (integer × integer), and the single division
 * by RATE_SCALE introduces at most 1 ULP of error.
 */
export const RATE_SCALE = 1_000_000;

/**
 * Compute a single cost component in **unrounded microdollars** (float).
 *
 * Pricing rates are integer µ$/MTok (see {@link RATE_SCALE}). The formula:
 *   microdollars = tokens × rateMicroPerMTok / RATE_SCALE
 *
 * The multiplication `tokens × rate` is exact (both integers), and the
 * single division introduces at most 1 ULP of error. The caller sums all
 * components and calls `Math.round()` once to get the final integer
 * microdollar value.
 *
 * Overflow safety: 10M tokens × 180,000,000 µ$/MTok = 1.8×10¹⁵, within
 * Number.MAX_SAFE_INTEGER (9×10¹⁵).
 */
export function costComponent(tokens: number, rateMicroPerMTok: number): number {
  if (tokens <= 0 || rateMicroPerMTok <= 0) return 0;
  return (tokens * rateMicroPerMTok) / RATE_SCALE;
}
