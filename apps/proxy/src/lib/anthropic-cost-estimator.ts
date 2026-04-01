import { getModelPricing, costComponent } from "@nullspend/cost-engine";

const SAFETY_MARGIN = 1.1;
const CHARS_PER_TOKEN = 4;
const UNKNOWN_MODEL_FALLBACK_MICRODOLLARS = 1_000_000; // $1

/**
 * Anthropic-specific maximum output token caps. Anthropic requires `max_tokens`
 * in the request body, so the explicit cap path is almost always taken. This map
 * is a defensive fallback only.
 */
const MODEL_OUTPUT_CAPS: Record<string, number> = {
  "claude-opus-4-6": 128_000,
  "claude-opus-4-6-20260205": 128_000,
  "claude-opus-4-5": 128_000,
  "claude-opus-4-5-20251101": 128_000,

  "claude-sonnet-4-6": 64_000,
  "claude-sonnet-4-6-20260217": 64_000,
  "claude-sonnet-4-5": 64_000,
  "claude-sonnet-4-5-20250929": 64_000,
  "claude-sonnet-4": 64_000,
  "claude-sonnet-4-20250514": 64_000,
  "claude-sonnet-4-0": 64_000,

  "claude-opus-4-1": 64_000,
  "claude-opus-4-1-20250805": 64_000,
  "claude-opus-4": 64_000,
  "claude-opus-4-20250514": 64_000,
  "claude-opus-4-0": 64_000,

  "claude-haiku-4-5": 64_000,
  "claude-haiku-4-5-20251001": 64_000,

  "claude-haiku-3.5": 8_000,
  "claude-3-5-haiku-20241022": 8_000,

  "claude-haiku-3": 4_000,
  "claude-3-haiku-20240307": 4_000,
};

const DEFAULT_OUTPUT_CAP = 64_000;

/**
 * Estimate the maximum cost of an Anthropic request in microdollars.
 *
 * Uses body byte-length as a rough proxy for input tokens (~4 chars/token)
 * and the explicit output cap (or model-specific default) for output tokens.
 * Multiplied by a 1.1x safety margin.
 *
 * Returns an integer (microdollars) for budget reservation.
 */
export function estimateAnthropicMaxCost(
  model: string,
  body: Record<string, unknown>,
  bodyByteLength?: number,
): number {
  const pricing = getModelPricing("anthropic", model);

  if (!pricing) {
    return UNKNOWN_MODEL_FALLBACK_MICRODOLLARS;
  }

  const inputTokenEstimate = Math.ceil((bodyByteLength ?? JSON.stringify(body).length) / CHARS_PER_TOKEN);

  const explicitOutputCap = body.max_tokens as number | undefined;

  const outputTokenEstimate =
    explicitOutputCap ?? (MODEL_OUTPUT_CAPS[model] ?? DEFAULT_OUTPUT_CAP);

  // Apply long-context multipliers (matches anthropic-cost-calculator.ts logic)
  const isLongContext = inputTokenEstimate > 200_000;
  const inputRate = isLongContext ? pricing.inputPerMTok * 2 : pricing.inputPerMTok;
  const outputRate = isLongContext ? pricing.outputPerMTok * 1.5 : pricing.outputPerMTok;

  const inputCost = costComponent(inputTokenEstimate, inputRate);
  const outputCost = costComponent(outputTokenEstimate, outputRate);

  return Math.round((inputCost + outputCost) * SAFETY_MARGIN);
}
