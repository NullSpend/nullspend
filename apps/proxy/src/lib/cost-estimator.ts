import { getModelPricing, costComponent } from "@nullspend/cost-engine";

const SAFETY_MARGIN = 1.1;
const CHARS_PER_TOKEN = 4;
const UNKNOWN_MODEL_FALLBACK_MICRODOLLARS = 1_000_000; // $1

/**
 * Model-specific maximum output token caps used when the request doesn't
 * specify `max_tokens` or `max_completion_tokens`. These represent the
 * model's actual output limit — the worst-case scenario for cost.
 */
const MODEL_OUTPUT_CAPS: Record<string, number> = {
  "gpt-4o": 16_384,
  "gpt-4o-mini": 16_384,
  "gpt-4.1": 16_384,
  "gpt-4.1-mini": 16_384,
  "gpt-4.1-nano": 16_384,
  "o3": 100_000,
  "o3-mini": 100_000,
  "o4-mini": 100_000,
  "o1": 100_000,
  "gpt-5": 16_384,
  "gpt-5-mini": 16_384,
  "gpt-5-nano": 16_384,
  "gpt-5.1": 16_384,
  "gpt-5.2": 16_384,
};

const DEFAULT_OUTPUT_CAP = 16_384;

/**
 * Estimate the maximum cost of a request in microdollars.
 *
 * Uses body byte-length as a rough proxy for input tokens (~4 chars/token)
 * and the explicit output cap (or model-specific default) for output tokens.
 * Multiplied by a 1.1x safety margin.
 *
 * Returns an integer (microdollars) for budget reservation.
 */
export function estimateMaxCost(
  model: string,
  body: Record<string, unknown>,
  bodyByteLength?: number,
): number {
  const pricing = getModelPricing("openai", model);

  if (!pricing) {
    return UNKNOWN_MODEL_FALLBACK_MICRODOLLARS;
  }

  const inputTokenEstimate = Math.ceil((bodyByteLength ?? JSON.stringify(body).length) / CHARS_PER_TOKEN);

  const explicitOutputCap =
    (body.max_completion_tokens as number | undefined) ??
    (body.max_tokens as number | undefined);

  const outputTokenEstimate = explicitOutputCap ?? (MODEL_OUTPUT_CAPS[model] ?? DEFAULT_OUTPUT_CAP);

  const inputCost = costComponent(inputTokenEstimate, pricing.inputPerMTok);
  const outputCost = costComponent(outputTokenEstimate, pricing.outputPerMTok);

  return Math.round((inputCost + outputCost) * SAFETY_MARGIN);
}
