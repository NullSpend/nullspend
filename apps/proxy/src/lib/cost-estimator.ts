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
 * Convert a user-supplied max-tokens-like field to a sanitized positive integer,
 * or `undefined` if the value is absent / malformed.
 *
 * Accepts: numbers (finite, positive), numeric strings ("1000" → 1000).
 * Rejects: booleans, arrays, objects, NaN, Infinity, negative, zero.
 * Clamps: values > 1M are clamped to 1M (sanity upper bound).
 */
function normalizeOutputCap(raw: unknown): number | undefined {
  if (typeof raw === "boolean" || Array.isArray(raw)) return undefined;
  if (raw === null || raw === undefined) return undefined;
  const asNumber = Number(raw);
  if (!Number.isFinite(asNumber) || asNumber <= 0) return undefined;
  return Math.min(Math.ceil(asNumber), 1_000_000);
}

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

  // P0-4: Sanitize user-supplied max_completion_tokens / max_tokens. Clamp to
  // finite positive integer. NaN, Infinity, negative, booleans, arrays, and
  // object values fall through to the model-specific default cap instead of
  // propagating NaN/negative/absurd-cap into the reservation math.
  //
  // Codex review P2: also reject boolean and array types explicitly, because
  // Number(true) === 1 and Number([5000]) === 5000. Those values would slide
  // past a pure `Number() + isFinite()` check and get treated as real caps.
  // Numeric strings ("1000" → 1000) remain accepted for OpenAI SDK parity.
  const explicitFromMaxCompletion = normalizeOutputCap(body.max_completion_tokens);
  const explicitFromMaxTokens = normalizeOutputCap(body.max_tokens);
  const explicitOutputCap = explicitFromMaxCompletion ?? explicitFromMaxTokens;

  const outputTokenEstimate = explicitOutputCap ?? (MODEL_OUTPUT_CAPS[model] ?? DEFAULT_OUTPUT_CAP);

  const inputCost = costComponent(inputTokenEstimate, pricing.inputPerMTok);
  const outputCost = costComponent(outputTokenEstimate, pricing.outputPerMTok);

  return Math.round((inputCost + outputCost) * SAFETY_MARGIN);
}
