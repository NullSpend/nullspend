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

  // P0-4: Sanitize user-supplied max_tokens. See cost-estimator.ts
  // normalizeOutputCap for rejection semantics (booleans, arrays, NaN,
  // Infinity, non-positive). Numeric strings are accepted for SDK parity.
  // Codex P2 review: explicit boolean + array rejection because
  // Number(true) === 1 and Number([5000]) === 5000 would otherwise slip
  // through a naive `Number()` coercion.
  const explicitOutputCap = normalizeOutputCap(body.max_tokens);

  const outputTokenEstimate =
    explicitOutputCap ?? (MODEL_OUTPUT_CAPS[model] ?? DEFAULT_OUTPUT_CAP);

  // P0-5: Apply long-context multipliers before the calculator does. The
  // calculator (anthropic-cost-calculator.ts:40,50) uses
  // `totalInputTokens = input_tokens + cache_creation + cache_read > 200_000`
  // — actual token counts from Anthropic's response.
  //
  // The estimator derives `inputTokenEstimate` from `bodyByteLength / 4`, which
  // can UNDER-count tokens by up to 50% for:
  //   - Multimodal requests (images add tokens but minimal body bytes —
  //     inline base64 or URL reference under-represents visual token cost)
  //   - Code / CJK / emoji content (higher token density than English —
  //     chars-per-token can be ~2-3 instead of 4)
  //   - Cache-heavy requests where the body shape differs from token distribution
  //
  // To guarantee the estimator applies the multiplier whenever the calculator
  // will, the threshold must account for the worst-case under-count:
  //   actual_tokens <= estimate_tokens / (1 - max_undercount_fraction)
  //   estimate_threshold = 200_000 * (1 - 0.5) = 100_000
  //
  // At 100K estimate tokens, even content that under-counts by 50% could have
  // up to 200K actual tokens — exactly at the calculator's long-context boundary.
  // Any higher threshold lets heavy-undercount requests slip through.
  //
  // Trade-off: over-reserves for cache-light text-only requests in the 100-200K
  // estimate range. Reconciliation releases the excess within seconds. The
  // alternative (under-reservation) is a budget-enforcement bug, not a UX
  // inconvenience. Codex P0-5 follow-up review confirmed 150K was insufficient.
  const LONG_CONTEXT_ESTIMATE_THRESHOLD = 100_000;
  const isLongContext = inputTokenEstimate > LONG_CONTEXT_ESTIMATE_THRESHOLD;
  const inputRate = isLongContext ? pricing.inputPerMTok * 2 : pricing.inputPerMTok;
  const outputRate = isLongContext ? pricing.outputPerMTok * 1.5 : pricing.outputPerMTok;

  const inputCost = costComponent(inputTokenEstimate, inputRate);
  const outputCost = costComponent(outputTokenEstimate, outputRate);

  return Math.round((inputCost + outputCost) * SAFETY_MARGIN);
}
