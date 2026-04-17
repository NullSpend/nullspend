/**
 * Gemini pre-request cost estimator.
 *
 * Estimates the maximum cost of a Gemini request for budget reservation.
 * Uses body byte-length as a rough proxy for input tokens (~4 chars/token)
 * and generationConfig.maxOutputTokens (or model-specific default) for output.
 *
 * Follows the same pattern as cost-estimator.ts (OpenAI).
 * Ref: https://ai.google.dev/gemini-api/docs/tokens
 */

import { getModelPricing, costComponent } from "@nullspend/cost-engine";

const SAFETY_MARGIN = 1.1;
const CHARS_PER_TOKEN = 4;
const UNKNOWN_MODEL_FALLBACK_MICRODOLLARS = 1_000_000; // $1

/**
 * Model-specific maximum output token caps.
 * Used when the request doesn't specify generationConfig.maxOutputTokens.
 */
const MODEL_OUTPUT_CAPS: Record<string, number> = {
  "gemini-2.5-pro": 65_536,
  "gemini-2.5-flash": 65_536,
  "gemini-2.5-flash-lite": 65_536,
  "gemini-2.0-flash": 8_192,
  "gemini-2.0-flash-lite": 8_192,
  "gemini-3-flash-preview": 65_536,
  "gemini-3-pro-preview": 65_536,
  "gemini-3.1-pro-preview": 65_536,
  "gemini-3.1-flash-lite-preview": 65_536,
};

const DEFAULT_OUTPUT_CAP = 8_192;

/**
 * Gemini model alias prefixes → canonical pricing model.
 * Dated aliases (gemini-2.5-flash-preview-04-17, gemini-2.5-pro-exp-03-25)
 * share pricing with their base model.
 *
 * ORDERING: Longer prefixes MUST come before shorter ones to prevent
 * false matches. e.g., "gemini-2.5-flash-lite" must precede
 * "gemini-2.5-flash", otherwise flash-lite models would match flash pricing.
 */
const GEMINI_MODEL_ALIASES: [string, string][] = [
  // Keep sorted by prefix length descending within each family
  // 3.1 series
  ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview"],
  ["gemini-3.1-pro-preview", "gemini-3.1-pro-preview"],
  // 3.0 series
  ["gemini-3-flash-preview", "gemini-3-flash-preview"],
  // 2.5 series (longer prefixes first)
  ["gemini-2.5-flash-lite", "gemini-2.5-flash-lite"],
  ["gemini-2.5-flash", "gemini-2.5-flash"],
  ["gemini-2.5-pro", "gemini-2.5-pro"],
  // 2.0 series (longer prefixes first)
  ["gemini-2.0-flash-lite", "gemini-2.0-flash-lite"],
  ["gemini-2.0-flash", "gemini-2.0-flash"],
];

/**
 * Look up pricing for a Gemini model, falling back to canonical name
 * for dated aliases (e.g., gemini-2.5-flash-preview-04-17 → gemini-2.5-flash).
 */
export function resolveGeminiPricing(model: string) {
  const direct = getModelPricing("google", model);
  if (direct) return { pricing: direct, resolvedModel: model };

  for (const [prefix, canonical] of GEMINI_MODEL_ALIASES) {
    if (model.startsWith(prefix)) {
      const pricing = getModelPricing("google", canonical);
      if (pricing) return { pricing, resolvedModel: canonical };
    }
  }

  return { pricing: null, resolvedModel: model };
}

/**
 * Estimate the maximum cost of a Gemini request in microdollars.
 */
export function estimateGeminiMaxCost(
  model: string,
  body: Record<string, unknown>,
  bodyByteLength?: number,
): number {
  const { pricing, resolvedModel } = resolveGeminiPricing(model);

  if (!pricing) {
    return UNKNOWN_MODEL_FALLBACK_MICRODOLLARS;
  }

  const inputTokenEstimate = Math.ceil((bodyByteLength ?? JSON.stringify(body).length) / CHARS_PER_TOKEN);

  const generationConfig = body.generationConfig as { maxOutputTokens?: number } | undefined;
  const rawOutputCap = Number(generationConfig?.maxOutputTokens);
  // Clamp to finite positive integer. NaN, Infinity, negative, and non-numeric
  // values fall through to the model-specific default cap.
  const explicitOutputCap = Number.isFinite(rawOutputCap) && rawOutputCap > 0
    ? Math.min(Math.ceil(rawOutputCap), 1_000_000) // cap at 1M tokens sanity limit
    : undefined;

  const outputTokenEstimate = explicitOutputCap ?? (MODEL_OUTPUT_CAPS[resolvedModel] ?? MODEL_OUTPUT_CAPS[model] ?? DEFAULT_OUTPUT_CAP);

  // Long-context pricing: only gemini-2.5-pro has the >200K multiplier
  // (2x input, 1.5x output). Must match the calculator in gemini-cost-calculator.ts
  // so reservations cover actual cost on long-context requests.
  const isLongContext = resolvedModel === "gemini-2.5-pro" && inputTokenEstimate > 200_000;
  const inputRate = isLongContext ? pricing.inputPerMTok * 2 : pricing.inputPerMTok;
  const outputRate = isLongContext ? pricing.outputPerMTok * 1.5 : pricing.outputPerMTok;

  const inputCost = costComponent(inputTokenEstimate, inputRate);
  const outputCost = costComponent(outputTokenEstimate, outputRate);

  return Math.round((inputCost + outputCost) * SAFETY_MARGIN);
}
