/**
 * Gemini cost calculator.
 *
 * Maps Gemini usageMetadata to a cost event ready for persistence.
 * Uses the shared cost-engine for pricing lookup and calculation.
 *
 * Gemini usage fields:
 *   promptTokenCount         — total input tokens (includes cached)
 *   candidatesTokenCount     — output tokens
 *   cachedContentTokenCount  — cached input tokens (subset of prompt)
 *   thoughtsTokenCount       — thinking/reasoning tokens (subset of output, informational)
 *
 * Cost formula:
 *   normalInput = promptTokenCount - cachedContentTokenCount
 *   cost = costComponent(normalInput, inputRate) + costComponent(cached, cachedRate) + costComponent(output, outputRate)
 *
 * Follows the same pattern as cost-calculator.ts (OpenAI) and anthropic-cost-calculator.ts.
 * Ref: https://ai.google.dev/gemini-api/docs/tokens
 */

import { costComponent } from "@nullspend/cost-engine";
import type { NewCostEventRow } from "@nullspend/db";
import type { GeminiUsageMetadata } from "./gemini-types.js";
import { resolveGeminiPricing } from "./gemini-cost-estimator.js";

type CostEventInsert = Omit<NewCostEventRow, "id" | "createdAt">;

/**
 * Calculate cost from a Gemini response's usageMetadata.
 *
 * Looks up pricing by requestModel first (including prefix-based alias
 * fallback for dated models like gemini-2.5-flash-preview-04-17).
 * Falls back to responseModel (Gemini may resolve aliases via modelVersion).
 */
export function calculateGeminiCost(
  requestModel: string,
  responseModel: string | null,
  usage: GeminiUsageMetadata | null,
  requestId: string,
  durationMs: number,
  attribution?: { userId: string | null; apiKeyId: string | null; actionId: string | null },
  toolDefinitionTokens?: number,
): CostEventInsert {
  // Clamp all token counts to finite non-negative integers.
  // Number("Infinity") → Infinity, Number("abc") → NaN — both must be caught.
  const safeInt = (v: unknown): number => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
  };
  const promptTokens = safeInt(usage?.promptTokenCount);
  const candidatesTokens = safeInt(usage?.candidatesTokenCount);
  const cachedTokens = Math.min(safeInt(usage?.cachedContentTokenCount), promptTokens); // can't exceed prompt
  const thoughtsTokens = safeInt(usage?.thoughtsTokenCount);

  const normalInputTokens = promptTokens - cachedTokens;

  // Use prefix-based alias fallback (gemini-2.5-flash-preview-04-17 → gemini-2.5-flash)
  let { pricing, resolvedModel } = resolveGeminiPricing(requestModel);

  if (!pricing && responseModel) {
    const fallback = resolveGeminiPricing(responseModel);
    pricing = fallback.pricing;
    if (pricing) resolvedModel = fallback.resolvedModel;
  }

  let costMicrodollars = 0;
  let costBreakdown: CostEventInsert["costBreakdown"] = null;
  if (pricing) {
    const input = costComponent(normalInputTokens, pricing.inputPerMTok);
    const cached = costComponent(cachedTokens, pricing.cachedInputPerMTok);
    const output = costComponent(candidatesTokens, pricing.outputPerMTok);
    costMicrodollars = Math.max(0, Math.round(input + cached + output));

    // Round components, distribute rounding residual to largest for exact sum
    const roundedInput = Math.round(input);
    const roundedCached = Math.round(cached);
    const roundedOutput = Math.round(output);
    const residual = costMicrodollars - (roundedInput + roundedCached + roundedOutput);
    let adjInput = roundedInput;
    let adjCached = roundedCached;
    let adjOutput = roundedOutput;
    if (residual !== 0) {
      if (output >= input && output >= cached) adjOutput += residual;
      else if (input >= cached) adjInput += residual;
      else adjCached += residual;
    }

    const bd: { input: number; output: number; cached: number; reasoning?: number; toolDefinition?: number } = {
      input: adjInput, output: adjOutput, cached: adjCached,
    };

    // Thinking tokens are a SUBSET of output cost (like OpenAI reasoning), stored for display
    if (thoughtsTokens > 0) {
      bd.reasoning = Math.round(costComponent(thoughtsTokens, pricing.outputPerMTok));
    }

    // Tool definition is a SUBSET of input cost, stored for attribution
    if (toolDefinitionTokens && toolDefinitionTokens > 0) {
      bd.toolDefinition = Math.round(costComponent(toolDefinitionTokens, pricing.inputPerMTok));
    }

    costBreakdown = bd;
  }

  return {
    requestId,
    provider: "google",
    model: resolvedModel,
    inputTokens: promptTokens,
    outputTokens: candidatesTokens,
    cachedInputTokens: cachedTokens,
    reasoningTokens: thoughtsTokens,
    costMicrodollars,
    costBreakdown,
    durationMs,
    userId: attribution?.userId ?? null,
    apiKeyId: attribution?.apiKeyId ?? null,
    actionId: attribution?.actionId ?? null,
    eventType: "llm" as const,
  };
}
