import { getModelPricing, costComponent } from "@nullspend/cost-engine";
import type { NewCostEventRow } from "@nullspend/db";
import type {
  AnthropicRawUsage,
  AnthropicCacheCreationDetail,
} from "./anthropic-types.js";

type CostEventInsert = Omit<NewCostEventRow, "id" | "createdAt">;

/**
 * Map an Anthropic Messages API usage object to a cost event ready for persistence.
 *
 * Key differences from the OpenAI calculator:
 * - `input_tokens` is already the uncached portion (do NOT subtract cache tokens)
 * - `totalInputTokens` includes cache creation + cache read for DB and long-context check
 * - Long context (>200K total input) doubles input/cache rates, 1.5x output
 * - Cache write cost is TTL-aware when detail is available
 * - Thinking tokens are included in `output_tokens` — no separate count
 */
export function calculateAnthropicCost(
  requestModel: string,
  responseModel: string | null,
  usage: AnthropicRawUsage,
  cacheCreationDetail: AnthropicCacheCreationDetail | null,
  requestId: string,
  durationMs: number,
  attribution?: {
    userId: string | null;
    apiKeyId: string | null;
    actionId: string | null;
  },
  toolDefinitionTokens?: number,
): CostEventInsert {
  const inputTokens = Math.max(0, Number(usage.input_tokens) || 0);
  const outputTokens = Math.max(0, Number(usage.output_tokens) || 0);
  const cacheCreationTokens =
    Math.max(0, Number(usage.cache_creation_input_tokens) || 0);
  const cacheReadTokens = Math.max(0, Number(usage.cache_read_input_tokens) || 0);

  const totalInputTokens = inputTokens + cacheCreationTokens + cacheReadTokens;

  let pricing = getModelPricing("anthropic", requestModel);
  let resolvedModel = requestModel;

  if (!pricing && responseModel) {
    pricing = getModelPricing("anthropic", responseModel);
    if (pricing) resolvedModel = responseModel;
  }

  const isLongContext = totalInputTokens > 200_000;
  const inputRate = pricing
    ? isLongContext ? pricing.inputPerMTok * 2 : pricing.inputPerMTok
    : 0;
  const cacheReadRate = pricing
    ? isLongContext ? pricing.cachedInputPerMTok * 2 : pricing.cachedInputPerMTok
    : 0;
  const cacheWrite5mRate = pricing?.cacheWrite5mPerMTok
    ? isLongContext ? pricing.cacheWrite5mPerMTok * 2 : pricing.cacheWrite5mPerMTok
    : 0;
  const cacheWrite1hRate = pricing?.cacheWrite1hPerMTok
    ? isLongContext ? pricing.cacheWrite1hPerMTok * 2 : pricing.cacheWrite1hPerMTok
    : 0;
  const outputRate = pricing
    ? isLongContext ? pricing.outputPerMTok * 1.5 : pricing.outputPerMTok
    : 0;

  let cacheWriteCost: number;

  if (cacheCreationDetail?.ephemeral_5m_input_tokens !== undefined) {
    const tokens5m =
      Math.max(0, Number(cacheCreationDetail.ephemeral_5m_input_tokens) || 0);
    const tokens1h =
      Math.max(0, Number(cacheCreationDetail.ephemeral_1h_input_tokens) || 0);
    cacheWriteCost =
      costComponent(tokens5m, cacheWrite5mRate) +
      costComponent(tokens1h, cacheWrite1hRate);
  } else {
    cacheWriteCost = costComponent(cacheCreationTokens, cacheWrite5mRate);
  }

  let costMicrodollars = 0;
  let costBreakdown: CostEventInsert["costBreakdown"] = null;
  if (pricing) {
    const input = costComponent(inputTokens, inputRate);
    const cached = cacheWriteCost + costComponent(cacheReadTokens, cacheReadRate);
    const output = costComponent(outputTokens, outputRate);
    costMicrodollars = Math.max(0, Math.round(input + cached + output));

    // Round components, then distribute rounding residual to largest to guarantee exact sum
    const roundedInput = Math.round(input);
    const roundedCached = Math.round(cached);
    const roundedOutput = Math.round(output);
    const residual = costMicrodollars - (roundedInput + roundedCached + roundedOutput);
    let adjInput = roundedInput;
    let adjCached = roundedCached;
    let adjOutput = roundedOutput;
    if (residual !== 0) {
      // Apply residual to the largest component
      if (output >= input && output >= cached) adjOutput += residual;
      else if (input >= cached) adjInput += residual;
      else adjCached += residual;
    }
    const bd: { input: number; output: number; cached: number; toolDefinition?: number } = {
      input: adjInput, output: adjOutput, cached: adjCached,
    };

    // Tool definition is a SUBSET of input cost (already counted in input_tokens), stored for attribution
    if (toolDefinitionTokens && toolDefinitionTokens > 0) {
      bd.toolDefinition = Math.round(costComponent(toolDefinitionTokens, inputRate));
    }

    costBreakdown = bd;
  }

  return {
    requestId,
    provider: "anthropic",
    model: resolvedModel,
    inputTokens: totalInputTokens,
    outputTokens,
    cachedInputTokens: cacheReadTokens,
    reasoningTokens: 0,
    costMicrodollars,
    costBreakdown,
    durationMs,
    userId: attribution?.userId ?? null,
    apiKeyId: attribution?.apiKeyId ?? null,
    actionId: attribution?.actionId ?? null,
    eventType: "llm" as const,
  };
}
