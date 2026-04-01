import { getModelPricing, costComponent } from "@nullspend/cost-engine";

import type { CostEventInput } from "./types.js";
import type {
  OpenAISSEUsage,
  AnthropicSSEUsage,
  AnthropicCacheCreationDetail,
} from "./sse-parser.js";

// ---------------------------------------------------------------------------
// OpenAI cost calculation
// ---------------------------------------------------------------------------

/**
 * Calculate a CostEventInput from OpenAI usage data.
 * Matches the proxy's `calculateOpenAICost` logic exactly.
 */
export function calculateOpenAICostEvent(
  model: string,
  usage: OpenAISSEUsage,
  durationMs: number,
  metadata: {
    sessionId?: string;
    traceId?: string;
    tags?: Record<string, string>;
  },
): CostEventInput {
  const promptTokens = Math.max(0, Number(usage.prompt_tokens) || 0);
  const completionTokens = Math.max(0, Number(usage.completion_tokens) || 0);
  const cachedTokens = Math.max(
    0,
    Number(usage.prompt_tokens_details?.cached_tokens) || 0,
  );
  const reasoningTokens = Math.max(
    0,
    Number(usage.completion_tokens_details?.reasoning_tokens) || 0,
  );

  const normalInputTokens = promptTokens - cachedTokens;

  const pricing = getModelPricing("openai", model);

  let costMicrodollars = 0;
  let costBreakdown:
    | { input: number; output: number; cached: number; reasoning?: number }
    | undefined;

  if (pricing) {
    const input = costComponent(normalInputTokens, pricing.inputPerMTok);
    const cached = costComponent(cachedTokens, pricing.cachedInputPerMTok);
    const output = costComponent(completionTokens, pricing.outputPerMTok);
    costMicrodollars = Math.max(0, Math.round(input + cached + output));

    // Round components, then distribute rounding residual to largest
    const roundedInput = Math.round(input);
    const roundedCached = Math.round(cached);
    const roundedOutput = Math.round(output);
    const residual =
      costMicrodollars - (roundedInput + roundedCached + roundedOutput);
    let adjInput = roundedInput;
    let adjCached = roundedCached;
    let adjOutput = roundedOutput;
    if (residual !== 0) {
      if (output >= input && output >= cached) adjOutput += residual;
      else if (input >= cached) adjInput += residual;
      else adjCached += residual;
    }

    costBreakdown = { input: adjInput, output: adjOutput, cached: adjCached };
    if (reasoningTokens > 0) {
      costBreakdown.reasoning = Math.round(
        costComponent(reasoningTokens, pricing.outputPerMTok),
      );
    }
  }

  return {
    provider: "openai",
    model,
    inputTokens: promptTokens,
    outputTokens: completionTokens,
    cachedInputTokens: cachedTokens,
    reasoningTokens,
    costMicrodollars,
    costBreakdown,
    durationMs,
    sessionId: metadata.sessionId,
    traceId: metadata.traceId,
    tags: metadata.tags,
    eventType: "llm",
  };
}

// ---------------------------------------------------------------------------
// Anthropic cost calculation
// ---------------------------------------------------------------------------

/**
 * Calculate a CostEventInput from Anthropic usage data.
 * Matches the proxy's `calculateAnthropicCost` logic exactly.
 */
export function calculateAnthropicCostEvent(
  model: string,
  usage: AnthropicSSEUsage,
  cacheCreationDetail: AnthropicCacheCreationDetail | null,
  durationMs: number,
  metadata: {
    sessionId?: string;
    traceId?: string;
    tags?: Record<string, string>;
  },
): CostEventInput {
  const inputTokens = Math.max(0, Number(usage.input_tokens) || 0);
  const outputTokens = Math.max(0, Number(usage.output_tokens) || 0);
  const cacheCreationTokens = Math.max(
    0,
    Number(usage.cache_creation_input_tokens) || 0,
  );
  const cacheReadTokens = Math.max(
    0,
    Number(usage.cache_read_input_tokens) || 0,
  );

  const totalInputTokens = inputTokens + cacheCreationTokens + cacheReadTokens;

  const pricing = getModelPricing("anthropic", model);

  const isLongContext = totalInputTokens > 200_000;
  const inputRate = pricing
    ? isLongContext
      ? pricing.inputPerMTok * 2
      : pricing.inputPerMTok
    : 0;
  const cacheReadRate = pricing
    ? isLongContext
      ? pricing.cachedInputPerMTok * 2
      : pricing.cachedInputPerMTok
    : 0;
  const cacheWrite5mRate = pricing?.cacheWrite5mPerMTok
    ? isLongContext
      ? pricing.cacheWrite5mPerMTok * 2
      : pricing.cacheWrite5mPerMTok
    : 0;
  const cacheWrite1hRate = pricing?.cacheWrite1hPerMTok
    ? isLongContext
      ? pricing.cacheWrite1hPerMTok * 2
      : pricing.cacheWrite1hPerMTok
    : 0;
  const outputRate = pricing
    ? isLongContext
      ? pricing.outputPerMTok * 1.5
      : pricing.outputPerMTok
    : 0;

  let cacheWriteCost: number;
  if (cacheCreationDetail?.ephemeral_5m_input_tokens !== undefined) {
    const tokens5m = Math.max(
      0,
      Number(cacheCreationDetail.ephemeral_5m_input_tokens) || 0,
    );
    const tokens1h = Math.max(
      0,
      Number(cacheCreationDetail.ephemeral_1h_input_tokens) || 0,
    );
    cacheWriteCost =
      costComponent(tokens5m, cacheWrite5mRate) +
      costComponent(tokens1h, cacheWrite1hRate);
  } else {
    cacheWriteCost = costComponent(cacheCreationTokens, cacheWrite5mRate);
  }

  let costMicrodollars = 0;
  let costBreakdown:
    | { input: number; output: number; cached: number }
    | undefined;

  if (pricing) {
    const input = costComponent(inputTokens, inputRate);
    const cached =
      cacheWriteCost + costComponent(cacheReadTokens, cacheReadRate);
    const output = costComponent(outputTokens, outputRate);
    costMicrodollars = Math.max(0, Math.round(input + cached + output));

    const roundedInput = Math.round(input);
    const roundedCached = Math.round(cached);
    const roundedOutput = Math.round(output);
    const residual =
      costMicrodollars - (roundedInput + roundedCached + roundedOutput);
    let adjInput = roundedInput;
    let adjCached = roundedCached;
    let adjOutput = roundedOutput;
    if (residual !== 0) {
      if (output >= input && output >= cached) adjOutput += residual;
      else if (input >= cached) adjInput += residual;
      else adjCached += residual;
    }

    costBreakdown = { input: adjInput, output: adjOutput, cached: adjCached };
  }

  return {
    provider: "anthropic",
    model,
    inputTokens: totalInputTokens,
    outputTokens,
    cachedInputTokens: cacheReadTokens,
    reasoningTokens: 0,
    costMicrodollars,
    costBreakdown,
    durationMs,
    sessionId: metadata.sessionId,
    traceId: metadata.traceId,
    tags: metadata.tags,
    eventType: "llm",
  };
}
