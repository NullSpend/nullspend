/**
 * OpenAI chat completions route — thin wrapper.
 *
 * Delegates to the generic handler with the OpenAI adapter.
 * Upstream URL validation (x-nullspend-upstream) is handled inside
 * the adapter's getUpstreamUrl, which runs AFTER mandate checks
 * (preserving original ordering from the pre-refactor route).
 */

import type { RequestContext } from "../lib/context.js";
import { openaiChatAdapter } from "../providers/openai.js";
import { handleProviderRequest } from "./provider-handler.js";

export async function handleChatCompletions(
  request: Request,
  env: Env,
  ctx: RequestContext,
): Promise<Response> {
  return handleProviderRequest(request, env, ctx, openaiChatAdapter);
}
