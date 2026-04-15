/**
 * Anthropic messages route — thin wrapper.
 *
 * Delegates directly to the generic handler with the Anthropic adapter.
 * No provider-specific pre-flight handling needed (unlike OpenAI's
 * upstream override).
 */

import type { RequestContext } from "../lib/context.js";
import { anthropicChatAdapter } from "../providers/anthropic.js";
import { handleProviderRequest } from "./provider-handler.js";

export async function handleAnthropicMessages(
  request: Request,
  env: Env,
  ctx: RequestContext,
): Promise<Response> {
  return handleProviderRequest(request, env, ctx, anthropicChatAdapter);
}
