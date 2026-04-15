/**
 * Provider endpoint registry.
 *
 * Maps URL paths to ProviderAdapter configs. The generic handler
 * (provider-handler.ts) uses the adapter to process the request.
 *
 * Adding a new provider endpoint:
 *   1. Create an adapter in providers/{provider}.ts
 *   2. For fixed paths: providerRoutes.set(path, adapter)
 *   3. For dynamic paths (model-in-URL): add to matchGeminiRoute()
 *   4. Done — no changes to index.ts or route wrappers needed
 *
 * MCP, internal, health, and policy routes are NOT registered here.
 * They have different lifecycles (no upstream fetch, different auth).
 */

import type { ProviderAdapter } from "../lib/provider-types.js";
import { openaiChatAdapter } from "./openai.js";
import { anthropicChatAdapter } from "./anthropic.js";
import { geminiChatAdapter } from "./gemini.js";

// ── Exact-match routes (O(1) lookup) ────────────────────────────────
const providerRoutes = new Map<string, ProviderAdapter>();
providerRoutes.set("/v1/chat/completions", openaiChatAdapter);
providerRoutes.set("/v1/messages", anthropicChatAdapter);

// ── Gemini dynamic path matching ────────────────────────────────────
//
// Gemini uses /v1beta/models/{model}:{method} where model varies.
// Exact-match Map can't handle this, so we use prefix + suffix checks.
// Only reached when exact match misses — zero cost for OpenAI/Anthropic.
//
//   :generateContent       → geminiChatAdapter (non-streaming)
//   :streamGenerateContent → geminiChatAdapter (streaming)
//
function matchGeminiRoute(pathname: string): ProviderAdapter | undefined {
  if (!pathname.startsWith("/v1beta/models/")) return undefined;
  if (pathname.endsWith(":generateContent")) return geminiChatAdapter;
  if (pathname.endsWith(":streamGenerateContent")) return geminiChatAdapter;
  return undefined;
}

/**
 * Look up a provider adapter for a URL path.
 * Returns the adapter if found, undefined if the path is not a provider route.
 *
 * Resolution order:
 *   1. Exact-match Map (O(1)) — OpenAI, Anthropic
 *   2. Gemini prefix+suffix check — dynamic model-in-URL paths
 */
export function matchProviderRoute(pathname: string): ProviderAdapter | undefined {
  return providerRoutes.get(pathname) ?? matchGeminiRoute(pathname);
}
