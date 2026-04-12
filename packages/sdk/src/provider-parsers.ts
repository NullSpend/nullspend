import type { OpenAISSEUsage, AnthropicSSEUsage, AnthropicCacheCreationDetail } from "./sse-parser.js";

// ---------------------------------------------------------------------------
// Route matching
// ---------------------------------------------------------------------------

/**
 * Determine if a request should be tracked for cost reporting.
 * Uses URL path (not hostname) to support custom baseURL / Azure.
 */
export function isTrackedRoute(
  provider: "openai" | "anthropic",
  url: string,
  method: string,
): boolean {
  if (method !== "POST") return false;
  try {
    const path = new URL(url).pathname;
    if (provider === "openai") {
      return (
        path.endsWith("/chat/completions") ||
        path.endsWith("/completions") ||
        path.endsWith("/embeddings")
        // Note: /responses API uses input_tokens/output_tokens field names
        // and a different SSE format — needs its own parser (future work).
      );
    }
    if (provider === "anthropic") {
      return path.endsWith("/messages");
    }
    return false;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Request body utilities
// ---------------------------------------------------------------------------

/**
 * Extract the model name from a JSON request body string.
 */
export function extractModelFromBody(body: string): string | null {
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed.model === "string" && parsed.model.length > 0) {
      return parsed.model;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Detect if a request body indicates streaming.
 */
export function isStreamingRequest(body: string): boolean {
  try {
    const parsed = JSON.parse(body);
    return parsed.stream === true;
  } catch {
    return false;
  }
}

/**
 * Detect if a response is streaming based on Content-Type header.
 */
export function isStreamingResponse(response: Response): boolean {
  const ct = response.headers.get("content-type") ?? "";
  return ct.includes("text/event-stream");
}

// ---------------------------------------------------------------------------
// JSON response usage extraction
// ---------------------------------------------------------------------------

/**
 * Extract usage from a non-streaming OpenAI JSON response.
 */
export function extractOpenAIUsageFromJSON(
  json: unknown,
): OpenAISSEUsage | null {
  if (!json || typeof json !== "object") return null;
  const obj = json as Record<string, unknown>;
  if (!obj.usage || typeof obj.usage !== "object") return null;
  const usage = obj.usage as Record<string, unknown>;
  if (typeof usage.prompt_tokens !== "number") return null;
  // completion_tokens is optional: embeddings responses have only prompt_tokens
  // and total_tokens. Default to 0 so the cost calculator works correctly.
  if (usage.completion_tokens === undefined || usage.completion_tokens === null) {
    usage.completion_tokens = 0;
  }
  if (typeof usage.completion_tokens !== "number") return null;
  return usage as unknown as OpenAISSEUsage;
}

/**
 * Extract usage from a non-streaming Anthropic JSON response.
 */
export function extractAnthropicUsageFromJSON(json: unknown): {
  usage: AnthropicSSEUsage;
  cacheDetail: AnthropicCacheCreationDetail | null;
} | null {
  if (!json || typeof json !== "object") return null;
  const obj = json as Record<string, unknown>;
  if (!obj.usage || typeof obj.usage !== "object") return null;
  const usage = obj.usage as Record<string, unknown>;
  if (typeof usage.input_tokens !== "number") return null;
  if (typeof usage.output_tokens !== "number") return null;

  let cacheDetail: AnthropicCacheCreationDetail | null = null;
  if (usage.cache_creation && typeof usage.cache_creation === "object") {
    const cc = usage.cache_creation as Record<string, unknown>;
    cacheDetail = {
      ephemeral_5m_input_tokens: typeof cc.ephemeral_5m_input_tokens === "number"
        ? cc.ephemeral_5m_input_tokens
        : undefined,
      ephemeral_1h_input_tokens: typeof cc.ephemeral_1h_input_tokens === "number"
        ? cc.ephemeral_1h_input_tokens
        : undefined,
    };
  }

  return {
    usage: usage as unknown as AnthropicSSEUsage,
    cacheDetail,
  };
}
