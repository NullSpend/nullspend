/**
 * Gemini-specific header builders.
 *
 * Follows the same pattern as anthropic-headers.ts:
 * - Upstream: extract API key from auth header, set provider-specific header
 * - Client: forward content-type, rate-limit headers, NullSpend version
 *
 * Gemini auth: x-goog-api-key header (not Authorization: Bearer).
 * Ref: https://ai.google.dev/gemini-api/docs/api-overview
 */

/**
 * Build headers for the upstream Gemini request.
 * Extracts API key from Bearer token or x-goog-api-key and forwards as x-goog-api-key.
 */
export function buildGeminiUpstreamHeaders(request: Request): Headers {
  const headers = new Headers();

  // Extract API key: prefer Bearer token (NullSpend convention), fall back to x-goog-api-key
  const authHeader = request.headers.get("authorization");
  if (authHeader?.startsWith("Bearer ")) {
    headers.set("x-goog-api-key", authHeader.slice(7));
  } else {
    const googleKey = request.headers.get("x-goog-api-key");
    if (googleKey) headers.set("x-goog-api-key", googleKey);
  }

  headers.set("content-type", "application/json");

  // Forward W3C trace context
  const traceparent = request.headers.get("traceparent");
  if (traceparent) headers.set("traceparent", traceparent);
  const tracestate = request.headers.get("tracestate");
  if (tracestate) headers.set("tracestate", tracestate);

  return headers;
}

/**
 * Build headers for the client response from a Gemini upstream response.
 * Forwards content-type, retry-after, and sets NullSpend version.
 *
 * Note: Gemini doesn't have standard rate-limit remaining-count headers
 * like OpenAI (x-ratelimit-*) or Anthropic (anthropic-ratelimit-*).
 * Note: Gemini has no upstream x-request-id header. The handler sets
 * x-request-id after this function returns (with the NullSpend-generated UUID).
 */
export function buildGeminiClientHeaders(
  upstreamResponse: Response,
  apiVersion?: string,
): Headers {
  const headers = new Headers();

  const ct = upstreamResponse.headers.get("content-type");
  if (ct) headers.set("content-type", ct);

  const retryAfter = upstreamResponse.headers.get("retry-after");
  if (retryAfter) headers.set("retry-after", retryAfter);

  if (apiVersion) {
    headers.set("NullSpend-Version", apiVersion);
  }

  return headers;
}
