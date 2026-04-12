/**
 * Build headers for the upstream Anthropic request.
 * Extracts API key from Bearer token or x-api-key and forwards as x-api-key.
 * Always injects anthropic-version and content-type.
 */
export function buildAnthropicUpstreamHeaders(request: Request): Headers {
  const headers = new Headers();

  const authHeader = request.headers.get("authorization");
  if (authHeader?.startsWith("Bearer ")) {
    headers.set("x-api-key", authHeader.slice(7));
  } else {
    const xApiKey = request.headers.get("x-api-key");
    if (xApiKey) headers.set("x-api-key", xApiKey);
  }

  headers.set("anthropic-version", request.headers.get("anthropic-version") ?? "2023-06-01");
  headers.set("content-type", "application/json");

  // PXY-14: Forward anthropic-beta but log for observability.
  // Beta features can change SSE response format and cost accounting.
  // We forward to avoid breaking clients, but track usage for audit.
  const beta = request.headers.get("anthropic-beta");
  if (beta) {
    headers.set("anthropic-beta", beta);
    // Return the beta value so the caller can emit a metric
  }

  // Forward W3C trace context (parity with OpenAI upstream headers)
  const traceparent = request.headers.get("traceparent");
  if (traceparent) headers.set("traceparent", traceparent);
  const tracestate = request.headers.get("tracestate");
  if (tracestate) headers.set("tracestate", tracestate);

  return headers;
}

/**
 * Build headers for the client response from an Anthropic upstream response.
 * Normalizes `request-id` to `x-request-id` and forwards rate-limit headers.
 */
export function buildAnthropicClientHeaders(
  upstreamResponse: Response,
  apiVersion?: string,
): Headers {
  const headers = new Headers();

  const ct = upstreamResponse.headers.get("content-type");
  if (ct) headers.set("content-type", ct);

  const requestId = upstreamResponse.headers.get("request-id");
  if (requestId) headers.set("x-request-id", requestId);

  for (const [name, value] of upstreamResponse.headers) {
    if (name.startsWith("anthropic-ratelimit-")) {
      headers.set(name, value);
    }
  }

  const retryAfter = upstreamResponse.headers.get("retry-after");
  if (retryAfter) headers.set("retry-after", retryAfter);

  if (apiVersion) {
    headers.set("NullSpend-Version", apiVersion);
  }

  return headers;
}
