/**
 * Sanitize upstream API error responses before forwarding to clients.
 *
 * Prevents leaking internal details like API key fragments, account identifiers,
 * rate limit tiers, or upstream-specific error structures.
 */
export async function sanitizeUpstreamError(
  upstreamResponse: Response,
  _provider: "openai" | "anthropic",
): Promise<string> {
  const status = upstreamResponse.status;

  // 401/403: Replace entirely — may contain API key fragments or account info
  if (status === 401) {
    return JSON.stringify({
      error: { type: "authentication_error", message: "Upstream authentication failed" },
    });
  }
  if (status === 403) {
    return JSON.stringify({
      error: { type: "permission_error", message: "Upstream permission denied" },
    });
  }

  // 5xx: Internal server errors may contain database details, connection
  // strings, stack traces. Never forward the message — use generic text.
  if (status >= 500) {
    return JSON.stringify({
      error: { type: "upstream_error", message: `Upstream returned ${status}` },
    });
  }

  // Other 4xx (not 401/403): extract type and code (machine-readable,
  // safe to forward) but sanitize the message to strip account-specific
  // details like org IDs, API key fragments, and internal identifiers.
  try {
    const text = await upstreamResponse.text();
    const parsed = JSON.parse(text);

    // OpenAI format: { error: { message, type, code } }
    // Anthropic format: { type, error: { type, message } }
    const errorObj = parsed?.error;
    if (errorObj && typeof errorObj === "object") {
      return JSON.stringify({
        error: {
          type: typeof errorObj.type === "string" ? errorObj.type : "upstream_error",
          message: typeof errorObj.message === "string" ? sanitizeMessage(errorObj.message, status) : "Upstream request failed",
          ...(typeof errorObj.code === "string" ? { code: errorObj.code } : {}),
        },
      });
    }

    // Anthropic top-level type field
    if (parsed?.type === "error" && parsed?.error) {
      return JSON.stringify({
        error: {
          type: typeof parsed.error.type === "string" ? parsed.error.type : "upstream_error",
          message: typeof parsed.error.message === "string" ? sanitizeMessage(parsed.error.message, status) : "Upstream request failed",
        },
      });
    }

    // Unrecognized format — return generic error
    return JSON.stringify({
      error: { type: "upstream_error", message: `Upstream returned ${status}` },
    });
  } catch {
    // Body wasn't JSON — return generic error
    return JSON.stringify({
      error: { type: "upstream_error", message: `Upstream returned ${status}` },
    });
  }
}

/**
 * Strip account-specific details from upstream error messages.
 * Preserves the message structure but redacts patterns that could
 * leak provider account info to NullSpend's clients.
 */
function sanitizeMessage(message: string, _status: number): string {
  return message
    // OpenAI org IDs: org-xxxx
    .replace(/\borg-[a-zA-Z0-9]{10,}\b/g, "org-***")
    // API key fragments: sk-xxxx, sk-proj-xxxx
    .replace(/\bsk-(?:proj-)?[a-zA-Z0-9]{4,}\b/g, "sk-***")
    // Long hex strings (internal IDs): 32+ hex chars
    .replace(/\b[0-9a-f]{32,}\b/gi, "***")
    // Email addresses
    .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b/gi, "***@***");
}
