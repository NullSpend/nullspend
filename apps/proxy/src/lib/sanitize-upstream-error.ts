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

  // Other errors: try to extract only safe fields
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
          message: typeof errorObj.message === "string" ? errorObj.message : "Upstream request failed",
          ...(typeof errorObj.code === "string" ? { code: errorObj.code } : {}),
        },
      });
    }

    // Anthropic top-level type field
    if (parsed?.type === "error" && parsed?.error) {
      return JSON.stringify({
        error: {
          type: typeof parsed.error.type === "string" ? parsed.error.type : "upstream_error",
          message: typeof parsed.error.message === "string" ? parsed.error.message : "Upstream request failed",
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
