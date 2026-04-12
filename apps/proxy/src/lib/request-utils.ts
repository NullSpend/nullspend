/**
 * Force-merge stream_options.include_usage = true when stream is enabled.
 * Cost tracking depends on the usage chunk; this overrides user settings.
 */
export function ensureStreamOptions(body: Record<string, unknown>): void {
  if (body.stream !== true) return;

  if (!body.stream_options || typeof body.stream_options !== "object") {
    body.stream_options = { include_usage: true };
    return;
  }

  (body.stream_options as Record<string, unknown>).include_usage = true;
}

export function extractModelFromBody(body: Record<string, unknown>): string {
  return typeof body.model === "string" ? body.model : "unknown";
}
