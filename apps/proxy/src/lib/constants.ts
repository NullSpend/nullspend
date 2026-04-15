export const OPENAI_BASE_URL = "https://api.openai.com";
export const ANTHROPIC_BASE_URL = "https://api.anthropic.com";
export const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

/** Default upstream fetch timeout (2 minutes). */
export const DEFAULT_UPSTREAM_TIMEOUT_MS = 120_000;

/**
 * Resolve the upstream timeout from an optional env override,
 * falling back to {@link DEFAULT_UPSTREAM_TIMEOUT_MS}.
 *
 * Workers cannot read env at module level, so this must be called
 * inside a request handler where `env` is available.
 */
export function resolveUpstreamTimeoutMs(env: Record<string, unknown>): number {
  const raw = env.UPSTREAM_TIMEOUT_MS;
  if (raw != null) {
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return DEFAULT_UPSTREAM_TIMEOUT_MS;
}
