import { optionalBinding } from "./env.js";

export const OPENAI_BASE_URL = "https://api.openai.com";
export const ANTHROPIC_BASE_URL = "https://api.anthropic.com";
export const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

/** Default upstream fetch timeout (2 minutes). */
export const DEFAULT_UPSTREAM_TIMEOUT_MS = 120_000;

/**
 * PR-2c: canonical NullSpend URLs surfaced in plan-limit denial envelopes.
 * Getters read env-backed overrides so self-hosted / white-labeled deploys
 * can expose their own URLs without a code fork. Cloud deploys leave the
 * overrides unset and the defaults apply.
 *
 * Access via `optionalBinding()` so the vars compile without regenerating
 * `worker-configuration.d.ts` (codex-round-2 H2 + codex-round-3 H2).
 */
const DEFAULT_PRICING_URL = "https://nullspend.dev/pricing";
const DEFAULT_SELF_HOST_URL = "https://github.com/NullSpend/nullspend";

/** Plan-limit upgrade URL — where denied Free-tier users go to upgrade. */
export function getPricingUrl(env: Env): string {
  const override = optionalBinding(env, "NULLSPEND_PRICING_URL_OVERRIDE");
  return typeof override === "string" && override.length > 0 ? override : DEFAULT_PRICING_URL;
}

/** Plan-limit self-host URL — alternative remediation for users who prefer self-hosting. */
export function getSelfHostUrl(env: Env): string {
  const override = optionalBinding(env, "NULLSPEND_SELF_HOST_URL_OVERRIDE");
  return typeof override === "string" && override.length > 0 ? override : DEFAULT_SELF_HOST_URL;
}

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
