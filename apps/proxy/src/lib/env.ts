/**
 * Typed accessor for optional Cloudflare Worker bindings.
 *
 * The generated `Env` interface (worker-configuration.d.ts) declares all
 * bindings as required, but several are only available in certain deploy
 * environments (production vs local dev). The code treats them as optional
 * via `as Record<string, unknown>` casts scattered across 15+ locations.
 *
 * This module centralises that pattern into a single typed interface so
 * that binding name typos produce a compile error instead of a silent
 * runtime `undefined`.
 */

/** Bindings that may be absent in local dev or non-production deploys. */
export interface OptionalEnv {
  BODY_STORAGE?: R2Bucket;
  COST_EVENT_QUEUE?: Queue;
  WEBHOOK_QUEUE?: Queue;
  METRICS?: AnalyticsEngineDataset;
  CF_ACCOUNT_ID?: string;
  CF_API_TOKEN?: string;
  UPSTREAM_TIMEOUT_MS?: string | number;
  /** Local dev: force DB writes even without Hyperdrive. */
  FORCE_DB_PERSIST?: string;
  /** Local dev: skip DB writes entirely. */
  SKIP_DB_PERSIST?: string;
  /**
   * PR-2c: plan-counter enforcement flag. `"true"` enables hard denials
   * on Free-tier orgs past the 100K governed-request cap. Default `"false"`
   * (shadow mode: counter increments + metrics fire but no denials). Flag
   * flip to `"true"` is a post-launch step (PR-2e).
   */
  PLAN_COUNTER_ENABLED?: string;
  /**
   * PR-2c: override for NullSpend pricing URL in plan-limit denial envelopes.
   * Cloud deploy leaves unset → default `https://nullspend.dev/pricing` applies.
   * Self-hosted / white-labeled deploys can override via wrangler.jsonc vars.
   */
  NULLSPEND_PRICING_URL_OVERRIDE?: string;
  /**
   * PR-2c: override for NullSpend self-host URL in plan-limit denial envelopes.
   * Cloud deploy leaves unset → default `https://github.com/NullSpend/nullspend` applies.
   */
  NULLSPEND_SELF_HOST_URL_OVERRIDE?: string;
}

/**
 * Access an optional env binding with type safety.
 * Returns `undefined` when the binding is absent.
 */
export function optionalBinding<K extends keyof OptionalEnv>(
  env: Env,
  key: K,
): OptionalEnv[K] {
  return (env as unknown as OptionalEnv)[key];
}
