/**
 * GET /health/feature-flags — shadow-mode alert signal source (PR-2d / Decision #34).
 *
 * Returns the Worker's live values of `PLAN_COUNTER_ENABLED`, `CACHE_SCHEMA_VERSION`,
 * and `build_sha` as JSON. No auth — these are operational metadata, not secrets.
 * The shadow-mode GitHub Action (plan §5 S4) polls this endpoint and fires if the
 * `/pricing` page serves 200s while `PLAN_COUNTER_ENABLED === "false"`.
 *
 * Response shape is stable — the alert consumer matches exact field names + string
 * values. Treat any change here as a contract change.
 *
 * `PLAN_COUNTER_ENABLED`: string value of the Worker env var ("true" / "false" /
 * absent → "false" default). Returning the raw string (not a boolean) keeps parity
 * with how ops reads/toggles the flag via `wrangler secret`.
 *
 * `CACHE_SCHEMA_VERSION`: hard-coded constant from `api-key-auth.ts`. Bumps every
 * time `ApiKeyIdentity` shape changes, invalidating all isolate-local caches.
 * Exposing here lets the alert sanity-check that a rolling deploy converged —
 * mismatched versions across colos indicate a stuck deploy.
 *
 * `build_sha`: resolved from `env.BUILD_SHA` (set by CI at `wrangler deploy`). Unset
 * during local dev → "unknown". Future work: wire Cloudflare `version_metadata`
 * binding when ops needs deploy-correlation beyond the alert signal.
 */

import { CACHE_SCHEMA_VERSION } from "../lib/api-key-auth.js";

// Typed via cast because `BUILD_SHA` is not part of the auto-generated
// `worker-configuration.d.ts` — it's a deploy-time env var that may or may
// not be set. Runtime default `"unknown"` handles absence.
interface FeatureFlagEnv {
  PLAN_COUNTER_ENABLED?: string;
  NULLSPEND_CLOUD?: string;
  BUILD_SHA?: string;
}

export function handleFeatureFlags(env: Env): Response {
  const e = env as unknown as FeatureFlagEnv;

  return Response.json(
    {
      PLAN_COUNTER_ENABLED: e.PLAN_COUNTER_ENABLED ?? "false",
      // PR-2e post-flip review: exposed for the launch-watcher's
      // cloud_flag_missing alert. PLAN_COUNTER_ENABLED="true" is a silent
      // no-op unless NULLSPEND_CLOUD="true" — the watcher must see both.
      NULLSPEND_CLOUD: e.NULLSPEND_CLOUD ?? "false",
      CACHE_SCHEMA_VERSION,
      build_sha: e.BUILD_SHA ?? "unknown",
    },
    {
      headers: {
        // no-store so the shadow-mode alert always reads the current flag
        // state — a cached "false" after a flip to "true" would cause spurious
        // alerts during the TTL window.
        "Cache-Control": "no-store",
      },
    },
  );
}
