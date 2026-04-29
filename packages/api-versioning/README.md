# @nullspend/api-versioning

Runtime-agnostic API versioning framework: registry + response transforms keyed
by CalVer date. Used by the NullSpend dashboard (Next.js 16) and proxy
(Cloudflare Workers).

## Why

Uses the "build latest, transform backwards" pattern: handler returns the
newest shape, the framework projects it back to the version the client
requested. Resolves the version per request from the `nullspend-version`
header (case-insensitive), defaults to `LATEST` when absent.

Strategy (Stripe pattern): URL `/v1/...` is frozen and signals the v1 API
surface; the `nullspend-version` header carries the dated CalVer (`2026-04-01`,
`2026-05-12`, etc.) within v1. Major rewrites would mint `/v2/`.

## Phase 0 scope (what this ships today)

- Single registered baseline version (`2026-04-01`)
- Response-side transform machinery wired through every API response (Phase 0
  short-circuits because zero transforms are registered)
- `withApiVersion(scope, handler)` Next.js Route Handler wrapper
- Discovery endpoint (`GET /api/_versions`, `GET /v1/_versions`)
- Strict 400 envelope on malformed / unknown `nullspend-version` headers

## Phase 1+ pending (NOT shipped, design notes only)

- Request-side transforms (`transformRequest` field, request-body parse step
  inside the wrapper). The framework currently has no request-transform path —
  Phase 1 will add it when the first request-shape change ships.
- Deprecation signaling (RFC 9745 `Deprecation` + RFC 8594 `Sunset` headers).
  Requires a `sunsetAt: string` field on `VersionChange` to populate Sunset
  with the actual scheduled removal date (not the version's birth date). Phase
  1 design item; see Audit E1 in `.claude/audit-artifacts/edge-case-audit-latest.md`.
- Side-effect change handling (`hasSideEffects` flag) for transforms that the
  handler must branch on. Will be re-added when the first such change exists.

## Quick start — register a new version with one transform

```ts
// In packages/api-versioning/src/registry-default.ts
NULLSPEND_REGISTRY.registerVersion("2026-05-12");

NULLSPEND_REGISTRY.registerChange<
  { event_id: string; spend_microdollars: number },
  { eventId: string; spendMicrodollars: number }
>({
  resource: "cost-events",
  oldVersion: "2026-04-01",
  newVersion: "2026-05-12",
  // Project NEW shape → OLD shape (clients pinned to old see old)
  transformResponse: (next) => ({
    event_id: next.eventId,
    spend_microdollars: next.spendMicrodollars,
  }),
});
```

After registering: handler returns the newest shape unconditionally; the
wrapper applies `transformResponse` for clients pinned to the old version.

## Coordinated rollout

Breaking changes to this package require synchronized proxy + dashboard
deploys; both surfaces import the same registry singleton, and a new version
flagged here will be visible from both sides on next deploy. Pin the SDK
release in lockstep — SDK installs that pinned the previous version continue
working via `transformResponse`, but the SDK release notes should call out
the new dated version.

## Resolution divergence: dashboard vs proxy (Phase 0 known gap)

The proxy's resolution chain is `header → keyVersion → LATEST` (where
`keyVersion` is `auth.apiVersion` persisted on the API key row). The dashboard
wrapper's resolution chain is `header → LATEST` — it does NOT consult the
per-key pinned version because the wrapper runs OUTSIDE the auth path.

Phase 0 impact: zero. Only one version is registered, so all paths resolve to
`2026-04-01` regardless of chain. Phase 1+ impact: a customer with an API key
pinned to an older version, calling a dashboard endpoint without sending the
`nullspend-version` header, would receive the LATEST shape from the dashboard
but the pinned shape from the proxy. Track this when shipping the first
transform; the auth-aware wrapper redesign waits until the actual transform
requirements are known.

## Discovery endpoint auth

`GET /v1/versions` (proxy) and `GET /api/versions` (dashboard) are
intentionally **unauthenticated** — they enumerate the registry, which is
public by design (Stripe + GitHub do the same with `/v1/api-versions` and
`/api/versions`). Both endpoints stamp `Cache-Control: public, max-age=300`
so CDN absorbs hammering at the edge.

**Path naming**: do NOT use `_versions`. Next.js App Router treats
`_`-prefixed folder names as private folders and excludes them from
routing — the route returns 404 in production with no error. The proxy
path was renamed to match for cross-surface consistency.

## Header conventions

- Request side: `nullspend-version` (lowercase, no `x-` prefix). Match Python
  SDK `_tracked_client.py` convention. HTTP headers are case-insensitive on
  the wire so `NullSpend-Version` from the TS SDK works equally.
- Response side: `NullSpend-Version` (mixed case) on every successful response
  AND on the wrapper's own 400 (so clients debugging a rejected version see
  what the server thinks LATEST is).
