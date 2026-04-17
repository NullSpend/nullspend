# Proxy Worker (@nullspend/proxy)

Cloudflare Workers proxy that sits between agents and LLM providers (OpenAI, Anthropic, Google Gemini). Authenticates requests, tracks costs, and enforces budgets.

## Commands

```bash
pnpm test                      # Proxy unit + contract tests (PR gate)
pnpm dev                       # Start wrangler dev server
pnpm deploy                    # Deploy to Cloudflare
SMOKE_LIVE=1 pnpm test:smoke   # Live smoke (manual/nightly; real API calls)
pnpm test:smoke                # Prints help + exits 0 without SMOKE_LIVE=1
pnpm smoke:record              # Refresh MSW cassettes (quarterly or on provider shape drift)
pnpm test:stress               # Stress tests — production-mutating, manual only
```

## Critical Rules

- **NEVER use `passThroughOnException()`** — proxy must fail closed (502), never forward unauthenticated/untracked requests to origin
- **NEVER add failover logic** that bypasses auth or cost tracking — this undermines the entire FinOps purpose
- Auth check must be the absolute first thing before any processing
- Body size limit (1MB) enforced both pre-read (Content-Length) and post-read (byte count)
- **Rate limiter fails OPEN** — if the Cloudflare rate limiter binding is unavailable, requests proceed without rate limiting. This is intentional (availability > rate limiting) but means a partial Cloudflare outage disables rate limits. The dashboard rate limiter (Upstash) fails CLOSED (503).

## Testing

Three tiers live in this package. Full tier definitions + decision tree live in `docs/internal/test-tier-taxonomy.md` — read that before adding a new test.

- **Unit + contract** (`src/__tests__/`) — PR gate, 1715 tests, ~10s, zero external calls. Includes Wave 3 `contract-*.test.ts` files that invoke route handlers in-process with MSW-intercepted upstream.
- **Live smoke** (`smoke-*.test.ts` at package root, 29 files as of 2026-04-16) — manual/nightly. Requires `SMOKE_LIVE=1`. Hits deployed proxy + real providers. Single-call-per-test contract assertions. See `vitest.smoke.config.ts` gate + retry.
- **Stress** (`stress-*.test.ts`, 10 files as of 2026-04-16) — production-mutating, manual only. Concurrency, latency benchmarks, race condition hunting, resource exhaustion. See "Stress tests" section below.

**Audit history:** the 2026-04-16 smoke-tier audit moved 4 whole files + 21 individual tests out of the smoke tier into stress. See the taxonomy doc's "Cleanup history" section for the full move log.

### Unit + contract test conventions

- Tests live in `src/__tests__/` directory.
- Mock `cloudflare:workers` with `vi.mock("cloudflare:workers", () => cloudflareWorkersMock())` from `test-helpers.ts`.
- Polyfill `crypto.subtle.timingSafeEqual` in `beforeAll` (Workers API not in Node).
- Existing tests use inline `makeEnv()` + `makeCtx()` helpers per file.
- **Wave 3 contract tests** use the shared `src/__tests__/msw/contract-helpers.ts` module (`makeContractEnv`, `makeContractCtx`, `makeContractRequest`, `invokeWorker`, `makeExecutionContext`). Prefer these for new contract tests.

### Contract tests + MSW (Wave 3)

Scaffolding lives in `src/__tests__/msw/`:
- `server.ts` — `setupServer(...handlers)` — attached via `setupFiles` in `vitest.config.ts`.
- `setup.ts` — `beforeAll(listen)` / `afterEach(resetHandlers)` / `afterAll(close)`. `onUnhandledRequest: "bypass"` so the existing 80+ tests using `globalThis.fetch = vi.fn()` overrides stay unaffected.
- `openai-handlers.ts` + `anthropic-handlers.ts` — handler builders (default success, error status, network error, streaming).
- `contract-helpers.ts` — invocation helpers + pattern rules (comment at top of file).
- Fixtures live in `src/__tests__/fixtures/cassettes/` (git-tracked JSON).

**Pattern rules for new contract tests:**
1. Prefer route-handler invocation (import `handleX` from `routes/x.js`) unless the behavior happens pre-routing (body parsing, top-level auth). Body-size is the only current example using top-level `invokeWorker`.
2. Do NOT overwrite `globalThis.fetch` in MSW-backed tests — it silently bypasses the interceptor. MSW's `onUnhandledRequest: "bypass"` means legacy tests doing fetch overrides are unaffected, but new MSW-dependent tests must leave fetch alone.
3. When testing denial paths, mock `budget-do-client.js` to return the relevant `DoCheckResult` shape. The DO itself is covered in `user-budget-do.do.test.ts` (workerd pool).

### Cassette recording

```bash
# From apps/proxy/. Hits real OpenAI + Anthropic (~$0.0001 total).
pnpm smoke:record
```

Guarded by `RECORD=1` (set automatically by the `smoke:record` script). Loads `.env.smoke` inline for `OPENAI_API_KEY` + `ANTHROPIC_API_KEY`. Writes normalized JSON to `src/__tests__/fixtures/cassettes/` — dynamic fields (id, created, system_fingerprint) are overwritten with stable values so git diffs surface only real provider shape drift.

Refresh cadence: quarterly, or when a provider ships a new model family.

### Live smoke (manual/nightly)

Wave 3 Phase 3 demoted live smoke from ambient tier to explicit opt-in. Defense in depth:

- **Script gate** (`scripts/smoke-gate.ts`): `pnpm test:smoke` prints a help message and exits 0 if `SMOKE_LIVE !== "1"`.
- **Config gate** (`vitest.smoke.config.ts`): `include: []` when `SMOKE_LIVE !== "1"`, so direct `npx vitest --config vitest.smoke.config.ts` invocations also short-circuit with exit 0 (via `passWithNoTests: true`).

When the gate is open, retry config mitigates rate-limit flake:
```ts
retry: {
  count: 3,
  delay: 1000,
  condition: /429|ECONNRESET|ETIMEDOUT|socket hang up|rate[_-]?limit/i,
}
```
Config-level `condition` must be `RegExp` (Vitest worker-thread serialization constraint). Permanent failures fail fast on the first attempt — the regex narrows retry to transient patterns.

**Smoke test org requirements:** the smoke test API key's org (`7f0521bb-...`) MUST have an active Pro subscription in `subscriptions` table for body-capture tests to pass (body capture is tier-gated via `request_logging_enabled` computed from `s.tier IN ('pro','enterprise')`). Fixture subscription id `a4f52e61-682d-4628-9d96-711117ddd037` — seeded 2026-04-16 during Phase B. See `docs/internal/body-capture-investigation-20260416.md`.

Manual-only smoke files that bail without extra prerequisites:
- `smoke-sdk-functional.test.ts` — requires `NULLSPEND_DASHBOARD_URL` pointing at a reachable dashboard (local `pnpm dev` at 127.0.0.1:3000 or a deployed URL). Without it, the whole file `describe.skip`s gracefully during full-suite runs.
- `smoke-margin-sync.test.ts` — requires Stripe test credentials + CRON_SECRET. Skips via `skip = true` flag if env vars missing.

## Stress tests

Stress tests live alongside smoke tests in this directory and run via
`vitest.stress.config.ts`. They hit the live deployed proxy + Hyperdrive
+ Postgres + Cloudflare Queue stack and **mutate real production data**.
Manual runs only — never wire into CI.

```bash
pnpm test:stress                       # all stress files (default medium intensity)
STRESS_INTENSITY=light pnpm test:stress # smaller fixtures, fewer concurrent reqs
STRESS_INTENSITY=heavy pnpm test:stress # max load
pnpm stress:cleanup                    # crash-recovery: purge stress-sdk-% leftovers
```

### `stress-sdk-features.test.ts`

Validates the `@nullspend/sdk` surface area against the deployed proxy
under concurrent load (Phase 0 transport matrix → Phase 1 functional
tests → Phase 2 concurrent stress → Phase 3 mid-test mutation → Phase 4
verification). See `docs/internal/test-plans/sdk-stress-test-plan.md`
§15/§15a/§15b for the design corrections this file implements.

**Prerequisites:**
- **Rebuild the SDK if you've changed any `packages/sdk/src/` file**:
  `pnpm --filter @nullspend/sdk build`. The stress test imports
  `@nullspend/sdk` which resolves to `packages/sdk/dist/`, NOT to source.
  Stale `dist/` will silently exercise the OLD SDK code and produce
  confusing test failures (e.g. a "fix verified by unit tests" failing
  the live test for no apparent reason). Lesson learned the hard way
  during the §15c-1 fix.
- `pnpm dev` running in another terminal (the SDK direct-mode tests
  hit `http://127.0.0.1:3000/api/cost-events`). Tests auto-skip the
  direct-mode subset if the dashboard is unreachable at startup. Not
  required for proxy-only stress tests like `§6.9`.
- `.env.smoke` populated with `PROXY_URL`, `NULLSPEND_API_KEY`,
  `NULLSPEND_SMOKE_KEY_ID`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `INTERNAL_SECRET`, `DATABASE_URL`, and `NULLSPEND_DASHBOARD_URL`.

**Production mutation warning:** the test creates an isolated stress
user + api_key in `beforeAll` and tears them down in `afterAll`, so all
attribution-level data is contained. **But** the proxy-side path still
writes real cost events through Cloudflare Queue → Hyperdrive → DO state
on the deployed worker. The infrastructure is real even though the
identity is isolated. Manual runs only — **never wire into CI**.

**Cost:** ~$0.02 per medium run, ~$0.05 per heavy run, when the test
works on the first try. Run light first.

**Findings log:** each run writes
`stress-sdk-findings-${TEST_RUN_ID}.json` alongside the test file with
every observation tagged `info`/`warn`/`bug`. Findings files are
git-ignored.

**Crash recovery:** if a run crashes mid-test, leftover fixtures stay in
the live DB. Run `pnpm stress:cleanup` to purge anything matching the
`stress-sdk-%` prefix.

### `smoke-sdk-functional.test.ts`

Functional E2E suite covering the SDK paths the stress test intentionally
skipped (HITL action lifecycle, read APIs, retry/timeout/apiVersion config,
HITL error class fields). Single-call tests, sequential execution — NOT a
stress test. Lives under the smoke config (`vitest.smoke.config.ts`),
runs via `pnpm test:smoke smoke-sdk-functional.test.ts`. Manual runs only,
never CI. See `docs/internal/test-plans/sdk-testing-gaps.md` "Functional
E2E suite" (F1–F11) for the canonical scope.

**Prerequisites:**
- **Rebuild the SDK** (same lesson as stress test): `pnpm --filter @nullspend/sdk build`.
- `pnpm dev` running OR `NULLSPEND_DASHBOARD_URL` pointing at a deployed Vercel dashboard.
- `.env.smoke` populated with `NULLSPEND_API_KEY`, `NULLSPEND_SMOKE_KEY_ID`,
  `NULLSPEND_SMOKE_USER_ID`, `DATABASE_URL`, `INTERNAL_SECRET`, `NULLSPEND_DASHBOARD_URL`.

**Approval mechanism:** Direct SQL `UPDATE actions` — the dashboard
`/api/actions/[id]/approve` route is session-cookie auth (admin role) and
the SDK API key cannot call it. The test mirrors what `lib/actions/resolve-action.ts`
does at the SQL level. Test actions are tagged `agent_id LIKE 'sdk-functional-test-%'`
and cleaned up symmetrically in `beforeAll` (orphan rows from prior crashed
runs) and `afterAll`.

## Architecture

**Entry & Routing**
- `src/index.ts` — entry point, routing, body parsing, session/trace extraction
- `src/routes/provider-handler.ts` — generic provider route handler (shared lifecycle for all LLM providers)
- `src/routes/openai.ts` — OpenAI thin wrapper (delegates to provider-handler with OpenAI adapter)
- `src/routes/anthropic.ts` — Anthropic thin wrapper (delegates to provider-handler with Anthropic adapter)
- `src/providers/openai.ts` — OpenAI ProviderAdapter (headers, SSE parser, cost calculator wrappers)
- `src/providers/anthropic.ts` — Anthropic ProviderAdapter (headers, SSE parser, cost calculator, cache tags)
- `src/providers/gemini.ts` — Gemini ProviderAdapter (native URL scheme, model-in-URL, x-goog-api-key auth)
- `src/providers/registry.ts` — maps URL paths to ProviderAdapter configs (exact-match + Gemini prefix matching)
- `src/lib/provider-types.ts` — ProviderAdapter interface, StreamResult, ParsedResponse types (includes optional extractModel/isStreaming hooks)
- `src/routes/mcp.ts` — MCP budget check + cost event ingestion (separate lifecycle, not adapter-based)
- `src/routes/internal.ts` — internal budget invalidation/sync endpoint
- `src/routes/shared.ts` — shared budget denial handling (6 denial types with `recovery` object + enriched messages), `buildRecovery()` helper, `fmtDollars()`, webhook dispatch helpers (used by all routes)

**Auth & Context**
- `src/lib/auth.ts` — API key auth (delegates to `api-key-auth.ts`)
- `src/lib/api-key-auth.ts` — SHA-256 hash lookup with positive/negative caching
- `src/lib/context.ts` — `RequestContext` (auth, connectionString, sessionId, traceId, tags)
- `src/lib/api-version.ts` — API version resolution (header → key → default)
- `src/lib/trace-context.ts` — W3C traceparent parsing, custom header fallback, auto-generation
- `src/lib/tags.ts` — `X-NullSpend-Tags` header parsing and validation
- `src/lib/validation.ts` — shared validation helpers (UUID regex, etc.)

**Budget Enforcement (Durable Object)**
- `src/durable-objects/user-budget.ts` — UserBudgetDO: SQLite tables (budgets, reservations, velocity_state, session_spend, loop_call_log), checkAndReserve (with velocity, loop detection, session limit, budget enforcement in that order), reconcile (with session spend correction), alarm cleanup (with session TTL + loop log pruning)
- `src/lib/budget-orchestrator.ts` — checkBudget + reconcileBudget orchestration (passes loopContext through)
- `src/lib/budget-do-client.ts` — DO RPC client (check, reconcile, upsert, remove, reset, velocity state)
- `src/lib/budget-do-lookup.ts` — Postgres → DOBudgetEntity lookup for DO population (includes loop config fields)
- `src/lib/budget-spend.ts` — Postgres atomic spend increment + period reset write-back

**Loop Detection (integrated into Budget Enforcement DO)**
- Default-on: 50 identical calls per 60s window per `provider:model:contentHash` key
- Aggregate: 5+ distinct keys with 3+ same-content repeats triggers multi-model loop detection
- Content hash: SHA-256 of first 8KB of request body, truncated to 8 hex chars
- Deferred INSERT: loop_call_log entry only committed after budget check passes (prevents budget-denied requests from inflating loop counter)
- Denial backoff: 5s in-memory cache with lazy eviction + 1K cap, stores original detection details
- Alarm pruning: respects max configured loopWindowSeconds, safety cap at 5000 rows
- Config: `loopMaxCalls` (0=disabled, null=default 50), `loopWindowSeconds` (null=default 60), `loopAggregateMaxKeys` (null=default 5)
- Response: 429 with `code: "loop_detected"`, `Retry-After: 5`, `X-NullSpend-Denied: 1`, `recovery: { retryable: true, retry_after_seconds: 5 }`
- Warning header: `X-NullSpend-Loop-Count: 40/50` on success responses at >=80% threshold
- Webhook: `loop.detected` event via existing signed dispatch
- Denial priority order: velocity > loop > session > tag > customer > budget

**Cost Calculation**
- `src/lib/cost-calculator.ts` — OpenAI token-to-cost conversion
- `src/lib/cost-estimator.ts` — OpenAI pre-request cost estimation
- `src/lib/anthropic-cost-calculator.ts` — Anthropic token-to-cost (cache write TTLs, long context 2x)
- `src/lib/anthropic-cost-estimator.ts` — Anthropic pre-request estimation
- `src/lib/gemini-cost-calculator.ts` — Gemini token-to-cost (usageMetadata → microdollars)
- `src/lib/gemini-cost-estimator.ts` — Gemini pre-request estimation
- `src/lib/gemini-sse-parser.ts` — Gemini SSE parser (complete-response-per-event, not delta)
- `src/lib/gemini-headers.ts` — Gemini header forwarding (x-goog-api-key auth)
- `src/lib/gemini-types.ts` — Gemini API type definitions (GeminiUsageMetadata, GeminiResponse)
- `src/lib/cost-logger.ts` — async DB write via `waitUntil()`

**Body Storage (Request/Response Logging)**
- `src/lib/body-storage.ts` — R2 storage for request/response bodies (Pro/Enterprise tier-gated via `requestLoggingEnabled`)
  - `storeRequestBody` / `storeResponseBody` — non-streaming JSON bodies
  - `storeStreamingResponseBody` — raw SSE text stored at `{ownerId}/{requestId}/response.sse`
  - `createStreamBodyAccumulator()` — passthrough TransformStream that accumulates decoded text up to 1MB; sits between upstream body and SSE parser: `upstream → accumulator → SSE parser → client`
  - `retrieveBodies()` — fetches request.json + response.json + response.sse from R2, prefers JSON over SSE

**Request/Response Processing**
- `src/lib/request-utils.ts` — `ensureStreamOptions`, `extractModelFromBody`
- `src/lib/sse-parser.ts` — OpenAI streaming response parser for usage extraction
- `src/lib/anthropic-sse-parser.ts` — Anthropic streaming parser
- `src/lib/headers.ts` — header sanitization (strip proxy headers, forward provider headers)
- `src/lib/anthropic-headers.ts` — Anthropic-specific header forwarding
- `src/lib/sanitize-upstream-error.ts` — strip API keys from upstream error responses
- `src/lib/errors.ts` — standardized error response builder
- `src/lib/upstream-allowlist.ts` — allowed upstream host validation

**Webhooks**
- `src/lib/webhook-events.ts` — event payload builders (15 event types)
- `src/lib/webhook-thresholds.ts` — `detectThresholdCrossings` (per-entity configurable thresholds)
- `src/lib/webhook-dispatch.ts` — dispatcher interface + Queue-based enqueue
- `src/lib/webhook-queue.ts` — webhook queue message type + enqueue helper
- `src/webhook-queue-handler.ts` — Queue consumer: fetch endpoint, sign, deliver, retry with exponential backoff
- `src/webhook-dlq-handler.ts` — DLQ consumer: log + metric + ack
- `src/lib/webhook-signer.ts` — HMAC-SHA256 signature generation
- `src/lib/webhook-cache.ts` — KV-cached endpoint lookup
- `src/lib/webhook-expiry.ts` — rotated secret expiry

**Infrastructure**
- `src/lib/db.ts` — Per-request postgres.js instance (max:1, prepare:false, fetch_types:false) — I/O context isolation
- `src/lib/timing-safe-equal.ts` — Constant-time string comparison (shared by internal auth + webhook signer)
- `src/lib/cache-kv.ts` — KV-backed caching helpers
- `src/routes/metrics.ts` — `GET /health/metrics` — AE SQL API query, KV caching (90s), negative caching (30s), JSON + Prometheus content negotiation
- `src/lib/write-metric.ts` — `writeLatencyDataPoint` — fire-and-forget AE data point write per request
- `src/lib/metrics.ts` — structured metric emission
- `src/lib/reconciliation-queue.ts` — Cloudflare Queue-based async reconciliation
- `src/lib/cost-event-queue.ts` — Cloudflare Queue-based async cost event logging (queue-first with direct fallback)
- `src/cost-event-queue-handler.ts` — Cost event queue consumer (batch INSERT + per-message fallback)
- `src/cost-event-dlq-handler.ts` — Cost event DLQ consumer (always-ack + best-effort write)
- `src/lib/constants.ts` — shared constants

## Cost Tracking Flow

```
Request → Resolve trace ID → Auth → Forward to provider → Parse response/stream → Extract usage → Calculate cost → Enqueue to COST_EVENT_QUEUE (fallback: direct DB write)
```

Cost events are enqueued to Cloudflare Queues via `logCostEventQueued()` / `logCostEventsBatchQueued()`. The queue consumer batch-INSERTs with `onConflictDoNothing` for idempotent re-delivery. Falls back to direct `logCostEvent()` when queue binding is absent (local dev).

Non-streaming: parse JSON response for `usage` field. Body stored as `response.json` in R2.
Streaming: SSE parser accumulates chunks, extracts final `usage` from `[DONE]`-adjacent message. When body logging is enabled, a `StreamBodyAccumulator` TransformStream sits between upstream and SSE parser (`upstream → accumulator → SSE parser → client`), passing chunks through immediately while accumulating text. After stream completes, the accumulated SSE text is stored as `response.sse` in R2 via `waitUntil`.
Cancelled streams: when the client aborts mid-stream, the SSE parser resolves with `cancelled: true` and no usage. The route writes an estimated cost event (tokens=0, cost=pre-request estimate) tagged with `_ns_estimated: "true"` and `_ns_cancelled: "true"` in the JSONB `tags` column, then reconciles the budget reservation with the estimate. Partial streaming bodies are stored for debugging. The cost event write is try/catch-wrapped so failures cannot block budget reconciliation.

## Telemetry

Cost events include enrichment fields populated per-request:
- `budget_status` — `skipped` (no budgets / hasBudgets flag), `approved`, or `denied`
- `stop_reason` — provider finish/stop reason (`stop`, `max_tokens`, `end_turn`, `tool_calls`, etc.)
- `estimated_cost_microdollars` — pre-request budget estimate for accuracy analysis

SSE parsers capture `firstChunkMs` (time of first upstream chunk) for TTFB tracking.

AE data points include 4 doubles: `[overheadMs, upstreamMs, totalMs, ttfbMs]`. The `/health/metrics` endpoint exposes p50/p95/p99 for all four.

Anthropic cost events include cache split tags: `_ns_cache_write_tokens`, `_ns_cache_read_tokens`.
Provider rate limit proximity captured in tags: `_ns_ratelimit_remaining_requests`, `_ns_ratelimit_remaining_tokens`.

Error classification: `emitMetric("request_error", { status, reason })` on all error paths in `index.ts`. `emitMetric("budget_denied", { reason, provider, entityType })` on all denial paths in `shared.ts` and `mcp.ts`.

Auth includes `hasBudgets` flag (EXISTS subquery on budgets table). When false, budget orchestrator skips DO RPC entirely — 17ms → 2-3ms overhead for tracking-only users.

Budget sync latency: dashboard sends `sentAt` timestamp on invalidation calls, proxy emits `budget_sync_latency_ms` metric.
Stale-cache detection: `budget_cache_stale` metric when auth's `hasBudgets` disagrees with DO state.
Request metadata tags: `_ns_max_tokens`, `_ns_temperature`, `_ns_tool_count` captured per request.
Long-context detection: `_ns_long_context: "true"` tag on Anthropic requests >200k total input tokens.
`costBreakdown.toolDefinition`: tool definition cost (subset of input cost) included in breakdown for both providers.
