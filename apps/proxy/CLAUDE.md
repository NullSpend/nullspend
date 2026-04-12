# Proxy Worker (@nullspend/proxy)

Cloudflare Workers proxy that sits between agents and OpenAI. Authenticates requests, tracks costs, and enforces budgets.

## Commands

```bash
pnpm test             # Run proxy tests (from this directory)
pnpm dev              # Start wrangler dev server
pnpm deploy           # Deploy to Cloudflare
```

## Critical Rules

- **NEVER use `passThroughOnException()`** — proxy must fail closed (502), never forward unauthenticated/untracked requests to origin
- **NEVER add failover logic** that bypasses auth or cost tracking — this undermines the entire FinOps purpose
- Auth check must be the absolute first thing before any processing
- Body size limit (1MB) enforced both pre-read (Content-Length) and post-read (byte count)

## Testing

- Tests live in `src/__tests__/` directory
- Mock `cloudflare:workers` with `vi.mock("cloudflare:workers", ...)`
- Polyfill `crypto.subtle.timingSafeEqual` in `beforeAll`
- `makeEnv()` helper returns typed `Env` with test values
- `makeCtx()` helper returns mock `ExecutionContext`

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
- `src/routes/openai.ts` — OpenAI chat completions handler
- `src/routes/anthropic.ts` — Anthropic messages handler
- `src/routes/mcp.ts` — MCP budget check + cost event ingestion
- `src/routes/internal.ts` — internal budget invalidation/sync endpoint
- `src/routes/shared.ts` — shared budget denial handling, webhook dispatch helpers (used by all routes)

**Auth & Context**
- `src/lib/auth.ts` — API key auth (delegates to `api-key-auth.ts`)
- `src/lib/api-key-auth.ts` — SHA-256 hash lookup with positive/negative caching
- `src/lib/context.ts` — `RequestContext` (auth, connectionString, sessionId, traceId, tags)
- `src/lib/api-version.ts` — API version resolution (header → key → default)
- `src/lib/trace-context.ts` — W3C traceparent parsing, custom header fallback, auto-generation
- `src/lib/tags.ts` — `X-NullSpend-Tags` header parsing and validation
- `src/lib/validation.ts` — shared validation helpers (UUID regex, etc.)

**Budget Enforcement (Durable Object)**
- `src/durable-objects/user-budget.ts` — UserBudgetDO: SQLite tables (budgets, reservations, velocity_state, session_spend), checkAndReserve (with session limit enforcement), reconcile (with session spend correction), alarm cleanup (with session TTL)
- `src/lib/budget-orchestrator.ts` — checkBudget + reconcileBudget orchestration
- `src/lib/budget-do-client.ts` — DO RPC client (check, reconcile, upsert, remove, reset, velocity state)
- `src/lib/budget-do-lookup.ts` — Postgres → DOBudgetEntity lookup for DO population
- `src/lib/budget-spend.ts` — Postgres atomic spend increment + period reset write-back

**Cost Calculation**
- `src/lib/cost-calculator.ts` — OpenAI token-to-cost conversion
- `src/lib/cost-estimator.ts` — OpenAI pre-request cost estimation
- `src/lib/anthropic-cost-calculator.ts` — Anthropic token-to-cost (cache write TTLs, long context 2x)
- `src/lib/anthropic-cost-estimator.ts` — Anthropic pre-request estimation
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
