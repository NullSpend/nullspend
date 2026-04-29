---
title: "Proxy Endpoints"
description: "Routes the NullSpend proxy exposes for upstream provider calls."
---


The NullSpend proxy sits between your agents and upstream providers. It authenticates requests, tracks costs, and enforces budgets transparently.

**Base URL:** `https://proxy.nullspend.dev`

---

## Provider Routes

| Method | Path | Provider | Default Upstream |
|---|---|---|---|
| POST | `/v1/chat/completions` | OpenAI | `https://api.openai.com` |
| POST | `/v1/messages` | Anthropic | `https://api.anthropic.com` |
| POST | `/v1beta/models/{model}:generateContent` | Google Gemini | `https://generativelanguage.googleapis.com` |
| POST | `/v1beta/models/{model}:streamGenerateContent` | Google Gemini (streaming) | `https://generativelanguage.googleapis.com` |
| POST | `/v1/mcp/budget/check` | MCP | None (local) |
| POST | `/v1/mcp/events` | MCP | None (local) |

All provider routes require an `X-NullSpend-Key` header. Unsupported paths return `404 not_found`. Non-POST methods return `404`.

### OpenAI (`/v1/chat/completions`)

Forwards to `https://api.openai.com/v1/chat/completions` (or custom upstream). Headers forwarded to the upstream provider:

- `authorization` — Your OpenAI API key
- `openai-organization`
- `openai-project`
- `traceparent`, `tracestate` — W3C trace context

Supports both streaming and non-streaming responses.

### Anthropic (`/v1/messages`)

Forwards to `https://api.anthropic.com/v1/messages` (or custom upstream). Headers forwarded:

- `x-api-key` or `authorization` — Your Anthropic API key
- `anthropic-version` — Defaults to `2023-06-01` if not provided
- `anthropic-beta`
- `traceparent`, `tracestate`

Supports both streaming and non-streaming responses.

### Google Gemini (`/v1beta/models/{model}:generateContent`, `:streamGenerateContent`)

Forwards to Google's Gemini API using the same URL structure as the native API. Replace the base URL and add your NullSpend key. The request body passes through unmodified in native Gemini format.

**Authentication:** The proxy extracts the Google API key from either:
- `Authorization: Bearer <key>` (NullSpend convention, works with all providers)
- `x-goog-api-key: <key>` (Google's native header)

Headers forwarded:
- `x-goog-api-key` — Your Google API key (extracted from `Authorization` or forwarded directly)
- `traceparent`, `tracestate` — W3C trace context

**Streaming:** The proxy automatically appends `?alt=sse` to the upstream URL for `:streamGenerateContent` requests. You don't need to include it yourself.

**Model in URL:** Unlike OpenAI/Anthropic where the model is in the request body, Gemini puts the model in the URL path. The proxy extracts it automatically for cost tracking. Dated model aliases (e.g., `gemini-2.5-flash-preview-04-17`) are resolved to their base model for pricing.

**Thinking tokens:** Gemini 2.5 models report `thoughtsTokenCount` in usage metadata. These are tracked as `_ns_thinking_tokens` in cost event tags, similar to OpenAI reasoning tokens.

**Response ID:** Since Gemini doesn't return a request ID in response headers, the proxy generates a UUID and sets `x-request-id` on the response. Google's `responseId` from the response body is captured as the `_ns_google_response_id` tag for cross-referencing.

```bash
# Non-streaming
curl "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:generateContent" \
  -H "x-goog-api-key: $GOOGLE_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'

# Streaming
curl "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:streamGenerateContent" \
  -H "x-goog-api-key: $GOOGLE_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "Content-Type: application/json" \
  --no-buffer \
  -d '{"contents":[{"parts":[{"text":"Count to 5"}]}]}'
```

### MCP (`/v1/mcp/budget/check`, `/v1/mcp/events`)

Local endpoints for MCP server integrations. `/budget/check` performs a pre-request budget check. `/events` ingests cost events from MCP tool calls.

---

## Policy Endpoint

`GET /v1/policy` returns the calling key's enforcement policy: current budget remaining/limit/spend, allowed models and providers, and the cheapest model per provider (filtered to the allow list). Used by SDKs to render upgrade prompts before a request is sent.

Auth: API key (same as provider routes). Returns `200` with a JSON body shaped:

```json
{
  "budget": { "remaining_microdollars": 9_000_000, "max_microdollars": 10_000_000, "spend_microdollars": 1_000_000, "period_end": "2026-05-01T00:00:00.000Z", "entity_type": "api_key", "entity_id": "..." },
  "allowed_models": ["gpt-4o-mini", "claude-haiku-4-5"],
  "allowed_providers": ["openai", "anthropic"],
  "cheapest_per_provider": { "openai": { "model": "gpt-4o-mini", "input_per_mtok": 150_000, "output_per_mtok": 600_000 } },
  "cheapest_overall": { "model": "gpt-4o-mini", "provider": "openai", "input_per_mtok": 150_000, "output_per_mtok": 600_000 },
  "restrictions_active": true
}
```

The dashboard mirror of this endpoint is `GET /api/policy` (same auth, same shape).

---

## Health Endpoints

No authentication required.

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `{ "status": "ok", "service": "nullspend-proxy" }` |
| GET | `/health/metrics` | Analytics Engine metrics (JSON or Prometheus, based on `Accept` header) |
| GET | `/health/ready` | `{ "status": "ok", "service": "nullspend-proxy" }` — simple readiness check |
| GET | `/health/feature-flags` | Live runtime values of `PLAN_COUNTER_ENABLED`, `NULLSPEND_CLOUD`, `CACHE_SCHEMA_VERSION`, and `build_sha`. Used by the launch-watcher and shadow-mode alerts. |

---

## Internal Endpoints

These use shared secret authentication (not API keys) and are not for external use.

| Method | Path | Purpose |
|---|---|---|
| POST | `/internal/budget/invalidate` | Invalidate budget cache for a user |
| GET | `/internal/budget/velocity-state` | Query velocity limit state |

---

## Upstream Allowlist

When overriding the upstream provider with the `X-NullSpend-Upstream` header, only these URLs are accepted:

| URL | Provider |
|---|---|
| `https://api.openai.com` | OpenAI (default) |
| `https://api.groq.com/openai` | Groq |
| `https://api.together.xyz` | Together AI |
| `https://api.fireworks.ai/inference` | Fireworks AI |
| `https://api.mistral.ai` | Mistral |
| `https://openrouter.ai/api` | OpenRouter |
| `https://generativelanguage.googleapis.com` | Google Gemini (default) |

Invalid upstream URLs return `400 invalid_upstream`. Entries must not include API version path segments (`/v1`, `/v1beta`), as the proxy appends these automatically.

---

## Body Size Limit

Maximum request body: **1 MB** (1,048,576 bytes).

Enforced in two places:
1. **Pre-read** — `Content-Length` header checked before reading the body
2. **Post-read** — Actual byte count verified after reading

Exceeding either check returns `413 payload_too_large`.

Response body logging (Pro/Enterprise) also caps at 1 MB. Streaming responses exceeding 1 MB are truncated in the stored body; the client receives the full response regardless.

---

## Request Processing Pipeline

Every request follows this exact order:

1. **Trace ID resolution** — Always runs, even on errors. Sets `X-NullSpend-Trace-Id` on the response.
2. **Health routes** — No auth. Returns immediately for `/health`, `/health/metrics`, `/health/ready`.
3. **Internal routes** — Shared secret auth. Returns immediately for `/internal/*`.
4. **Route lookup** — POST only. Unknown `/v1/*` paths return `404`.
5. **Rate limiting + API key authentication** — Run in parallel via `Promise.all`. Rate limiting checks IP then key limits ([Rate Limits](rate-limits.md)). Auth does SHA-256 hash lookup ([Authentication](authentication.md)).
6. **Body parsing** — JSON validation and size check. Runs sequentially after auth and rate limiting complete.
7. **Context construction** — Resolves webhooks, API version, session ID, tags, and trace context.
8. **Route handler** — Budget check → upstream call → cost tracking → reconciliation.

---

## Related

- [Custom Headers](custom-headers.md) — request and response headers
- [Authentication](authentication.md) — key lifecycle and validation
- [Rate Limits](rate-limits.md) — enforcement order and failure modes
- [Errors](errors.md) — error codes and response format
