# Proxy Endpoints

The NullSpend proxy sits between your agents and upstream providers. It authenticates requests, tracks costs, and enforces budgets transparently.

**Base URL:** `https://proxy.nullspend.com`

---

## Provider Routes

| Method | Path | Provider | Default Upstream |
|---|---|---|---|
| POST | `/v1/chat/completions` | OpenAI | `https://api.openai.com` |
| POST | `/v1/messages` | Anthropic | `https://api.anthropic.com` |
| POST | `/v1/mcp/budget/check` | MCP | None (local) |
| POST | `/v1/mcp/events` | MCP | None (local) |

All provider routes require an `X-NullSpend-Key` header. Unsupported `/v1/*` paths return `404 not_found`. Non-POST methods return `404`.

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

### MCP (`/v1/mcp/budget/check`, `/v1/mcp/events`)

Local endpoints for MCP server integrations. `/budget/check` performs a pre-request budget check. `/events` ingests cost events from MCP tool calls.

---

## Health Endpoints

No authentication required.

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `{ "status": "ok", "service": "nullspend-proxy" }` |
| GET | `/health/metrics` | Analytics Engine metrics (JSON or Prometheus, based on `Accept` header) |
| GET | `/health/ready` | `{ "status": "ok", "service": "nullspend-proxy" }` — simple readiness check |

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

Invalid upstream URLs return `400 invalid_upstream`. Perplexity is excluded because it doesn't use the `/v1/` prefix in its URL structure.

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
