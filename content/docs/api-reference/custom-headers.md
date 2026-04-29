---
title: "Custom Headers"
description: "NullSpend-specific request and response headers for cost attribution and routing."
---


NullSpend uses HTTP headers as its primary API surface. All NullSpend-specific headers are optional except `X-NullSpend-Key`.

## Request Headers

### `X-NullSpend-Key` (required)

Your NullSpend API key. Authenticates the request and determines which account costs are attributed to.

| Property | Value |
|---|---|
| Format | `ns_live_sk_` + 32 hex characters (43 characters total) |
| Required | Yes |
| If missing | `401` with `error.code: "unauthorized"` |
| If invalid | `401` with `error.code: "unauthorized"` |

```bash
X-NullSpend-Key: ns_live_sk_a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
```

Keys are hashed with SHA-256 before storage and validated with timing-safe comparison. Create and revoke keys in the dashboard under **Settings**, or see the [API Keys API](api-keys-api.md) for programmatic management. For key lifecycle and caching details, see [Authentication](authentication.md).

---

### `X-NullSpend-Tags`

Attach metadata to a request for cost attribution. Tags appear in the dashboard and are included in webhook payloads.

| Property | Value |
|---|---|
| Format | JSON object |
| Max keys | 10 |
| Key pattern | `[a-zA-Z0-9_-]+` (max 64 characters) |
| Value max length | 256 characters |
| Reserved prefix | `_ns_` — keys starting with this are silently dropped |
| If invalid JSON | Silently ignored (request proceeds with no tags) |
| If a single key/value is invalid | That key is silently dropped; valid keys are kept |

```bash
X-NullSpend-Tags: {"team":"billing","env":"production","feature":"summarizer"}
```

Tags are never a reason for request rejection. Invalid tags are dropped silently — the request always proceeds. Null bytes (`\0`) in values cause the tag to be dropped.

---

### `X-NullSpend-Session`

Groups requests into a session for session-level spend limits.

| Property | Value |
|---|---|
| Format | String |
| Max length | 256 characters (`400` if longer) |
| If omitted | Session limits are not enforced for this request |

```bash
X-NullSpend-Session: conv_abc123
```

When a session limit is configured on the budget and this header is present, the proxy tracks cumulative spend per session ID. Once the limit is reached, the proxy returns `429` with `error.code: "session_limit_exceeded"` and details including `session_id`, `session_spend_microdollars`, and `session_limit_microdollars`.

The proxy echoes `X-NullSpend-Session` in the response headers when present, so agent frameworks can confirm the session ID was captured.

---

### `X-NullSpend-Finalize`

Signal that this is a finalization request — unlock the finalization reserve for this request.

| Property | Value |
|---|---|
| Format | `"1"` |
| If omitted | Normal budget enforcement (reserve is subtracted from remaining) |
| If set but entity is not in reserve zone | Ignored — reserve still applies |

```bash
X-NullSpend-Finalize: 1
```

The proxy only honors this header when the budget entity has entered the reserve zone (spend + reservations >= limit - reserve). This prevents callers from burning through the reserve before reaching it. See [Finalization Reserve](../features/budgets.md#finalization-reserve).

---

### `X-NullSpend-Customer`

Associate this request with a customer for per-customer cost tracking and budget enforcement.

| Property | Value |
|---|---|
| Format | String, 1-256 characters |
| Allowed characters | `[a-zA-Z0-9._:-]` |
| If invalid | Request proceeds but customer ID is ignored. `X-NullSpend-Warning: invalid_customer` response header set. |
| If omitted | Falls back to `tags.customer` if present. Otherwise no customer association. |

```bash
X-NullSpend-Customer: acme-corp
```

When a customer-level budget is configured and this header is present, the proxy tracks spend per customer. Once the customer budget limit is reached, the proxy returns `429` with `error.code: "customer_budget_exceeded"` and details including `customer_id`.

The customer ID is stored in the `customer_id` column of cost events and can be used for per-customer cost attribution and margin analysis.

---

### `X-NullSpend-Trace-Id`

Set a custom trace ID for request correlation. If omitted, the proxy auto-generates one. See [Tracing](../features/tracing.md) for the full resolution chain and usage examples.

| Property | Value |
|---|---|
| Format | 32-character lowercase hex string |
| Regex | `^[0-9a-f]{32}$` |
| If invalid | Silently ignored; proxy auto-generates a trace ID |
| If omitted | Proxy generates one via `crypto.randomUUID()` (dashes removed) |

```bash
X-NullSpend-Trace-Id: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
```

The all-zeros ID (`00000000000000000000000000000000`) is rejected per W3C spec.

---

### `X-NullSpend-Action-Id`

Links this request to a [HITL (human-in-the-loop)](../features/human-in-the-loop.md) approval action.

| Property | Value |
|---|---|
| Format | `ns_act_` + UUID |
| If omitted | No HITL association |

```bash
X-NullSpend-Action-Id: ns_act_550e8400-e29b-41d4-a716-446655440000
```

---

### `X-NullSpend-Request-Id`

Idempotency / correlation key for this request. Pass a stable ID (UUID or ULID) on retries so the proxy and downstream cost ingest can de-duplicate. The same ID is echoed back in the response under `X-NullSpend-Request-Id`.

| Property | Value |
|---|---|
| Format | UUID or ULID, max 64 characters |
| If invalid or oversize | Silently ignored; proxy generates a fresh ID |
| If omitted | Proxy auto-generates one |

```bash
X-NullSpend-Request-Id: 01J9F6X3R3HM6E3D6N5N0M0G7Y
```

---

### `Idempotency-Key`

Deduplication key for dashboard mutating endpoints (`POST /api/cost-events`, `POST /api/cost-events/batch`, `POST /api/actions`, `POST /api/tool-costs/discover`, `POST /api/actions/:id/result`). Send the same key on a retry to receive the original response without re-executing the handler.

| Property | Value |
|---|---|
| Format | Free-form string (caller-defined; UUID or ULID recommended) |
| TTL | Successful response cached for **24 hours** |
| Scope | `(API key hash, route path, key)` — cannot collide across tenants or endpoints |
| If omitted | Handler runs normally, no caching |
| Replay response | Includes `X-Idempotent-Replayed: true` header |
| Concurrent duplicate | Polls up to 1 second; returns `503 request_in_progress` with `Retry-After: 1` if still in flight |
| Failure caching | 5xx responses are NOT cached — retries proceed normally |

```bash
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
```

Alternative: many endpoints accept an `idempotencyKey` field in the request body; the header takes precedence when both are present.

For proxy-side deduplication (cost-event idempotency at ingest), use [`X-NullSpend-Request-Id`](#x-nullspend-request-id) instead.

---

### `X-NullSpend-Upstream`

Override the upstream provider URL for this request. Only URLs in the proxy's allowlist are accepted.

| Property | Value |
|---|---|
| Format | Full URL |
| If invalid | `400` with `error.code: "invalid_upstream"` |
| If omitted | Default upstream for the provider is used |

---

### `NullSpend-Version`

Pin the API version for this request. Can also be set at the key level in the dashboard.

| Property | Value |
|---|---|
| Format | ISO date string (e.g., `2026-04-01`) |
| Resolution order | This header → key-level setting → default (`2026-04-01`) |
| If omitted | Uses key-level or default version |

```bash
NullSpend-Version: 2026-04-01
```

---

### `traceparent`

Standard W3C trace context header. If present, the proxy extracts the trace ID from it (taking priority over `X-NullSpend-Trace-Id`) and forwards both `traceparent` and `tracestate` to the upstream provider.

| Property | Value |
|---|---|
| Format | `{version}-{trace-id}-{span-id}-{flags}` |
| Version | `00` (version `ff` is rejected per W3C spec) |
| Trace ID | 32-character lowercase hex (all-zeros rejected) |
| Span ID | 16-character lowercase hex (all-zeros rejected) |

```bash
traceparent: 00-a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6-b7c8d9e0f1a2b3c4-01
```

---

## Response Headers

Every response from the proxy includes these headers:

### `X-NullSpend-Trace-Id`

The trace ID for this request. Use this to correlate requests across your system and in the NullSpend dashboard.

```
X-NullSpend-Trace-Id: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
```

### `X-NullSpend-Request-Id`

The NullSpend request ID for this response. Echoes the inbound `X-NullSpend-Request-Id` if it was provided and valid; otherwise a freshly generated ID.

```
X-NullSpend-Request-Id: 01J9F6X3R3HM6E3D6N5N0M0G7Y
```

### `X-NullSpend-Effective-Tags`

The merged tag set actually applied to this request — inbound `X-NullSpend-Tags` plus the API key's `default_tags`. Useful for verifying tag merge behavior.

```
X-NullSpend-Effective-Tags: {"team":"billing","env":"production"}
```

Omitted when the merged tag set is empty.

### `X-NullSpend-Warning`

Non-fatal warning about the request. The request still proceeds. Currently emitted values:

| Value | Meaning |
|---|---|
| `invalid_customer` | `X-NullSpend-Customer` failed format validation; customer was dropped |

```
X-NullSpend-Warning: invalid_customer
```

### `X-NullSpend-Denied`

Set to `1` on every response that was blocked by enforcement (budget, velocity, session, tag-budget, customer-budget, plan-limit, loop-detection, mandate). Use this to short-circuit response handling without parsing the JSON body.

```
X-NullSpend-Denied: 1
```

Only present on denied responses.

### `NullSpend-Version`

The API version used to process this request.

```
NullSpend-Version: 2026-04-01
```

### `x-nullspend-overhead-ms`

Proxy processing overhead in milliseconds. This is the time NullSpend added on top of the upstream provider's latency (budget checks, cost calculation, logging).

```
x-nullspend-overhead-ms: 12
```

### `Server-Timing`

W3C Server-Timing header with per-step latency breakdown:

```
Server-Timing: preflight;dur=0;desc="Auth + rate limit",body;dur=0;desc="Body parse",budget;dur=12;desc="Budget check",overhead;dur=12;desc="Proxy overhead",upstream;dur=834;desc="Provider latency",total;dur=846;desc="Total"
```

Steps (`preflight`, `body`, `budget`) are only included when measured. The `overhead`, `upstream`, and `total` entries are always present.

### Budget Proximity Headers

When the request matches a budget with enforcement, the proxy includes budget proximity headers. These signal how close the agent is to hitting the budget wall.

| Header | Value | When |
|---|---|---|
| `X-NullSpend-Budget-Limit` | Total budget in microdollars | Always (when budgets exist) |
| `X-NullSpend-Budget-Spent` | Current spend + reservations in microdollars | Always |
| `X-NullSpend-Budget-Remaining` | Raw remaining in microdollars | Always |
| `X-NullSpend-Budget-Entity` | `{entityType}:{entityId}` of the tightest budget | Always |
| `X-NullSpend-Budget-Finalization-Reserve` | Reserve amount in microdollars | Only when reserve > 0 |
| `X-NullSpend-Budget-Effective-Remaining` | Remaining minus reserve in microdollars | Only when reserve > 0 |
| `X-NullSpend-Budget-Requests-Remaining` | Estimated requests remaining (e.g., `~12`) | Only when reserve > 0 and avg cost > 0 |

These are snapshot values at check time. On streaming responses, they reflect pre-reconciliation state.

### Rate Limit Headers (on `429` responses only)

When you hit a rate limit, the response includes:

| Header | Value |
|---|---|
| `Retry-After` | Seconds until it's safe to retry (currently `60`) |

### Upstream Headers

The proxy forwards these headers from the upstream provider when present:

- `x-request-id` — Provider's request ID
- All `x-ratelimit-*` headers — Provider's own rate limit info
- `retry-after` — Provider's retry guidance
