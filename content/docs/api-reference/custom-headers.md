# Custom Headers

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
