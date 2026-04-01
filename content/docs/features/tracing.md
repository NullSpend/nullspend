---
title: "Tracing"
description: "Tracing links related LLM requests so you can see what a multi-step agent run cost as a whole. Every request through the proxy gets a trace ID — either one yo"
---

Tracing links related LLM requests so you can see what a multi-step agent run cost as a whole. Every request through the proxy gets a trace ID — either one you provide or one generated automatically.

## How Trace IDs Are Resolved

The proxy resolves a trace ID using a priority chain. It never throws — invalid headers are silently ignored and fall through to the next option.

| Priority | Source | Format |
|---|---|---|
| 1 | `traceparent` header | W3C trace context — trace ID extracted from the second field |
| 2 | `X-NullSpend-Trace-Id` header | 32-character lowercase hex string |
| 3 | Auto-generated | `crypto.randomUUID()` with dashes removed |

The resolved trace ID is returned in the `X-NullSpend-Trace-Id` response header on every request.

## Setting a Trace ID

### W3C `traceparent` (recommended)

The standard W3C trace context header. If your system already propagates `traceparent`, NullSpend picks it up automatically — no extra headers needed.

**Format:** `{version}-{trace-id}-{span-id}-{flags}`

```
traceparent: 00-a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6-b7c8d9e0f1a2b3c4-01
              │  │                                │                  │
              │  └─ 32-char lowercase hex         └─ 16-char hex    └─ flags
              └─ version (00)
```

**Validation rules:**
- Version `ff` is rejected (reserved by the W3C spec)
- All-zeros trace ID (`00000000000000000000000000000000`) is rejected
- All-zeros span ID (`0000000000000000`) is rejected
- Malformed headers are silently ignored

Both `traceparent` and `tracestate` are forwarded to the upstream provider.

**TypeScript:**

```typescript
const response = await fetch("https://proxy.nullspend.com/v1/chat/completions", {
  method: "POST",
  headers: {
    "X-NullSpend-Key": "ns_live_sk_...",
    "Authorization": "Bearer sk-...",
    "Content-Type": "application/json",
    "traceparent": "00-a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6-b7c8d9e0f1a2b3c4-01",
  },
  body: JSON.stringify({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  }),
});
```

**cURL:**

```bash
curl https://proxy.nullspend.com/v1/chat/completions \
  -H "X-NullSpend-Key: ns_live_sk_..." \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -H "traceparent: 00-a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6-b7c8d9e0f1a2b3c4-01" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

### `X-NullSpend-Trace-Id`

Use this if you don't have W3C trace context but want to group requests under a shared ID.

**Format:** 32 lowercase hex characters, matching `^[0-9a-f]{32}$`.

```bash
X-NullSpend-Trace-Id: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
```

Invalid values (wrong length, uppercase, non-hex characters) are silently ignored and the proxy auto-generates a trace ID instead.

### Auto-generated

If neither header is present (or both are invalid), the proxy generates a trace ID via `crypto.randomUUID()` with dashes removed. The generated ID is returned in the `X-NullSpend-Trace-Id` response header so you can capture it for subsequent requests in the same trace.

```typescript
const response = await fetch("https://proxy.nullspend.com/v1/chat/completions", {
  method: "POST",
  headers: {
    "X-NullSpend-Key": "ns_live_sk_...",
    "Authorization": "Bearer sk-...",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  }),
});

// Capture the auto-generated trace ID for subsequent requests
const traceId = response.headers.get("X-NullSpend-Trace-Id");
```

## Claude Agent SDK

The [`@nullspend/claude-agent`](../sdks/claude-agent.md) adapter accepts a `traceId` option:

```typescript
import { withNullSpend } from "@nullspend/claude-agent";

const options = withNullSpend({
  apiKey: "ns_live_sk_...",
  traceId: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  // ... other Claude Agent SDK options
});
```

The `traceId` must be a 32-character lowercase hex string. Invalid values throw at configuration time rather than silently falling through.

## Querying by Trace

Once requests share a trace ID, you can query them together:

- **Dashboard** — filter the cost events table by trace ID to see all requests in a trace
- **API** — `GET /api/cost-events?traceId=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6` returns all cost events for that trace
- **Summary** — `GET /api/cost-events/summary` returns the top 25 traces by spend
- **Webhooks** — every `cost_event.created` payload includes a `trace_id` field

## Related

- [Custom Headers](../api-reference/custom-headers.md) — full header reference including `traceparent` and `X-NullSpend-Trace-Id`
- [Cost Tracking](cost-tracking.md) — how cost events are recorded (trace ID is one of the recorded fields)
- [Cost Events API](../api-reference/cost-events-api.md) — query cost events by trace ID
- [Tags](tags.md) — another way to group and attribute costs
