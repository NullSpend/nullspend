---
title: "Claude Agent Adapter"
description: "Adapter that routes Claude Agent SDK calls through the NullSpend proxy for automatic cost tracking and budget enforcement."
---

Adapter that routes Claude Agent SDK calls through the NullSpend proxy for automatic cost tracking and budget enforcement.

## Installation

```bash
npm install @nullspend/claude-agent
```

Peer dependency: `@anthropic-ai/claude-agent-sdk`

## Usage

`withNullSpend()` takes your Claude Agent SDK options plus NullSpend-specific fields, and returns modified `Options` with the proxy URL and headers injected.

```typescript
import { withNullSpend } from "@nullspend/claude-agent";

const options = withNullSpend({
  // NullSpend options
  apiKey: process.env.NULLSPEND_API_KEY!,
  budgetSessionId: "session-abc-123",
  tags: { project: "customer-support", environment: "production" },
  traceId: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  actionId: "ns_act_550e8400-e29b-41d4-a716-446655440000",
  proxyUrl: "https://proxy.nullspend.dev", // default

  // Claude Agent SDK options (passed through)
  model: "claude-sonnet-4-20250514",
  prompt: "You are a helpful assistant.",
  maxTurns: 10,
});
```

## Options

`NullSpendAgentOptions` — all NullSpend-specific fields:

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `apiKey` | `string` | Yes | — | NullSpend API key (`ns_live_sk_...` or `ns_test_sk_...`) |
| `budgetSessionId` | `string` | No | Auto-generated | Session ID for per-conversation budget enforcement. If omitted, a unique ID is auto-generated (e.g., `ses_mncqabm4_a7f3`). |
| `autoSession` | `boolean` | No | `true` | Auto-generate a session ID when `budgetSessionId` is not provided. Set to `false` to disable session tracking entirely. |
| `tags` | `Record<string, string>` | No | — | Key-value tags for cost attribution |
| `traceId` | `string` | No | — | 32-char lowercase hex trace ID for request correlation |
| `actionId` | `string` | No | — | NullSpend action ID (`ns_act_<UUID>`) to correlate costs with an approved action |
| `proxyUrl` | `string` | No | `https://proxy.nullspend.dev` | Override the proxy URL |

All other fields are passed through to the Claude Agent SDK as-is.

## Validation

`withNullSpend` validates inputs eagerly and throws on invalid values:

| Field | Rule | Error |
|---|---|---|
| `apiKey` | Required, no newlines | `"withNullSpend: apiKey is required"` |
| `traceId` | Must match `^[0-9a-f]{32}$` | `"traceId must be a 32-char lowercase hex string"` |
| `actionId` | Must match `^ns_act_<UUID>$` (case-insensitive) | `"actionId must be in ns_act_<UUID> format"` |
| `tags` keys | Max 10 keys, each matching `[a-zA-Z0-9_-]+`, max 64 chars | Key-specific error message |
| `tags` values | Max 256 chars per value | Value-specific error message |
| `apiKey`, `budgetSessionId` | No `\r` or `\n` characters | `"must not contain newline characters"` |

## How It Works

`withNullSpend` sets two environment variables in the returned options:

- **`ANTHROPIC_BASE_URL`** — set to `proxyUrl` (default `https://proxy.nullspend.dev`), routing all Anthropic API calls through the proxy
- **`ANTHROPIC_CUSTOM_HEADERS`** — newline-delimited custom headers injected into every request

### Headers Set

| Header | Source | Always Present |
|---|---|---|
| `x-nullspend-key` | `apiKey` | Yes |
| `x-nullspend-session` | `budgetSessionId` or auto-generated | Yes (unless `autoSession: false`) |
| `x-nullspend-tags` | `JSON.stringify(tags)` | Only if provided and non-empty |
| `x-nullspend-trace-id` | `traceId` | Only if provided |
| `x-nullspend-action-id` | `actionId` | Only if provided |

### Environment Merging

The returned `env` object is built by layering:

1. `process.env` (base — preserves `PATH`, `HOME`, `ANTHROPIC_API_KEY`, etc.)
2. Any `env` from the caller's SDK options (overrides)
3. NullSpend's `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS` (final layer)

If the caller already set `ANTHROPIC_CUSTOM_HEADERS`, NullSpend's headers are appended after the existing value.

## Session Tracking

Every `withNullSpend()` call auto-generates a unique session ID by default. This means every agent invocation gets session tracking without any configuration.

```typescript
// Session tracking is automatic — no config needed
const options = withNullSpend({
  apiKey: process.env.NULLSPEND_API_KEY!,
  prompt: "Debug this auth bug",
});
// A session ID like "ses_mncqabm4_a7f3" is generated and sent
// with every request. View sessions at /app/sessions in the dashboard.
```

**Override with your own session ID** when you want to control the grouping:

```typescript
const options = withNullSpend({
  apiKey: process.env.NULLSPEND_API_KEY!,
  budgetSessionId: `task-${taskId}`,  // your own session boundary
  prompt: "Deploy v2.3",
});
```

**Disable session tracking** for fire-and-forget requests:

```typescript
const options = withNullSpend({
  apiKey: process.env.NULLSPEND_API_KEY!,
  autoSession: false,
  prompt: "Quick one-off query",
});
```

Session IDs are used for:
- **Cost grouping** — see total spend per task/conversation in the Sessions page
- **Session limits** — cap spend per session (e.g., $2 per agent invocation) via budget configuration
- **Debugging** — trace all requests in a single agent run

## Related

- [Claude Code Quickstart](../quickstart/claude-code.md) — get started in 2 minutes
- [Tracing](../features/tracing.md) — trace ID format and resolution
- [Tags](../features/tags.md) — tag format and cost attribution
- [Human-in-the-Loop](../features/human-in-the-loop.md) — action ID correlation
- [JavaScript SDK](javascript.md) — full NullSpend API client
