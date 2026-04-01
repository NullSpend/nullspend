# JavaScript SDK

TypeScript/JavaScript client for the NullSpend API.

## Installation

```bash
npm install @nullspend/sdk
```

## Quick Start

```typescript
import { NullSpend } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://nullspend.com",
  apiKey: "ns_live_sk_...",
});

// Report a cost event
await ns.reportCost({
  provider: "openai",
  model: "gpt-4o",
  inputTokens: 500,
  outputTokens: 150,
  costMicrodollars: 4625,
});
```

## Configuration

The `NullSpend` constructor accepts a `NullSpendConfig` object:

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | **required** | NullSpend dashboard URL (e.g. `https://nullspend.com`) |
| `apiKey` | `string` | **required** | API key (`ns_live_sk_...`) |
| `apiVersion` | `string` | `"2026-04-01"` | API version sent via `NullSpend-Version` header |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |
| `requestTimeoutMs` | `number` | `30000` | Per-request timeout in ms. Set to `0` to disable |
| `maxRetries` | `number` | `2` | Max retries on transient failures. Clamped to `[0, 10]` |
| `retryBaseDelayMs` | `number` | `500` | Base delay between retries in ms |
| `maxRetryTimeMs` | `number` | `0` | Total wall-time cap for all retry attempts. `0` = no cap |
| `onRetry` | `(info: RetryInfo) => void \| boolean` | — | Called before each retry. Return `false` to abort |
| `costReporting` | `CostReportingConfig` | — | Enable client-side cost event batching (see below) |

## Actions (Human-in-the-Loop)

The SDK provides methods for the full [HITL approval workflow](../features/human-in-the-loop.md).

### `createAction(input)`

Create a new action for human approval.

```typescript
const action = await ns.createAction({
  agentId: "support-agent",
  actionType: "send_email",
  payload: { to: "user@example.com", subject: "Refund" },
  metadata: { ticketId: "T-1234" },
  expiresInSeconds: 1800,
});
console.log(action.id, action.status, action.expiresAt);
```

### `getAction(id)`

Fetch the current state of an action.

```typescript
const action = await ns.getAction("ns_act_550e8400-...");
console.log(action.status); // "pending" | "approved" | "rejected" | ...
```

### `markResult(id, input)`

Report execution status back to NullSpend.

```typescript
// Start executing
await ns.markResult(id, { status: "executing" });

// Report success
await ns.markResult(id, {
  status: "executed",
  result: { rowsDeleted: 42 },
});

// Or report failure
await ns.markResult(id, {
  status: "failed",
  errorMessage: "Connection timeout",
});
```

### `waitForDecision(id, options?)`

Poll until the action leaves `pending` status or the timeout elapses.

```typescript
const decision = await ns.waitForDecision(id, {
  pollIntervalMs: 2000,   // default: 2000 (2s)
  timeoutMs: 300000,       // default: 300000 (5 min)
  onPoll: (action) => console.log(action.status),
});
```

Throws `TimeoutError` if the timeout elapses while still `pending`.

### `proposeAndWait<T>(options)`

High-level orchestrator that combines create → poll → execute → report:

```typescript
const result = await ns.proposeAndWait({
  agentId: "data-agent",
  actionType: "db_write",
  payload: { query: "DELETE FROM logs WHERE age > 90" },
  expiresInSeconds: 3600,

  execute: async ({ actionId }) => {
    // Runs only after human approval.
    // actionId can be sent as X-NullSpend-Action-Id to correlate costs.
    return await deleteOldLogs();
  },

  pollIntervalMs: 2000,  // default: 2000
  timeoutMs: 300000,      // default: 300000 (5 min)
  onPoll: (action) => {},
});
```

- On approval: marks `executing`, calls `execute()`, marks `executed` with result
- On rejection/expiry: throws `RejectedError`
- On execute failure: marks `failed`, re-throws the original error
- Handles `409` conflicts from concurrent writes gracefully

## Cost Reporting

Three approaches for reporting cost events.

### `reportCost(event)` — Single Event

```typescript
const result = await ns.reportCost({
  provider: "anthropic",
  model: "claude-sonnet-4-20250514",
  inputTokens: 1000,
  outputTokens: 500,
  costMicrodollars: 6750,
  // Optional fields:
  cachedInputTokens: 200,
  reasoningTokens: 0,
  durationMs: 1200,
  sessionId: "session-123",
  traceId: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  eventType: "llm",        // "llm" | "tool" | "custom"
  tags: { team: "backend" },
});
console.log(result.id, result.createdAt);
```

### `reportCostBatch(events)` — Batch

```typescript
const { inserted, ids } = await ns.reportCostBatch([
  { provider: "openai", model: "gpt-4o", inputTokens: 500, outputTokens: 150, costMicrodollars: 4625 },
  { provider: "openai", model: "gpt-4o-mini", inputTokens: 1000, outputTokens: 300, costMicrodollars: 225 },
]);
```

### Client-Side Batching

Enable automatic batching by passing `costReporting` in the constructor:

```typescript
const ns = new NullSpend({
  baseUrl: "https://nullspend.com",
  apiKey: "ns_live_sk_...",
  costReporting: {
    batchSize: 10,          // default: 10 (clamped [1, 100])
    flushIntervalMs: 5000,  // default: 5000 (min 100)
    maxQueueSize: 1000,     // default: 1000 (min 1)
    onDropped: (count) => console.warn(`Dropped ${count} events`),
    onFlushError: (error, events) => console.error("Flush failed", error),
  },
});

// Queue events — they flush automatically
ns.queueCost({ provider: "openai", model: "gpt-4o", inputTokens: 500, outputTokens: 150, costMicrodollars: 4625 });

// Force an immediate flush
await ns.flush();

// Flush remaining events and stop the timer
await ns.shutdown();
```

When the queue overflows `maxQueueSize`, the oldest events are dropped and `onDropped` is called.

## Tracked Fetch (Auto Cost Tracking)

`createTrackedFetch` wraps `globalThis.fetch` to automatically track cost events for OpenAI and Anthropic API calls. Call the provider directly — the SDK intercepts responses, extracts token counts, calculates cost, and reports it to NullSpend. No proxy required.

Requires `costReporting` in the constructor config.

### Basic Usage

```typescript
import OpenAI from "openai";
import { NullSpend } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://app.nullspend.com",
  apiKey: "ns_live_sk_...",
  costReporting: {},
});

// Pass the tracked fetch to the OpenAI client
const openai = new OpenAI({
  fetch: ns.createTrackedFetch("openai"),
});

// Use OpenAI as normal — costs are tracked automatically
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});

// Flush pending cost events before exit
await ns.shutdown();
```

Works with Anthropic too:

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  fetch: ns.createTrackedFetch("anthropic"),
});
```

### Options

`createTrackedFetch(provider, options?)` accepts a `TrackedFetchOptions` object:

| Option | Type | Default | Description |
|---|---|---|---|
| `sessionId` | `string` | — | Session ID for cost grouping and session limits |
| `tags` | `Record<string, string>` | — | Tags for cost attribution |
| `traceId` | `string` | — | 32-char hex trace ID for request correlation |
| `actionId` | `string` | — | NullSpend action ID to correlate with HITL approval |
| `enforcement` | `boolean` | `false` | Enable cooperative budget and mandate enforcement |
| `onCostError` | `(error: Error) => void` | stderr log | Called on non-fatal cost tracking errors |
| `onDenied` | `(reason: DenialReason) => void` | — | Called before throwing `BudgetExceededError` or `MandateViolationError` |

### Enforcement Mode

When `enforcement: true`, the SDK fetches and caches your key's policy before each request. If the request would violate a budget or mandate, it throws instead of calling the provider:

```typescript
import { BudgetExceededError, MandateViolationError } from "@nullspend/sdk";

const fetch = ns.createTrackedFetch("openai", {
  enforcement: true,
  onDenied: (reason) => {
    if (reason.type === "budget") {
      console.log(`Only $${reason.remaining / 1_000_000} remaining`);
    } else {
      console.log(`Mandate violation: ${reason.mandate}`);
    }
  },
});

try {
  const openai = new OpenAI({ fetch });
  await openai.chat.completions.create({ model: "gpt-4o", messages: [...] });
} catch (err) {
  if (err instanceof BudgetExceededError) {
    // Budget would be exceeded — request was NOT sent to OpenAI
  }
  if (err instanceof MandateViolationError) {
    // Model or provider not allowed by key policy
  }
}
```

### How It Works

1. Intercepts `fetch` calls to OpenAI/Anthropic chat/messages endpoints
2. Non-tracked routes (GET requests, non-LLM endpoints) pass through untouched
3. For tracked requests: parses the response (streaming or non-streaming), extracts token usage, calculates cost using the built-in pricing engine, and queues a cost event
4. Cost events are batched and flushed according to `costReporting` config
5. If the request goes through `proxy.nullspend.com`, tracking is skipped (proxy already handles it)

### Source Field

Cost events from `createTrackedFetch` are reported with `source: "api"`. This appears as "SDK" in the dashboard's source filter and analytics.

## Budget Status

```typescript
const status = await ns.checkBudget();

for (const entity of status.entities) {
  console.log(
    `${entity.entityType}/${entity.entityId}: ` +
    `$${entity.spendMicrodollars / 1_000_000} / $${entity.limitMicrodollars / 1_000_000}`
  );
}
```

Returns a `BudgetStatus` with an `entities` array. Each `BudgetEntity` contains:

| Field | Type | Description |
|---|---|---|
| `entityType` | `string` | Budget entity type (e.g. `"user"`, `"key"`, `"tag"`) |
| `entityId` | `string` | Entity identifier |
| `limitMicrodollars` | `number` | Budget ceiling |
| `spendMicrodollars` | `number` | Current spend |
| `remainingMicrodollars` | `number` | Remaining budget |
| `policy` | `string` | Enforcement policy |
| `resetInterval` | `string \| null` | Reset period (e.g. `"daily"`, `"monthly"`) |
| `currentPeriodStart` | `string \| null` | ISO 8601 timestamp of current period start |

## Cost Awareness (Read APIs)

Query your spend data programmatically — useful for cost-aware agents and dashboards.

### `listBudgets()`

Fetch all budgets for the authenticated org.

```typescript
const { data: budgets } = await ns.listBudgets();

for (const budget of budgets) {
  const spent = budget.spendMicrodollars / 1_000_000;
  const limit = budget.maxBudgetMicrodollars / 1_000_000;
  console.log(`${budget.entityType}/${budget.entityId}: $${spent} / $${limit}`);
}
```

Each `BudgetRecord` contains:

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Budget ID |
| `entityType` | `string` | `"user"`, `"api_key"`, or `"tag"` |
| `entityId` | `string` | Entity identifier |
| `maxBudgetMicrodollars` | `number` | Budget ceiling |
| `spendMicrodollars` | `number` | Current spend |
| `policy` | `string` | `"strict_block"`, `"soft_block"`, or `"warn"` |
| `resetInterval` | `string \| null` | `"daily"`, `"monthly"`, etc. |
| `thresholdPercentages` | `number[]` | Webhook alert thresholds |
| `velocityLimitMicrodollars` | `number \| null` | Per-window spend limit |
| `sessionLimitMicrodollars` | `number \| null` | Per-session spend limit |

### `getCostSummary(period?)`

Get aggregated spend data for a time period.

```typescript
const summary = await ns.getCostSummary("30d"); // "7d" | "30d" | "90d"

console.log(`Period: ${summary.totals.period}`); // "7d" | "30d" | "90d"
console.log(`Total spend: $${summary.totals.totalCostMicrodollars / 1_000_000}`);
console.log(`Total requests: ${summary.totals.totalRequests}`);

// Spend by model
summary.models.forEach(m => {
  console.log(`  ${m.model}: $${m.totalCostMicrodollars / 1_000_000}`);
});

// Daily trend
for (const day of summary.daily) {
  console.log(`  ${day.date}: $${day.totalCostMicrodollars / 1_000_000}`);
}
```

### `listCostEvents(options?)`

Fetch recent cost events with pagination.

```typescript
// Get the last 10 cost events
const { data: events, cursor } = await ns.listCostEvents({ limit: 10 });

for (const event of events) {
  console.log(`${event.model}: ${event.inputTokens} in / ${event.outputTokens} out — $${event.costMicrodollars / 1_000_000}`);
}

// Paginate with cursor
if (cursor) {
  const nextPage = await ns.listCostEvents({ limit: 10, cursor: JSON.stringify(cursor) });
}
```

## Retry Behavior

The SDK automatically retries on transient failures:

**Retryable:** `429`, `500`, `502`, `503`, `504`, network errors (`TypeError`), timeout errors (`AbortSignal.timeout`)

**Not retryable:** user-initiated abort (`AbortError`), `4xx` errors other than `429`

**Backoff:** Full-jitter exponential — `floor(random() * min(base * 2^attempt, 5000ms))`. The `Retry-After` header is respected when present (used once, then back to exponential).

**Idempotency:** Mutating requests (`POST`) include an `Idempotency-Key` header generated once and reused across retries.

The `onRetry` callback receives a `RetryInfo` object:

```typescript
const ns = new NullSpend({
  baseUrl: "https://nullspend.com",
  apiKey: "ns_live_sk_...",
  maxRetries: 3,
  onRetry: ({ attempt, delayMs, error, method, path }) => {
    console.log(`Retry ${attempt} for ${method} ${path} in ${delayMs}ms: ${error.message}`);
    // Return false to abort retrying
  },
});
```

## Error Handling

Three error classes, all extending `Error`:

### `NullSpendError`

Base error for all SDK errors. Properties:

| Property | Type | Description |
|---|---|---|
| `statusCode` | `number \| undefined` | HTTP status code (if from an API response) |
| `code` | `string \| undefined` | Machine-readable error code from the API |

```typescript
try {
  await ns.createAction({ ... });
} catch (err) {
  if (err instanceof NullSpendError) {
    console.log(err.statusCode); // 409
    console.log(err.code);       // "invalid_action_transition"
  }
}
```

### `TimeoutError`

Thrown by `waitForDecision` when the timeout elapses. Extends `NullSpendError`.

### `RejectedError`

Thrown by `proposeAndWait` when the action is rejected or expired. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `actionId` | `string` | The action that was rejected |
| `actionStatus` | `string` | The terminal status (`"rejected"` or `"expired"`) |

```typescript
try {
  await ns.proposeAndWait({ ... });
} catch (err) {
  if (err instanceof RejectedError) {
    console.log(`${err.actionId} was ${err.actionStatus}`);
  }
}
```

## Types

All types are exported from the package:

```typescript
import type {
  // Configuration
  NullSpendConfig,
  CostReportingConfig,
  RetryInfo,

  // Actions
  CreateActionInput,
  CreateActionResponse,
  ActionRecord,
  MarkResultInput,
  MutateActionResponse,
  ProposeAndWaitOptions,
  ExecuteContext,
  WaitForDecisionOptions,

  // Cost reporting
  CostEventInput,
  ReportCostResponse,
  ReportCostBatchResponse,

  // Budgets
  BudgetStatus,
  BudgetEntity,
  BudgetRecord,
  ListBudgetsResponse,

  // Cost awareness (read)
  CostEventRecord,
  ListCostEventsResponse,
  ListCostEventsOptions,
  CostSummaryResponse,
  CostSummaryPeriod,

  // Enums
  ActionType,
  ActionStatus,
} from "@nullspend/sdk";
```

**Constants:**

```typescript
import {
  ACTION_TYPES,       // readonly tuple of valid action types
  ACTION_STATUSES,    // readonly tuple of all statuses
  TERMINAL_STATUSES,  // ReadonlySet of terminal statuses
} from "@nullspend/sdk";
```

**Utilities:**

```typescript
import {
  waitWithAbort,       // waitForDecision with AbortSignal support
  interruptibleSleep,  // sleep that can be cancelled via AbortSignal
} from "@nullspend/sdk";
```

## Related

- [Human-in-the-Loop](../features/human-in-the-loop.md) — approval workflow concepts and best practices
- [Cost Tracking](../features/cost-tracking.md) — how cost events are recorded
- [Actions API](../api-reference/actions-api.md) — raw HTTP endpoint reference
- [Budgets API](../api-reference/budgets-api.md) — budget management endpoints
- [Claude Agent Adapter](claude-agent.md) — adapter for the Claude Agent SDK
