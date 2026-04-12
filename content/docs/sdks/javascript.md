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
  baseUrl: "https://nullspend.dev",
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
| `baseUrl` | `string` | **required** | NullSpend dashboard URL (e.g. `https://nullspend.dev`) |
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
const { id, status, expiresAt } = await ns.createAction({
  agentId: "support-agent",
  actionType: "send_email",
  payload: { to: "user@example.com", subject: "Refund" },
  metadata: { ticketId: "T-1234" },
  expiresInSeconds: 1800,
});
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
const { id, createdAt } = await ns.reportCost({
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
  baseUrl: "https://nullspend.dev",
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

console.log(`Total spend: $${summary.totals.totalCostMicrodollars / 1_000_000}`);
console.log(`Total requests: ${summary.totals.totalRequests}`);

// Spend by model
for (const [model, cost] of Object.entries(summary.models)) {
  console.log(`  ${model}: $${cost / 1_000_000}`);
}

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
  const nextPage = await ns.listCostEvents({ limit: 10, cursor: `${cursor.createdAt},${cursor.id}` });
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
  baseUrl: "https://nullspend.dev",
  apiKey: "ns_live_sk_...",
  maxRetries: 3,
  onRetry: ({ attempt, delayMs, error, method, path }) => {
    console.log(`Retry ${attempt} for ${method} ${path} in ${delayMs}ms: ${error.message}`);
    // Return false to abort retrying
  },
});
```

## Customer Sessions

Scope cost tracking and enforcement to a specific customer for per-customer profitability tracking.

```typescript
const session = ns.customer("acme-corp");

// Pre-configured fetch functions for each provider
const openai = new OpenAI({ fetch: session.openai });
const anthropic = new Anthropic({ fetch: session.anthropic });

// Or get a tracked fetch for a specific provider
const fetch = session.fetch("openai");
```

The `customer()` method validates and normalizes the customer ID (trimmed, max 256 chars, alphanumeric + `._:-`). All cost events from the session's tracked fetch functions are tagged with the customer ID, enabling per-customer margin tracking on the dashboard.

```typescript
// With plan tagging and enforcement options
const session = ns.customer("acme-corp", {
  plan: "pro",
  enforcement: true,
  sessionId: "task-042",
  tags: { team: "backend" },
  onDenied: (reason) => console.log(reason),
});
```

## Tracked Fetch (Provider Wrappers)

Wrap your LLM provider's `fetch` to automatically track costs and enforce policies client-side.

### Basic Setup

```typescript
const ns = new NullSpend({
  baseUrl: "https://nullspend.dev",
  apiKey: "ns_live_sk_...",
  costReporting: {},  // required for createTrackedFetch
});

const openai = new OpenAI({ fetch: ns.createTrackedFetch("openai") });
const anthropic = new Anthropic({ fetch: ns.createTrackedFetch("anthropic") });
```

Cost events are calculated locally using the built-in pricing engine and reported asynchronously in batches. Your requests go directly to the provider — no proxy required.

The SDK tracks chat completions, legacy completions, and embeddings endpoints. The OpenAI Responses API (`/responses`) uses a different SSE format and is not yet tracked — use the proxy for responses API cost tracking.

### Enforcement Mode

Enable `enforcement: true` to check budgets, model mandates, and session limits before each request:

```typescript
const openai = new OpenAI({
  fetch: ns.createTrackedFetch("openai", {
    enforcement: true,
    sessionId: "task-042",
    sessionLimitMicrodollars: 5_000_000, // $5 per session
    tags: { team: "backend", customer: "acme" },
    onDenied: (reason) => {
      if (reason.type === "budget") console.log(`Budget: ${reason.remaining} remaining`);
      if (reason.type === "mandate") console.log(`Mandate: ${reason.mandate} blocks ${reason.requested}`);
      if (reason.type === "session_limit") console.log(`Session: ${reason.sessionSpend} of ${reason.sessionLimit}`);
    },
    onCostError: (err) => console.warn("Cost tracking error:", err.message),
  }),
});
```

### `TrackedFetchOptions`

| Option | Type | Default | Description |
|---|---|---|---|
| `customer` | `string` | — | Customer ID for per-customer cost attribution and margin tracking |
| `enforcement` | `boolean` | `false` | Enable budget, mandate, and session limit checks |
| `sessionId` | `string` | — | Session identifier for cost correlation and session limits |
| `sessionLimitMicrodollars` | `number` | — | Manual per-session spend cap (takes precedence over policy) |
| `tags` | `Record<string, string>` | — | Tags attached to every cost event |
| `traceId` | `string` | — | Distributed trace ID |
| `actionId` | `string` | — | HITL action ID for cost correlation |
| `onDenied` | `(reason: DenialReason) => void` | — | Called before throwing enforcement errors |
| `onCostError` | `(error: Error) => void` | `console.warn` | Called on non-fatal cost tracking errors (including policy cache failures) |

### Enforcement Flow

When `enforcement: true`, each request goes through:

1. **Mandate check** — is this model/provider allowed by key policy?
2. **Budget check** — does estimated cost fit within remaining budget?
3. **Session limit check** — does `sessionSpend + estimate` exceed the session limit?

If any check fails, the SDK throws the corresponding error **before** calling the provider. If the policy endpoint is unreachable, the SDK falls open (requests proceed) and calls `onCostError` so failures aren't silently swallowed — except for manual session limits, which are always enforced.

### Proxy Mode vs Direct Mode

The SDK detects whether requests go through the NullSpend proxy (by comparing the request URL origin against `proxyUrl`) or directly to the provider:

- **Proxy mode:** The proxy handles cost tracking and budget enforcement server-side. The SDK intercepts proxy 429 responses with the `X-NullSpend-Denied: 1` header and throws the corresponding error (`BudgetExceededError`, `VelocityExceededError`, `SessionLimitExceededError`, `TagBudgetExceededError`). Non-NullSpend 429s (e.g., upstream provider rate limits) pass through as raw responses.
- **Direct mode:** The SDK tracks costs client-side using the built-in pricing engine. No 429 interception occurs — the SDK only enforces locally via the policy cache. This prevents trust-boundary violations from arbitrary upstream headers.

### Session Limit Enforcement

Session limits track cumulative spend per `createTrackedFetch()` instance:

- Each instance starts at 0 spend
- Actual cost from each successful response is accumulated
- Before each request, the SDK checks `sessionSpend + estimate > sessionLimit`
- The limit comes from `sessionLimitMicrodollars` (manual) or the policy endpoint (from budget config), with manual taking precedence
- Streaming cost is accumulated asynchronously — a concurrent request may slip through before the first stream's cost is counted
- Failed responses (4xx/5xx) don't count toward session spend

> **Note:** SDK session limits are cooperative — each `createTrackedFetch()` instance tracks independently. For fleet-wide authoritative enforcement, use the proxy.

## Error Handling

Eight error classes, all extending `Error`. All constructors validate arguments — `NaN`, `Infinity`, and negative numbers are clamped to `0`.

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

| Property | Type | Description |
|---|---|---|
| `actionId` | `string` | The action that timed out |
| `timeoutMs` | `number` | The timeout duration in milliseconds |

### `RejectedError`

Thrown by `proposeAndWait` when the action is rejected or expired. Extends `NullSpendError`.

| Property | Type | Description |
|---|---|---|
| `actionId` | `string` | The action that was rejected |
| `actionStatus` | `string` | The terminal status (`"rejected"` or `"expired"`) |

### `BudgetExceededError`

Thrown when enforcement is enabled and the estimated cost exceeds remaining budget. In proxy mode, also thrown when the proxy returns a `budget_exceeded` or `customer_budget_exceeded` 429.

| Property | Type | Description |
|---|---|---|
| `remainingMicrodollars` | `number` | Budget remaining when denial occurred |
| `entityType` | `string \| undefined` | Budget entity type (`"api_key"`, `"customer"`, `"tag"`) |
| `entityId` | `string \| undefined` | Entity identifier |
| `limitMicrodollars` | `number \| undefined` | Budget ceiling |
| `spendMicrodollars` | `number \| undefined` | Current spend |
| `upgradeUrl` | `string \| undefined` | Plan upgrade URL (may contain `{customer_id}` placeholder) |

### `MandateViolationError`

Thrown when the requested model or provider is not allowed by key policy.

| Property | Type | Description |
|---|---|---|
| `mandate` | `string` | Which mandate was violated (`"allowed_models"` or `"allowed_providers"`) |
| `requested` | `string` | The model or provider that was denied |
| `allowed` | `string[]` | The allowed values |

### `SessionLimitExceededError`

Thrown when session spend plus estimated cost exceeds the session limit.

| Property | Type | Description |
|---|---|---|
| `sessionSpendMicrodollars` | `number` | Accumulated session spend at denial time |
| `sessionLimitMicrodollars` | `number` | Configured session limit |

### `VelocityExceededError`

Thrown in proxy mode when the proxy returns a `velocity_exceeded` 429. Indicates spending rate exceeded the configured velocity limit.

| Property | Type | Description |
|---|---|---|
| `retryAfterSeconds` | `number \| undefined` | Seconds until the velocity window resets |
| `limitMicrodollars` | `number \| undefined` | Velocity limit per window |
| `windowSeconds` | `number \| undefined` | Velocity window duration |
| `currentMicrodollars` | `number \| undefined` | Current spend in window |

### `TagBudgetExceededError`

Thrown in proxy mode when the proxy returns a `tag_budget_exceeded` 429. Indicates a tag-scoped budget was exhausted.

| Property | Type | Description |
|---|---|---|
| `tagKey` | `string \| undefined` | Tag key (e.g., `"project"`) |
| `tagValue` | `string \| undefined` | Tag value (e.g., `"docs-processor"`) |
| `remainingMicrodollars` | `number \| undefined` | Remaining tag budget |
| `limitMicrodollars` | `number \| undefined` | Tag budget ceiling |
| `spendMicrodollars` | `number \| undefined` | Current tag spend |

### `DenialReason` (callback type)

The `onDenied` callback receives a `DenialReason` union before the error is thrown:

| `type` | Fields | Description |
|---|---|---|
| `"budget"` | `remaining`, `entityType?`, `entityId?`, `limit?`, `spend?` | Budget entity denied |
| `"mandate"` | `mandate`, `requested`, `allowed` | Policy rule violation |
| `"session_limit"` | `sessionSpend`, `sessionLimit` | Session spend cap exceeded |
| `"velocity"` | `retryAfterSeconds?`, `limit?`, `window?`, `current?` | Velocity rate limit hit |
| `"tag_budget"` | `tagKey?`, `tagValue?`, `remaining?`, `limit?`, `spend?` | Tag-scoped budget denied |

### Example

```typescript
import {
  BudgetExceededError,
  MandateViolationError,
  SessionLimitExceededError,
  VelocityExceededError,
  TagBudgetExceededError,
} from "@nullspend/sdk";

try {
  await openai.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "Hi" }] });
} catch (err) {
  if (err instanceof BudgetExceededError) {
    console.log(`Budget exhausted: $${err.remainingMicrodollars / 1_000_000} remaining`);
    if (err.upgradeUrl) console.log(`Upgrade: ${err.upgradeUrl}`);
  } else if (err instanceof VelocityExceededError) {
    console.log(`Rate limited — retry in ${err.retryAfterSeconds}s`);
  } else if (err instanceof TagBudgetExceededError) {
    console.log(`Tag ${err.tagKey}=${err.tagValue} budget exceeded`);
  } else if (err instanceof SessionLimitExceededError) {
    console.log(`Session spent $${err.sessionSpendMicrodollars / 1_000_000} of $${err.sessionLimitMicrodollars / 1_000_000} limit`);
  } else if (err instanceof MandateViolationError) {
    console.log(`${err.mandate} blocks "${err.requested}". Allowed: ${err.allowed.join(", ")}`);
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

  // Tracked fetch & customer sessions
  TrackedFetchOptions,
  TrackedProvider,
  DenialReason,
  CustomerSession,
  CustomerSessionOptions,

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
