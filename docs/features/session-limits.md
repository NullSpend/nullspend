# Session Limits

Session limits cap how much a single agent conversation can spend, regardless of the overall budget. A runaway agent loop that stays under velocity limits can still accumulate significant cost across a long session — session limits provide a per-conversation ceiling.

See [Budgets](budgets.md) for overall budget configuration.

## How It Works

```
Request arrives with X-NullSpend-Session header
  │
  ▼
Lookup session spend in DO SQLite
  │
  ▼
currentSpend + estimate > sessionLimit?
  ├─ NO → Continue to velocity + budget checks
  │
  ▼
429 (no Retry-After) → agent should start a new session
```

If the `X-NullSpend-Session` header is absent, session limit enforcement is skipped entirely.

## Configuration

Set this field when creating or updating a budget via the API:

| Field | Type | Range | Default |
|---|---|---|---|
| `sessionLimitMicrodollars` | integer or null | > 0 | null (disabled) |

Setting `sessionLimitMicrodollars` to `null` disables session limit enforcement for that budget entity.

## Setting the Session Header

Send the `X-NullSpend-Session` header with each request to identify the conversation:

**TypeScript:**

```typescript
const response = await fetch("https://proxy.nullspend.dev/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${apiKey}`,
    "X-NullSpend-Session": "task-042",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  }),
});
```

**Python:**

```python
response = requests.post(
    "https://proxy.nullspend.dev/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "X-NullSpend-Session": "task-042",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    },
)
```

**Claude Agent SDK:**

```typescript
const client = withNullSpend(baseClient, {
  budgetSessionId: "task-042",  // NOT the SDK's conversation sessionId
});
```

The header value is truncated to 256 characters (not rejected). Choose short, meaningful IDs.

## Session Tracking

Session spend is tracked in the Durable Object's SQLite database:

| Column | Type | Description |
|---|---|---|
| `entity_key` | text | Budget entity (`user:{id}` or `api_key:{id}`) |
| `session_id` | text | Your session identifier |
| `spend` | integer | Cumulative spend in microdollars |
| `request_count` | integer | Number of requests in this session |
| `last_seen` | integer | Timestamp of last request (ms) |

**Lifecycle:**

- **Reservation:** When a request is approved, the estimated cost is added to `spend`
- **Reconciliation:** When the actual cost is known, the delta (`actual - estimate`) is applied: `spend = MAX(0, spend + delta)`
- **Expired reservations:** If a reservation expires without reconciliation (crash/timeout), the DO alarm reverses the reservation from session spend
- **Cleanup:** Sessions with `last_seen` older than 24 hours are deleted by the DO alarm

## 429 Response

When a session limit is exceeded, the proxy returns:

```json
{
  "error": {
    "code": "session_limit_exceeded",
    "message": "Request blocked: session spend exceeds session limit. Start a new session.",
    "details": {
      "session_id": "conv_abc123",
      "session_spend_microdollars": 4800000,
      "session_limit_microdollars": 5000000
    }
  }
}
```

**No `Retry-After` header.** Unlike velocity limits, the session is done — retrying won't help. The agent should start a new session (new `X-NullSpend-Session` value) to continue.

## Webhooks

### `session.limit_exceeded`

Fires when a request is denied because the session spend cap is reached. Key fields in `data.object`:

| Field | Description |
|---|---|
| `budget_entity_type` | `user` or `api_key` |
| `budget_entity_id` | The entity whose session limit was hit |
| `session_id` | The session that exceeded the limit |
| `session_spend_microdollars` | Current session spend at denial time |
| `session_limit_microdollars` | Configured session limit |
| `model` | Model of the denied request |
| `provider` | `openai` or `anthropic` |
| `blocked_at` | ISO 8601 timestamp |

See [Event Types](../webhooks/event-types.md) for the full JSON example.

## Key Behaviors

- **No header = no enforcement.** Session limits only apply when `X-NullSpend-Session` is present on the request.
- **Client-defined sessions.** The proxy never creates, invalidates, or manages session IDs — your agent decides when to start a new session.
- **Independent of budget resets.** Session spend does NOT reset when the budget period resets. A session that spans a daily reset carries its full cumulative spend.
- **Always strict.** Session limits are hard caps regardless of the budget policy (`warn_only` does not apply).
- **24-hour cleanup.** Stale session data is automatically cleaned up after 24 hours of inactivity via the DO alarm.

## Enforcement Order

The enforcement pipeline runs in this order:

1. **Period reset** — if the budget period has elapsed, reset spend before any checks run
2. **Session limit** — deny before touching velocity counters
3. **Velocity limit** — sliding window + circuit breaker
4. **Budget exhaustion** — is there enough budget remaining?
5. **Reservation** — reserve estimated cost

Session is checked before velocity so that denied requests don't inflate velocity counters or affect budget accounting.

## Example

**Scenario:** $5 session limit, agent conversation "task-042".

1. Agent starts conversation "task-042", sending `X-NullSpend-Session: task-042`
2. First 10 requests cost $0.45 each — session spend reaches $4.50
3. Request 11 has an estimated cost of $0.60
4. `$4.50 + $0.60 = $5.10 > $5.00` → **denied**
5. `session.limit_exceeded` webhook fires
6. Agent receives `429` with `session_limit_exceeded` error
7. Agent starts a new conversation with `X-NullSpend-Session: task-043`
8. New session starts at $0 spend — requests resume

## SDK Cooperative Enforcement

The `@nullspend/sdk` provides client-side session limit enforcement via `createTrackedFetch()`. This is cooperative — the SDK tracks session spend locally and denies requests that would exceed the limit before calling the provider.

```typescript
import { NullSpend, SessionLimitExceededError } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://app.nullspend.dev",
  apiKey: "ns_live_sk_...",
  costReporting: {},
});

const openai = new OpenAI({
  fetch: ns.createTrackedFetch("openai", {
    enforcement: true,
    sessionId: "task-042",
    sessionLimitMicrodollars: 5_000_000, // $5 manual limit
    onDenied: (reason) => {
      if (reason.type === "session_limit") {
        console.log(`Session spent ${reason.sessionSpend}, limit is ${reason.sessionLimit}`);
      }
    },
  }),
});
```

### How It Works

1. Each `createTrackedFetch()` call creates an independent session spend accumulator starting at 0
2. Before each request, the SDK estimates cost and checks: `sessionSpend + estimate > sessionLimit`
3. If over limit, throws `SessionLimitExceededError` without calling the provider
4. After each successful response, the actual cost is accumulated into the session spend counter
5. The limit comes from `sessionLimitMicrodollars` (manual option) or the policy endpoint (from budget config), with manual taking precedence

### Differences from Proxy Enforcement

| Aspect | Proxy | SDK |
|---|---|---|
| Tracking | Server-side Durable Object, fleet-wide | Client-side closure, per-instance |
| Accuracy | Atomic reservation + reconciliation | Estimate-based, streaming cost is async |
| Multiple instances | Shared session state | Each instance tracks independently |
| Bypass | Cannot bypass (network-level) | Cooperative — raw `fetch` bypasses |
| Policy source | Database (authoritative) | Cached policy endpoint or manual option |

### Key Behaviors

- **Requires `enforcement: true`** — without it, session limits are not checked
- **Requires `sessionId`** — mirrors proxy behavior; no session ID means no session enforcement
- **Fails open on policy error** — if the policy endpoint is unreachable, requests proceed (but manual limits are still enforced)
- **Streaming is async** — cost from a streaming response is accumulated after the stream completes. A concurrent second request may slip through before the first stream's cost is counted
- **Failed responses don't count** — 4xx/5xx responses don't accumulate session spend (consistent with proxy)

## Related

- [Budgets](budgets.md) — overall budget configuration and enforcement
- [Velocity Limits](velocity-limits.md) — spending-rate circuit breaker
- [Event Types](../webhooks/event-types.md) — full webhook payload examples
- [Errors](../api-reference/errors.md) — all error codes and response shapes
- [JavaScript SDK](../sdks/javascript.md) — full SDK reference
- [Claude Agent SDK](../sdks/claude-agent.md) — `budgetSessionId` option
