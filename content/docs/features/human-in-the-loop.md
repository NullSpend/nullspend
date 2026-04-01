---
title: "Human-in-the-Loop Approvals"
description: "Agents pause before sensitive operations and wait for human approval. NullSpend provides the coordination layer — your agent proposes an action, a human appro"
---

Agents pause before sensitive operations and wait for human approval. NullSpend provides the coordination layer — your agent proposes an action, a human approves or rejects it, and the agent proceeds or stops.

## How It Works

```
Agent                    NullSpend                  Human
  │                         │                         │
  ├─ POST /api/actions ────►│                         │
  │                         ├─ Slack notification ───►│
  │                         │                         │
  │   (polls GET /api/      │    Reviews in dashboard │
  │    actions/:id)         │                         │
  │◄────────────────────────┤◄── Approves ────────────┤
  │                         │                         │
  ├─ Executes action        │                         │
  │                         │                         │
  ├─ POST result ──────────►│                         │
  │                         │                         │
```

NullSpend does not execute the action itself. It manages the approval workflow — state transitions, polling, expiration, and notifications. Your agent is responsible for performing the actual operation after receiving approval.

## State Machine

```
pending ──► approved ──► executing ──► executed (terminal)
  │              │             │
  │              │             └──────► failed   (terminal)
  ├──► rejected (terminal)
  └──► expired  (terminal)
```

| From | To | Triggered By |
|---|---|---|
| `pending` | `approved` | Human approves in dashboard |
| `pending` | `rejected` | Human rejects in dashboard |
| `pending` | `expired` | TTL elapses |
| `approved` | `executing` | Agent calls `markResult({ status: "executing" })` |
| `executing` | `executed` | Agent calls `markResult({ status: "executed", result })` |
| `executing` | `failed` | Agent calls `markResult({ status: "failed", errorMessage })` |

Terminal states (`rejected`, `expired`, `executed`, `failed`) cannot transition further. Attempting an invalid transition returns `409` with `error.code: "invalid_action_transition"`.

## Action Types

| Type | Description |
|---|---|
| `send_email` | Send an email |
| `http_post` | Make an HTTP POST request |
| `http_delete` | Make an HTTP DELETE request |
| `shell_command` | Execute a shell command |
| `db_write` | Write to a database |
| `file_write` | Write to a file |
| `file_delete` | Delete a file |

Action types are informational labels — NullSpend does not enforce or validate what the agent actually does after approval. Only the values listed above are accepted; arbitrary strings are rejected by validation.

## Quick Start with the SDK

The `proposeAndWait` method handles the full lifecycle: create action, poll for decision, execute on approval, report result. See the [JavaScript SDK reference](../sdks/javascript.md) for full method signatures and configuration options.

```typescript
import { NullSpend, RejectedError, TimeoutError } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://nullspend.com",
  apiKey: "ns_live_sk_...",
});

try {
  const result = await ns.proposeAndWait({
    agentId: "support-agent",
    actionType: "send_email",
    payload: {
      to: "customer@example.com",
      subject: "Refund Confirmation",
      body: "Your refund of $49.99 has been processed.",
    },
    metadata: {
      ticketId: "TICKET-1234",
      refundAmount: 4999,
    },
    expiresInSeconds: 1800, // 30 minutes

    execute: async ({ actionId }) => {
      // This runs only after human approval.
      // Use actionId as X-NullSpend-Action-Id header to correlate costs.
      const response = await sendEmail({
        to: "customer@example.com",
        subject: "Refund Confirmation",
        body: "Your refund of $49.99 has been processed.",
      });
      return { messageId: response.id };
    },
  });

  console.log("Email sent:", result.messageId);
} catch (err) {
  if (err instanceof RejectedError) {
    console.log(`Action ${err.actionId} was ${err.actionStatus}`);
  } else if (err instanceof TimeoutError) {
    console.log("No decision within timeout");
  } else {
    throw err;
  }
}
```

**Python equivalent** (raw HTTP):

```python
import time
import requests

BASE = "https://nullspend.com"
HEADERS = {"X-NullSpend-Key": "ns_live_sk_..."}

# 1. Create the action
action = requests.post(f"{BASE}/api/actions", headers=HEADERS, json={
    "agentId": "support-agent",
    "actionType": "send_email",
    "payload": {"to": "customer@example.com", "subject": "Refund Confirmation"},
    "expiresInSeconds": 1800,
}).json()

action_id = action["id"]

# 2. Poll for decision
while True:
    status = requests.get(f"{BASE}/api/actions/{action_id}", headers=HEADERS).json()
    if status["status"] != "pending":
        break
    time.sleep(2)

# 3. Execute if approved
if status["status"] == "approved":
    requests.post(f"{BASE}/api/actions/{action_id}/result", headers=HEADERS,
                  json={"status": "executing"})
    try:
        result = send_email(...)
        requests.post(f"{BASE}/api/actions/{action_id}/result", headers=HEADERS,
                      json={"status": "executed", "result": {"messageId": result.id}})
    except Exception as e:
        requests.post(f"{BASE}/api/actions/{action_id}/result", headers=HEADERS,
                      json={"status": "failed", "errorMessage": str(e)})
```

## Step-by-Step (Low-Level)

If you need more control than `proposeAndWait`, use the low-level methods directly.

### 1. Create an action

```typescript
const { id, expiresAt } = await ns.createAction({
  agentId: "data-pipeline",
  actionType: "db_write",
  payload: { query: "DELETE FROM users WHERE inactive_days > 365" },
  metadata: { estimatedRows: 1200 },
  expiresInSeconds: 3600,
});
```

### 2. Wait for a decision

```typescript
const decision = await ns.waitForDecision(id, {
  pollIntervalMs: 2000,  // default: 2000 (2 seconds)
  timeoutMs: 300000,     // default: 300000 (5 minutes)
  onPoll: (action) => console.log(`Status: ${action.status}`),
});
```

The SDK polls `GET /api/actions/:id` every `pollIntervalMs` until the status leaves `pending` or the timeout elapses.

### 3. Execute and report

```typescript
if (decision.status === "approved") {
  await ns.markResult(id, { status: "executing" });

  try {
    const result = await performDatabaseWrite();
    await ns.markResult(id, { status: "executed", result: { rowsDeleted: 1200 } });
  } catch (err) {
    await ns.markResult(id, { status: "failed", errorMessage: err.message });
  }
}
```

See the [Actions API](../api-reference/actions-api.md) for full request/response schemas.

## Expiration

Actions expire automatically if no decision is made within the TTL.

| `expiresInSeconds` Value | Behavior |
|---|---|
| Omitted / `undefined` | Default: 3600 seconds (1 hour) |
| `0` or `null` | No expiration — action stays pending indefinitely |
| Positive number | Expires in that many seconds from creation |

Maximum expiration is 30 days (2,592,000 seconds).

When an action expires, its status transitions to `expired` and an `action.expired` webhook is fired. The SDK's `waitForDecision` resolves with the expired action (it doesn't throw — check `action.status`). `proposeAndWait` throws a `RejectedError` with `actionStatus: "expired"`.

## Notifications

### Slack

If Slack is configured (Settings → Slack), NullSpend sends a notification to your channel when an action is created. This is fire-and-forget — Slack delivery failures don't affect the action.

### Webhooks

Four webhook event types cover the action lifecycle:

| Event | Fires When |
|---|---|
| `action.created` | Action is created (status: `pending`) |
| `action.approved` | Human approves the action |
| `action.rejected` | Human rejects the action |
| `action.expired` | TTL elapses without a decision |

See [Event Types](../webhooks/event-types.md) for full payload examples.

## Dashboard

The Actions page in the dashboard shows all actions with their current status. You can:

- Filter by status (`pending`, `approved`, `rejected`, `expired`, `executing`, `executed`, `failed`)
- Approve or reject pending actions with one click
- View the action payload, metadata, and result
- See associated cost events (when the agent sends `X-NullSpend-Action-Id` with subsequent requests)

## Error Handling

### SDK Errors

| Error | When |
|---|---|
| `RejectedError` | `proposeAndWait`: action was rejected or expired. Check `err.actionStatus`. |
| `TimeoutError` | `waitForDecision` or `proposeAndWait`: no decision within `timeoutMs`. |
| `NullSpendError` | Network errors, invalid responses, or API errors. Check `err.statusCode` and `err.code`. |

### API Error Codes

| Code | Status | Meaning |
|---|---|---|
| `invalid_action_transition` | 409 | Invalid state transition (e.g., approving an already-rejected action) |
| `action_expired` | 409 | Action has expired |
| `stale_action` | 409 | Concurrent modification detected |
| `not_found` | 404 | Action doesn't exist or belongs to another user |

## Best Practices

- **Set an expiration.** Pending actions that never resolve waste attention. Default is 1 hour; use shorter TTLs for time-sensitive operations.
- **Use `metadata` for context.** Include enough information for the reviewer to make a decision without leaving the dashboard — ticket IDs, affected records, estimated impact.
- **Handle rejection gracefully.** Your agent should have a fallback path when an action is rejected — not just crash.
- **Correlate costs with `X-NullSpend-Action-Id`.** After approval, send the action ID as a header on subsequent proxy requests to link cost events to the approved action.
- **Use `onPoll` for logging.** Track how long your agent waits and whether decisions are taking longer than expected.

## Related

- [Actions API](../api-reference/actions-api.md) — full endpoint reference for creating, polling, approving, rejecting, and reporting results
- [Webhook Event Types](../webhooks/event-types.md) — `action.created`, `action.approved`, `action.rejected`, `action.expired` payloads
- [Budgets](budgets.md) — budget enforcement that might trigger HITL workflows for high-cost operations
- [JavaScript SDK](../sdks/javascript.md) — full `NullSpend` client reference with `proposeAndWait`, `waitForDecision`, and error classes
- [MCP Server](../sdks/mcp-server.md) — MCP server exposing `propose_action` and `check_action` tools
- [MCP Proxy](../sdks/mcp-proxy.md) — MCP proxy that gates upstream tool calls through approval
