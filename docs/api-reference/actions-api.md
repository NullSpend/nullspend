# Actions API

Create, approve, reject, and track human-in-the-loop (HITL) actions. Actions let agents request human approval before executing sensitive operations.

See [API Overview](overview.md) for authentication, pagination, errors, and ID formats.

---

## List Actions

`GET /api/actions`

Retrieve actions for the authenticated user with optional status filtering.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `status` | query | string | No | Filter by single status. |
| `statuses` | query | string | No | Comma-separated status list (e.g., `"pending,approved"`). |
| `limit` | query | integer | No | Page size. 1–100, default 50. |
| `cursor` | query | string | No | JSON-encoded cursor from a previous response. |

**Valid statuses**: `pending`, `approved`, `rejected`, `expired`, `executing`, `executed`, `failed`

### Request

```bash
# Requires dashboard session cookie
curl "https://nullspend.com/api/actions?statuses=pending,approved&limit=20" \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "agentId": "support-bot",
      "actionType": "send_email",
      "status": "pending",
      "payload": {
        "to": "customer@example.com",
        "subject": "Your refund has been processed"
      },
      "metadata": { "ticketId": "T-1234" },
      "createdAt": "2026-03-20T14:30:00.000Z",
      "approvedAt": null,
      "rejectedAt": null,
      "executedAt": null,
      "expiresAt": "2026-03-21T14:30:00.000Z",
      "expiredAt": null,
      "approvedBy": null,
      "rejectedBy": null,
      "result": null,
      "errorMessage": null,
      "environment": null,
      "sourceFramework": null
    }
  ],
  "cursor": null
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid status, limit, or cursor |
| `authentication_required` | 401 | No valid session |

---

## Create Action

`POST /api/actions`

Request human approval for a sensitive operation. The action starts in `pending` status and must be approved or rejected from the dashboard.

### Authentication

API key

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `agentId` | body | string | Yes | Agent identifier. 1–255 chars. |
| `actionType` | body | string | Yes | One of: `send_email`, `http_post`, `http_delete`, `shell_command`, `db_write`, `file_write`, `file_delete`. |
| `payload` | body | object | Yes | Action payload. Max 64 KB serialized, max 20 nesting levels. |
| `metadata` | body | object | No | Additional metadata. Max 16 KB serialized, max 20 nesting levels. |
| `expiresInSeconds` | body | integer | No | Seconds until the action expires. 0–2,592,000 (30 days). `null` for no expiry. |
| `Idempotency-Key` | header | string | No | Deduplication key for idempotent retries. |

### Request

```typescript
const res = await fetch("https://nullspend.com/api/actions", {
  method: "POST",
  headers: {
    "X-NullSpend-Key": "ns_live_sk_abc123...",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    agentId: "support-bot",
    actionType: "send_email",
    payload: {
      to: "customer@example.com",
      subject: "Your refund has been processed",
      body: "We've processed your $50 refund...",
    },
    metadata: { ticketId: "T-1234" },
    expiresInSeconds: 86400,
  }),
});
```

```python
import requests

resp = requests.post(
    "https://nullspend.com/api/actions",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
    json={
        "agentId": "support-bot",
        "actionType": "send_email",
        "payload": {
            "to": "customer@example.com",
            "subject": "Your refund has been processed",
            "body": "We've processed your $50 refund...",
        },
        "metadata": {"ticketId": "T-1234"},
        "expiresInSeconds": 86400,
    },
)
```

```bash
curl -X POST https://nullspend.com/api/actions \
  -H "X-NullSpend-Key: ns_live_sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "agentId": "support-bot",
    "actionType": "send_email",
    "payload": {"to":"customer@example.com","subject":"Your refund has been processed"},
    "expiresInSeconds": 86400
  }'
```

### Response

**201 Created**:

```json
{
  "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "expiresAt": "2026-03-21T14:30:00.000Z"
}
```

Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

Side effect: sends a Slack notification (if configured) to alert the human approver.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid action type, payload too large, nesting too deep |
| `invalid_json` | 400 | Malformed JSON body |
| `unsupported_media_type` | 415 | Content-Type is not `application/json` |
| `payload_too_large` | 413 | Body exceeds 1 MB |
| `authentication_required` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## Get Action

`GET /api/actions/:id`

Retrieve a single action by ID. Agents can poll this endpoint to check if their action has been approved.

### Authentication

Dual (API key or session)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Action ID (`ns_act_*`). |

### Request

```typescript
// Agent polling for approval
const res = await fetch(
  "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  { headers: { "X-NullSpend-Key": "ns_live_sk_abc123..." } }
);
```

```python
import requests

resp = requests.get(
    "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
)
```

```bash
curl https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-NullSpend-Key: ns_live_sk_abc123..."
```

### Response

**200 OK**:

```json
{
  "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "agentId": "support-bot",
  "actionType": "send_email",
  "status": "approved",
  "payload": {
    "to": "customer@example.com",
    "subject": "Your refund has been processed"
  },
  "metadata": { "ticketId": "T-1234" },
  "createdAt": "2026-03-20T14:30:00.000Z",
  "approvedAt": "2026-03-20T14:35:00.000Z",
  "rejectedAt": null,
  "executedAt": null,
  "expiresAt": "2026-03-21T14:30:00.000Z",
  "expiredAt": null,
  "approvedBy": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
  "rejectedBy": null,
  "result": null,
  "errorMessage": null,
  "environment": null,
  "sourceFramework": null
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | Invalid API key or no session |
| `not_found` | 404 | Action not found or not owned by user |
| `rate_limit_exceeded` | 429 | Per-key rate limit (API key auth only) |

---

## Approve Action

`POST /api/actions/:id/approve`

Approve a pending action. Only a human in the dashboard can approve — this endpoint requires session auth.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Action ID (`ns_act_*`). |

No request body required.

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/approve \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "approved",
  "approvedAt": "2026-03-20T14:35:00.000Z"
}
```

### State Transitions

Only valid from `pending` status. The action must not have expired.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Action not found or not owned by user |
| `invalid_action_transition` | 409 | Action not in `pending` status |
| `action_expired` | 409 | Action has expired |
| `stale_action` | 409 | Concurrent modification detected |

---

## Reject Action

`POST /api/actions/:id/reject`

Reject a pending action. Only a human in the dashboard can reject — this endpoint requires session auth.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Action ID (`ns_act_*`). |

No request body required.

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/reject \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "rejected",
  "rejectedAt": "2026-03-20T14:35:00.000Z"
}
```

### State Transitions

Only valid from `pending` status. The action must not have expired.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Action not found or not owned by user |
| `invalid_action_transition` | 409 | Action not in `pending` status |
| `action_expired` | 409 | Action has expired |
| `stale_action` | 409 | Concurrent modification detected |

---

## Mark Action Result

`POST /api/actions/:id/result`

Report the outcome of an approved action. Called by the agent after executing (or failing to execute) the action.

### Authentication

API key

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Action ID (`ns_act_*`). |
| `status` | body | string | Yes | `"executing"`, `"executed"`, or `"failed"`. |
| `result` | body | object | No | Execution result. Max 64 KB, max 20 nesting levels. **Forbidden** when `status` is `"executing"`. |
| `errorMessage` | body | string | No | Error description. Max 4,000 chars. **Required** when `status` is `"failed"`. **Forbidden** when `status` is `"executing"` or `"executed"`. |
| `Idempotency-Key` | header | string | No | Deduplication key for idempotent retries. |

### Request

```typescript
// Mark as executing
await fetch(
  "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/result",
  {
    method: "POST",
    headers: {
      "X-NullSpend-Key": "ns_live_sk_abc123...",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ status: "executing" }),
  }
);

// Mark as executed with result
await fetch(
  "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/result",
  {
    method: "POST",
    headers: {
      "X-NullSpend-Key": "ns_live_sk_abc123...",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      status: "executed",
      result: { emailId: "msg_abc123", sentAt: "2026-03-20T14:36:00.000Z" },
    }),
  }
);
```

```python
import requests

# Mark as failed
resp = requests.post(
    "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/result",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
    json={
        "status": "failed",
        "errorMessage": "SMTP connection refused: relay.example.com:587",
    },
)
```

```bash
curl -X POST https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/result \
  -H "X-NullSpend-Key: ns_live_sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"status": "executed", "result": {"emailId": "msg_abc123"}}'
```

### Response

**200 OK**:

```json
{
  "id": "ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "executed",
  "executedAt": "2026-03-20T14:36:00.000Z"
}
```

Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### State Transitions

```
pending → approved → executing → executed
                  ↘            ↘ failed
                   → executed
                   → failed
```

- `approved` → `executing`, `executed`, or `failed`
- `executing` → `executed` or `failed`
- Terminal states (`executed`, `failed`, `rejected`, `expired`) cannot transition further.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid status, missing errorMessage for failed, result set for executing |
| `invalid_json` | 400 | Malformed JSON body |
| `unsupported_media_type` | 415 | Content-Type is not `application/json` |
| `payload_too_large` | 413 | Body exceeds 1 MB |
| `authentication_required` | 401 | Missing or invalid API key |
| `not_found` | 404 | Action not found or not owned by user |
| `invalid_action_transition` | 409 | Action in a terminal state |
| `stale_action` | 409 | Concurrent modification detected |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## Get Action Costs

`GET /api/actions/:id/costs`

Retrieve cost events associated with an action.

### Authentication

Dual (API key or session)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Action ID (`ns_act_*`). |

### Request

```typescript
const res = await fetch(
  "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/costs",
  { headers: { "X-NullSpend-Key": "ns_live_sk_abc123..." } }
);
```

```python
import requests

resp = requests.get(
    "https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/costs",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
)
```

```bash
curl https://nullspend.com/api/actions/ns_act_a1b2c3d4-e5f6-7890-abcd-ef1234567890/costs \
  -H "X-NullSpend-Key: ns_live_sk_abc123..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_evt_b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "provider": "openai",
      "model": "gpt-4o",
      "inputTokens": 1500,
      "outputTokens": 500,
      "cachedInputTokens": 0,
      "reasoningTokens": 0,
      "costMicrodollars": 6750,
      "tags": { "agent": "support-bot" },
      "createdAt": "2026-03-20T14:35:30.000Z"
    }
  ]
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | Invalid API key or no session |
| `not_found` | 404 | Action not found or not owned by user |
| `rate_limit_exceeded` | 429 | Per-key rate limit (API key auth only) |

---

## Related

- [Cost Events API](cost-events-api.md) — cost event schema and ingest
- [Budgets API](budgets-api.md) — spending limits that interact with action costs
- [Error Reference](errors.md) — full error catalog
