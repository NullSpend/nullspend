---
title: "Webhooks API"
description: "Create, manage, and test webhook endpoints. All webhook management endpoints require session authentication (dashboard only)."
---

Create, manage, and test webhook endpoints. All webhook management endpoints require session authentication (dashboard only).

See [API Overview](overview.md) for authentication, pagination, errors, and ID formats.

---

## List Webhook Endpoints

`GET /api/webhooks`

Retrieve all webhook endpoints for the current organization.

### Authentication

Session (dashboard)

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.com/api/webhooks \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "url": "https://example.com/webhooks/nullspend",
      "description": "Production webhook",
      "eventTypes": ["cost_event.created", "budget.exceeded"],
      "enabled": true,
      "apiVersion": "2026-04-01",
      "payloadMode": "full",
      "createdAt": "2026-03-01T09:00:00.000Z",
      "updatedAt": "2026-03-15T12:00:00.000Z"
    }
  ]
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |

---

## Create Webhook Endpoint

`POST /api/webhooks`

Register a new webhook endpoint. The signing secret is returned only in this response — store it securely.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `url` | body | string | Yes | HTTPS URL. No private IPs (10.x, 192.168.x, 172.16-31.x, 127.x, 169.254.x, .local). No IPv6 literals. |
| `description` | body | string | No | Human-readable description. Max 200 chars. |
| `eventTypes` | body | string[] | No | Event types to subscribe to. Default `[]` (receives all). |
| `payloadMode` | body | string | No | `"full"` or `"thin"`. Default `"full"`. |

**Valid event types** (15):

`cost_event.created`, `budget.threshold.warning`, `budget.threshold.critical`, `budget.exceeded`, `budget.reset`, `request.blocked`, `action.created`, `action.approved`, `action.rejected`, `action.expired`, `velocity.exceeded`, `velocity.recovered`, `session.limit_exceeded`, `tag_budget.exceeded`, `test.ping`

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.com/api/webhooks \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhooks/nullspend",
    "description": "Production webhook",
    "eventTypes": ["cost_event.created", "budget.exceeded"],
    "payloadMode": "full"
  }'
```

### Response

**201 Created**:

```json
{
  "data": {
    "id": "ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "url": "https://example.com/webhooks/nullspend",
    "description": "Production webhook",
    "eventTypes": ["cost_event.created", "budget.exceeded"],
    "enabled": true,
    "apiVersion": "2026-04-01",
    "payloadMode": "full",
    "createdAt": "2026-03-20T14:30:00.000Z",
    "updatedAt": "2026-03-20T14:30:00.000Z",
    "signingSecret": "whsec_your_signing_secret_will_appear_here_shown_only_once_at_create"
  }
}
```

> **Warning**: `signingSecret` is shown **only at creation**. Format: `whsec_` + 64 hex chars.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid URL, invalid event types, or other field errors |
| `authentication_required` | 401 | No valid session |
| `limit_exceeded` | 409 | Organization has reached the endpoint limit for its tier (Free: 2, Pro: 25, Enterprise: unlimited) |

---

## Update Webhook Endpoint

`PATCH /api/webhooks/:id`

Update one or more fields of a webhook endpoint. At least one field must be provided.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Webhook ID (`ns_wh_*`). |
| `url` | body | string | No | New HTTPS URL. Same validation as create. |
| `description` | body | string \| null | No | New description. Max 200 chars. `null` clears it. |
| `eventTypes` | body | string[] | No | New event type list. |
| `enabled` | body | boolean | No | Enable or disable the endpoint. |
| `payloadMode` | body | string | No | `"full"` or `"thin"`. |

### Request

```bash
# Requires dashboard session cookie
curl -X PATCH https://nullspend.com/api/webhooks/ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Response

**200 OK**:

```json
{
  "data": {
    "id": "ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "url": "https://example.com/webhooks/nullspend",
    "description": "Production webhook",
    "eventTypes": ["cost_event.created", "budget.exceeded"],
    "enabled": false,
    "apiVersion": "2026-04-01",
    "payloadMode": "full",
    "createdAt": "2026-03-01T09:00:00.000Z",
    "updatedAt": "2026-03-20T14:30:00.000Z"
  }
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | No fields provided or invalid field values |
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Webhook not found or not owned by user |

---

## Delete Webhook Endpoint

`DELETE /api/webhooks/:id`

Permanently delete a webhook endpoint and all its delivery history.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Webhook ID (`ns_wh_*`). |

### Request

```bash
# Requires dashboard session cookie
curl -X DELETE https://nullspend.com/api/webhooks/ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "success": true
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Webhook not found or not owned by user |

---

## Send Test Ping

`POST /api/webhooks/:id/test`

Send a `test.ping` event to a webhook endpoint with a proper HMAC signature. Useful for verifying your endpoint is reachable and correctly validates signatures.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Webhook ID (`ns_wh_*`). |

No request body required.

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.com/api/webhooks/ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890/test \
  -H "Cookie: session=..."
```

### Response

**200 OK** (success):

```json
{
  "success": true,
  "statusCode": 200,
  "responsePreview": "OK"
}
```

**200 OK** (endpoint unreachable or error):

```json
{
  "success": false,
  "statusCode": null,
  "responsePreview": "connect ECONNREFUSED 203.0.113.1:443"
}
```

The test request has a 5-second timeout. Headers sent to your endpoint: `Content-Type`, `X-NullSpend-Signature`, `X-NullSpend-Webhook-Id`, `X-NullSpend-Webhook-Timestamp`, `User-Agent: NullSpend-Webhooks/1.0`.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Webhook not found or not owned by user |

---

## Rotate Signing Secret

`POST /api/webhooks/:id/rotate-secret`

Generate a new signing secret. The previous secret remains valid for 24 hours to allow zero-downtime rotation.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Webhook ID (`ns_wh_*`). |

No request body required.

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.com/api/webhooks/ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890/rotate-secret \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": {
    "signingSecret": "whsec_your_new_rotated_secret_will_appear_here_shown_only_at_rotate",
    "secretRotatedAt": "2026-03-20T15:00:00.000Z"
  }
}
```

> **Note**: The old secret is preserved for 24 hours. During that window, NullSpend signs deliveries with both secrets. Update your verification code within 24 hours.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Webhook not found or not owned by user |

---

## List Deliveries

`GET /api/webhooks/:id/deliveries`

Retrieve the most recent 50 deliveries for a webhook endpoint, sorted by creation time (newest first).

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Webhook ID (`ns_wh_*`). |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.com/api/webhooks/ns_wh_a1b2c3d4-e5f6-7890-abcd-ef1234567890/deliveries \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_del_b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "eventType": "cost_event.created",
      "eventId": "ns_evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "delivered",
      "attempts": 1,
      "lastAttemptAt": "2026-03-20T14:30:01.000Z",
      "responseStatus": 200,
      "createdAt": "2026-03-20T14:30:00.000Z"
    },
    {
      "id": "ns_del_c3d4e5f6-a7b8-9012-cdef-123456789012",
      "eventType": "budget.exceeded",
      "eventId": "ns_bgt_aabbccdd-eeff-0011-2233-445566778899",
      "status": "failed",
      "attempts": 3,
      "lastAttemptAt": "2026-03-20T14:35:00.000Z",
      "responseStatus": 500,
      "createdAt": "2026-03-20T14:30:00.000Z"
    }
  ]
}
```

Delivery `status` values: `delivered`, `failed`, `pending`.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Webhook not found or not owned by user |

---

## Related

- [Webhooks Overview](../webhooks/overview.md) — event types, payload formats, security
- [Budgets](../features/budgets.md) — budget events that trigger webhooks
- [Error Reference](errors.md) — full error catalog
