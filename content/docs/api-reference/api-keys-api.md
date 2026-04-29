---
title: "API Keys API"
description: "Create, list, and revoke API keys. Keys authenticate agent and SDK requests to NullSpend."
---

Create, list, and revoke API keys. Keys authenticate agent and SDK requests to NullSpend.

See [API Overview](overview.md) for authentication, pagination, errors, and ID formats.

---

## List API Keys

`GET /api/keys`

Retrieve all active (non-revoked) API keys for the current organization.

### Authentication

Session (viewer role)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `limit` | query | integer | No | Page size. 1–100, default 50. |
| `cursor` | query | string | No | JSON-encoded cursor from a previous response. |

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/keys?limit=10 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
      "name": "production-key",
      "keyPrefix": "ns_live_",
      "defaultTags": {},
      "allowedModels": null,
      "allowedProviders": null,
      "allowedCustomers": null,
      "requireCustomerId": false,
      "lastUsedAt": "2026-03-20T14:30:00.000Z",
      "createdAt": "2026-03-01T09:00:00.000Z"
    }
  ],
  "cursor": null
}
```

`allowedModels`, `allowedProviders`, `allowedCustomers` are `null` when no restriction is set; an empty array means "none allowed". `requireCustomerId` enforces that every proxy request must include the `X-NullSpend-Customer` header (or `tags.customer`).

Headers: `NullSpend-Version: 2026-04-01`

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid query parameters |
| `authentication_required` | 401 | No valid session |

---

## Create API Key

`POST /api/keys`

Generate a new API key. The raw key is returned only in this response — store it securely.

### Authentication

Session (member role)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `name` | body | string | Yes | Human-readable name. 1–50 chars, trimmed. |
| `defaultTags` | body | object | No | Default tags merged into every request made with this key. Max 10 keys (`[a-zA-Z0-9_-]+`, max 64 chars), values max 256 chars. Keys starting with `_ns_` are reserved. |
| `allowedModels` | body | string[] \| null | No | Restrict this key to a specific model allow-list (max 50 entries). `null` = no restriction. Empty array means "none allowed". |
| `allowedProviders` | body | string[] \| null | No | Restrict to specific providers — currently `"openai"` and `"anthropic"`. |
| `allowedCustomers` | body | string[] \| null | No | Restrict to specific customer IDs (max 100 entries; `[a-zA-Z0-9._:-]+`, max 256 chars each). |
| `requireCustomerId` | body | boolean | No | When `true`, requests without `X-NullSpend-Customer` (or `tags.customer`) are rejected with `mandate_violation`. Default `false`. |rved. Defaults to `{}`. |

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.dev/api/keys \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{"name": "production-key", "defaultTags": {"team": "billing"}}'
```

### Response

**201 Created**:

```json
{
  "data": {
    "id": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "name": "production-key",
    "keyPrefix": "ns_live_",
    "defaultTags": { "team": "billing" },
    "allowedModels": null,
    "allowedProviders": null,
    "allowedCustomers": null,
    "requireCustomerId": false,
    "rawKey": "ns_live_sk_a1b2c3d4e5f67890...",
    "createdAt": "2026-03-20T14:30:00.000Z"
  }
}
```

> **Warning**: `data.rawKey` is the full API key. It is shown **only once** at creation and cannot be retrieved again.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Name missing or too long |
| `authentication_required` | 401 | No valid session |
| `limit_exceeded` | 409 | Organization has reached the key limit for its tier (Free: 10, Pro+: unlimited) |

---

## Revoke API Key

`DELETE /api/keys/:id`

Revoke an API key. The key stops working immediately (soft delete — sets `revokedAt`).

### Authentication

Session (admin role)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Key ID (`ns_key_*`). |

### Request

```bash
# Requires dashboard session cookie
curl -X DELETE https://nullspend.dev/api/keys/ns_key_11223344-5566-7788-99aa-bbccddeeff00 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": {
    "id": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "revokedAt": "2026-03-20T15:00:00.000Z"
  }
}
```

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Key not found or already revoked |

---

## Update API Key

`PATCH /api/keys/:id`

Update a key's name, default tags, or policy restrictions. At least one field is required.

### Authentication

Session (admin role)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Key ID (`ns_key_*`). |
| `name` | body | string | No | New name. 1–50 chars, trimmed. |
| `defaultTags` | body | object | No | New default tags. Max 10 keys, same validation as `X-NullSpend-Tags`. |
| `allowedModels` | body | string[] \| null | No | Replace the model allow-list. `null` clears the restriction. |
| `allowedProviders` | body | string[] \| null | No | Replace the provider allow-list. `null` clears the restriction. |
| `allowedCustomers` | body | string[] \| null | No | Replace the customer allow-list. `null` clears the restriction. |
| `requireCustomerId` | body | boolean | No | Toggle the require-customer-id mandate. |

### Request

```bash
# Requires dashboard session cookie
curl -X PATCH https://nullspend.dev/api/keys/ns_key_11223344-5566-7788-99aa-bbccddeeff00 \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{"name": "renamed-key", "defaultTags": {"team": "billing"}}'
```

### Response

**200 OK**:

```json
{
  "data": {
    "id": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "name": "renamed-key",
    "keyPrefix": "ns_live_",
    "defaultTags": { "team": "billing" },
    "allowedModels": null,
    "allowedProviders": null,
    "allowedCustomers": null,
    "requireCustomerId": false,
    "lastUsedAt": "2026-03-20T14:30:00.000Z",
    "createdAt": "2026-03-01T09:00:00.000Z"
  }
}
```

Updating `defaultTags` invalidates the proxy's auth cache so the new tags take effect immediately.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | No fields provided, name too long, or invalid tags |
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Key not found or already revoked |

---

## Introspect Identity

`GET /api/auth/introspect`

Check which user and key an API key resolves to. Useful for agents to verify their own identity.

### Authentication

API key

### Request

```typescript
const res = await fetch("https://nullspend.dev/api/auth/introspect", {
  headers: { "X-NullSpend-Key": "ns_live_sk_abc123..." },
});
```

```python
import requests

resp = requests.get(
    "https://nullspend.dev/api/auth/introspect",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
)
```

```bash
curl https://nullspend.dev/api/auth/introspect \
  -H "X-NullSpend-Key: ns_live_sk_abc123..."
```

### Response

**200 OK**:

```json
{
  "userId": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
  "keyId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00"
}
```

In dev mode (local development):

```json
{
  "userId": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
  "keyId": "dev",
  "dev": true
}
```

Headers: `NullSpend-Version: 2026-04-01`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## Related

- [Custom Headers](custom-headers.md) — `X-NullSpend-Key` and other headers
- [Budgets API](budgets-api.md) — set spending limits per key
- [Error Reference](errors.md) — full error catalog
