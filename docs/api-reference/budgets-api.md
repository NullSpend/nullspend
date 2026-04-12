# Budgets API

Create, manage, and query spending budgets. Budgets enforce cost limits on users and API keys with optional velocity limits and session limits.

See [API Overview](overview.md) for authentication, pagination, errors, and ID formats.

---

## List Budgets

`GET /api/budgets`

Retrieve all budgets for the current organization, including API key and tag budgets.

### Authentication

Session (dashboard)

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/budgets \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "data": [
    {
      "id": "ns_bgt_aabbccdd-eeff-0011-2233-445566778899",
      "entityType": "user",
      "entityId": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
      "maxBudgetMicrodollars": 10000000,
      "spendMicrodollars": 2450000,
      "policy": "strict_block",
      "resetInterval": "monthly",
      "currentPeriodStart": "2026-03-01T00:00:00.000Z",
      "thresholdPercentages": [50, 80, 90],
      "velocityLimitMicrodollars": 500000,
      "velocityWindowSeconds": 60,
      "velocityCooldownSeconds": 300,
      "sessionLimitMicrodollars": 100000,
      "createdAt": "2026-02-15T10:00:00.000Z",
      "updatedAt": "2026-03-20T14:30:00.000Z"
    },
    {
      "id": "ns_bgt_11223344-5566-7788-99aa-bbccddeeff00",
      "entityType": "api_key",
      "entityId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
      "maxBudgetMicrodollars": 5000000,
      "spendMicrodollars": 800000,
      "policy": "strict_block",
      "resetInterval": null,
      "currentPeriodStart": null,
      "thresholdPercentages": [75, 90],
      "velocityLimitMicrodollars": null,
      "velocityWindowSeconds": null,
      "velocityCooldownSeconds": null,
      "sessionLimitMicrodollars": null,
      "createdAt": "2026-03-01T09:00:00.000Z",
      "updatedAt": "2026-03-10T12:00:00.000Z"
    }
  ]
}
```

Headers: `NullSpend-Version: 2026-04-01`

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |

---

## Create or Update Budget

`POST /api/budgets`

Create a new budget or update an existing one. Upserts on the `(entityType, entityId)` pair — if a budget already exists for that entity, it is updated.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `entityType` | body | string | Yes | `"user"`, `"api_key"`, `"tag"`, or `"customer"`. |
| `entityId` | body | string | Yes | Entity ID. Format depends on entity type: `"user"`: `ns_usr_*` prefixed UUID, `"api_key"`: `ns_key_*` prefixed UUID, `"tag"`: `"key=value"` format (max 321 chars), `"customer"`: plain string (max 256 chars). |
| `maxBudgetMicrodollars` | body | integer | Yes | Spending limit in microdollars. Must be positive. |
| `policy` | body | string | No | Enforcement policy: `"strict_block"`, `"soft_block"`, or `"warn"`. Default `"strict_block"`. |
| `resetInterval` | body | string | No | `"daily"`, `"weekly"`, or `"monthly"`. Omit for no auto-reset. |
| `thresholdPercentages` | body | integer[] | No | Alert thresholds (1–100). Max 10 values, ascending, no duplicates. |
| `velocityLimitMicrodollars` | body | integer \| null | No | Max spend per velocity window. `null` removes the limit. |
| `velocityWindowSeconds` | body | integer | No | Velocity window duration. 10–3600. |
| `velocityCooldownSeconds` | body | integer | No | Cooldown after velocity breach. 10–3600. |
| `sessionLimitMicrodollars` | body | integer \| null | No | Per-session spending limit. `null` removes the limit. |

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.dev/api/budgets \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d '{
    "entityType": "api_key",
    "entityId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
    "maxBudgetMicrodollars": 5000000,
    "resetInterval": "monthly",
    "thresholdPercentages": [50, 80, 90]
  }'
```

### Response

**201 Created**:

```json
{
  "id": "ns_bgt_11223344-5566-7788-99aa-bbccddeeff00",
  "entityType": "api_key",
  "entityId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
  "maxBudgetMicrodollars": 5000000,
  "spendMicrodollars": 0,
  "policy": "strict_block",
  "resetInterval": "monthly",
  "currentPeriodStart": "2026-03-20T14:30:00.000Z",
  "thresholdPercentages": [50, 80, 90],
  "velocityLimitMicrodollars": null,
  "velocityWindowSeconds": null,
  "velocityCooldownSeconds": null,
  "sessionLimitMicrodollars": null,
  "createdAt": "2026-03-20T14:30:00.000Z",
  "updatedAt": "2026-03-20T14:30:00.000Z"
}
```

Side effect: invalidates the proxy cache so enforcement picks up the new budget immediately.

### Errors

| Code | HTTP | When |
|---|---|---|
| `validation_error` | 400 | Invalid fields (e.g., non-positive budget, thresholds not ascending) |
| `spend_cap_exceeded` | 400 | Budget exceeds the tier-based spending cap |
| `forbidden` | 403 | Entity not owned by user, unsupported entity type, or budget count limit for tier |
| `authentication_required` | 401 | No valid session |

---

## Delete Budget

`DELETE /api/budgets/:id`

Permanently delete a budget. Enforcement stops immediately.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Budget ID (`ns_bgt_*`). |

### Request

```bash
# Requires dashboard session cookie
curl -X DELETE https://nullspend.dev/api/budgets/ns_bgt_aabbccdd-eeff-0011-2233-445566778899 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "deleted": true
}
```

Side effect: invalidates the proxy cache.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Budget not found |

---

## Reset Budget Spend

`POST /api/budgets/:id`

Reset a budget's spend counter to zero and start a new period. This is a manual reset — it does not modify the budget's configuration.

### Authentication

Session (dashboard)

### Parameters

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `id` | path | string | Yes | Budget ID (`ns_bgt_*`). |

### Request

```bash
# Requires dashboard session cookie
curl -X POST https://nullspend.dev/api/budgets/ns_bgt_aabbccdd-eeff-0011-2233-445566778899 \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "id": "ns_bgt_aabbccdd-eeff-0011-2233-445566778899",
  "entityType": "user",
  "entityId": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
  "maxBudgetMicrodollars": 10000000,
  "spendMicrodollars": 0,
  "policy": "strict_block",
  "resetInterval": "monthly",
  "currentPeriodStart": "2026-03-20T15:00:00.000Z",
  "thresholdPercentages": [50, 80, 90],
  "velocityLimitMicrodollars": 500000,
  "velocityWindowSeconds": 60,
  "velocityCooldownSeconds": 300,
  "sessionLimitMicrodollars": 100000,
  "createdAt": "2026-02-15T10:00:00.000Z",
  "updatedAt": "2026-03-20T15:00:00.000Z"
}
```

Side effect: invalidates the proxy cache.

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |
| `not_found` | 404 | Budget not found |

---

## Get Budget Status

`GET /api/budgets/status`

Check the live budget status for the authenticated API key's user and key entities. Returns remaining balance computed from the current spend. This is the only budget endpoint callable programmatically (via API key).

### Authentication

API key

### Request

```typescript
const res = await fetch("https://nullspend.dev/api/budgets/status", {
  headers: { "X-NullSpend-Key": "ns_live_sk_abc123..." },
});
```

```python
import requests

resp = requests.get(
    "https://nullspend.dev/api/budgets/status",
    headers={"X-NullSpend-Key": "ns_live_sk_abc123..."},
)
```

```bash
curl https://nullspend.dev/api/budgets/status \
  -H "X-NullSpend-Key: ns_live_sk_abc123..."
```

### Response

**200 OK**:

```json
{
  "entities": [
    {
      "entityType": "user",
      "entityId": "ns_usr_aabbccdd-eeff-0011-2233-445566778899",
      "limitMicrodollars": 10000000,
      "spendMicrodollars": 2450000,
      "remainingMicrodollars": 7550000,
      "policy": "strict_block",
      "resetInterval": "monthly",
      "currentPeriodStart": "2026-03-01T00:00:00.000Z",
      "thresholdPercentages": [50, 80, 90],
      "velocityLimitMicrodollars": 500000,
      "velocityWindowSeconds": 60,
      "velocityCooldownSeconds": 300,
      "sessionLimitMicrodollars": 100000
    },
    {
      "entityType": "api_key",
      "entityId": "ns_key_11223344-5566-7788-99aa-bbccddeeff00",
      "limitMicrodollars": 5000000,
      "spendMicrodollars": 800000,
      "remainingMicrodollars": 4200000,
      "policy": "strict_block",
      "resetInterval": null,
      "currentPeriodStart": null,
      "thresholdPercentages": [75, 90],
      "velocityLimitMicrodollars": null,
      "velocityWindowSeconds": null,
      "velocityCooldownSeconds": null,
      "sessionLimitMicrodollars": null
    }
  ]
}
```

`remainingMicrodollars` is always >= 0 (clamped, never negative).

> **Note:** The API response includes additional budget configuration fields (e.g., `policy`, `resetInterval`, `thresholdPercentages`, velocity and session limit settings) beyond what the SDK's `BudgetEntity` type surfaces.

Headers: `NullSpend-Version: 2026-04-01`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Per-key rate limit exceeded |

---

## Get Velocity State

`GET /api/budgets/velocity-status`

Poll the live velocity state from the proxy worker. Used by the dashboard to show real-time velocity limit status.

### Authentication

Session (dashboard)

### Request

```bash
# Requires dashboard session cookie
curl https://nullspend.dev/api/budgets/velocity-status \
  -H "Cookie: session=..."
```

### Response

**200 OK**:

```json
{
  "velocityState": []
}
```

This endpoint has a 3-second timeout and degrades gracefully — it returns `{ "velocityState": [] }` on any error (proxy unreachable, timeout, non-2xx response, local dev).

### Errors

| Code | HTTP | When |
|---|---|---|
| `authentication_required` | 401 | No valid session |

---

## Related

- [Budgets](../features/budgets.md) — feature overview (velocity limits, session limits, thresholds)
- [Budget Configuration](../guides/budget-configuration.md) — setup guide
- [API Keys API](api-keys-api.md) — manage the keys that budgets enforce on
- [Error Reference](errors.md) — full error catalog
