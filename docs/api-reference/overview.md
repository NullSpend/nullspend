# API Reference Overview

NullSpend exposes a REST API for cost tracking, budget management, and human-in-the-loop workflows. All endpoints live under `https://nullspend.dev/api/`.

---

## Authentication

NullSpend uses two authentication modes depending on the endpoint. For key lifecycle and caching details, see [Authentication](authentication.md).

### API Key (agent/SDK use)

Pass your API key in the `X-NullSpend-Key` header:

```
X-NullSpend-Key: ns_live_sk_...
```

API keys are created in the dashboard and identified by a `ns_key_` prefix. The raw key is shown only once at creation. See [API Keys API](api-keys-api.md) for management endpoints.

### Session (dashboard use)

Session-authenticated endpoints require a browser session cookie from the NullSpend dashboard. These endpoints power the dashboard UI and are not callable from external scripts without a valid session.

### Endpoint Auth Summary

| Endpoint | Method | Auth |
|---|---|---|
| `POST /api/cost-events` | Ingest event | API key |
| `POST /api/cost-events/batch` | Batch ingest | API key |
| `GET /api/cost-events` | List events | Session |
| `GET /api/cost-events/:id` | Get event | Session |
| `GET /api/cost-events/summary` | Analytics | Session |
| `GET /api/budgets/status` | Budget status | API key |
| `GET /api/auth/introspect` | Key introspection | API key |
| `POST /api/actions` | Create action | API key |
| `POST /api/actions/:id/result` | Mark result | API key |
| `GET /api/actions/:id` | Get action | API key or Session |
| `GET /api/actions/:id/costs` | Action costs | API key or Session |
| `GET /api/budgets` | List budgets | Session |
| `POST /api/budgets` | Create/update budget | Session |
| `DELETE /api/budgets/:id` | Delete budget | Session |
| `POST /api/budgets/:id` | Reset budget spend | Session |
| `GET /api/budgets/velocity-status` | Velocity state | Session |
| `GET /api/keys` | List keys | Session |
| `POST /api/keys` | Create key | Session |
| `DELETE /api/keys/:id` | Revoke key | Session |
| `POST /api/actions/:id/approve` | Approve action | Session |
| `POST /api/actions/:id/reject` | Reject action | Session |
| `GET /api/actions` | List actions | Session |
| `GET /api/webhooks` | List webhooks | Session |
| `POST /api/webhooks` | Create webhook | Session |
| `PATCH /api/webhooks/:id` | Update webhook | Session |
| `DELETE /api/webhooks/:id` | Delete webhook | Session |
| `POST /api/webhooks/:id/test` | Test webhook | Session |
| `POST /api/webhooks/:id/rotate-secret` | Rotate secret | Session |
| `GET /api/webhooks/:id/deliveries` | Delivery history | Session |
| `GET /api/margins` | Margin table | Session |
| `GET /api/margins/:customer` | Customer detail | Session |
| `GET /api/margins/unmatched` | Unmatched customers | Session |
| `POST /api/stripe/connect` | Connect Stripe | Session (admin) |
| `DELETE /api/stripe/disconnect` | Disconnect Stripe | Session (admin) |
| `GET /api/stripe/revenue-sync` | Trigger sync | Cron or Session |
| `GET /api/customer-mappings` | List mappings | Session |
| `POST /api/customer-mappings` | Create mapping | Session |
| `DELETE /api/customer-mappings` | Delete mapping | Session |

---

## ID Formats

All NullSpend IDs use the `ns_` prefix followed by a type identifier and a UUID:

| Prefix | Resource |
|---|---|
| `ns_act_` | Action |
| `ns_key_` | API key |
| `ns_evt_` | Cost event |
| `ns_bgt_` | Budget |
| `ns_wh_` | Webhook endpoint |
| `ns_del_` | Webhook delivery |
| `ns_usr_` | User |
| `ns_tc_` | Trace |

---

## Error Format

All errors follow a consistent shape:

```json
{
  "error": {
    "code": "machine_readable_code",
    "message": "Human-readable description.",
    "details": null
  }
}
```

Validation errors include field-level issues in `details`:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "details": {
      "issues": [
        { "path": ["fieldName"], "message": "Expected number, received string" }
      ]
    }
  }
}
```

See [Error Reference](errors.md) for the full error catalog.

---

## Pagination

List endpoints use cursor-based pagination. The response includes a `cursor` field — pass it as the `cursor` query parameter to fetch the next page.

```json
{
  "data": [...],
  "cursor": { "createdAt": "2026-03-15T10:00:00.000Z", "id": "ns_evt_..." }
}
```

- `limit` controls page size (default varies by endpoint, max 100)
- `cursor` is a JSON-encoded object containing `createdAt` and `id`
- When `cursor` is `null`, there are no more results
- Pagination is forward-only, sorted by `createdAt DESC`

---

## Rate Limiting

For enforcement order and failure modes, see [Rate Limits](rate-limits.md).

API-key-authenticated endpoints return rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 97
X-RateLimit-Reset: 1711036800000
```

When the limit is exceeded, the endpoint returns `429 Too Many Requests` with a `Retry-After` header (in seconds).

---

## Idempotency

POST endpoints that accept API key auth support idempotent requests. Pass an `Idempotency-Key` header or include `idempotencyKey` in the request body. Duplicate requests return `200` with the original result instead of creating a new resource.

---

## Versioning

For the version resolution chain and strategy, see [Versioning](versioning.md).

Responses include a `NullSpend-Version` header indicating the API version:

```
NullSpend-Version: 2026-04-01
```
