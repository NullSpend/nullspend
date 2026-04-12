# Audit Log API

Query the audit trail of administrative actions in your organization. Every sensitive operation (key creation, budget changes, webhook updates, etc.) is recorded as an audit event.

## List audit events

```
GET /api/audit-log
```

Returns audit events for your organization, newest first. Cursor-paginated.

**Auth:** Session only (admin role).

### Query parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | integer | 50 | Results per page (1–100) |
| `cursor` | string | — | JSON cursor from a previous response for pagination |
| `action` | string | — | Filter to a specific action type (e.g., `"api_key.created"`) |

### Response

```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "actorId": "usr_...",
      "action": "api_key.created",
      "resourceType": "api_key",
      "resourceId": "key_...",
      "metadata": { "name": "Production Key" },
      "createdAt": "2026-04-10T14:30:00.000Z"
    }
  ],
  "cursor": {
    "createdAt": "2026-04-10T14:30:00.000Z",
    "id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

When `cursor` is `null`, there are no more pages.

### Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Event UUID |
| `actorId` | string | User who performed the action |
| `action` | string | Action type (see below) |
| `resourceType` | string | Type of resource affected |
| `resourceId` | string | ID of the affected resource |
| `metadata` | object | Action-specific details |
| `createdAt` | string | ISO 8601 timestamp |

### Pagination

Pass the `cursor` object from the response as a JSON string in the next request:

```bash
curl "https://www.nullspend.dev/api/audit-log?cursor=%7B%22createdAt%22%3A%222026-04-10T14%3A30%3A00.000Z%22%2C%22id%22%3A%22550e8400-...%22%7D"
```

### Filtering by action

```bash
curl "https://www.nullspend.dev/api/audit-log?action=budget.updated"
```

## Action types

Audit events are recorded for administrative operations including:

- `api_key.created`, `api_key.revoked`
- `budget.created`, `budget.updated`, `budget.deleted`
- `webhook.created`, `webhook.updated`, `webhook.deleted`
- `slack.configured`, `slack.deleted`
- `action.approved`, `action.rejected`
- `budget_increase.approved`, `budget_increase.rejected`

The exact set of action types grows as new features ship. Use the `action` filter parameter to query specific types.

## Related

- [Authentication](authentication.md) — session auth details
- [Errors](errors.md) — standard error format
