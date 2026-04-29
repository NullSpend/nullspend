---
title: "Audit Log API"
description: "API reference for querying organization audit events."
---

Query the audit trail of administrative actions in your organization. The audit log records org membership and ownership changes that affect access — the kinds of events you'd need for a compliance review or post-incident investigation.

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
curl "https://nullspend.dev/api/audit-log?cursor=%7B%22createdAt%22%3A%222026-04-10T14%3A30%3A00.000Z%22%2C%22id%22%3A%22550e8400-...%22%7D"
```

### Filtering by action

```bash
curl "https://nullspend.dev/api/audit-log?action=budget.updated"
```

## Action types

The following action codes are currently emitted. The set grows as new features ship — always treat the `action` field as a free-form string in your consumers.

### Organizations

- `org.created` — user created a new organization
- `org.deleted` — owner deleted the organization
- `org.ownership_transferred` — ownership transferred to another member (`metadata.newOwnerUserId`)
- `org_upgrade_url.updated` — org-level upgrade URL changed
- `customer_upgrade_url.updated` — per-customer upgrade URL changed

### Members & invitations

- `invitation.created` — admin invited a new member (`metadata.email`, `metadata.role`)
- `invitation.accepted` — invitee accepted the invitation
- `invitation.revoked` — admin revoked a pending invitation
- `member.role_changed` — admin changed a member's role (`metadata.newRole`)
- `member.removed` — admin removed a member from the org
- `member.left` — member left the org voluntarily

### What is NOT audited (today)

Key, budget, webhook, and HITL action mutations are not currently written to the audit log. They appear in their own surfaces (`/api/keys`, `/api/budgets`, `/api/webhooks`, `/api/actions`) but you cannot query them through `/api/audit-log`. If you need that coverage today, log them on your side via the SDK — first-party audit support is on the roadmap.

Use the `action` filter parameter to query specific types.

## Related

- [Authentication](/docs/api-reference/authentication) — session auth details
- [Errors](/docs/api-reference/errors) — standard error format
