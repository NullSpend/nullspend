---
title: "Customer Mappings API"
description: "API reference for managing Stripe-to-NullSpend customer mappings for margin tracking."
---

Manage mappings between Stripe customers and NullSpend cost attribution tags. These mappings power the [Margins](/docs/features/margins) feature by linking Stripe revenue to NullSpend cost data.

## List mappings

```
GET /api/customer-mappings
```

Returns all customer mappings for your organization.

**Auth:** Session (viewer role).

### Response

```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "orgId": "org_...",
      "stripeCustomerId": "cus_abc123",
      "tagKey": "customer",
      "tagValue": "acme-corp",
      "matchType": "manual",
      "confidence": 1.0,
      "createdAt": "2026-04-10T12:00:00.000Z"
    }
  ]
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Mapping UUID |
| `stripeCustomerId` | string | Stripe customer ID (e.g., `cus_abc123`) |
| `tagKey` | string | NullSpend tag key (default: `"customer"`) |
| `tagValue` | string | NullSpend tag value to match against cost events |
| `matchType` | string | `"manual"` (user-created) or `"auto"` (system-suggested) |
| `confidence` | number or null | Match confidence (1.0 for manual, variable for auto) |
| `createdAt` | string | ISO 8601 timestamp |

## Create mapping

```
POST /api/customer-mappings
```

Create or update a customer mapping. If a mapping already exists for the same `(orgId, stripeCustomerId, tagKey)`, the `tagValue` and `matchType` are updated.

**Auth:** Session (admin role).

### Request body

```json
{
  "stripeCustomerId": "cus_abc123",
  "tagValue": "acme-corp",
  "tagKey": "customer",
  "matchType": "manual"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `stripeCustomerId` | string | Yes | Stripe customer ID |
| `tagValue` | string | Yes | NullSpend tag value |
| `tagKey` | string | No | Tag key (defaults to `"customer"`) |
| `matchType` | string | No | `"manual"` (default) or `"auto"` |

Both `stripeCustomerId` and `tagValue` must be under 255 characters with no control characters.

### Response

Returns the created mapping with status `201`.

## Delete mapping

```
DELETE /api/customer-mappings?id=<uuid>
```

Delete a customer mapping by its UUID.

**Auth:** Session (admin role).

### Query parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Mapping UUID |

### Response

```json
{
  "data": { "deleted": true }
}
```

Returns `404` if the mapping doesn't exist or doesn't belong to your organization.

## How mappings work

Customer mappings connect two data sources:

1. **Stripe revenue** — invoices from your Stripe account, attributed to Stripe customer IDs
2. **NullSpend costs** — cost events tagged with `X-NullSpend-Tags: customer=<value>`

When you create a mapping like `cus_abc123 → acme-corp`, NullSpend links all Stripe invoices for `cus_abc123` to all cost events tagged `customer=acme-corp`. This lets the [Margins](/docs/features/margins) page show revenue vs cost and compute margin per customer.

NullSpend can also auto-detect mappings when Stripe customer metadata contains NullSpend tag values. Use the [Unmatched Customers](/docs/api-reference/margins-api) endpoint to see which Stripe customers don't have mappings yet.

## Related

- [Margins](/docs/features/margins) — margin tracking overview
- [Margins API](/docs/api-reference/margins-api) — margin data and unmatched customers
- [Cost Attribution](/docs/features/cost-attribution) — how tags drive attribution
