# Margins API

Dashboard endpoints for customer margin analytics. All endpoints require session authentication.

## GET /api/margins

Returns the margin table for a given period.

**Query Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `period` | string | Current month | Calendar month in `YYYY-MM` format |
| `format` | string | `json` | Response format. `csv` returns a downloadable CSV file. |

**Response (JSON):**

```json
{
  "data": {
    "summary": {
      "blendedMarginPercent": 42.5,
      "totalRevenueMicrodollars": 500000000,
      "totalCostMicrodollars": 287500000,
      "criticalCount": 1,
      "atRiskCount": 2,
      "lastSyncAt": "2026-04-05T10:00:00.000Z",
      "syncStatus": "active",
      "skippedCurrencies": { "eur": 3 }
    },
    "customers": [
      {
        "stripeCustomerId": "cus_abc123",
        "customerName": "Acme Corp",
        "avatarUrl": null,
        "tagValue": "acme-corp",
        "revenueMicrodollars": 100000000,
        "costMicrodollars": 30000000,
        "marginMicrodollars": 70000000,
        "marginPercent": 70,
        "healthTier": "healthy",
        "sparkline": [
          { "period": "2026-02", "marginPercent": 65 },
          { "period": "2026-03", "marginPercent": 68 },
          { "period": "2026-04", "marginPercent": 70 },
          { "period": "2026-05", "marginPercent": 72, "projected": true }
        ],
        "projectedTierWorsening": false,
        "budgetSuggestionMicrodollars": null
      }
    ]
  }
}
```

**Response (CSV):**

Returns `Content-Type: text/csv` with `Content-Disposition: attachment; filename="margins-2026-04.csv"`.

Columns: Customer, Stripe ID, Tag Value, Revenue ($), Cost ($), Margin (%), Margin ($), Health Tier.

**Errors:**

| Status | Code | When |
|---|---|---|
| 400 | `validation_error` | Invalid period format |
| 401 | `authentication_required` | No session |
| 403 | `forbidden` | Insufficient role |

---

## GET /api/margins/:customer

Returns detailed margin data for a single customer.

**Path Parameters:**

| Param | Type | Description |
|---|---|---|
| `customer` | string | URL-encoded tag value (e.g., `acme-corp`) |

**Query Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `period` | string | Current month | Calendar month in `YYYY-MM` format |

**Response:**

```json
{
  "data": {
    "stripeCustomerId": "cus_abc123",
    "customerName": "Acme Corp",
    "avatarUrl": null,
    "tagValue": "acme-corp",
    "healthTier": "healthy",
    "marginPercent": 70,
    "revenueMicrodollars": 100000000,
    "costMicrodollars": 30000000,
    "revenueOverTime": [
      { "period": "2026-02", "revenue": 90000000, "cost": 25000000 },
      { "period": "2026-03", "revenue": 95000000, "cost": 28000000 },
      { "period": "2026-04", "revenue": 100000000, "cost": 30000000 }
    ],
    "modelBreakdown": [
      { "model": "gpt-4o", "cost": 20000000, "requestCount": 150 },
      { "model": "gpt-4o-mini", "cost": 10000000, "requestCount": 800 }
    ]
  }
}
```

**Errors:**

| Status | Code | When |
|---|---|---|
| 404 | `not_found` | Customer mapping not found |

---

## GET /api/margins/unmatched

Returns unmatched Stripe customers, unmapped cost tags, and pending auto-matches for the mapping management UI.

**Response:**

```json
{
  "data": {
    "unmatchedStripeCustomers": [
      {
        "stripeCustomerId": "cus_xyz",
        "customerName": "BetaCo",
        "customerEmail": "billing@beta.co",
        "totalRevenueMicrodollars": 50000000
      }
    ],
    "unmappedTagValues": [
      {
        "tagValue": "gamma-inc",
        "totalCostMicrodollars": 15000000,
        "requestCount": 200
      }
    ],
    "pendingAutoMatches": [
      {
        "id": "uuid",
        "stripeCustomerId": "cus_def",
        "customerName": "Delta LLC",
        "tagValue": "cus_def",
        "confidence": 0.9
      }
    ],
    "customerNames": {
      "cus_abc": "Acme Corp",
      "cus_def": "Delta LLC"
    }
  }
}
```

---

## POST /api/stripe/connect

Connect a Stripe restricted key. Validates the key with a minimal Stripe API call before storing.

**Request Body:**

```json
{
  "stripeKey": "rk_live_..."
}
```

**Validation:**
- Key must start with `rk_` (restricted key). `sk_test_` is allowed in non-production environments.
- Key is tested with `stripe.customers.list({ limit: 1 })`.

**Response (201):**

```json
{
  "data": {
    "id": "uuid",
    "keyPrefix": "rk_live_abcd...wxyz",
    "status": "active",
    "createdAt": "2026-04-05T10:00:00.000Z"
  }
}
```

**Errors:**

| Status | Code | When |
|---|---|---|
| 400 | `validation_error` | Missing or invalid key format |
| 400 | `stripe_validation_failed` | Key doesn't authenticate with Stripe |
| 409 | `conflict` | Stripe already connected (disconnect first) |

---

## DELETE /api/stripe/disconnect

Removes the Stripe connection and cascades: deletes all revenue data and customer mappings for the org.

**Response:**

```json
{
  "data": { "deleted": true }
}
```

---

## GET /api/stripe/revenue-sync

Triggers a revenue sync. Called by Vercel Cron (with `Bearer CRON_SECRET`) or manually from the dashboard (with session auth, requires `member` role).

**Cron Response:**

```json
{
  "data": { "synced": 5, "errors": 0 }
}
```

**Manual Response:**

```json
{
  "data": {
    "orgId": "uuid",
    "customersProcessed": 12,
    "periodsUpdated": 15,
    "autoMatchesCreated": 2,
    "invoicesFetched": 48,
    "invoicesSkipped": 3,
    "skippedCurrencies": { "eur": 3 },
    "durationMs": 4521
  }
}
```

---

## Customer Mappings

### GET /api/customer-mappings

Returns all customer-to-tag mappings for the org.

### POST /api/customer-mappings

Create or update a mapping. Upserts on `(orgId, stripeCustomerId, tagKey)`.

**Request Body:**

```json
{
  "stripeCustomerId": "cus_abc123",
  "tagValue": "acme-corp",
  "tagKey": "customer",
  "matchType": "manual"
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `stripeCustomerId` | Yes | | Stripe customer ID |
| `tagValue` | Yes | | Cost event tag value |
| `tagKey` | No | `"customer"` | Tag key (almost always `customer`) |
| `matchType` | No | `"manual"` | `"manual"` or `"auto"` |

### DELETE /api/customer-mappings?id=UUID

Delete a mapping by ID. Returns 404 if not found or not owned by the org.

---

## Related

- [Margins Feature Guide](../features/margins.md) — setup, health tiers, Slack alerts
- [Customer Attribution](../features/tags.md#customer-attribution) — `X-NullSpend-Customer` header (recommended) or tag fallback
- [Webhook Event Types](../webhooks/event-types.md) — `margin.threshold_crossed`
