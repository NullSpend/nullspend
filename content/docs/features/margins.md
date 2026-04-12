---
title: "Margins"
description: "Per-customer profitability analytics powered by Stripe revenue sync."
---

# Margins

Margins connect your Stripe revenue to your AI costs, showing per-customer profitability. Connect Stripe once and NullSpend auto-matches customers, calculates margins, and alerts you when profitability drops.

## How It Works

```
Stripe (paid invoices)
    │
    ▼
Revenue Sync (every 2 hours)
    │
    ├─ Fetch invoices (last 3 months + current month)
    ├─ Group by customer + calendar month
    ├─ Auto-match Stripe customers → cost event tags
    │
    ▼
Margin Table
    │
    ├─ Revenue per customer per month (from Stripe)
    ├─ AI cost per customer per month (from cost events)
    ├─ Margin % = (Revenue - Cost) / Revenue × 100
    ├─ Health tier classification
    ├─ 3-month sparkline + trajectory projection
    │
    ▼
Alerts
    ├─ Webhook: margin.threshold_crossed
    └─ Slack: rich message with action buttons
```

## Setup

### 1. Connect Stripe

Go to **Margins** in the dashboard. Paste a Stripe restricted key with invoice and customer read access.

Use a restricted key (`rk_live_...`), not your secret key. NullSpend only needs to read invoices and customers. The key is encrypted with AES-256-GCM before storage.

### 2. Identify Your Customers

Add the `X-NullSpend-Customer` header to your AI requests so NullSpend can match them to Stripe customers:

```
X-NullSpend-Customer: acme-corp
```

The value is the identifier you use for this customer: a slug, a Stripe customer ID, or a company name. Alternatively, you can use the tag fallback: `X-NullSpend-Tags: {"customer":"acme-corp"}`. See [Customer Attribution](tags.md#customer-attribution) for details.

### 3. Auto-Match

After the first sync, NullSpend tries to automatically match Stripe customers to your cost tags using two methods:

| Match Method | How It Works | Confidence |
|---|---|---|
| Metadata match | Stripe `customer.metadata.nullspend_customer` equals a tag value | 1.0 |
| Customer ID match | Stripe customer ID (`cus_xxx`) equals a tag value | 0.9 |

Auto-matches appear in the **Pending Auto-Matches** section on the Customers page. Confirm or reject each one. You can also manually map any unmatched Stripe customer to a tag value via the dropdown.

### 4. View Margins

The margin table shows every matched customer with:

- **Revenue** (from Stripe invoices, USD only)
- **AI Cost** (from cost events tagged with `customer=...`)
- **Margin %** and **Margin $**
- **Health tier** badge
- **3-month sparkline** with trajectory projection (dashed line)
- **Budget suggestion** for critical customers

## Health Tiers

| Tier | Margin % | What It Means |
|---|---|---|
| Healthy | >= 50% | Cost is well below revenue |
| Moderate | 20% to 49% | Watch this customer |
| At Risk | 0% to 19% | Barely profitable |
| Critical | < 0% | Losing money on this customer |

## Trajectory Projection

The sparkline shows 3 months of margin history. If all 3 months have data, NullSpend projects a trend line forward one month using linear regression. A warning icon appears next to the health badge if the projected margin crosses into a worse tier.

This is directional, not a forecast. It shows where the trend is heading so you can act before a customer goes critical.

## Slack Alerts

When a customer's margin crosses into a worse tier (e.g., moderate to at-risk), NullSpend sends a Slack alert to your configured webhook. The message includes:

- Customer name and period
- Previous and current margin with tier emojis
- Revenue and cost amounts
- **View Margins** and **Set Budget Cap** buttons (deep links to the dashboard)

Slack alerts require a Slack webhook configured in Settings. They fire independently of webhook events, so a Slack outage never blocks your webhooks.

## CSV Export

Click the **CSV** button in the header to download the margin table as a CSV file. Columns: Customer, Stripe ID, Tag Value, Revenue ($), Cost ($), Margin (%), Margin ($), Health Tier.

The CSV uses RFC 4180 escaping and includes formula injection defense (values starting with `=`, `+`, `-`, or `@` are prefixed with a single quote).

## Multi-Currency

Revenue sync currently supports USD invoices only. Non-USD invoices are skipped and counted. If any invoices were skipped, an amber banner appears on the Customers page showing the count per currency (e.g., "4 invoices in EUR, 1 in GBP skipped. Margins show USD revenue only.").

## Revenue Sync

Sync runs automatically every 2 hours via Vercel Cron. You can also click **Sync Now** on the Customers page to trigger an immediate sync.

Each sync:
1. Fetches all paid invoices from the last 3 calendar months + current month
2. Groups by customer and calendar month
3. Replaces (DELETE + re-INSERT) revenue data per customer per period
4. Runs auto-match against cost event tags
5. Detects margin threshold crossings and dispatches webhook + Slack alerts

Sync is idempotent. Running it multiple times produces the same result.

## Webhook Event

Margin crossings emit a `margin.threshold_crossed` webhook event. See [Webhook Event Types](../webhooks/event-types.md#marginthreshold_crossed) for the payload format.

Only **worsening** crossings fire events (healthy to moderate, moderate to at-risk, etc.). Improving margins do not trigger webhooks to avoid alert fatigue.

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/margins` | GET | Session (viewer) | Margin table with summary and per-customer data |
| `/api/margins?format=csv` | GET | Session (viewer) | Download margin table as CSV |
| `/api/margins/:customer` | GET | Session (viewer) | Customer detail with revenue over time and model breakdown |
| `/api/margins/unmatched` | GET | Session (viewer) | Unmatched Stripe customers, unmapped tags, pending auto-matches |
| `/api/stripe/connect` | GET, POST | Session (viewer/admin) | Get or create Stripe connection |
| `/api/stripe/disconnect` | DELETE | Session (admin) | Remove connection + cascade delete revenue and mappings |
| `/api/stripe/revenue-sync` | GET | Cron or Session (member) | Trigger revenue sync |
| `/api/customer-mappings` | GET, POST, DELETE | Session (viewer/member) | Manage customer-to-tag mappings |

## Related

- [Tags](tags.md) — how to tag requests with customer identifiers
- [Budgets](budgets.md) — set spending caps for customers with tag budgets
- [Webhook Event Types](../webhooks/event-types.md) — `margin.threshold_crossed` event
