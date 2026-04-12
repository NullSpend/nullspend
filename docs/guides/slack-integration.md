# Slack Integration

Get real-time NullSpend alerts in Slack. Budget thresholds, human-in-the-loop approvals, and margin warnings — all delivered to a channel your team already watches.

## Setup

1. **Create a Slack Incoming Webhook** — go to [api.slack.com/apps](https://api.slack.com/apps), create (or pick) an app, enable Incoming Webhooks, and add one to the channel you want alerts in. Copy the webhook URL.

2. **Add the webhook in NullSpend** — open **Settings > Slack** in the dashboard. Paste the webhook URL and save. The URL must be `https://hooks.slack.com/services/...`, `/workflows/...`, or `/triggers/...`.

3. **Test it** — click **Send Test** in the dashboard. You should see a test message appear in your Slack channel within a few seconds.

That's it. NullSpend will now send alerts to that channel automatically.

### Optional fields

| Field | Purpose |
|---|---|
| **Channel name** | Display label in the dashboard (cosmetic, does not control routing) |
| **Slack User ID** | If set, only this Slack user can approve/reject actions via button clicks. Find your ID in Slack: Profile > ⋯ > Copy member ID. |

## What triggers alerts

NullSpend sends three categories of Slack alerts:

### Budget threshold alerts

Sent when a budget entity's spend crosses a threshold:

| Event | When |
|---|---|
| **Warning** | Spend crosses 50% or 75% of the budget limit |
| **Critical** | Spend crosses 90% of the budget limit |
| **Exceeded** | Spend exceeds 100% of the budget limit |

These thresholds are configurable per-budget. The message includes the entity name, current spend, limit, and a **View Budgets** button linking to the dashboard.

Custom thresholds can be set per-budget via the API — see [Budgets API](../api-reference/budgets-api.md).

### Human-in-the-loop (HITL) action alerts

Sent when an agent proposes an action that requires human approval:

| Event | Message |
|---|---|
| **Action pending** | Shows action type, agent ID, payload summary, and **Approve** / **Reject** buttons |
| **Budget increase requested** | Shows current spend, current limit, requested amount, and approve/reject buttons |
| **Action decided** | Threaded reply confirming who approved or rejected, with a link to the action detail |
| **Budget increase completed** | Shows old limit → new limit after approval |

If you configured a **Slack User ID**, only that user's button clicks are accepted. Others see an ephemeral "not authorized" message.

### Margin alerts

Sent when a customer's margin health tier worsens (requires Stripe connection):

| Transition | Example |
|---|---|
| healthy → moderate | Margin dropped from 35% to 22% |
| moderate → at_risk | Margin dropped from 20% to 8% |
| at_risk → critical | Margin dropped below 0% (losing money) |

The message includes customer name, previous/current margin, revenue, cost, and a **View Margins** button.

## Message format

All Slack messages use [Block Kit](https://api.slack.com/block-kit) for rich formatting:

- **Header block** with severity emoji and title
- **Section fields** with key metrics (spend, limit, margin, etc.)
- **Action buttons** that link to the relevant dashboard page (or trigger approve/reject for HITL)

Emojis indicate severity: :warning: warning, :red_circle: critical, :rotating_light: exceeded, :large_green_circle: healthy, :large_blue_circle: moderate.

## Configuration API

Manage Slack config programmatically (session auth required, admin role):

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/slack/config` | Get current config (webhook URL is masked) |
| `POST` | `/api/slack/config` | Create or update config |
| `DELETE` | `/api/slack/config` | Remove config |
| `POST` | `/api/slack/test` | Send a test notification |

### Create/update config

```bash
curl -X POST https://www.nullspend.dev/api/slack/config \
  -H "Content-Type: application/json" \
  -H "Cookie: <session>" \
  -d '{
    "webhookUrl": "https://hooks.slack.com/services/T00/B00/xxxx",
    "channelName": "#ops-alerts",
    "slackUserId": "U0123ABCDEF",
    "isActive": true
  }'
```

### Response

```json
{
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "webhookUrl": "https://hooks.slack.com/services/****/****/xxxx****",
    "channelName": "#ops-alerts",
    "slackUserId": "U0123ABCDEF",
    "isActive": true,
    "createdAt": "2026-04-10T12:00:00.000Z",
    "updatedAt": "2026-04-10T12:00:00.000Z"
  }
}
```

The webhook URL is always masked in responses — only the first few characters of the last path segment are visible.

## Security

- Webhook URLs must be HTTPS and hosted at `hooks.slack.com`
- Non-HTTPS URLs are silently rejected (SSRF defense)
- The Slack callback endpoint (`/api/slack/callback`) verifies Slack request signatures using your app's signing secret
- Button interactions are scoped to the configured Slack User ID when set
- All Slack alert dispatch is **fail-open** — a Slack delivery failure never blocks the underlying operation (cost logging, budget enforcement, etc.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| No messages appearing | Check that the webhook is **active** in Settings > Slack. Click **Send Test** to verify. |
| "Not authorized" on button click | The Slack User ID in your config doesn't match your Slack account. Update it or remove it to allow any team member. |
| Test works but no budget alerts | Budget alerts only fire when thresholds are crossed. Make a test request through the proxy to trigger a cost event. |
| Margin alerts not appearing | Requires an active Stripe connection and at least one customer mapping. Check Settings > Stripe. |

## Related

- [Budgets](../features/budgets.md) — how budget limits and thresholds work
- [Human-in-the-Loop](../features/human-in-the-loop.md) — action lifecycle and approval flow
- [Margins](../features/margins.md) — Stripe margin tracking
- [Webhooks Overview](../webhooks/overview.md) — HTTP webhook alternative for programmatic integrations
