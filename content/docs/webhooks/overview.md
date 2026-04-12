---
title: "Webhooks"
description: "Webhooks send real-time HTTP POST notifications to your server when events happen in NullSpend — cost events, budget alerts, velocity trips, and more."
---

Webhooks send real-time HTTP POST notifications to your server when events happen in NullSpend — cost events, budget alerts, velocity trips, and more.

## Quick Setup

1. Open the [NullSpend dashboard](https://nullspend.dev/app/settings)
2. Go to **Settings** → **Webhooks** → **Add Endpoint**
3. Enter your HTTPS endpoint URL
4. Choose which event types to receive (or leave empty for all events)
5. Choose a payload mode: **full** (complete data) or **thin** (reference + fetch-back URL)
6. Copy the signing secret — you'll need it to [verify signatures](security.md)
7. Click **Create** and test with `test.ping`

## Event Types

NullSpend emits 18 event types:

| Event | Fires When |
|---|---|
| `cost_event.created` | A cost event is recorded (every proxied request) |
| `budget.threshold.warning` | Spend crosses a threshold < 90% |
| `budget.threshold.critical` | Spend crosses a threshold ≥ 90% |
| `budget.exceeded` | Budget ceiling is hit |
| `budget.increased` | Budget limit increased via HITL approval |
| `budget.reset` | Budget period resets (daily/weekly/monthly) |
| `request.blocked` | A request is blocked (budget, rate limit, or policy) |
| `velocity.exceeded` | Velocity limit tripped — circuit breaker open |
| `velocity.recovered` | Velocity cooldown expired — circuit breaker closed |
| `session.limit_exceeded` | Session spend cap exceeded |
| `tag_budget.exceeded` | Tag-level budget exceeded |
| `customer_budget.exceeded` | Customer-scoped budget exceeded |
| `margin.threshold_crossed` | Customer margin worsened to a lower health tier |
| `action.created` | HITL approval action created |
| `action.approved` | HITL action approved |
| `action.rejected` | HITL action rejected |
| `action.expired` | HITL action expired (TTL elapsed) |
| `test.ping` | Manual test event |

See [Event Types](event-types.md) for full payload examples.

## Payload Modes

### Full Mode (default)

The complete event data is included in the POST body:

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "data": {
    "object": {
      "request_id": "req_xyz",
      "provider": "openai",
      "model": "gpt-4o",
      "input_tokens": 1000,
      "output_tokens": 500,
      "cost_microdollars": 7,
      "tags": { "team": "billing" }
    }
  }
}
```

### Thin Mode

Only a reference to the object is included. Fetch the full data from the API when needed. Thin mode is available for `cost_event.created` only — all other event types use full payloads regardless of the endpoint's payload mode setting.

```json
{
  "id": "evt_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "cost_event.created",
  "api_version": "2026-04-01",
  "created_at": 1711036800,
  "related_object": {
    "id": "req_xyz",
    "type": "cost_event",
    "url": "/api/cost-events?requestId=req_xyz&provider=openai"
  }
}
```

Thin mode reduces payload size and avoids sending potentially sensitive data (like tags) to your webhook endpoint. Fetch the full event from the `url` when you need it.

## Transport

Webhooks are delivered via Cloudflare Queues with automatic retries:

- **5 retry attempts** with exponential backoff on failure
- Your endpoint must return a **2xx** response within **5 seconds**
- Non-2xx responses or timeouts trigger a retry

See [Delivery](delivery.md) for details on both delivery paths, failure handling, and endpoint caching.

## Headers

Every webhook POST includes these headers:

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `X-NullSpend-Signature` | HMAC-SHA256 signature (see [Security](security.md)) |
| `X-NullSpend-Webhook-Id` | Event ID (e.g., `evt_a1b2c3d4-...`) |
| `X-NullSpend-Webhook-Timestamp` | Unix timestamp (seconds) when the event was signed |
| `User-Agent` | `NullSpend-Webhooks/1.0` |

## Security

Every webhook is signed with HMAC-SHA256 using your endpoint's signing secret. Always verify the signature before processing. See [Webhook Security](security.md) for verification code in TypeScript and Python.

## Limits

| Limit | Value |
|---|---|
| Max endpoints per organization | Free: 2, Pro: 25, Enterprise: unlimited |
| URL protocol | HTTPS required |
| Blocked URLs | Private IPs (10.x, 172.16–31.x, 192.168.x), localhost, 127.x, 0.0.0.0, IPv6 literals, `.local` domains |
| Response timeout | 5 seconds |
| Retry attempts | 5 |

## Best Practices

- **Return 2xx quickly.** Do your processing asynchronously. If your handler takes more than 5 seconds, the delivery is retried.
- **Verify the signature.** Always check `X-NullSpend-Signature` before trusting the payload. See [Security](security.md).
- **Deduplicate with `event.id`.** Use the `id` field (e.g., `evt_a1b2c3d4-...`) to detect and skip duplicate deliveries.
- **Don't rely on ordering.** Events may arrive out of order. Use `created_at` to determine event sequence.
- **Use event type filtering.** Subscribe only to the events you need. An endpoint with `eventTypes: []` receives everything.
- **Test with `test.ping`.** Send a test event from the dashboard to verify your endpoint is reachable and signature verification works.
- **Use thin mode for high-volume endpoints.** If you process hundreds of cost events per minute, thin mode reduces bandwidth and lets you fetch details on demand.

For expanded best practices with code examples, see [Best Practices](best-practices.md).

## Related

- [Event Types](event-types.md) — full catalog of all 18 events with JSON examples
- [Delivery](delivery.md) — transport, retries, failure handling, and endpoint caching
- [Webhook Security](security.md) — HMAC verification with code examples
- [Webhooks API](../api-reference/webhooks-api.md) — create, update, test, and manage endpoints programmatically
- [Best Practices](best-practices.md) — expanded patterns with code examples
- [Tags](../features/tags.md) — tags included in cost event payloads
- [Budgets](../features/budgets.md) — budget events (exceeded, threshold, reset)
