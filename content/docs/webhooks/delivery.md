---
title: "Webhook Delivery"
description: "How NullSpend delivers webhook events, including transport, retries, and failure handling."
---

How NullSpend delivers webhook events, including transport, retries, and failure handling.

## Two Delivery Paths

NullSpend has two independent systems that dispatch webhooks, depending on where the event originates:

| Path | Events | Transport | Retries |
|---|---|---|---|
| **Proxy-side** | `cost_event.created`, all budget events, `velocity.*`, `session.*`, `tag_budget.*`, `request.blocked` | Cloudflare Queue | 5 retries, exponential backoff |
| **Dashboard-side** | `action.*`, `test.ping`, dashboard-originated `cost_event.created` | Direct HTTP POST | None (fire-and-forget) |

Both paths sign payloads identically — your verification code works the same regardless of which path delivered the event.

## Proxy-Side Delivery

Events from the proxy worker are delivered via a Cloudflare Queue-based dispatch pipeline.

**How it works:**

1. The proxy builds the event payload and enqueues it to the webhook delivery queue
2. The queue consumer fetches the endpoint metadata, signs the payload, and delivers the HTTP POST
3. If your endpoint returns a non-2xx response or doesn't respond within 5 seconds, the consumer retries with exponential backoff

**Retry behavior:**

- **5 retry attempts** with exponential backoff
- Your endpoint must return a **2xx** status code within **5 seconds**
- After all retries are exhausted, the event goes to the dead-letter queue (DLQ)

**Event type filtering:** Each endpoint can subscribe to specific event types. If an endpoint's `eventTypes` array is empty, it receives all events. If it lists specific types, only matching events are dispatched.

## Dashboard-Side Delivery

Events from the Next.js dashboard (action lifecycle events, test pings) are delivered directly via HTTP POST.

**How it works:**

1. The dashboard queries the database for your active webhook endpoints
2. Each matching endpoint receives a signed HTTP POST
3. The request has a **5-second timeout** — if your endpoint doesn't respond in time, the attempt is abandoned
4. **No retries.** Dashboard-side delivery is fire-and-forget.

**Secret rotation cleanup:** After dispatching, the dashboard checks for endpoints with expired rotation windows (older than 24 hours) and clears the previous signing secret. This cleanup is lazy and fire-and-forget — it doesn't affect delivery.

## Headers

Every webhook POST — from both paths — includes these headers:

| Header | Value | Purpose |
|---|---|---|
| `Content-Type` | `application/json` | Always JSON |
| `X-NullSpend-Signature` | HMAC-SHA256 signature | Verify authenticity (see [Security](security.md)) |
| `X-NullSpend-Webhook-Id` | Event ID (e.g., `evt_a1b2c3d4-...`) | Deduplicate deliveries |
| `X-NullSpend-Webhook-Timestamp` | Unix timestamp (seconds) | Detect replay attacks |
| `User-Agent` | `NullSpend-Webhooks/1.0` | Identify NullSpend traffic |

## Payload Modes

Endpoints can be configured with a payload mode that controls how `cost_event.created` events are delivered:

| Mode | Behavior |
|---|---|
| **full** (default) | Complete event data in `data.object` |
| **thin** | Reference only in `related_object` — fetch full data from the API |

Payload mode only affects `cost_event.created` events. All other event types always use full payloads regardless of the endpoint's mode setting.

### When to use thin mode

**Use thin** when you process high volumes of cost events (100+/minute) and want to minimize webhook bandwidth. Thin payloads are ~200 bytes vs ~2KB for full payloads. Your handler receives a reference and fetches the full event on demand — useful when you only need to process a subset of events or want to batch-fetch.

**Use full (default)** when you need all event data immediately and volume is manageable. Most integrations should start here. No extra API call needed to process the event.

| Scenario | Recommended mode |
|---|---|
| Logging/analytics pipeline that processes every event | Full |
| High-volume stream, filter then fetch | Thin |
| Slack/PagerDuty alerting on specific conditions | Full |
| Billing reconciliation (batch, periodic) | Thin |

### Thin payload example

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

### Fetching the full event from a thin payload

Use the `related_object.url` path with your API key:

```bash
curl "https://www.nullspend.dev/api/cost-events?requestId=req_xyz&provider=openai" \
  -H "Authorization: Bearer ns_live_..."
```

The response contains the full cost event data — the same shape as a full-mode webhook payload's `data.object`.

**Tip:** If you receive many thin events, batch your fetch calls rather than calling the API per-event. The cost events list endpoint supports filtering by `requestId`, so you can fetch multiple events efficiently.

## Failure Handling

Webhook delivery is **fail-open** — errors are logged but never block the operation that triggered the event.

- A failed webhook dispatch never prevents a cost event from being written
- A failed webhook dispatch never blocks or delays an API response
- If the endpoint lookup fails (cache miss + DB error), the event is silently dropped

**DLQ:** On the proxy side, events that fail all 5 retry attempts land in the dead-letter queue. These are not automatically retried.

**Dashboard side:** No DLQ. Failed deliveries are logged server-side but not retried or stored.

## Endpoint Caching

### Proxy

The proxy caches endpoint metadata (ID, URL, event types) to avoid querying the database on every request:

- **Workers KV** with a **5-minute TTL**
- **Signing secrets are never cached** — they are always fetched from the database at dispatch time

Cache invalidation happens when you create, update, or delete an endpoint via the dashboard API.

### Dashboard

The dashboard queries the database directly for each dispatch — no caching layer.

## Best Practices

- **Return 200 fast.** Do your processing asynchronously. Proxy-side events retry on timeout; dashboard-side events are lost.
- **Deduplicate by event ID.** Use the `X-NullSpend-Webhook-Id` header (same as `event.id`) to skip duplicate deliveries from retries.
- **Don't rely on ordering.** Events may arrive out of order, especially with retries. Use `created_at` to determine sequence.
- **Use thin mode for high volume.** If you process hundreds of cost events per minute, thin mode reduces bandwidth and latency. Fetch full details on demand.
- **Monitor for DLQ entries.** If your endpoint has persistent failures, events accumulate in the dead-letter queue.

For expanded best practices with code examples, see [Best Practices](best-practices.md).

## Related

- [Webhooks Overview](overview.md) — setup, event types, and quick start
- [Webhook Security](security.md) — HMAC signature verification with code examples
- [Best Practices](best-practices.md) — expanded patterns with code examples
- [Event Types](event-types.md) — full catalog of all 18 events with JSON examples
- [Webhooks API](../api-reference/webhooks-api.md) — manage endpoints programmatically
