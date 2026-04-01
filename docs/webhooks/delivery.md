# Webhook Delivery

How NullSpend delivers webhook events, including transport, retries, and failure handling.

## Two Delivery Paths

NullSpend has two independent systems that dispatch webhooks, depending on where the event originates:

| Path | Events | Transport | Retries |
|---|---|---|---|
| **Proxy-side** | `cost_event.created`, all budget events, `velocity.*`, `session.*`, `tag_budget.*`, `request.blocked` | Cloudflare Queues | Exponential backoff (10s to 1hr) |
| **Dashboard-side** | `action.*`, `test.ping`, dashboard-originated `cost_event.created` | Direct HTTP POST | None (fire-and-forget) |

Both paths sign payloads identically — your verification code works the same regardless of which path delivered the event.

## Proxy-Side Delivery

Events from the proxy worker are delivered via [Cloudflare Queues](https://developers.cloudflare.com/queues/).

**How it works:**

1. The proxy enqueues a thin message (userId, endpointId, event) to the webhook queue
2. The queue consumer fetches the endpoint's signing secret from the database
3. The payload is signed with a fresh timestamp and delivered via HTTP POST
4. If your endpoint returns a non-2xx response or doesn't respond within 30 seconds, the message is retried

**Retry behavior:**

- **Exponential backoff** — 10s, 20s, 40s, ... up to 1 hour max delay
- **2xx** = success (acked), **429 or 5xx** = transient failure (retried), **4xx** (non-429) = permanent failure (acked, not retried)
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

**Thin payload example:**

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

Fetch the full event data using the `url` field with your API key. See [Webhooks Overview](overview.md) for more on payload modes.

## Failure Handling

Webhook delivery is **fail-open** — errors are logged but never block the operation that triggered the event.

- A failed webhook dispatch never prevents a cost event from being written
- A failed webhook dispatch never blocks or delays an API response
- If the endpoint lookup fails (cache miss + DB error), the event is silently dropped

**Webhook DLQ:** On the proxy side, events that fail all retry attempts land in the Cloudflare Queue dead-letter queue. Failed deliveries are logged with a `webhook_delivery_failed` metric and always acked to prevent infinite retries.

**Dashboard side:** No DLQ. Failed deliveries are logged server-side but not retried or stored.

## Endpoint Caching

### Proxy

The proxy caches endpoint metadata (ID, URL, event types) to avoid querying the database on every request:

- **Workers KV** with a **5-minute TTL** (preferred when the KV binding is available)
- **Redis** with a 5-minute TTL (fallback)
- **Signing secrets are never cached** — they are always fetched from the database at dispatch time

Cache invalidation happens when you create, update, or delete an endpoint via the dashboard API.

### Dashboard

The dashboard queries the database directly for each dispatch — no caching layer.

## Best Practices

- **Return 200 fast.** Do your processing asynchronously. Proxy-side events retry on timeout; dashboard-side events are lost.
- **Deduplicate by event ID.** Use the `X-NullSpend-Webhook-Id` header (same as `event.id`) to skip duplicate deliveries from retries.
- **Don't rely on ordering.** Events may arrive out of order, especially with retries. Use `created_at` to determine sequence.
- **Use thin mode for high volume.** If you process hundreds of cost events per minute, thin mode reduces bandwidth and latency. Fetch full details on demand.
- **Monitor for DLQ entries.** If your endpoint has persistent failures, events accumulate in the dead-letter queue. Check delivery logs periodically.

For expanded best practices with code examples, see [Best Practices](best-practices.md).

## Related

- [Webhooks Overview](overview.md) — setup, event types, and quick start
- [Webhook Security](security.md) — HMAC signature verification with code examples
- [Best Practices](best-practices.md) — expanded patterns with code examples
- [Event Types](event-types.md) — full catalog of all 15 events with JSON examples
- [Webhooks API](../api-reference/webhooks-api.md) — manage endpoints programmatically
