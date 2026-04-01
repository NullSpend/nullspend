---
title: "Webhook Best Practices"
description: "Practical patterns for reliable webhook consumption. These expand on the quick tips in Webhooks Overview and Delivery."
---

Practical patterns for reliable webhook consumption. These expand on the quick tips in [Webhooks Overview](overview.md) and [Delivery](delivery.md).

## 1. Return 2xx Fast

Proxy-side webhooks are delivered via Cloudflare Queues with a **5-second timeout**. Dashboard-side webhooks are fire-and-forget — if your handler is slow, the event is lost.

**Pattern:** Accept the event, enqueue it for processing, and return `200` immediately.

**TypeScript (Express):**

```typescript
app.post("/webhooks/nullspend", async (req, res) => {
  // Verify signature first (see section 2)
  const isValid = await verifySignature(req);
  if (!isValid) return res.status(401).send("Invalid signature");

  // Enqueue for async processing — don't block the response
  await queue.add("process-webhook", { event: req.body });

  res.status(200).json({ received: true });
});
```

**Python (FastAPI):**

```python
@app.post("/webhooks/nullspend")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    if not verify_signature(request.headers, body):
        raise HTTPException(status_code=401)

    event = json.loads(body)
    background_tasks.add_task(process_event, event)

    return {"received": True}
```

## 2. Verify Signatures

Always verify the `X-NullSpend-Signature` header before trusting the payload. See [Webhook Security](security.md) for full verification code in TypeScript and Python.

**Key reminder:** Verify against the **raw request body string**, not re-serialized JSON. JSON serialization is not stable — key order may differ, and the signature will fail.

## 3. Deduplicate by Event ID

Every webhook event has a unique `id` field (format: `evt_` + UUID). Retries can deliver the same event multiple times. Use the event ID as an idempotency key.

**TypeScript (Redis):**

```typescript
async function isDuplicate(eventId: string): Promise<boolean> {
  // SETNX returns true only if the key was newly set
  const isNew = await redis.set(eventId, "1", { NX: true, EX: 86400 });
  return !isNew;
}

// In your handler:
if (await isDuplicate(event.id)) {
  return res.status(200).json({ skipped: true });
}
```

**Python (Redis):**

```python
def is_duplicate(event_id: str) -> bool:
    return not redis.set(event_id, "1", nx=True, ex=86400)
```

## 4. Handle Events Idempotently

Beyond deduplication, make your processing logic itself idempotent. If the same event is processed twice, the result should be identical.

**Example:** When ingesting cost events, use `request_id` as a unique constraint:

```sql
INSERT INTO cost_events (request_id, provider, model, cost_microdollars)
VALUES ($1, $2, $3, $4)
ON CONFLICT (request_id) DO NOTHING;
```

This way, even if dedup fails (Redis eviction, race condition), the database enforces correctness.

## 5. Don't Rely on Event Ordering

Events may arrive out of order, especially with retries. A `budget.threshold.critical` event might arrive before `budget.threshold.warning` if the warning delivery was retried.

**Use `created_at` for sequencing**, not arrival time:

```typescript
// Wrong: processing order = arrival order
events.forEach(processEvent);

// Right: sort by event timestamp
events.sort((a, b) => a.created_at - b.created_at).forEach(processEvent);
```

## 6. Choose the Right Payload Mode

Payload mode only affects `cost_event.created` events. All other event types always use full payloads.

| Use Case | Mode | Why |
|---|---|---|
| Low-volume alerting (< 100/min) | Full | Simplest — all data in one request |
| High-volume analytics ingestion | Thin | Less bandwidth, batch fetch-back |
| Security-sensitive (tags may contain PII) | Thin | Sensitive data stays in your API, not in transit |
| Mixed (alerts + cost ingestion) | Both | Create two endpoints with different modes |

**Thin payloads** include a `related_object.url` field — fetch the full event from your NullSpend API when you need it.

## 7. Filter Event Types

Each endpoint can subscribe to specific event types. Leave `eventTypes` empty to receive everything.

**Common filter sets:**

| Purpose | Event Types |
|---|---|
| Alerting | `budget.exceeded`, `budget.threshold.warning`, `budget.threshold.critical`, `velocity.exceeded` |
| Cost ingestion | `cost_event.created` |
| HITL workflow | `action.created`, `action.approved`, `action.rejected`, `action.expired` |
| Session monitoring | `session.limit_exceeded` |

Filtering at the endpoint level is more efficient than filtering in your handler — unmatched events are never dispatched.

## 8. Monitor for Failures

**Proxy-side:** Events that fail all 5 retry attempts land in the dead-letter queue (DLQ). These are not automatically retried.

**Dashboard-side:** No DLQ. Failed deliveries are logged server-side but not retried or stored. If you need reliable delivery for `action.*` events, build a polling fallback against the actions API.

## 9. Rotate Secrets Safely

NullSpend supports zero-downtime secret rotation with a 24-hour dual-signing window:

1. **Rotate** the secret in the dashboard (Settings → Webhooks → endpoint → Rotate Secret)
2. During the next **24 hours**, every webhook is signed with **both** the old and new secrets
3. **Update** your verification code with the new secret
4. After 24 hours, the old secret is automatically cleared

Your existing verification code already checks any `v1` signature value in the header — as long as you update within 24 hours, there's no downtime.

## 10. Test Before Going Live

1. Create your endpoint in the dashboard
2. Click **Test** to send a `test.ping` event
3. Log the full headers and body in your handler
4. Verify:
   - Signature verification passes
   - Event ID format is `evt_` + UUID
   - `created_at` is a recent Unix timestamp
   - Your handler returns `200` within 5 seconds

## Related

- [Webhook Security](security.md) — HMAC signature verification with code examples
- [Delivery](delivery.md) — transport, retries, failure handling
- [Event Types](event-types.md) — full catalog of all 15 events with JSON examples
- [Webhooks Overview](overview.md) — setup and quick start
