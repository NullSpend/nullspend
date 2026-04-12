# Rate Limits

The NullSpend proxy enforces rate limits to protect against abuse. Two layers run concurrently: per-IP and per-key.

---

## Two Layers

| Layer | Limit | Identifier | Scope |
|---|---|---|---|
| Per-IP | 120/min | `cf-connecting-ip` | All requests |
| Per-key | 600/min | `x-nullspend-key` | Authenticated requests |

Both layers use Cloudflare's native rate limiting binding, which runs on the same machine as the Worker with ~0ms overhead. Limits are configured in `wrangler.jsonc`.

---

## Enforcement Order

1. **IP rate limit + auth** — Run in parallel. Neither depends on the other.
2. **Key rate limit** — Checked as part of the rate limiting step, only if a valid `x-nullspend-key` header is present.
3. Either failure → `429` immediately. The request never reaches the upstream provider.

Rate limiting runs **concurrently with** authentication in the [request pipeline](proxy-endpoints.md#request-processing-pipeline).

---

## Edge Cases

- **Missing or empty `x-nullspend-key`** — Only the IP limit is checked. The key limit is skipped.
- **Key longer than 128 characters** — Treated as invalid for rate limiting purposes. Only the IP limit is checked.
- **Per-colo counting** — Cloudflare native rate limiting counts per data center (colo), not globally. A user hitting multiple colos gets separate counters. This is acceptable for abuse protection.

---

## Response Headers

When a rate limit is exceeded, the `429` response includes:

| Header | Value |
|---|---|
| `Retry-After` | Seconds until it's safe to retry (currently `60`) |

The response body follows the standard error format:

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Too many requests",
    "details": null
  }
}
```

---

## Failure Mode: Fail-Open

If the rate limiting binding is unavailable, rate limiting is **skipped silently**. The request proceeds as if no rate limit exists.

This is intentional: rate limiting is protective, not a correctness requirement. Budget enforcement — which is independent of rate limiting — still applies and **fails closed** (returns `503`).

---

## Rate Limits vs Budget Enforcement

| | Rate Limits | Budget Enforcement |
|---|---|---|
| What it caps | Request count per minute | Dollar spend per period |
| Failure mode | Fail-open (skip if binding unavailable) | Fail-closed (`503` if unavailable) |
| Error code | `rate_limited` | `budget_exceeded` / `velocity_exceeded` |
| Scope | Per-IP + per-key | Per-user + per-key + per-tag |
| Pipeline position | Parallel with auth | Inside route handler |

---

## Related

- [Errors](errors.md#rate-limiting) — rate limit error format
- [Custom Headers](custom-headers.md) — response header reference
- [Proxy Endpoints](proxy-endpoints.md) — request pipeline order
