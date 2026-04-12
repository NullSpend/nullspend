# Errors

Every error response from NullSpend — both the proxy and the dashboard API — uses the same format:

```json
{
  "error": {
    "code": "budget_exceeded",
    "message": "Request blocked: estimated cost exceeds remaining budget",
    "details": null
  }
}
```

- `code` — Machine-readable error identifier. Use this for programmatic handling.
- `message` — Human-readable explanation.
- `details` — Additional context (object or `null`). Present on validation errors, session limits, and velocity limits.

## Proxy Errors

These errors are returned by the NullSpend proxy (`proxy.nullspend.dev`).

### Authentication

| Code | HTTP | When | Fix |
|---|---|---|---|
| `unauthorized` | 401 | `X-NullSpend-Key` header is missing, malformed, or the key has been revoked | Verify the header is present and the key is active in Settings |

### Budget Enforcement

| Code | HTTP | When | Fix |
|---|---|---|---|
| `budget_exceeded` | 429 | Estimated cost of this request exceeds the remaining budget | Increase the budget ceiling, wait for the period to reset, or remove the budget |
| `customer_budget_exceeded` | 429 | Estimated cost exceeds the customer-level budget limit | Increase the customer budget or remove the limit. The response `details` includes `customer_id`, `budget_limit_microdollars`, and `budget_spend_microdollars` |
| `velocity_exceeded` | 429 | Spend rate within the velocity window exceeds the configured limit | Wait for the cooldown period (check `Retry-After` header). The response `details` includes `limitMicrodollars`, `windowSeconds`, and `currentMicrodollars` |
| `session_limit_exceeded` | 429 | Cumulative spend for this session ID exceeds the session limit | Start a new session (new `X-NullSpend-Session` value) or increase the session limit. The response `details` includes `session_id`, `session_spend_microdollars`, and `session_limit_microdollars` |
| `tag_budget_exceeded` | 429 | Estimated cost exceeds a tag-level budget limit | Adjust the tag budget. The response `details` includes `tag_key`, `tag_value`, `budget_limit_microdollars`, and `budget_spend_microdollars` |
| `budget_unavailable` | 503 | Budget enforcement service is temporarily unavailable | Retry after a brief delay. The proxy fails closed — requests are blocked, not passed through |

### Request Validation

| Code | HTTP | When | Fix |
|---|---|---|---|
| `bad_request` | 400 | Request body is not valid JSON or is missing required fields | Check the request body format |
| `invalid_model` | 400 | The `model` field is not in the pricing catalog | Check [supported models](../reference/supported-models.md) |
| `payload_too_large` | 413 | Request body exceeds 1 MB | Reduce the request body size |
| `invalid_upstream` | 400 | `X-NullSpend-Upstream` URL is not in the allowlist | Use the default upstream or contact support to add your URL |

### Rate Limiting

| Code | HTTP | When | Fix |
|---|---|---|---|
| `rate_limited` | 429 | Too many requests. Default limits: 120/min per IP, 600/min per API key | Reduce request rate. Check the `Retry-After` header for when to retry |

Rate limit responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and `Retry-After` headers. See the [headers reference](custom-headers.md#rate-limit-headers-on-429-responses-only). For the full rate limiting reference, see [Rate Limits](rate-limits.md).

### Upstream & Server

| Code | HTTP | When | Fix |
|---|---|---|---|
| `upstream_error` | 502 | The upstream provider returned an error or no response body | Check the provider's status page (e.g., [status.openai.com](https://status.openai.com)) |
| `not_found` | 404 | The requested endpoint is not supported by the proxy | Supported endpoints: `POST /v1/chat/completions` (OpenAI), `POST /v1/messages` (Anthropic), `POST /v1/mcp/budget/check`, `POST /v1/mcp/events` |
| `internal_error` | 500 | Unexpected server error | Retry the request. If persistent, contact support with the `X-NullSpend-Trace-Id` from the response |

## Dashboard API Errors

These errors are returned by the NullSpend dashboard API (`nullspend.dev/api/`).

### Validation

| Code | HTTP | When | Fix |
|---|---|---|---|
| `invalid_json` | 400 | Request body is not valid JSON | Send valid JSON with `Content-Type: application/json` |
| `validation_error` | 400 | Request failed schema validation | Check `details.issues` for specific field errors |
| `unsupported_media_type` | 415 | `Content-Type` is not `application/json` | Set the `Content-Type` header |
| `payload_too_large` | 413 | Request body exceeds the max size | Reduce the request body |

Validation error details include an `issues` array:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "details": {
      "issues": [
        { "path": ["amount"], "message": "Expected number, received string" }
      ]
    }
  }
}
```

### Resources

| Code | HTTP | When | Fix |
|---|---|---|---|
| `not_found` | 404 | The requested resource does not exist | Verify the resource ID |
| `limit_exceeded` | 409 | Resource limit reached for the organization's tier (e.g., Free: 10 keys, 2 webhooks, 3 budgets) | Delete unused resources or upgrade to a higher tier |

### HITL Actions

| Code | HTTP | When | Fix |
|---|---|---|---|
| `invalid_action_transition` | 409 | Invalid state transition (e.g., approving an already-rejected action) | Check the action's current state before transitioning |
| `stale_action` | 409 | The action was modified by another actor since you last fetched it | Re-fetch the action and retry |
| `action_expired` | 409 | The action's TTL has expired | Create a new action |

### Rate Limiting

| Code | HTTP | When | Fix |
|---|---|---|---|
| `rate_limit_exceeded` | 429 | Too many requests (per-IP or per-key) | Reduce request rate. Check the `Retry-After` header |

> **Note:** The proxy uses `rate_limited` while the dashboard API uses `rate_limit_exceeded`. Handle both codes if your application calls both services.

### Authentication & Authorization

| Code | HTTP | When | Fix |
|---|---|---|---|
| `authentication_required` | 401 | No valid session or API key | Log in or provide a valid `X-NullSpend-Key` header |
| `forbidden` | 403 | Authenticated but not authorized to access this resource | Verify you own the resource |

### Server

| Code | HTTP | When | Fix |
|---|---|---|---|
| `service_unavailable` | 503 | A downstream service is temporarily unavailable | Retry after a brief delay |
| `internal_error` | 500 | Unexpected server error | Retry the request |

## HTTP Status Code Summary

| Status | Meaning |
|---|---|
| 400 | Bad request — check your input |
| 401 | Identity unknown — check your API key or session |
| 403 | Identity known but not authorized |
| 404 | Resource or endpoint not found |
| 409 | Conflict — resource limit or state conflict |
| 413 | Request body too large |
| 415 | Wrong content type |
| 429 | Rate or budget limit exceeded — check `Retry-After` |
| 500 | Server error — retry |
| 502 | Upstream provider error |
| 503 | Service temporarily unavailable — retry |

## Handling Errors Programmatically

```typescript
const response = await fetch("https://proxy.nullspend.dev/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
    "X-NullSpend-Key": process.env.NULLSPEND_API_KEY,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "Hello" }] }),
});

if (!response.ok) {
  const { error } = await response.json();
  switch (error.code) {
    case "budget_exceeded":
      // Notify the user, queue for later, or request budget increase
      break;
    case "velocity_exceeded":
      // Back off and retry after Retry-After seconds
      const retryAfter = response.headers.get("Retry-After");
      break;
    case "rate_limited":
      // Reduce request rate
      break;
    default:
      console.error(`NullSpend error: ${error.code} — ${error.message}`);
  }
}
```

## API Reference

- [API Overview](overview.md) — authentication, pagination, ID formats
- [Cost Events API](cost-events-api.md) — ingest, list, and analyze cost events
- [API Keys API](api-keys-api.md) — create, list, revoke keys, and introspect identity
- [Budgets API](budgets-api.md) — create, manage, and query budgets
- [Webhooks API](webhooks-api.md) — manage webhook endpoints and deliveries
- [Actions API](actions-api.md) — human-in-the-loop approval workflows
