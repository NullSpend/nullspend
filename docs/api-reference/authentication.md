# Authentication

NullSpend uses two authentication modes:

- **API key** — For the proxy and SDK. Pass `X-NullSpend-Key` in the request header.
- **Session** — For the dashboard. Uses a browser session cookie.

For the full endpoint auth breakdown, see the [API Overview](overview.md#endpoint-auth-summary).

---

## API Key Authentication

Header: `X-NullSpend-Key`

Key format: `ns_live_sk_` + 32 hex characters (43 characters total). See [Custom Headers](custom-headers.md#x-nullspend-key-required) for validation rules and examples.

---

## Key Lifecycle

### Creating Keys

Create keys in the dashboard under **Settings → API Keys**, or programmatically via `POST /api/keys` (see [API Keys API](api-keys-api.md)).

The raw key is shown **once** at creation — store it immediately. Maximum **20 keys** per user.

### Key-Level Settings

Each key can have:

- **API version** — Overrides the default version. Can be overridden per-request by the `NullSpend-Version` header. See [Versioning](versioning.md).
- **Name** — Human-readable label (1–50 characters).
- **Default tags** — Tags automatically applied to every request made with this key.

### Revoking Keys

Revoke keys in the dashboard or via `DELETE /api/keys/:id`. Revocation is a soft delete (`revoked_at` timestamp).

**Propagation delay: up to 120 seconds.** The proxy caches valid keys with a 120-second TTL (with ±10s jitter), so a revoked key may continue to authenticate for up to ~130 seconds after revocation. Use the dashboard's invalidation endpoint for immediate revocation.

---

## How Key Validation Works

When a request arrives at the proxy:

1. Read the `x-nullspend-key` header
2. SHA-256 hash the raw key
3. Check the **positive cache** (256 entries, 120s TTL ±10s jitter) — if found and not expired, return the cached identity
4. Check the **negative cache** (2,048 entries, 30s TTL) — if found and not expired, reject immediately
5. If both miss, query the database: `WHERE key_hash = $1 AND revoked_at IS NULL`
6. The DB query also checks for enabled webhooks (`has_webhooks` flag) and loads key-level settings
7. Cache the result in the appropriate cache (positive for valid keys, negative for invalid)

Timing-safe comparison prevents timing attacks. Database connection timeout: 5,000ms.

---

## Session Authentication

The dashboard uses browser session cookies from Supabase Auth. Session-authenticated endpoints power the dashboard UI and are not callable from external scripts.

For the full list of which endpoints use session vs API key auth, see the [API Overview](overview.md#endpoint-auth-summary).

---

## Error Responses

| Scenario | HTTP | Code |
|---|---|---|
| Missing `X-NullSpend-Key` header | 401 | `unauthorized` |
| Malformed or invalid key | 401 | `unauthorized` |
| Revoked key (after cache expiry) | 401 | `unauthorized` |

See [Errors](errors.md#authentication) for the full error format.

---

## Related

- [API Overview](overview.md) — endpoint auth summary
- [Custom Headers](custom-headers.md) — header format and validation
- [API Keys API](api-keys-api.md) — create, list, and revoke keys
- [Errors](errors.md) — error codes and response format
