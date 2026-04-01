# Versioning

NullSpend uses date-based API versioning (the Stripe pattern). Each version is an ISO date string.

---

## Current Version

```
2026-04-01
```

## Supported Versions

| Version | Status |
|---|---|
| `2026-04-01` | Current (default) |

---

## Version Resolution Chain

The proxy resolves the API version using a three-step priority chain:

| Priority | Source | Set By |
|---|---|---|
| 1 (highest) | `NullSpend-Version` request header | Per-request override |
| 2 | Key-level version | Set at key creation or updated in dashboard |
| 3 (lowest) | Default (`2026-04-01`) | Automatic fallback |

If a source provides a version not in the supported list, it's ignored and the next source is tried.

---

## Setting the Version

### Per-Request

Send the `NullSpend-Version` header with any proxy request:

```bash
curl https://proxy.nullspend.com/v1/chat/completions \
  -H "NullSpend-Version: 2026-04-01" \
  -H "X-NullSpend-Key: ns_live_sk_..." \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

See [Custom Headers](custom-headers.md#nullspend-version) for the header reference.

### Per-Key

Set a version when creating a key in the dashboard or via the [API Keys API](api-keys-api.md). All requests using that key will use the pinned version unless overridden by the `NullSpend-Version` header.

---

## Response Header

Every proxy response includes the `NullSpend-Version` header indicating which version processed the request:

```
NullSpend-Version: 2026-04-01
```

---

## Versioning Strategy

When a breaking change is introduced:

1. A new version is added to the supported list
2. Existing keys continue using their pinned version
3. The default version moves to the latest for newly created keys

This ensures existing integrations are not broken by API changes.

---

## Related

- [Custom Headers](custom-headers.md) — header format and resolution
- [API Keys API](api-keys-api.md) — set version at key level
- [API Overview](overview.md) — general API reference
