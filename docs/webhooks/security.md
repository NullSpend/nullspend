# Webhook Security

Every webhook is signed with HMAC-SHA256. Always verify the signature before processing a webhook event.

## How Signing Works

When NullSpend sends a webhook, it:

1. Creates a signed content string: `{timestamp}.{JSON payload}`
2. Computes HMAC-SHA256 of that string using your endpoint's signing secret
3. Sets the `X-NullSpend-Signature` header to: `t={timestamp},v1={hex digest}`

```
X-NullSpend-Signature: t=1711036800,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8f9
```

## Verification Steps

1. Parse the `X-NullSpend-Signature` header to extract `t` (timestamp) and `v1` (signature)
2. Reconstruct the signed content: `{t}.{raw request body}`
3. Compute HMAC-SHA256 of the signed content using your signing secret
4. Compare your computed signature to `v1` using a timing-safe comparison
5. Check that `|current_time - t|` ≤ 300 seconds (replay protection)

## TypeScript (Node.js)

```typescript
import { createHmac, timingSafeEqual } from "crypto";

function verifyWebhook(
  payload: string,
  signatureHeader: string,
  secret: string
): boolean {
  // 1. Parse the signature header
  const parts = signatureHeader.split(",");
  const timestamp = parts
    .find((p) => p.startsWith("t="))
    ?.slice(2);
  const signatures = parts
    .filter((p) => p.startsWith("v1="))
    .map((p) => p.slice(3));

  if (!timestamp || signatures.length === 0) {
    return false;
  }

  // 2. Check timestamp (reject if older than 5 minutes)
  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - parseInt(timestamp, 10)) > 300) {
    return false;
  }

  // 3. Compute expected signature
  const signedContent = `${timestamp}.${payload}`;
  const expected = createHmac("sha256", secret)
    .update(signedContent)
    .digest("hex");

  // 4. Timing-safe compare against any v1 value (supports secret rotation)
  const expectedBuf = Buffer.from(expected, "hex");
  return signatures.some((sig) => {
    const sigBuf = Buffer.from(sig, "hex");
    return (
      sigBuf.length === expectedBuf.length &&
      timingSafeEqual(sigBuf, expectedBuf)
    );
  });
}

// Usage in an Express handler
app.post("/webhooks/nullspend", (req, res) => {
  const payload = req.body; // raw string, not parsed JSON
  const signature = req.headers["x-nullspend-signature"] as string;

  if (!verifyWebhook(payload, signature, process.env.WEBHOOK_SECRET!)) {
    return res.status(401).send("Invalid signature");
  }

  const event = JSON.parse(payload);
  console.log(`Received ${event.type}: ${event.id}`);

  // Process asynchronously — return 200 quickly
  res.status(200).send("OK");
});
```

**Important:** Use the raw request body string for verification, not `JSON.stringify(parsedBody)`. JSON serialization may reorder keys or change whitespace, breaking the signature.

## TypeScript (Web Crypto API / Edge)

For Cloudflare Workers, Vercel Edge Functions, or other edge runtimes:

```typescript
async function verifyWebhook(
  payload: string,
  signatureHeader: string,
  secret: string
): Promise<boolean> {
  const parts = signatureHeader.split(",");
  const timestamp = parts
    .find((p) => p.startsWith("t="))
    ?.slice(2);
  const signatures = parts
    .filter((p) => p.startsWith("v1="))
    .map((p) => p.slice(3));

  if (!timestamp || signatures.length === 0) {
    return false;
  }

  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - parseInt(timestamp, 10)) > 300) {
    return false;
  }

  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );

  const signedContent = `${timestamp}.${payload}`;
  const mac = await crypto.subtle.sign("HMAC", key, encoder.encode(signedContent));
  const expected = Array.from(new Uint8Array(mac))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  // Compare against any v1 value (supports secret rotation)
  return signatures.some((sig) => {
    if (sig.length !== expected.length) return false;
    // Constant-time comparison
    let result = 0;
    for (let i = 0; i < sig.length; i++) {
      result |= sig.charCodeAt(i) ^ expected.charCodeAt(i);
    }
    return result === 0;
  });
}
```

## Python

```python
import hashlib
import hmac
import time


def verify_webhook(payload: str, signature_header: str, secret: str) -> bool:
    """Verify a NullSpend webhook signature."""
    parts = signature_header.split(",")

    # 1. Parse timestamp and signatures
    timestamp = None
    signatures = []
    for part in parts:
        if part.startswith("t="):
            timestamp = part[2:]
        elif part.startswith("v1="):
            signatures.append(part[3:])

    if not timestamp or not signatures:
        return False

    # 2. Check timestamp (reject if older than 5 minutes)
    now = int(time.time())
    if abs(now - int(timestamp)) > 300:
        return False

    # 3. Compute expected signature
    signed_content = f"{timestamp}.{payload}"
    expected = hmac.new(
        secret.encode("utf-8"),
        signed_content.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # 4. Timing-safe compare against any v1 value (supports secret rotation)
    return any(hmac.compare_digest(expected, sig) for sig in signatures)


# Usage in a Flask handler
@app.route("/webhooks/nullspend", methods=["POST"])
def handle_webhook():
    payload = request.get_data(as_text=True)
    signature = request.headers.get("X-NullSpend-Signature", "")

    if not verify_webhook(payload, signature, os.environ["WEBHOOK_SECRET"]):
        return "Invalid signature", 401

    event = request.get_json()
    print(f"Received {event['type']}: {event['id']}")

    return "OK", 200
```

## Replay Protection

Always check the timestamp in the signature header. Reject events where the timestamp is more than **300 seconds** (5 minutes) from the current time:

```
|current_time - timestamp| > 300 → reject
```

This prevents an attacker who captures a valid webhook from replaying it later.

## Secret Rotation

When you rotate a webhook endpoint's signing secret:

1. NullSpend stores the old secret alongside the new one
2. For **24 hours**, every webhook is signed with **both** secrets
3. The `X-NullSpend-Signature` header contains two `v1` values during rotation:

```
X-NullSpend-Signature: t=1711036800,v1={new_secret_sig},v1={old_secret_sig}
```

4. Your verification code should check against **any** `v1` value (all code examples above already do this)
5. After 24 hours, the old secret is automatically cleared

This gives you a 24-hour window to update your verification code with the new secret without dropping any events.

## URL Restrictions

NullSpend validates webhook endpoint URLs to prevent SSRF:

| Restriction | Details |
|---|---|
| Protocol | HTTPS only (HTTP rejected) |
| Private IPs | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` blocked |
| Loopback | `127.0.0.0/8`, `0.0.0.0`, `localhost` blocked |
| Link-local | `169.254.0.0/16` blocked |
| IPv6 literals | Hostnames starting with `[` blocked |
| Local domains | `.local` TLD blocked |

## Related

- [Webhooks Overview](overview.md) — setup, payload modes, transport
- [Event Types](event-types.md) — full catalog of all 15 events
- [Custom Headers](../api-reference/custom-headers.md) — `X-NullSpend-Signature` and other webhook headers
