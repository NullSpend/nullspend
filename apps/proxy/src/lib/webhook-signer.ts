import { toHex } from "./hex.js";
import { timingSafeStringEqual } from "./timing-safe-equal.js";

const SIGNATURE_VERSION = "v1";

/** 24-hour dual-signing window for secret rotation. SYNC: lib/webhooks/signer.ts */
export const SECRET_ROTATION_WINDOW_SECONDS = 86_400;

/**
 * Compute HMAC-SHA256 hex for a payload+timestamp using Web Crypto API.
 */
async function computeHmacHex(
  payload: string,
  secret: string,
  timestamp: number,
): Promise<string> {
  const signedContent = `${timestamp}.${payload}`;
  const encoder = new TextEncoder();

  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(signedContent),
  );

  return toHex(signature);
}

/**
 * Sign a webhook payload using HMAC-SHA256.
 * Returns a Stripe-format signature: `t={timestamp},v1={hex}`
 */
export async function signWebhookPayload(
  payload: string,
  secret: string,
  timestamp: number,
): Promise<string> {
  const hex = await computeHmacHex(payload, secret, timestamp);
  return `t=${timestamp},${SIGNATURE_VERSION}=${hex}`;
}

/**
 * Dual-sign a webhook payload with current and (optionally) previous secret.
 * During rotation window: `t={ts},v1={currentHex},v1={previousHex}`
 * After rotation window (previousSecret is null): `t={ts},v1={currentHex}`
 */
export async function dualSignWebhookPayload(
  payload: string,
  currentSecret: string,
  previousSecret: string | null,
  timestamp: number,
): Promise<string> {
  const currentHex = await computeHmacHex(payload, currentSecret, timestamp);
  if (!previousSecret) {
    return `t=${timestamp},${SIGNATURE_VERSION}=${currentHex}`;
  }
  const previousHex = await computeHmacHex(payload, previousSecret, timestamp);
  return `t=${timestamp},${SIGNATURE_VERSION}=${currentHex},${SIGNATURE_VERSION}=${previousHex}`;
}

/**
 * Parse a signature header into its components.
 * Expected format: `t={timestamp},v1={hex}`
 */
export function parseSignature(
  header: string,
): { timestamp: number; signatures: string[] } | null {
  const parts = header.split(",");
  let timestamp: number | null = null;
  const signatures: string[] = [];

  for (const part of parts) {
    const [key, value] = part.split("=", 2);
    if (key === "t") {
      timestamp = parseInt(value, 10);
      if (isNaN(timestamp) || !Number.isSafeInteger(timestamp) || timestamp < 0) return null;
    } else if (key === SIGNATURE_VERSION) {
      signatures.push(value);
    }
  }

  if (timestamp === null || signatures.length === 0) return null;
  return { timestamp, signatures };
}

/**
 * Verify a webhook signature against the payload.
 * Rejects events older than `toleranceSeconds` (default 300 = 5 minutes).
 */
export async function verifyWebhookSignature(
  payload: string,
  signatureHeader: string,
  secret: string,
  toleranceSeconds = 300,
): Promise<boolean> {
  const parsed = parseSignature(signatureHeader);
  if (!parsed) return false;

  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - parsed.timestamp) > toleranceSeconds) {
    return false;
  }

  const expected = await signWebhookPayload(payload, secret, parsed.timestamp);
  const expectedParsed = parseSignature(expected);
  if (!expectedParsed) return false;

  return parsed.signatures.some((sig) =>
    timingSafeStringEqual(sig, expectedParsed.signatures[0]),
  );
}

