import { describe, it, expect, beforeAll } from "vitest";
import {
  signWebhookPayload,
  dualSignWebhookPayload,
  parseSignature,
  verifyWebhookSignature,
} from "../lib/webhook-signer.js";

beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as any).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.byteLength !== viewB.byteLength) return false;
      let result = 0;
      for (let i = 0; i < viewA.byteLength; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

describe("signWebhookPayload", () => {
  it("produces a signature in Stripe format", async () => {
    const sig = await signWebhookPayload('{"test":true}', "whsec_secret123", 1710612181);
    expect(sig).toMatch(/^t=1710612181,v1=[0-9a-f]{64}$/);
  });

  it("produces deterministic output for same inputs", async () => {
    const sig1 = await signWebhookPayload("hello", "secret", 1000);
    const sig2 = await signWebhookPayload("hello", "secret", 1000);
    expect(sig1).toBe(sig2);
  });

  it("produces different output for different payloads", async () => {
    const sig1 = await signWebhookPayload("hello", "secret", 1000);
    const sig2 = await signWebhookPayload("world", "secret", 1000);
    expect(sig1).not.toBe(sig2);
  });

  it("produces different output for different secrets", async () => {
    const sig1 = await signWebhookPayload("hello", "secret1", 1000);
    const sig2 = await signWebhookPayload("hello", "secret2", 1000);
    expect(sig1).not.toBe(sig2);
  });

  it("produces different output for different timestamps", async () => {
    const sig1 = await signWebhookPayload("hello", "secret", 1000);
    const sig2 = await signWebhookPayload("hello", "secret", 2000);
    expect(sig1).not.toBe(sig2);
  });
});

describe("parseSignature", () => {
  it("parses valid signature header", () => {
    const result = parseSignature("t=1710612181,v1=abcdef0123456789");
    expect(result).toEqual({
      timestamp: 1710612181,
      signatures: ["abcdef0123456789"],
    });
  });

  it("returns null for missing timestamp", () => {
    expect(parseSignature("v1=abc123")).toBeNull();
  });

  it("returns null for missing signature", () => {
    expect(parseSignature("t=1000")).toBeNull();
  });

  it("returns null for invalid timestamp", () => {
    expect(parseSignature("t=notanumber,v1=abc123")).toBeNull();
  });
});

describe("verifyWebhookSignature", () => {
  it("verifies a valid signature", async () => {
    const secret = "whsec_testsecret";
    const payload = '{"event":"test"}';
    const timestamp = Math.floor(Date.now() / 1000);
    const sig = await signWebhookPayload(payload, secret, timestamp);

    const valid = await verifyWebhookSignature(payload, sig, secret);
    expect(valid).toBe(true);
  });

  it("rejects tampered payload", async () => {
    const secret = "whsec_testsecret";
    const timestamp = Math.floor(Date.now() / 1000);
    const sig = await signWebhookPayload('{"event":"original"}', secret, timestamp);

    const valid = await verifyWebhookSignature('{"event":"tampered"}', sig, secret);
    expect(valid).toBe(false);
  });

  it("rejects expired timestamp", async () => {
    const secret = "whsec_testsecret";
    const payload = '{"event":"test"}';
    const oldTimestamp = Math.floor(Date.now() / 1000) - 600; // 10 minutes ago
    const sig = await signWebhookPayload(payload, secret, oldTimestamp);

    const valid = await verifyWebhookSignature(payload, sig, secret, 300);
    expect(valid).toBe(false);
  });

  it("accepts timestamp within tolerance", async () => {
    const secret = "whsec_testsecret";
    const payload = '{"event":"test"}';
    const recentTimestamp = Math.floor(Date.now() / 1000) - 60; // 1 minute ago
    const sig = await signWebhookPayload(payload, secret, recentTimestamp);

    const valid = await verifyWebhookSignature(payload, sig, secret, 300);
    expect(valid).toBe(true);
  });
});

describe("dualSignWebhookPayload", () => {
  it("with null previousSecret produces single v1 (matches signWebhookPayload)", async () => {
    const payload = '{"test":true}';
    const secret = "whsec_current";
    const ts = 1710612181;

    const dualSig = await dualSignWebhookPayload(payload, secret, null, ts);
    const singleSig = await signWebhookPayload(payload, secret, ts);

    expect(dualSig).toBe(singleSig);
    const parsed = parseSignature(dualSig);
    expect(parsed).not.toBeNull();
    expect(parsed!.signatures).toHaveLength(1);
  });

  it("with previousSecret produces two v1 values", async () => {
    const payload = '{"test":true}';
    const currentSecret = "whsec_current";
    const previousSecret = "whsec_previous";
    const ts = 1710612181;

    const sig = await dualSignWebhookPayload(payload, currentSecret, previousSecret, ts);
    const parsed = parseSignature(sig);

    expect(parsed).not.toBeNull();
    expect(parsed!.signatures).toHaveLength(2);
    expect(parsed!.timestamp).toBe(ts);
  });

  it("first v1 is current secret, second is previous", async () => {
    const payload = '{"data":"hello"}';
    const currentSecret = "whsec_current";
    const previousSecret = "whsec_previous";
    const ts = 1710612181;

    const dualSig = await dualSignWebhookPayload(payload, currentSecret, previousSecret, ts);
    const parsed = parseSignature(dualSig);

    // Generate individual signatures for comparison
    const currentSig = await signWebhookPayload(payload, currentSecret, ts);
    const previousSig = await signWebhookPayload(payload, previousSecret, ts);
    const currentParsed = parseSignature(currentSig);
    const previousParsed = parseSignature(previousSig);

    expect(parsed!.signatures[0]).toBe(currentParsed!.signatures[0]);
    expect(parsed!.signatures[1]).toBe(previousParsed!.signatures[0]);
  });

  it("verifyWebhookSignature succeeds against dual header with current secret", async () => {
    const payload = '{"event":"rotation_test"}';
    const currentSecret = "whsec_current";
    const previousSecret = "whsec_previous";
    const ts = Math.floor(Date.now() / 1000);

    const dualSig = await dualSignWebhookPayload(payload, currentSecret, previousSecret, ts);
    const valid = await verifyWebhookSignature(payload, dualSig, currentSecret);
    expect(valid).toBe(true);
  });

  it("verifyWebhookSignature succeeds against dual header with previous secret", async () => {
    const payload = '{"event":"rotation_test"}';
    const currentSecret = "whsec_current";
    const previousSecret = "whsec_previous";
    const ts = Math.floor(Date.now() / 1000);

    const dualSig = await dualSignWebhookPayload(payload, currentSecret, previousSecret, ts);
    const valid = await verifyWebhookSignature(payload, dualSig, previousSecret);
    expect(valid).toBe(true);
  });
});
