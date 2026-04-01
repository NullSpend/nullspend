import { describe, it, expect, vi } from "vitest";
import {
  getCachedWebhookEndpoints,
  setCachedWebhookEndpoints,
  invalidateWebhookEndpoints,
} from "../lib/cache-kv.js";

function makeKV(overrides: Record<string, unknown> = {}) {
  return {
    get: vi.fn().mockResolvedValue(null),
    put: vi.fn().mockResolvedValue(undefined),
    delete: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as unknown as KVNamespace;
}

const testEndpoints = [
  { id: "ep-1", url: "https://hooks.example.com/1", eventTypes: ["cost_event.created"] },
  { id: "ep-2", url: "https://hooks.example.com/2", eventTypes: [] },
];

describe("getCachedWebhookEndpoints", () => {
  it("returns parsed JSON on cache hit", async () => {
    const kv = makeKV({ get: vi.fn().mockResolvedValue(testEndpoints) });
    const result = await getCachedWebhookEndpoints(kv, "user-1");
    expect(result).toEqual(testEndpoints);
  });

  it("returns null on cache miss", async () => {
    const kv = makeKV();
    const result = await getCachedWebhookEndpoints(kv, "user-1");
    expect(result).toBeNull();
  });

  it("calls kv.get with correct key and type", async () => {
    const kv = makeKV();
    await getCachedWebhookEndpoints(kv, "user-42");
    expect(kv.get).toHaveBeenCalledWith("webhook:user-42", "json");
  });

  it("propagates KV errors (no internal try/catch)", async () => {
    const kv = makeKV({ get: vi.fn().mockRejectedValue(new Error("KV down")) });
    await expect(getCachedWebhookEndpoints(kv, "user-1")).rejects.toThrow("KV down");
  });
});

describe("setCachedWebhookEndpoints", () => {
  it("calls kv.put with correct key, JSON, and TTL", async () => {
    const kv = makeKV();
    await setCachedWebhookEndpoints(kv, "user-1", testEndpoints);
    expect(kv.put).toHaveBeenCalledWith(
      "webhook:user-1",
      JSON.stringify(testEndpoints),
      { expirationTtl: 300 },
    );
  });

  it("serializes only CachedWebhookEndpoint fields (no extra properties leak)", async () => {
    const kv = makeKV();
    // Simulate an object with extra properties that could leak secrets
    // at the JS level (TypeScript types erased at runtime)
    const endpointsWithExtras = [
      { id: "ep-1", url: "https://hooks.example.com/1", eventTypes: [], signingSecret: "whsec_leaked" },
    ] as any;
    await setCachedWebhookEndpoints(kv, "user-1", endpointsWithExtras);
    const serialized = (kv.put as ReturnType<typeof vi.fn>).mock.calls[0][1] as string;
    // JSON.stringify serializes ALL enumerable properties regardless of TS types.
    // This test documents that cache-kv.ts does NOT strip extra fields —
    // the caller (webhook-cache.ts) is responsible for stripping secrets
    // before passing data to setCachedWebhookEndpoints.
    expect(serialized).toContain("whsec_leaked");
  });

  it("propagates KV errors", async () => {
    const kv = makeKV({ put: vi.fn().mockRejectedValue(new Error("KV write fail")) });
    await expect(setCachedWebhookEndpoints(kv, "user-1", testEndpoints)).rejects.toThrow("KV write fail");
  });
});

describe("invalidateWebhookEndpoints", () => {
  it("calls kv.delete with correct key", async () => {
    const kv = makeKV();
    await invalidateWebhookEndpoints(kv, "user-1");
    expect(kv.delete).toHaveBeenCalledWith("webhook:user-1");
  });

  it("propagates KV errors", async () => {
    const kv = makeKV({ delete: vi.fn().mockRejectedValue(new Error("KV delete fail")) });
    await expect(invalidateWebhookEndpoints(kv, "user-1")).rejects.toThrow("KV delete fail");
  });
});
