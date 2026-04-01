import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockSql } = vi.hoisted(() => {
  const mockSql = vi.fn().mockResolvedValue([]);
  return { mockSql };
});

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

const mockKvFns = vi.hoisted(() => ({
  getCachedWebhookEndpoints: vi.fn().mockResolvedValue(null),
  setCachedWebhookEndpoints: vi.fn().mockResolvedValue(undefined),
  invalidateWebhookEndpoints: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/cache-kv.js", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../lib/cache-kv.js")>();
  return {
    ...orig,
    getCachedWebhookEndpoints: mockKvFns.getCachedWebhookEndpoints,
    setCachedWebhookEndpoints: mockKvFns.setCachedWebhookEndpoints,
    invalidateWebhookEndpoints: mockKvFns.invalidateWebhookEndpoints,
  };
});

import {
  getWebhookEndpoints,
  getWebhookEndpointsWithSecrets,
  invalidateWebhookCache,
} from "../lib/webhook-cache.js";

const mockKv = {} as KVNamespace;

const mockDbRows = [
  {
    id: "ep-1",
    url: "https://hooks.example.com/1",
    signing_secret: "whsec_secret1",
    previous_signing_secret: null,
    secret_rotated_at: null,
    event_types: ["cost_event.created"],
    api_version: "2026-04-01",
    payload_mode: "full",
  },
  {
    id: "ep-2",
    url: "https://hooks.example.com/2",
    signing_secret: "whsec_secret2",
    previous_signing_secret: null,
    secret_rotated_at: null,
    event_types: [],
    api_version: "2026-04-01",
    payload_mode: "full",
  },
];

describe("getWebhookEndpoints", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("returns data from KV on cache hit without querying DB", async () => {
    const kvData = [
      { id: "ep-1", url: "https://hooks.example.com/1", eventTypes: ["cost_event.created"] },
    ];
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(kvData);

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);

    expect(result).toEqual(kvData);
    expect(mockSql).not.toHaveBeenCalled();
  });

  it("queries DB on KV miss and writes result to KV", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(null);
    mockSql.mockResolvedValueOnce(mockDbRows);

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({
      id: "ep-1",
      url: "https://hooks.example.com/1",
      eventTypes: ["cost_event.created"],
    });

    // Secrets must NOT be in the cached value
    expect(result[0]).not.toHaveProperty("signingSecret");

    // Verify KV cache was written
    expect(mockKvFns.setCachedWebhookEndpoints).toHaveBeenCalledWith(
      mockKv,
      "user-1",
      expect.arrayContaining([
        expect.objectContaining({ id: "ep-1" }),
        expect.objectContaining({ id: "ep-2" }),
      ]),
    );
  });

  it("does not include secrets in KV write", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(null);
    mockSql.mockResolvedValueOnce(mockDbRows);

    await getWebhookEndpoints("postgresql://test", "user-1", mockKv);

    const writtenData = mockKvFns.setCachedWebhookEndpoints.mock.calls[0][2];
    const serialized = JSON.stringify(writtenData);
    expect(serialized).not.toContain("whsec_secret1");
    expect(serialized).not.toContain("signingSecret");
  });

  it("falls through to DB on KV read error (fail-open)", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockRejectedValue(new Error("KV read fail"));
    mockSql.mockResolvedValueOnce(mockDbRows);

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);
    expect(result).toHaveLength(2);
  });

  it("returns endpoints even if KV write fails (fail-open)", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(null);
    mockKvFns.setCachedWebhookEndpoints.mockRejectedValue(new Error("KV write fail"));
    mockSql.mockResolvedValueOnce(mockDbRows);

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);
    expect(result).toHaveLength(2);
  });

  it("returns empty array on DB error (fail-open)", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(null);
    mockSql.mockRejectedValueOnce(new Error("db down"));

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);
    expect(result).toEqual([]);
  });

  it("returns empty array when user has no endpoints", async () => {
    mockKvFns.getCachedWebhookEndpoints.mockResolvedValue(null);
    mockSql.mockResolvedValueOnce([]);

    const result = await getWebhookEndpoints("postgresql://test", "user-1", mockKv);
    expect(result).toEqual([]);
  });
});

describe("getWebhookEndpointsWithSecrets", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("maps payload_mode from DB row to payloadMode", async () => {
    mockSql.mockResolvedValueOnce([
      { ...mockDbRows[0], payload_mode: "thin" },
    ]);

    const result = await getWebhookEndpointsWithSecrets("postgresql://test", "user-1");
    expect(result).toHaveLength(1);
    expect(result[0].payloadMode).toBe("thin");
  });

  it("falls back to 'full' when payload_mode is null/missing", async () => {
    mockSql.mockResolvedValueOnce([
      { ...mockDbRows[0], payload_mode: null },
      { ...mockDbRows[1] }, // payload_mode already "full"
    ]);

    const result = await getWebhookEndpointsWithSecrets("postgresql://test", "user-1");
    expect(result).toHaveLength(2);
    expect(result[0].payloadMode).toBe("full");
    expect(result[1].payloadMode).toBe("full");
  });

  it("returns endpoints with signing secrets from DB", async () => {
    mockSql.mockResolvedValueOnce(mockDbRows);

    const result = await getWebhookEndpointsWithSecrets("postgresql://test", "user-1");
    expect(result).toHaveLength(2);
    expect(result[0].signingSecret).toBe("whsec_secret1");
    expect(result[1].signingSecret).toBe("whsec_secret2");
    expect(result[0].previousSigningSecret).toBeNull();
    expect(result[0].secretRotatedAt).toBeNull();
  });

  it("returns empty array on database error (fail-open)", async () => {
    mockSql.mockRejectedValueOnce(new Error("connection failed"));

    const result = await getWebhookEndpointsWithSecrets("postgresql://test", "user-1");
    expect(result).toEqual([]);
  });
});

describe("invalidateWebhookCache", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("invalidates KV cache for user", async () => {
    await invalidateWebhookCache("user-1", mockKv);
    expect(mockKvFns.invalidateWebhookEndpoints).toHaveBeenCalledWith(mockKv, "user-1");
  });

  it("does not throw on KV invalidation error", async () => {
    mockKvFns.invalidateWebhookEndpoints.mockRejectedValue(new Error("KV delete fail"));
    await expect(invalidateWebhookCache("user-1", mockKv)).resolves.not.toThrow();
  });
});
