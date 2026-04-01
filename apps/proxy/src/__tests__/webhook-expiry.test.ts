import { describe, it, expect, vi, beforeEach } from "vitest";

const { mockSql } = vi.hoisted(() => {
  const mockSql = vi.fn().mockResolvedValue([]);
  return { mockSql };
});

vi.mock("../lib/db.js", () => ({
  getSql: () => mockSql,
}));

vi.mock("../lib/webhook-signer.js", () => ({
  SECRET_ROTATION_WINDOW_SECONDS: 86_400,
}));

import { expireRotatedSecrets } from "../lib/webhook-expiry.js";
import type { WebhookEndpointWithSecret } from "../lib/webhook-cache.js";

function makeEndpoint(overrides: Partial<WebhookEndpointWithSecret> = {}): WebhookEndpointWithSecret {
  return {
    id: "ep-1",
    url: "https://hooks.example.com/webhook",
    signingSecret: "whsec_current",
    previousSigningSecret: null,
    secretRotatedAt: null,
    eventTypes: [],
    apiVersion: "2026-04-01",
    payloadMode: "full" as const,
    ...overrides,
  };
}

describe("expireRotatedSecrets", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("does nothing when no endpoints have secretRotatedAt", async () => {
    await expireRotatedSecrets("postgresql://test", [
      makeEndpoint(),
      makeEndpoint({ id: "ep-2" }),
    ]);

    expect(mockSql).not.toHaveBeenCalled();
  });

  it("does nothing when all rotations are within the 24h window", async () => {
    await expireRotatedSecrets("postgresql://test", [
      makeEndpoint({
        previousSigningSecret: "whsec_old",
        secretRotatedAt: new Date().toISOString(), // just now
      }),
    ]);

    expect(mockSql).not.toHaveBeenCalled();
  });

  it("issues UPDATE for endpoints past the 24h window", async () => {
    const expired = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();

    await expireRotatedSecrets("postgresql://test", [
      makeEndpoint({
        id: "ep-expired",
        previousSigningSecret: "whsec_old",
        secretRotatedAt: expired,
      }),
    ]);

    expect(mockSql).toHaveBeenCalledTimes(1);
    // Tagged template receives (strings[], ...values) — first value is the expiredIds array
    const callArgs = mockSql.mock.calls[0];
    expect(callArgs[0].join("")).toContain("previous_signing_secret");
    expect(callArgs[1]).toEqual(["ep-expired"]);
  });

  it("batches multiple expired endpoint IDs into a single query", async () => {
    const expired = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();

    await expireRotatedSecrets("postgresql://test", [
      makeEndpoint({ id: "ep-1", previousSigningSecret: "old1", secretRotatedAt: expired }),
      makeEndpoint({ id: "ep-2", previousSigningSecret: "old2", secretRotatedAt: expired }),
      makeEndpoint({ id: "ep-3" }), // no rotation — filtered out
    ]);

    expect(mockSql).toHaveBeenCalledTimes(1);
    const callArgs = mockSql.mock.calls[0];
    expect(callArgs[1]).toEqual(["ep-1", "ep-2"]);
  });

  it("boundary: exactly 24h elapsed is expired (>= not >)", async () => {
    const exactBoundary = new Date(Date.now() - 86_400 * 1000).toISOString();

    await expireRotatedSecrets("postgresql://test", [
      makeEndpoint({ id: "ep-boundary", previousSigningSecret: "old", secretRotatedAt: exactBoundary }),
    ]);

    expect(mockSql).toHaveBeenCalledTimes(1);
    const callArgs = mockSql.mock.calls[0];
    expect(callArgs[1]).toEqual(["ep-boundary"]);
  });

  it("throws on DB error (caller expected to .catch)", async () => {
    mockSql.mockRejectedValueOnce(new Error("connection refused"));

    const expired = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();

    await expect(
      expireRotatedSecrets("postgresql://test", [
        makeEndpoint({ id: "ep-1", previousSigningSecret: "old", secretRotatedAt: expired }),
      ]),
    ).rejects.toThrow();
  });
});
