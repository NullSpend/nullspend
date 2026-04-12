import { describe, it, expect, vi } from "vitest";

const mockAuthenticateApiKey = vi.fn();
vi.mock("../lib/api-key-auth.js", () => ({
  authenticateApiKey: (...args: unknown[]) => mockAuthenticateApiKey(...args),
}));

import { authenticateRequest } from "../lib/auth.js";
import { beforeEach } from "vitest";

describe("authenticateRequest", () => {
  beforeEach(() => {
    mockAuthenticateApiKey.mockReset();
  });

  it("returns identity for valid API key", async () => {
    mockAuthenticateApiKey.mockResolvedValue({
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: null,
      apiVersion: "2026-04-01",
      defaultTags: { project: "alpha" },
    });

    const request = new Request("http://localhost/v1/chat/completions", {
      headers: { "x-nullspend-key": "ns_live_sk_valid_key" },
    });

    const result = await authenticateRequest(request, "postgresql://localhost");
    expect(result).toEqual({
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: null,
      apiVersion: "2026-04-01",
      defaultTags: { project: "alpha" },
    });
  });

  it("returns null when x-nullspend-key header is missing", async () => {
    const request = new Request("http://localhost/v1/chat/completions");

    const result = await authenticateRequest(request, "postgresql://localhost");
    expect(result).toBeNull();
    expect(mockAuthenticateApiKey).not.toHaveBeenCalled();
  });

  it("returns null when API key is invalid", async () => {
    mockAuthenticateApiKey.mockResolvedValue(null);

    const request = new Request("http://localhost/v1/chat/completions", {
      headers: { "x-nullspend-key": "ns_live_sk_invalid_key" },
    });

    const result = await authenticateRequest(request, "postgresql://localhost");
    expect(result).toBeNull();
  });

});
