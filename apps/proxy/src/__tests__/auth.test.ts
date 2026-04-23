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
      // PR-2a fields — mock returns them so the wrapper forwards them (per codex PR-2a-N2)
      planLimitBlockAt: 100_000,
      planLimitMode: "hard",
      tierLabel: "free",
      subscriptionPeriodStart: null,
      subscriptionPeriodEnd: null,
    });

    const request = new Request("http://localhost/v1/chat/completions", {
      headers: { "x-nullspend-key": "ns_live_sk_valid_key" },
    });

    const result = await authenticateRequest(request, { NULLSPEND_CLOUD: "true" }, "postgresql://localhost");
    expect(result).toMatchObject({
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: null,
      apiVersion: "2026-04-01",
      defaultTags: { project: "alpha" },
      // Assert PR-2a fields made it through the wrapper — closes the Decision #37 / codex N2 gap
      planLimitBlockAt: 100_000,
      planLimitMode: "hard",
      tierLabel: "free",
      subscriptionPeriodStart: null,
      subscriptionPeriodEnd: null,
    });
  });

  it("forwards env to authenticateApiKey (self-hosted bypass via real fetch path — codex PR-2a-N2)", async () => {
    mockAuthenticateApiKey.mockResolvedValue({
      userId: "user-sh",
      keyId: "key-sh",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: "org-sh",
      apiVersion: "2026-04-01",
      defaultTags: {},
      planLimitBlockAt: null,
      planLimitMode: "soft",
      tierLabel: "enterprise",
      subscriptionPeriodStart: null,
      subscriptionPeriodEnd: null,
      requestLoggingEnabled: true,
    });
    const request = new Request("http://localhost/v1/chat/completions", {
      headers: { "x-nullspend-key": "ns_live_sk_sh" },
    });
    const env = { NULLSPEND_CLOUD: "false" };

    await authenticateRequest(request, env, "postgresql://localhost");

    // Assert the env object was passed through to authenticateApiKey so the
    // self-hosted short-circuit can actually activate in prod.
    expect(mockAuthenticateApiKey).toHaveBeenCalledWith(
      "ns_live_sk_sh",
      "postgresql://localhost",
      env,
    );
  });

  it("returns null when x-nullspend-key header is missing", async () => {
    const request = new Request("http://localhost/v1/chat/completions");

    const result = await authenticateRequest(request, { NULLSPEND_CLOUD: "true" }, "postgresql://localhost");
    expect(result).toBeNull();
    expect(mockAuthenticateApiKey).not.toHaveBeenCalled();
  });

  it("returns null when API key is invalid", async () => {
    mockAuthenticateApiKey.mockResolvedValue(null);

    const request = new Request("http://localhost/v1/chat/completions", {
      headers: { "x-nullspend-key": "ns_live_sk_invalid_key" },
    });

    const result = await authenticateRequest(request, { NULLSPEND_CLOUD: "true" }, "postgresql://localhost");
    expect(result).toBeNull();
  });

});
