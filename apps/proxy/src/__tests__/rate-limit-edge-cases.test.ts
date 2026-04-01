import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";

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

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

const mockAuthenticateRequest = vi.fn();
vi.mock("../lib/auth.js", () => ({
  authenticateRequest: (...args: unknown[]) => mockAuthenticateRequest(...args),
}));

vi.mock("../routes/openai.js", () => ({
  handleChatCompletions: vi.fn().mockResolvedValue(
    Response.json({ id: "test", choices: [] }, { status: 200 }),
  ),
}));

vi.mock("../routes/anthropic.js", () => ({
  handleAnthropicMessages: vi.fn().mockResolvedValue(
    Response.json({ id: "test", content: [] }, { status: 200 }),
  ),
}));

import handler from "../index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockIpLimit = vi.fn();
const mockKeyLimit = vi.fn();

function makeRequest(
  path: string,
  headers: Record<string, string> = {},
): Request {
  return new Request(`http://localhost${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-connecting-ip": "1.2.3.4",
      ...headers,
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
}

function makeEnv(): Env {
  return {
    HYPERDRIVE: { connectionString: "postgresql://localhost:5432/test" },
    IP_RATE_LIMITER: { limit: mockIpLimit },
    KEY_RATE_LIMITER: { limit: mockKeyLimit },
  } as unknown as Env;
}

function makeCtx(): ExecutionContext {
  return {
    waitUntil: vi.fn(),
    passThroughOnException: vi.fn(),
  } as unknown as ExecutionContext;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("per-key rate limiting edge cases", () => {
  beforeEach(() => {
    mockIpLimit.mockReset().mockResolvedValue({ success: true });
    mockKeyLimit.mockReset().mockResolvedValue({ success: true });
    mockAuthenticateRequest.mockReset().mockResolvedValue({
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: null,
      apiVersion: "2026-04-01",
      defaultTags: {},
    });
  });

  it("key header absent — only IP rate limit applied", async () => {
    const req = makeRequest("/v1/chat/completions");
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(mockIpLimit).toHaveBeenCalledOnce();
    expect(mockKeyLimit).not.toHaveBeenCalled();
    expect(res.status).toBe(200);
  });

  it("key header present — both IP and key rate limits applied", async () => {
    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": "key-123",
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(mockIpLimit).toHaveBeenCalledOnce();
    expect(mockKeyLimit).toHaveBeenCalledOnce();
    expect(mockKeyLimit).toHaveBeenCalledWith({ key: "key-123" });
    expect(res.status).toBe(200);
  });

  it("IP rate limit hit — returns 429 with Retry-After", async () => {
    mockIpLimit.mockResolvedValue({ success: false });

    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": "key-123",
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(res.status).toBe(429);

    const body = await res.json();
    expect(body.error.code).toBe("rate_limited");
    expect(res.headers.get("Retry-After")).toBe("60");
  });

  it("IP passes but key rate limit hit — returns 429", async () => {
    mockIpLimit.mockResolvedValue({ success: true });
    mockKeyLimit.mockResolvedValue({ success: false });

    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": "key-456",
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(res.status).toBe(429);
    expect(res.headers.get("Retry-After")).toBe("60");
  });

  it("key header exceeds 128 chars — key rate limit skipped", async () => {
    const longKeyId = "a".repeat(129);
    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": longKeyId,
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(mockKeyLimit).not.toHaveBeenCalled();
    expect(res.status).toBe(200);
  });

  it("key header exactly 128 chars — key rate limit applied", async () => {
    const exactKeyId = "a".repeat(128);
    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": exactKeyId,
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(mockKeyLimit).toHaveBeenCalledOnce();
    expect(mockKeyLimit).toHaveBeenCalledWith({ key: exactKeyId });
    expect(res.status).toBe(200);
  });

  it("empty key header — key rate limit skipped", async () => {
    const req = makeRequest("/v1/chat/completions", {
      "x-nullspend-key": "",
    });
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(mockKeyLimit).not.toHaveBeenCalled();
    expect(res.status).toBe(200);
  });

  it("rate limiter error — request proceeds (fail-open)", async () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockIpLimit.mockRejectedValue(new Error("Rate limiter binding error"));

    const req = makeRequest("/v1/chat/completions");
    const res = await handler.fetch(req, makeEnv(), makeCtx());

    expect(res.status).toBe(200);
    expect(consoleSpy).toHaveBeenCalled();

    consoleSpy.mockRestore();
  });
});
