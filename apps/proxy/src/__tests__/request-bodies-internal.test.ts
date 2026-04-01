import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn((p: Promise<unknown>) => { p.catch(() => {}); }),
}));

// Mock body-storage
const mockRetrieveBodies = vi.fn();
vi.mock("../lib/body-storage.js", () => ({
  retrieveBodies: (...args: unknown[]) => mockRetrieveBodies(...args),
}));

// Mock timing-safe-equal
vi.mock("../lib/timing-safe-equal.js", () => ({
  timingSafeStringEqual: (a: string, b: string) => a === b,
}));

import { handleRequestBodies } from "../routes/internal.js";

// Polyfill crypto.subtle.timingSafeEqual for test env
beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as Record<string, unknown>).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.length !== viewB.length) return false;
      return viewA.every((v, i) => v === viewB[i]);
    };
  }
});

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    INTERNAL_SECRET: "test-secret",
    HYPERDRIVE: { connectionString: "postgresql://localhost:5432/test" },
    BODY_STORAGE: { get: vi.fn().mockResolvedValue(null), put: vi.fn() },
    ...overrides,
  } as unknown as Env;
}

function makeRequest(path: string, headers: Record<string, string> = {}): Request {
  return new Request(`https://proxy.test${path}`, {
    method: "GET",
    headers: {
      Authorization: "Bearer test-secret",
      ...headers,
    },
  });
}

describe("handleRequestBodies", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 401 when no auth header", async () => {
    const req = new Request("https://proxy.test/internal/request-bodies/req-123?ownerId=org_1", {
      method: "GET",
    });
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 401 when auth token is wrong", async () => {
    const req = new Request("https://proxy.test/internal/request-bodies/req-123?ownerId=org_1", {
      method: "GET",
      headers: { Authorization: "Bearer wrong-secret" },
    });
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(401);
  });

  it("returns 500 when INTERNAL_SECRET is not configured", async () => {
    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv({ INTERNAL_SECRET: undefined }));
    expect(res.status).toBe(500);
  });

  it("returns 400 when ownerId is missing", async () => {
    const req = makeRequest("/internal/request-bodies/req-123");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(400);
    const body = await res.json() as { error: { code: string } };
    expect(body.error.code).toBe("bad_request");
  });

  it("returns 500 when BODY_STORAGE is not configured", async () => {
    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const env = makeEnv();
    delete (env as Record<string, unknown>).BODY_STORAGE;
    const res = await handleRequestBodies(req, env);
    expect(res.status).toBe(500);
    const body = await res.json() as { error: { code: string } };
    expect(body.error.code).toBe("internal_error");
  });

  it("returns stored bodies on success", async () => {
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: '{"model":"gpt-4"}',
      responseBody: '{"choices":[{"message":{"content":"Hello"}}]}',
      responseFormat: "json",
    });

    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(200);

    const body = await res.json() as { requestBody: unknown; responseBody: unknown };
    expect(body.requestBody).toEqual({ model: "gpt-4" });
    expect(body.responseBody).toEqual({ choices: [{ message: { content: "Hello" } }] });
  });

  it("returns SSE body wrapped with _format when responseFormat is sse", async () => {
    const sseText = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\n";
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: '{"model":"gpt-4","stream":true}',
      responseBody: sseText,
      responseFormat: "sse",
    });

    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(200);

    const body = await res.json() as { requestBody: unknown; responseBody: unknown };
    expect(body.requestBody).toEqual({ model: "gpt-4", stream: true });
    expect(body.responseBody).toEqual({ _format: "sse", text: sseText });
  });

  it("returns nulls when no bodies stored", async () => {
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: null,
      responseBody: null,
      responseFormat: null,
    });

    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(200);

    const body = await res.json() as { requestBody: unknown; responseBody: unknown };
    expect(body.requestBody).toBeNull();
    expect(body.responseBody).toBeNull();
  });

  it("returns 500 on R2 retrieval failure", async () => {
    mockRetrieveBodies.mockRejectedValueOnce(new Error("R2 down"));

    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(500);
  });

  it("passes correct ownerId and requestId to retrieveBodies", async () => {
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: null,
      responseBody: null,
    });

    const req = makeRequest("/internal/request-bodies/my-req-id?ownerId=org_456");
    await handleRequestBodies(req, makeEnv());

    expect(mockRetrieveBodies).toHaveBeenCalledWith(
      expect.anything(),
      "org_456",
      "my-req-id",
    );
  });

  it("returns null for corrupt JSON in one body without failing the other", async () => {
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: '{"model":"gpt-4"}',
      responseBody: "NOT_VALID_JSON{{{",
    });

    const req = makeRequest("/internal/request-bodies/req-123?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(200);

    const body = await res.json() as { requestBody: unknown; responseBody: unknown };
    expect(body.requestBody).toEqual({ model: "gpt-4" });
    expect(body.responseBody).toBeNull();
  });

  it("returns 400 for ownerId with path traversal characters", async () => {
    const req = makeRequest("/internal/request-bodies/req-123?ownerId=../../etc");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("returns 400 for requestId with slash characters", async () => {
    // Note: ../../ is resolved by URL parser, so test a literal slash
    const req = makeRequest("/internal/request-bodies/req%2F..%2F..%2Fsecret?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(400);
  });

  it("accepts requestId with dots and colons (common in UUIDs and timestamps)", async () => {
    mockRetrieveBodies.mockResolvedValueOnce({
      requestBody: null,
      responseBody: null,
    });

    const req = makeRequest("/internal/request-bodies/req-2026-03-25T10:30:00.000Z?ownerId=org_1");
    const res = await handleRequestBodies(req, makeEnv());
    expect(res.status).toBe(200);
  });
});
