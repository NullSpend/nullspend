/**
 * Adversarial stress tests for POST /internal/budget/invalidate
 *
 * Pushes auth boundary, body validation, concurrency, and request-level
 * edge cases that the happy-path test file does not cover.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

// ── Mocks (same pattern as internal-route.test.ts) ──────────────────────
const { mockDoBudgetRemove, mockDoBudgetResetSpend, mockEmitMetric } = vi.hoisted(() => ({
  mockDoBudgetRemove: vi.fn(),
  mockDoBudgetResetSpend: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetRemove: (...args: unknown[]) => mockDoBudgetRemove(...args),
  doBudgetResetSpend: (...args: unknown[]) => mockDoBudgetResetSpend(...args),
  doBudgetUpsertEntities: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../durable-objects/user-budget.js", () => ({}));

import { handleBudgetInvalidation } from "../routes/internal.js";

// ── Helpers ──────────────────────────────────────────────────────────────
function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    INTERNAL_SECRET: "test-secret-value",
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
    ...overrides,
  } as unknown as Env;
}

function makeRequest(options: {
  auth?: string;
  body?: unknown;
  rawBody?: string;
  headers?: Record<string, string>;
  omitContentType?: boolean;
} = {}): Request {
  const headers: Record<string, string> = {};
  if (!options.omitContentType) {
    headers["Content-Type"] = "application/json";
  }
  if (options.auth !== undefined) {
    headers["Authorization"] = options.auth;
  }
  if (options.headers) {
    Object.assign(headers, options.headers);
  }

  let bodyStr: string;
  if (options.rawBody !== undefined) {
    bodyStr = options.rawBody;
  } else if (options.body !== undefined) {
    bodyStr = JSON.stringify(options.body);
  } else {
    bodyStr = "{}";
  }

  return new Request("https://proxy.test/internal/budget/invalidate", {
    method: "POST",
    headers,
    body: bodyStr,
  });
}

const validBody = {
  action: "remove" as const,
  ownerId: "user-1",
  entityType: "api_key",
  entityId: "key-1",
};

// Polyfill timingSafeEqual for test environment
beforeAll(() => {
  if (!crypto.subtle.timingSafeEqual) {
    (crypto.subtle as Record<string, unknown>).timingSafeEqual = (a: ArrayBuffer, b: ArrayBuffer) => {
      const viewA = new Uint8Array(a);
      const viewB = new Uint8Array(b);
      if (viewA.length !== viewB.length) return false;
      let result = 0;
      for (let i = 0; i < viewA.length; i++) {
        result |= viewA[i] ^ viewB[i];
      }
      return result === 0;
    };
  }
});

beforeEach(() => {
  vi.clearAllMocks();
  mockDoBudgetRemove.mockResolvedValue(undefined);
  mockDoBudgetResetSpend.mockResolvedValue(undefined);
});

// =========================================================================
// 1. AUTH BOUNDARY ATTACKS
// =========================================================================
describe("Auth boundary attacks", () => {
  it("401: empty Bearer token ('Bearer ' with nothing after it)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer ", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
    const json = (await res.json()) as { error: { code: string } };
    expect(json.error.code).toBe("unauthorized");
  });

  it("FINDING: trailing whitespace on token is STRIPPED by HTTP Headers API — auth passes (should it?)", async () => {
    // The Headers API strips trailing whitespace from header values per the HTTP spec.
    // This means "Bearer test-secret-value   " becomes "Bearer test-secret-value"
    // and .slice(7) yields "test-secret-value" which matches the secret.
    // FINDING: An attacker who knows the secret can pad it with trailing whitespace
    // and still authenticate. Not exploitable on its own, but surprising behavior
    // that the handler has no control over.
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value   ", body: validBody }),
      makeEnv(),
    );
    // Documenting actual behavior: 200 because Headers strips trailing whitespace
    expect(res.status).toBe(200);
  });

  it("FINDING: trailing newline on token is STRIPPED by HTTP Headers API — auth passes", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value\n", body: validBody }),
      makeEnv(),
    );
    // Documenting actual behavior: 200 because Headers strips trailing whitespace/newlines
    expect(res.status).toBe(200);
  });

  it("FINDING: trailing \\r\\n on token is STRIPPED by HTTP Headers API — auth passes", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value\r\n", body: validBody }),
      makeEnv(),
    );
    // Documenting actual behavior: 200 because Headers strips trailing CRLF
    expect(res.status).toBe(200);
  });

  it("handles very long token (10KB+) without crashing", async () => {
    const longToken = "a".repeat(10240);
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: `Bearer ${longToken}`, body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("FINDING: token with null bytes — Request constructor rejects (HTTP spec)", async () => {
    // The HTTP spec forbids null bytes in header values.
    // This means the Request constructor throws before the handler ever runs.
    // In production, the CF runtime would reject the request at the HTTP layer.
    // FINDING: Not reachable in the handler — but if someone bypasses the HTTP layer
    // (e.g., direct function call), there's no null-byte sanitization in the handler itself.
    expect(() =>
      makeRequest({ auth: "Bearer test-secret\x00-value", body: validBody }),
    ).toThrow();
  });

  it("401: case sensitivity — lowercase 'bearer' prefix", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "bearer test-secret-value", body: validBody }),
      makeEnv(),
    );
    // `startsWith("Bearer ")` is case-sensitive, so "bearer " should fail
    expect(res.status).toBe(401);
  });

  it("401: case sensitivity — uppercase 'BEARER' prefix", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "BEARER test-secret-value", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: mixed case 'BeArEr' prefix", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "BeArEr test-secret-value", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: 'Bearer' with no space (token jammed together)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearertest-secret-value", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: 'Bearer' with multiple spaces before token", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer  test-secret-value", body: validBody }),
      makeEnv(),
    );
    // Token extracted would be " test-secret-value" (starts with space) — should not match
    expect(res.status).toBe(401);
  });

  it("handles token that is an exact prefix of the secret", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("handles token that is the secret with extra appended chars", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value-extra", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: Authorization header is just whitespace", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "   ", body: validBody }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401: Authorization header is empty string", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "", body: validBody }),
      makeEnv(),
    );
    // Empty string — should be treated as missing/malformed
    expect(res.status).toBe(401);
  });

  it("500: INTERNAL_SECRET is undefined (not just empty)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer test-secret-value", body: validBody }),
      makeEnv({ INTERNAL_SECRET: undefined }),
    );
    expect(res.status).toBe(500);
  });

  it("auth bypass: token matches when INTERNAL_SECRET is falsy empty string and token is empty", async () => {
    // If INTERNAL_SECRET is "" and token is "" — does the timing-safe comparison pass?
    // The code checks `if (!env.INTERNAL_SECRET)` first, so it should return 500 before reaching comparison.
    vi.spyOn(console, "error").mockImplementation(() => {});
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer ", body: validBody }),
      makeEnv({ INTERNAL_SECRET: "" }),
    );
    // Should be 500 (misconfigured), not 200 (auth bypass)
    expect(res.status).toBe(500);
    expect(mockDoBudgetRemove).not.toHaveBeenCalled();
  });
});

// =========================================================================
// 2. BODY VALIDATION ATTACKS
// =========================================================================
describe("Body validation attacks", () => {
  const auth = "Bearer test-secret-value";

  describe("empty string fields", () => {
    it("400: ownerId is empty string", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: "" } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: entityType is empty string", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityType: "" } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: entityId is empty string", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityId: "" } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: action is empty string", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, action: "" } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("type confusion — numeric values where strings expected", () => {
    it("400: ownerId is a number", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: 12345 } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: entityType is a number", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityType: 0 } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: entityId is a number", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityId: 999 } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("type confusion — nested objects where strings expected", () => {
    it("400: ownerId is an object", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: { nested: "val" } } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: entityType is an object", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityType: { $gt: "" } } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("type confusion — arrays where strings expected", () => {
    it("400: ownerId is an array", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: ["user-1", "user-2"] } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: action is an array", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, action: ["remove", "reset_spend"] } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("boolean values where strings expected", () => {
    it("400: ownerId is true", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: true } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: ownerId is null", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: null } }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("very long string fields (100KB)", () => {
    const longStr = "x".repeat(100 * 1024);

    it("accepts or rejects 100KB ownerId (no crash, no memory bomb to downstream)", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: longStr } }),
        makeEnv(),
      );
      // The code doesn't enforce length limits — it will pass validation and call DO with a 100KB ownerId.
      // This test documents the behavior. If it returns 200, that's a finding (no length limit).
      const status = res.status;
      // Record what actually happens
      if (status === 200) {
        expect(mockDoBudgetRemove).toHaveBeenCalledWith(
          expect.anything(),
          longStr,
          expect.any(String),
          expect.any(String),
        );
        // FINDING: 100KB ownerId is accepted and forwarded to DO
      }
      expect([200, 400, 413]).toContain(status);
    });

    it("accepts or rejects 100KB entityId (no crash)", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityId: longStr } }),
        makeEnv(),
      );
      const status = res.status;
      expect([200, 400, 413]).toContain(status);
    });
  });

  describe("extra unexpected fields", () => {
    it("200: extra fields are silently ignored (no strict validation)", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, evil: "payload", admin: true, role: "superadmin" },
        }),
        makeEnv(),
      );
      // parseBody only picks the 4 known fields — extras are ignored
      expect(res.status).toBe(200);
    });
  });

  describe("unicode/emoji in entity fields", () => {
    it("200: emoji in entityType (accepted as valid string)", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityType: "api_key_\u{1F4A3}" } }),
        makeEnv(),
      );
      // No validation of entityType format — just typeof string && truthy
      expect(res.status).toBe(200);
      expect(mockDoBudgetRemove).toHaveBeenCalledWith(
        expect.anything(),
        "user-1",
        "api_key_\u{1F4A3}",
        "key-1",
      );
    });

    it("200: unicode RTL override in entityId", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityId: "key-\u202E-1" } }),
        makeEnv(),
      );
      expect(res.status).toBe(200);
    });

    it("200: zero-width characters in ownerId", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: "user\u200B-\u200B1" } }),
        makeEnv(),
      );
      // Zero-width chars make the string truthy — parseBody accepts it
      expect(res.status).toBe(200);
    });
  });

  describe("SQL injection attempts", () => {
    it("200: SQL injection in entityType (passed raw to DO)", async () => {
      const sqlInjection = "'; DROP TABLE budgets; --";
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, entityType: sqlInjection } }),
        makeEnv(),
      );
      // No sanitization — raw value forwarded to DO
      expect(res.status).toBe(200);
      expect(mockDoBudgetRemove).toHaveBeenCalledWith(
        expect.anything(),
        "user-1",
        sqlInjection,
        "key-1",
      );
    });

    it("200: SQL injection in entityId", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, entityId: "1 OR 1=1" },
        }),
        makeEnv(),
      );
      expect(res.status).toBe(200);
    });

    it("200: SQL injection in ownerId", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, ownerId: "user-1'; DELETE FROM users WHERE '1'='1" },
        }),
        makeEnv(),
      );
      expect(res.status).toBe(200);
    });
  });

  describe("prototype pollution", () => {
    it("__proto__ field is ignored (parseBody extracts only known fields)", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          rawBody: JSON.stringify({
            ...validBody,
            __proto__: { isAdmin: true },
          }),
        }),
        makeEnv(),
      );
      expect(res.status).toBe(200);
    });

    it("constructor field is ignored", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, constructor: { prototype: { isAdmin: true } } },
        }),
        makeEnv(),
      );
      expect(res.status).toBe(200);
    });

    it("400: action set to 'constructor' is rejected", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, action: "constructor" },
        }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: action set to '__proto__' is rejected", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({
          auth,
          body: { ...validBody, action: "__proto__" },
        }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("body is non-object JSON", () => {
    it("400: body is a JSON string", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, rawBody: '"just a string"' }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: body is a JSON number", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, rawBody: "42" }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: body is a JSON array", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, rawBody: JSON.stringify([validBody]) }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: body is JSON null", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, rawBody: "null" }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });

    it("400: body is JSON true", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, rawBody: "true" }),
        makeEnv(),
      );
      expect(res.status).toBe(400);
    });
  });

  describe("whitespace-only string fields", () => {
    it("200 or 400: ownerId is whitespace only", async () => {
      const res = await handleBudgetInvalidation(
        makeRequest({ auth, body: { ...validBody, ownerId: "   " } }),
        makeEnv(),
      );
      // "   " is a truthy string, so parseBody accepts it.
      // FINDING if 200: whitespace-only ownerId is accepted.
      const status = res.status;
      if (status === 200) {
        expect(mockDoBudgetRemove).toHaveBeenCalledWith(
          expect.anything(), "   ", "api_key", "key-1",
        );
      }
      expect([200, 400]).toContain(status);
    });
  });
});

// =========================================================================
// 3. CONCURRENCY
// =========================================================================
describe("Concurrency", () => {
  it("100 concurrent valid requests all succeed", async () => {
    const env = makeEnv();
    const promises = Array.from({ length: 100 }, (_, i) =>
      handleBudgetInvalidation(
        makeRequest({
          auth: "Bearer test-secret-value",
          body: { ...validBody, ownerId: `user-${i}` },
        }),
        env,
      ),
    );

    const results = await Promise.all(promises);
    const statuses = results.map((r) => r.status);
    expect(statuses.every((s) => s === 200)).toBe(true);
    expect(mockDoBudgetRemove).toHaveBeenCalledTimes(100);
  });

  it("mix of valid and invalid concurrent requests are correctly segregated", async () => {
    const env = makeEnv();
    const requests = [
      // 5 valid
      ...Array.from({ length: 5 }, (_, i) =>
        makeRequest({ auth: "Bearer test-secret-value", body: { ...validBody, ownerId: `valid-${i}` } }),
      ),
      // 5 wrong token
      ...Array.from({ length: 5 }, () =>
        makeRequest({ auth: "Bearer wrong", body: validBody }),
      ),
      // 5 bad body
      ...Array.from({ length: 5 }, () =>
        makeRequest({ auth: "Bearer test-secret-value", body: { action: "invalid" } }),
      ),
      // 5 missing auth
      ...Array.from({ length: 5 }, () =>
        makeRequest({ body: validBody }),
      ),
    ];

    const results = await Promise.all(
      requests.map((req) => handleBudgetInvalidation(req, env)),
    );

    const statuses = results.map((r) => r.status);
    // First 5 should be 200
    expect(statuses.slice(0, 5).every((s) => s === 200)).toBe(true);
    // Next 5 should be 401
    expect(statuses.slice(5, 10).every((s) => s === 401)).toBe(true);
    // Next 5 should be 400
    expect(statuses.slice(10, 15).every((s) => s === 400)).toBe(true);
    // Last 5 should be 401
    expect(statuses.slice(15, 20).every((s) => s === 401)).toBe(true);

    expect(mockDoBudgetRemove).toHaveBeenCalledTimes(5);
  });

  it("concurrent requests where DO throws intermittently", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    let callCount = 0;
    mockDoBudgetRemove.mockImplementation(() => {
      callCount++;
      if (callCount % 3 === 0) return Promise.reject(new Error("DO overloaded"));
      return Promise.resolve(undefined);
    });

    const env = makeEnv();
    const promises = Array.from({ length: 30 }, (_, i) =>
      handleBudgetInvalidation(
        makeRequest({
          auth: "Bearer test-secret-value",
          body: { ...validBody, ownerId: `user-${i}` },
        }),
        env,
      ),
    );

    const results = await Promise.all(promises);
    const statuses = results.map((r) => r.status);
    const successes = statuses.filter((s) => s === 200).length;
    const failures = statuses.filter((s) => s === 500).length;
    expect(successes).toBe(20);
    expect(failures).toBe(10);
  });
});

// =========================================================================
// 4. REQUEST BODY EDGE CASES
// =========================================================================
describe("Request body edge cases", () => {
  const auth = "Bearer test-secret-value";

  it("handles very large JSON body (>1MB) without crashing", async () => {
    // Build a valid body with a 1.5MB ownerId
    const bigUserId = "u".repeat(1.5 * 1024 * 1024);
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, ownerId: bigUserId } }),
      makeEnv(),
    );
    // No body size limit in the handler — it relies on upstream CF limits.
    // In unit tests the request.json() will succeed.
    const status = res.status;
    expect([200, 400, 413]).toContain(status);
    if (status === 200) {
      // FINDING: 1.5MB ownerId accepted
      expect(mockDoBudgetRemove).toHaveBeenCalled();
    }
  });

  it("400: Content-Type is text/plain with JSON body", async () => {
    const req = new Request("https://proxy.test/internal/budget/invalidate", {
      method: "POST",
      headers: {
        "Content-Type": "text/plain",
        Authorization: auth,
      },
      body: JSON.stringify(validBody),
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    // request.json() may or may not fail depending on runtime.
    // In Workers, request.json() parses regardless of Content-Type.
    // In Node/vitest, it usually also works. Record actual behavior.
    const status = res.status;
    // If the runtime parses it anyway, we get 200 (no Content-Type enforcement).
    // If it throws, we get 400.
    expect([200, 400]).toContain(status);
  });

  it("handles missing Content-Type header with JSON body", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: validBody, omitContentType: true }),
      makeEnv(),
    );
    // request.json() doesn't check Content-Type — it just tries to parse.
    const status = res.status;
    expect([200, 400]).toContain(status);
  });

  it("400: completely empty body", async () => {
    const req = new Request("https://proxy.test/internal/budget/invalidate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: auth,
      },
      body: "",
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    // Empty string will fail JSON.parse → 400
    expect(res.status).toBe(400);
  });

  it("400: body is just whitespace", async () => {
    const req = new Request("https://proxy.test/internal/budget/invalidate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: auth,
      },
      body: "   \n\t  ",
    });

    const res = await handleBudgetInvalidation(req, makeEnv());
    // Whitespace fails JSON.parse → 400
    expect(res.status).toBe(400);
  });

  it("handles duplicate JSON keys (last wins in standard JSON.parse)", async () => {
    // JSON with duplicate "action" key — standard JSON.parse takes the last one
    const rawBody = '{"action":"remove","ownerId":"user-1","entityType":"api_key","entityId":"key-1","action":"reset_spend"}';
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, rawBody }),
      makeEnv(),
    );
    // Last "action" wins → "reset_spend"
    expect(res.status).toBe(200);
    expect(mockDoBudgetResetSpend).toHaveBeenCalled();
    expect(mockDoBudgetRemove).not.toHaveBeenCalled();
  });

  it("400: body with BOM (byte order mark) prefix", async () => {
    const rawBody = "\uFEFF" + JSON.stringify(validBody);
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, rawBody }),
      makeEnv(),
    );
    // Some JSON parsers handle BOM, some don't. Record behavior.
    const status = res.status;
    expect([200, 400]).toContain(status);
  });

  it("400: body with trailing comma (invalid JSON)", async () => {
    const rawBody = '{"action":"remove","ownerId":"user-1","entityType":"api_key","entityId":"key-1",}';
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, rawBody }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: body with comments (invalid JSON)", async () => {
    const rawBody = '{"action":"remove"/* comment */,"ownerId":"user-1","entityType":"api_key","entityId":"key-1"}';
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, rawBody }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: body with single quotes (invalid JSON)", async () => {
    const rawBody = "{'action':'remove','ownerId':'user-1','entityType':'api_key','entityId':'key-1'}";
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, rawBody }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("handles deeply nested object in a field (no stack overflow)", async () => {
    // Create a deeply nested object
    let nested: unknown = "leaf";
    for (let i = 0; i < 100; i++) {
      nested = { deeper: nested };
    }
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, entityId: nested } }),
      makeEnv(),
    );
    // entityId is an object → typeof !== "string" → 400
    expect(res.status).toBe(400);
  });
});

// =========================================================================
// 5. ACTION FIELD EDGE CASES
// =========================================================================
describe("Action field edge cases", () => {
  const auth = "Bearer test-secret-value";

  it("400: action is 'Remove' (wrong case)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, action: "Remove" } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: action is 'REMOVE' (uppercase)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, action: "REMOVE" } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: action is 'reset-spend' (hyphen instead of underscore)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, action: "reset-spend" } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: action is 'remove ' (with trailing space)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, action: "remove " } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });

  it("400: action is ' remove' (with leading space)", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth, body: { ...validBody, action: " remove" } }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  });
});

// =========================================================================
// 6. TIMING-SAFE COMPARISON EDGE CASES
// =========================================================================
describe("Timing-safe comparison edge cases", () => {
  it("FINDING: unicode emoji in INTERNAL_SECRET — Request constructor rejects (not ByteString-safe)", async () => {
    // HTTP headers can only contain ISO-8859-1 characters (ByteString).
    // Emoji (multi-byte) causes the Request constructor to throw.
    // FINDING: If INTERNAL_SECRET is set to a value with emoji/multi-byte chars,
    // the endpoint becomes permanently unreachable — no one can authenticate.
    const unicodeSecret = "secret-\u{1F600}-value";
    expect(() =>
      makeRequest({ auth: `Bearer ${unicodeSecret}`, body: validBody }),
    ).toThrow();
  });

  it("FINDING: null byte in token — Request constructor rejects", async () => {
    // HTTP headers forbid null bytes. Request constructor throws.
    expect(() =>
      makeRequest({ auth: "Bearer test\x00secret", body: validBody }),
    ).toThrow();
  });

  it("handles single character INTERNAL_SECRET", async () => {
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: "Bearer x", body: validBody }),
      makeEnv({ INTERNAL_SECRET: "x" }),
    );
    expect(res.status).toBe(200);
  });

  it("handles very long INTERNAL_SECRET (10KB)", async () => {
    const longSecret = "s".repeat(10240);
    const res = await handleBudgetInvalidation(
      makeRequest({ auth: `Bearer ${longSecret}`, body: validBody }),
      makeEnv({ INTERNAL_SECRET: longSecret }),
    );
    expect(res.status).toBe(200);
  });
});
