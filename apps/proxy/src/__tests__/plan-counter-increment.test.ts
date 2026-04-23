/**
 * PR-2d tests for POST /internal/plan-counter/increment.
 *
 * C33 / C33b / C34 / C35 / C36 / C37 / C38 / C58 / C58b / C58c / C58d.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

const { mockAuthenticateApiKey, mockDoIncrementPlanCounter, mockEmitMetric } = vi.hoisted(() => ({
  mockAuthenticateApiKey: vi.fn(),
  mockDoIncrementPlanCounter: vi.fn(),
  mockEmitMetric: vi.fn(),
}));

vi.mock("../lib/api-key-auth.js", () => ({
  authenticateApiKey: (...args: unknown[]) => mockAuthenticateApiKey(...args),
  invalidateAuthCacheForOwner: vi.fn(),
}));

vi.mock("../lib/budget-do-client.js", () => ({
  doBudgetRemove: vi.fn(),
  doBudgetResetSpend: vi.fn(),
  doBudgetUpsertEntities: vi.fn(),
  doBudgetGetVelocityState: vi.fn(),
  doIncrementPlanCounter: (...args: unknown[]) => mockDoIncrementPlanCounter(...args),
}));

vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

vi.mock("../durable-objects/user-budget.js", () => ({}));

import { handlePlanCounterIncrement } from "../routes/internal.js";
import type { ApiKeyIdentity } from "../lib/api-key-auth.js";

const ORG_ID = "11111111-2222-3333-4444-555555555555";

function makeIdentity(overrides: Partial<ApiKeyIdentity> = {}): ApiKeyIdentity {
  return {
    userId: "user-abc",
    orgId: ORG_ID,
    keyId: "key-123",
    hasWebhooks: false,
    hasBudgets: true,
    requestLoggingEnabled: false,
    apiVersion: "2026-03-01",
    defaultTags: {},
    allowedModels: null,
    allowedProviders: null,
    allowedCustomers: null,
    requireCustomerId: false,
    orgUpgradeUrl: null,
    planLimitBlockAt: 100_000,
    planLimitMode: "hard",
    tierLabel: "free",
    subscriptionPeriodStart: null,
    subscriptionPeriodEnd: null,
    ...overrides,
  };
}

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    INTERNAL_SECRET: "test-secret-value",
    HYPERDRIVE: { connectionString: "postgres://test:test@localhost:5432/test" },
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
} = {}): Request {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.auth !== undefined) headers["Authorization"] = options.auth;

  return new Request("https://proxy.test/internal/plan-counter/increment", {
    method: "POST",
    headers,
    body: options.rawBody ?? (options.body !== undefined ? JSON.stringify(options.body) : "{}"),
  });
}

const VALID_AUTH = "Bearer test-secret-value";
const VALID_BODY = { apiKey: "ns_test_key_raw", idempotencyKeys: ["abc"] };

async function sha256Hex(input: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
}

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
  mockAuthenticateApiKey.mockResolvedValue(makeIdentity());
  // Default DO behavior: return approved w/ count=1. Tests that care about
  // dedup / distinct-key math install their own stateful mock.
  mockDoIncrementPlanCounter.mockResolvedValue({ status: "approved", count: 1 });
});

// ---------------------------------------------------------------------------
// C33 — auth
// ---------------------------------------------------------------------------

describe("C33: auth", () => {
  it("401 without Authorization header", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401 with wrong Bearer token", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Bearer wrong-secret", body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("401 with malformed Authorization (no Bearer prefix)", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Basic test-secret-value", body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
  });

  it("500 when neither INTERNAL_SECRET nor INTERNAL_SECRET_NEXT is configured", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv({ INTERNAL_SECRET: "" }),
    );
    expect(res.status).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// C33b / C34 — body validation
// ---------------------------------------------------------------------------

describe("C33b / C34: body validation", () => {
  const bad = async (body: unknown) => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body }),
      makeEnv(),
    );
    expect(res.status).toBe(400);
  };

  it("400 when apiKey is missing", async () => {
    await bad({ idempotencyKeys: ["k"] });
  });

  it("400 when apiKey is non-string", async () => {
    await bad({ apiKey: 42, idempotencyKeys: ["k"] });
  });

  it("400 when apiKey is empty string", async () => {
    await bad({ apiKey: "", idempotencyKeys: ["k"] });
  });

  it("400 when apiKey exceeds 256 chars", async () => {
    await bad({ apiKey: "a".repeat(257), idempotencyKeys: ["k"] });
  });

  it("400 when idempotencyKeys is missing", async () => {
    await bad({ apiKey: "k" });
  });

  it("400 when idempotencyKeys is not an array", async () => {
    await bad({ apiKey: "k", idempotencyKeys: "abc" });
  });

  it("400 when idempotencyKeys is empty array", async () => {
    await bad({ apiKey: "k", idempotencyKeys: [] });
  });

  it("400 when idempotencyKeys exceeds 100 entries", async () => {
    await bad({ apiKey: "k", idempotencyKeys: new Array(101).fill("k") });
  });

  it("400 when any idempotencyKey is non-string", async () => {
    await bad({ apiKey: "k", idempotencyKeys: ["ok", 42, "also-ok"] });
  });

  it("400 when any idempotencyKey is empty string", async () => {
    await bad({ apiKey: "k", idempotencyKeys: ["ok", ""] });
  });

  it("400 when any idempotencyKey exceeds 256 chars", async () => {
    await bad({ apiKey: "k", idempotencyKeys: ["a".repeat(257)] });
  });

  it("400 when body is JSON array instead of object", async () => {
    await bad([{ apiKey: "k", idempotencyKeys: ["x"] }]);
  });

  it("400 for invalid JSON", async () => {
    const req = new Request("https://proxy.test/internal/plan-counter/increment", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: VALID_AUTH },
      body: "not-json{{",
    });
    const res = await handlePlanCounterIncrement(req, makeEnv());
    expect(res.status).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// C35 — happy path
// ---------------------------------------------------------------------------

describe("C35: happy path", () => {
  it("returns { status, count } for valid single-key request", async () => {
    mockDoIncrementPlanCounter.mockResolvedValueOnce({ status: "approved", count: 42 });

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "ns_test", idempotencyKeys: ["abc"] } }),
      makeEnv(),
    );
    expect(res.status).toBe(200);
    const json = (await res.json()) as { status: string; count: number };
    expect(json.status).toBe("approved");
    expect(json.count).toBe(42);

    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(1);
  });

  it("returns 401 when apiKey is unknown", async () => {
    mockAuthenticateApiKey.mockResolvedValueOnce(null);

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(401);
    expect(mockDoIncrementPlanCounter).not.toHaveBeenCalled();
  });

  it("returns 503 when DB lookup throws", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockAuthenticateApiKey.mockRejectedValueOnce(new Error("DB down"));

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(503);
  });

  it("returns 500 when DO increment throws", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockDoIncrementPlanCounter.mockRejectedValueOnce(new Error("DO unavailable"));

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(res.status).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// C36 — apiKey-scoped: identity resolved server-side, body has NO orgId
// ---------------------------------------------------------------------------

describe("C36: apiKey-scoped authorization", () => {
  it("DO increment uses the resolved identity's orgId, NOT a caller-supplied value", async () => {
    const victimOrg = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    mockAuthenticateApiKey.mockResolvedValueOnce(makeIdentity({ orgId: victimOrg }));

    // Attacker attempts to smuggle orgId; endpoint ignores unknown fields and
    // passes the identity's orgId to the DO client.
    const attackerBody = {
      apiKey: "ns_leaked_key",
      idempotencyKeys: ["attack"],
      orgId: "ffffffff-0000-0000-0000-000000000000",  // ignored by endpoint
    };

    await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: attackerBody }),
      makeEnv(),
    );

    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(1);
    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    expect((opts as { orgId: string }).orgId).toBe(victimOrg);
    expect((opts as { orgId: string }).orgId).not.toBe(attackerBody.orgId);
  });

  it("DO increment carries identity's planLimitBlockAt/Mode/tier (not caller-supplied)", async () => {
    mockAuthenticateApiKey.mockResolvedValueOnce(
      makeIdentity({ planLimitBlockAt: 500_000, planLimitMode: "soft", tierLabel: "pro" }),
    );

    await handlePlanCounterIncrement(
      makeRequest({
        auth: VALID_AUTH,
        body: {
          apiKey: "ns_k",
          idempotencyKeys: ["k"],
          planLimitBlockAt: 1,  // smuggling attempt
          planLimitMode: "hard",
          tier: "enterprise",
        },
      }),
      makeEnv(),
    );

    const [, opts] = mockDoIncrementPlanCounter.mock.calls[0];
    expect(opts).toMatchObject({
      planLimitBlockAt: 500_000,
      planLimitMode: "soft",
      tier: "pro",
    });
  });
});

// ---------------------------------------------------------------------------
// C37 — dual-secret rotation
// ---------------------------------------------------------------------------

describe("C37: dual-secret rotation", () => {
  it("accepts INTERNAL_SECRET", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Bearer current-secret", body: VALID_BODY }),
      makeEnv({ INTERNAL_SECRET: "current-secret", INTERNAL_SECRET_NEXT: "next-secret" }),
    );
    expect(res.status).toBe(200);
  });

  it("accepts INTERNAL_SECRET_NEXT", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Bearer next-secret", body: VALID_BODY }),
      makeEnv({ INTERNAL_SECRET: "current-secret", INTERNAL_SECRET_NEXT: "next-secret" }),
    );
    expect(res.status).toBe(200);
  });

  it("rejects a token matching neither secret", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Bearer neither", body: VALID_BODY }),
      makeEnv({ INTERNAL_SECRET: "current-secret", INTERNAL_SECRET_NEXT: "next-secret" }),
    );
    expect(res.status).toBe(401);
  });

  it("works when only INTERNAL_SECRET_NEXT is set (rotation tail)", async () => {
    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: "Bearer next-only", body: VALID_BODY }),
      makeEnv({ INTERNAL_SECRET: "", INTERNAL_SECRET_NEXT: "next-only" }),
    );
    expect(res.status).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// C38 — per-key idempotency round-trip (DO dedup simulated)
// ---------------------------------------------------------------------------

describe("C38: per-key idempotency round-trip", () => {
  it("replayed key returns same count (DO dedup)", async () => {
    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    // First call with key "k1"
    const r1 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: ["k1"] } }),
      makeEnv(),
    );
    expect((await r1.json() as { count: number }).count).toBe(1);

    // Replay same key
    const r2 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: ["k1"] } }),
      makeEnv(),
    );
    expect((await r2.json() as { count: number }).count).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// C58 / C58b / C58c / C58d — batch + hash safety
// ---------------------------------------------------------------------------

describe("C58: batch per-event count correctness", () => {
  it("100 distinct sha256 keys → DO increment called 100 times with each", async () => {
    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    const keys = await Promise.all(
      Array.from({ length: 100 }, (_, i) => sha256Hex(JSON.stringify([`r${i}`, "openai"]))),
    );
    // Distinct hashes (no accidental collisions in the input space).
    expect(new Set(keys).size).toBe(100);
    // Each is exactly 64 hex chars (under the DO's 256-char limit).
    for (const k of keys) expect(k).toMatch(/^[0-9a-f]{64}$/);

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: keys } }),
      makeEnv(),
    );
    expect(res.status).toBe(200);
    expect((await res.json() as { count: number }).count).toBe(100);
    expect(mockDoIncrementPlanCounter).toHaveBeenCalledTimes(100);

    // Replay the batch — no new keys, count unchanged.
    const res2 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: keys } }),
      makeEnv(),
    );
    expect((await res2.json() as { count: number }).count).toBe(100);

    // Half replay + half fresh — only the fresh ones advance the counter.
    const fresh = await Promise.all(
      Array.from({ length: 50 }, (_, i) => sha256Hex(JSON.stringify([`r${100 + i}`, "openai"]))),
    );
    const mixed = [...keys.slice(0, 50), ...fresh];
    const res3 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: mixed } }),
      makeEnv(),
    );
    expect((await res3.json() as { count: number }).count).toBe(150);
  });
});

describe("C58b: compound-key same-requestId-different-provider", () => {
  it("identical requestId with different providers produces distinct hashes", async () => {
    const openaiKey = await sha256Hex(JSON.stringify(["r1", "openai"]));
    const anthropicKey = await sha256Hex(JSON.stringify(["r1", "anthropic"]));
    expect(openaiKey).not.toBe(anthropicKey);

    let currentCount = 0;
    const seen = new Set<string>();
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    const res = await handlePlanCounterIncrement(
      makeRequest({
        auth: VALID_AUTH,
        body: { apiKey: "k", idempotencyKeys: [openaiKey, anthropicKey] },
      }),
      makeEnv(),
    );
    expect((await res.json() as { count: number }).count).toBe(2);
  });
});

describe("C58c: JSON-tuple collision safety", () => {
  it('("a::b", "c") vs ("a", "b::c") — raw string-join would collide, JSON tuple does not', async () => {
    // A naive `requestId + "::" + provider` compound would map both to "a::b::c".
    // JSON tuple encoding serializes as ["a::b","c"] vs ["a","b::c"] — distinct bytes.
    const hashLeft = await sha256Hex(JSON.stringify(["a::b", "c"]));
    const hashRight = await sha256Hex(JSON.stringify(["a", "b::c"]));
    expect(hashLeft).not.toBe(hashRight);

    let currentCount = 0;
    const seen = new Set<string>();
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    const res = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: [hashLeft, hashRight] } }),
      makeEnv(),
    );
    expect((await res.json() as { count: number }).count).toBe(2);
  });
});

describe("C58d: SHA-256 length safety for oversized tuples", () => {
  it("200-char requestId + 100-char provider → hash stays 64 chars and is accepted", async () => {
    const bigRequestId = "r".repeat(200);
    const bigProvider = "p".repeat(100);
    const raw = JSON.stringify([bigRequestId, bigProvider]);
    // The raw tuple exceeds the DO's 256-char idempotency-key limit.
    expect(raw.length).toBeGreaterThan(256);

    const hash = await sha256Hex(raw);
    // Fixed 64-char hex digest — under the limit.
    expect(hash).toHaveLength(64);

    const seen = new Set<string>();
    let currentCount = 0;
    mockDoIncrementPlanCounter.mockImplementation((_env, opts: { idempotencyKey: string }) => {
      if (!seen.has(opts.idempotencyKey)) {
        seen.add(opts.idempotencyKey);
        currentCount++;
      }
      return Promise.resolve({ status: "approved", count: currentCount });
    });

    const r1 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: [hash] } }),
      makeEnv(),
    );
    expect(r1.status).toBe(200);
    expect((await r1.json() as { count: number }).count).toBe(1);

    // Replay — count unchanged (DO dedup via hash identity).
    const r2 = await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: { apiKey: "k", idempotencyKeys: [hash] } }),
      makeEnv(),
    );
    expect((await r2.json() as { count: number }).count).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Metrics observability (supporting regression coverage)
// ---------------------------------------------------------------------------

describe("plan_counter_endpoint metric emission", () => {
  it("emits { status: 'ok' } on successful increment", async () => {
    await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("plan_counter_endpoint", expect.objectContaining({
      status: "ok",
      tier: "free",
      keyCount: 1,
    }));
  });

  it("emits { status: 'unknown_key' } when identity lookup returns null", async () => {
    mockAuthenticateApiKey.mockResolvedValueOnce(null);
    await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("plan_counter_endpoint", { status: "unknown_key" });
  });

  it("emits { status: 'do_error' } when DO increment throws", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockDoIncrementPlanCounter.mockRejectedValueOnce(new Error("DO down"));
    await handlePlanCounterIncrement(
      makeRequest({ auth: VALID_AUTH, body: VALID_BODY }),
      makeEnv(),
    );
    expect(mockEmitMetric).toHaveBeenCalledWith("plan_counter_endpoint", expect.objectContaining({
      status: "do_error",
    }));
  });
});
