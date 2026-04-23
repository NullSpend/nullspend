/**
 * PR-2c unit tests for new helpers introduced across the 4-round codex iteration.
 *
 * Covers:
 * - `isTerminalPlanCounterFkError` (FK allowlist classifier) — codex-round-1 H5 + codex-round-2 H5
 * - `deletePlanCounterEntryTerminal` (outbox delete helper) — plan-audit F2
 * - `resolveNullspendRequestId` (ingress request-id validator) — codex-round-4 M2
 * - `stampNullspendHeaders` (response stamping helper) — codex-round-3 H3
 * - `getPricingUrl` / `getSelfHostUrl` (env-override getters) — codex-round-1 H6 + codex-round-4 H2
 * - `buildPlanLimitExceededPayload` (webhook builder) — plan-audit F1
 * - `buildRecovery("plan_limit_exceeded")` (already in recovery-field.test.ts — not duplicated here)
 *
 * These are pure-function unit tests. Integration coverage (C-INV-1, C-IDEM-3/5,
 * C-HEADER-MATRIX, etc) is deferred to a follow-up PR per plan §5.
 */

import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({ waitUntil: vi.fn() }));

import {
  isTerminalPlanCounterFkError,
  deletePlanCounterEntryTerminal,
} from "../lib/pg-sync-outbox-plan-counter.js";
import { resolveNullspendRequestId, type IngressMetadata } from "../lib/context.js";
import { stampNullspendHeaders } from "../lib/headers.js";
import { getPricingUrl, getSelfHostUrl } from "../lib/constants.js";
import { buildPlanLimitExceededPayload } from "../lib/webhook-events.js";

// ---------------------------------------------------------------------------
// isTerminalPlanCounterFkError
// ---------------------------------------------------------------------------

describe("isTerminalPlanCounterFkError", () => {
  it("returns true for 23503 + allowlisted org_period_usage constraint (postgres.js field name)", () => {
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint_name: "org_period_usage_org_id_fkey",
    })).toBe(true);
  });

  it("returns true for 23503 + allowlisted plan_counter_sync_requests constraint", () => {
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint_name: "plan_counter_sync_requests_org_id_fkey",
    })).toBe(true);
  });

  it("accepts legacy `.constraint` field from node-postgres / pg-native (edge-case-audit E1)", () => {
    // Other Postgres clients expose the FK name as `.constraint` (no underscore).
    // We accept both for compatibility across any direct-connection path.
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint: "org_period_usage_org_id_fkey",
    })).toBe(true);
  });

  it("prefers constraint_name when both fields are set (defensive)", () => {
    // Should match based on constraint_name even if .constraint happens to contain
    // a non-allowlisted value (wouldn't happen in practice but guards against drift).
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint_name: "org_period_usage_org_id_fkey",
      constraint: "some_other_constraint",
    })).toBe(true);
  });

  it("returns false for 23503 + non-allowlisted constraint (retry path)", () => {
    // Critical: a 23503 on a DIFFERENT FK (e.g., webhooks.org_id) must NOT be
    // classified as terminal — the outbox entry should retry. Regression guard
    // against codex-round-2 H5 (generic-classifier overmatch).
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint_name: "webhooks_org_id_fkey",
    })).toBe(false);
  });

  it("returns false for non-23503 error code (transient network, etc)", () => {
    expect(isTerminalPlanCounterFkError({
      code: "08006", // connection_failure
      constraint_name: "org_period_usage_org_id_fkey",
    })).toBe(false);
  });

  it("returns false for malformed error (missing code)", () => {
    expect(isTerminalPlanCounterFkError({
      constraint_name: "org_period_usage_org_id_fkey",
    })).toBe(false);
  });

  it("returns false for malformed error (missing constraint_name AND constraint)", () => {
    expect(isTerminalPlanCounterFkError({
      code: "23503",
    })).toBe(false);
  });

  it("returns false for non-string constraint_name value", () => {
    expect(isTerminalPlanCounterFkError({
      code: "23503",
      constraint_name: 42,
    })).toBe(false);
  });

  it("returns false for null / undefined / primitive errors", () => {
    expect(isTerminalPlanCounterFkError(null)).toBe(false);
    expect(isTerminalPlanCounterFkError(undefined)).toBe(false);
    expect(isTerminalPlanCounterFkError("some string")).toBe(false);
    expect(isTerminalPlanCounterFkError(42)).toBe(false);
  });

  it("returns false for standard Error instance (no pg fields)", () => {
    expect(isTerminalPlanCounterFkError(new Error("some failure"))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// deletePlanCounterEntryTerminal
// ---------------------------------------------------------------------------

describe("deletePlanCounterEntryTerminal", () => {
  it("issues DELETE with id binding", () => {
    const calls: Array<{ query: string; bindings: unknown[] }> = [];
    // Minimal stub matching the SqlStorage signature fields we use.
    const sql = {
      exec: (query: string, ...bindings: unknown[]) => {
        calls.push({ query, bindings });
        return { toArray: () => [], rowsWritten: 1 };
      },
    } as any;
    deletePlanCounterEntryTerminal(sql, 42);
    expect(calls).toHaveLength(1);
    expect(calls[0].query).toBe("DELETE FROM pg_sync_outbox_plan_counter WHERE id = ?");
    expect(calls[0].bindings).toEqual([42]);
  });
});

// ---------------------------------------------------------------------------
// resolveNullspendRequestId
// ---------------------------------------------------------------------------

describe("resolveNullspendRequestId", () => {
  const makeRequest = (headers: Record<string, string> = {}): Request =>
    new Request("https://example.com/", { headers });
  const noopEmit = vi.fn();

  it("auto-generates UUID when header is absent", () => {
    const id = resolveNullspendRequestId(makeRequest(), noopEmit);
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
    expect(noopEmit).not.toHaveBeenCalled();
  });

  it("accepts valid UUID from client", () => {
    const uuid = "12345678-1234-1234-1234-123456789012";
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": uuid }), noopEmit);
    expect(id).toBe(uuid);
  });

  it("accepts ULID / alphanumeric+hyphen/underscore formats", () => {
    const ulid = "01HN8ZK3QR4F5G6H7J8K9L0M1N";
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": ulid }), noopEmit);
    expect(id).toBe(ulid);
  });

  it("rejects empty string and emits metric", () => {
    const emit = vi.fn();
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": "" }), emit);
    expect(id).toMatch(/^[0-9a-f]{8}-/i);
    expect(emit).toHaveBeenCalledWith("empty");
  });

  it("rejects oversize (> 256 chars) and emits metric", () => {
    const emit = vi.fn();
    const oversize = "a".repeat(257);
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": oversize }), emit);
    expect(id).toMatch(/^[0-9a-f]{8}-/i);
    expect(emit).toHaveBeenCalledWith("too_long");
  });

  it("accepts at-limit length (exactly 256 chars)", () => {
    const emit = vi.fn();
    const atLimit = "a".repeat(256);
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": atLimit }), emit);
    expect(id).toBe(atLimit);
    expect(emit).not.toHaveBeenCalled();
  });

  it("rejects bad charset and emits metric", () => {
    const emit = vi.fn();
    // Contains forward-slash + dot + colon — not in [A-Za-z0-9_-]
    const bad = "req/abc.123:xyz";
    const id = resolveNullspendRequestId(makeRequest({ "x-nullspend-request-id": bad }), emit);
    expect(id).toMatch(/^[0-9a-f]{8}-/i);
    expect(emit).toHaveBeenCalledWith("bad_charset");
  });
});

// ---------------------------------------------------------------------------
// stampNullspendHeaders
// ---------------------------------------------------------------------------

describe("stampNullspendHeaders", () => {
  const makeMeta = (overrides: Partial<IngressMetadata> = {}): IngressMetadata => ({
    traceId: "trace-abc",
    nullspendRequestId: "req-123",
    sessionId: null,
    ...overrides,
  });

  it("sets X-NullSpend-Trace-Id and X-NullSpend-Request-Id on every call", () => {
    const res = new Response("ok");
    stampNullspendHeaders(res, makeMeta());
    expect(res.headers.get("X-NullSpend-Trace-Id")).toBe("trace-abc");
    expect(res.headers.get("X-NullSpend-Request-Id")).toBe("req-123");
  });

  it("omits X-NullSpend-Session when sessionId is null", () => {
    const res = new Response("ok");
    stampNullspendHeaders(res, makeMeta({ sessionId: null }));
    expect(res.headers.get("X-NullSpend-Session")).toBeNull();
  });

  it("sets X-NullSpend-Session when sessionId is provided", () => {
    const res = new Response("ok");
    stampNullspendHeaders(res, makeMeta({ sessionId: "session-xyz" }));
    expect(res.headers.get("X-NullSpend-Session")).toBe("session-xyz");
  });

  it("returns the same Response (mutation, not clone)", () => {
    const res = new Response("ok");
    const result = stampNullspendHeaders(res, makeMeta());
    expect(result).toBe(res);
  });
});

// ---------------------------------------------------------------------------
// getPricingUrl / getSelfHostUrl
// ---------------------------------------------------------------------------

describe("getPricingUrl / getSelfHostUrl", () => {
  const DEFAULT_PRICING = "https://nullspend.dev/pricing";
  const DEFAULT_SELF_HOST = "https://github.com/NullSpend/nullspend";

  it("returns default pricing URL when override is absent", () => {
    const env = {} as any;
    expect(getPricingUrl(env)).toBe(DEFAULT_PRICING);
  });

  it("returns default self-host URL when override is absent", () => {
    const env = {} as any;
    expect(getSelfHostUrl(env)).toBe(DEFAULT_SELF_HOST);
  });

  it("returns override pricing URL when set", () => {
    const env = { NULLSPEND_PRICING_URL_OVERRIDE: "https://example.com/upgrade" } as any;
    expect(getPricingUrl(env)).toBe("https://example.com/upgrade");
  });

  it("returns override self-host URL when set", () => {
    const env = { NULLSPEND_SELF_HOST_URL_OVERRIDE: "https://example.com/self-host" } as any;
    expect(getSelfHostUrl(env)).toBe("https://example.com/self-host");
  });

  it("falls back to default when override is empty string", () => {
    const env = {
      NULLSPEND_PRICING_URL_OVERRIDE: "",
      NULLSPEND_SELF_HOST_URL_OVERRIDE: "",
    } as any;
    expect(getPricingUrl(env)).toBe(DEFAULT_PRICING);
    expect(getSelfHostUrl(env)).toBe(DEFAULT_SELF_HOST);
  });

  it("falls back to default when override is non-string", () => {
    const env = {
      NULLSPEND_PRICING_URL_OVERRIDE: 42,
      NULLSPEND_SELF_HOST_URL_OVERRIDE: null,
    } as any;
    expect(getPricingUrl(env)).toBe(DEFAULT_PRICING);
    expect(getSelfHostUrl(env)).toBe(DEFAULT_SELF_HOST);
  });
});

// ---------------------------------------------------------------------------
// buildPlanLimitExceededPayload
// ---------------------------------------------------------------------------

describe("buildPlanLimitExceededPayload", () => {
  const baseData = {
    count: 100_001,
    blockAt: 100_000,
    tier: "free",
    upgradeUrl: "https://nullspend.dev/pricing",
    selfHostUrl: "https://github.com/NullSpend/nullspend",
    model: "gpt-4o-mini",
    provider: "openai",
  };

  it("builds envelope with type plan_limit.exceeded", () => {
    const event = buildPlanLimitExceededPayload(baseData);
    expect(event.type).toBe("plan_limit.exceeded");
    expect(event.id).toMatch(/^evt_[0-9a-f]{8}-/i);
    expect(event.api_version).toBeDefined();
    expect(event.created_at).toBeGreaterThan(0);
  });

  it("preserves all input fields in data.object (wire-shape check)", () => {
    const event = buildPlanLimitExceededPayload(baseData);
    const obj = event.data.object;
    expect(obj.current_count).toBe(100_001);
    expect(obj.block_at).toBe(100_000);
    expect(obj.tier).toBe("free");
    expect(obj.upgrade_url).toBe("https://nullspend.dev/pricing");
    expect(obj.self_host_url).toBe("https://github.com/NullSpend/nullspend");
    expect(obj.model).toBe("gpt-4o-mini");
    expect(obj.provider).toBe("openai");
    expect(obj.blocked_at).toMatch(/^\d{4}-\d{2}-\d{2}T/); // ISO-8601
  });

  it("accepts custom api_version", () => {
    const event = buildPlanLimitExceededPayload(baseData, "2026-04-01");
    expect(event.api_version).toBe("2026-04-01");
  });

  it("emits unique IDs across calls", () => {
    const a = buildPlanLimitExceededPayload(baseData);
    const b = buildPlanLimitExceededPayload(baseData);
    expect(a.id).not.toBe(b.id);
  });
});
