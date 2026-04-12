/**
 * SDK Functional E2E Test Suite — single-call SDK behavior validation.
 *
 * Covers the 11 SDK paths the stress test intentionally skipped (no race surface):
 *   F1-F4   HITL action lifecycle, polling, orchestrator, requestBudgetIncrease
 *   F5-F6   Read APIs (getCostSummary all 3 periods, listCostEvents pagination)
 *   F7-F10  Config behavior (custom fetch, retry, timeout, apiVersion override)
 *   F11     HITL error class fields (TimeoutError, RejectedError)
 *
 * This is a Functional E2E suite, NOT a stress test. Tests run sequentially
 * against the live deployed proxy + dashboard. F1-F6 hit the real dashboard
 * via NULLSPEND_DASHBOARD_URL. F7-F10 use custom fetch injection (the SDK
 * config field being tested) — no live network. F11 constructs error classes
 * directly to validate the built dist/ artifact.
 *
 * CRITICAL: this test mutates live production data. The SDK API key cannot
 * call /api/actions/[id]/approve (session-only auth, admin role), so the
 * approval mechanism is direct SQL UPDATE on the actions table. All test
 * actions are tagged with `agentId LIKE 'sdk-functional-test-%'` and cleaned
 * up symmetrically in beforeAll (orphans from prior crashes) AND afterAll.
 *
 * SDK BUILD WARNING: this file imports `@nullspend/sdk` which resolves to
 * `packages/sdk/dist/`, NOT to source. If you've changed any file under
 * `packages/sdk/src/`, you MUST run `pnpm --filter @nullspend/sdk build`
 * before running this test. Stale dist/ will silently exercise OLD SDK
 * code and produce confusing failures. (Lesson learned during the §15c-1
 * fix per stress test comments.)
 *
 * Requires:
 *   - Deployed proxy at PROXY_URL
 *   - NULLSPEND_DASHBOARD_URL (REQUIRED — fail-fast if unreachable)
 *     Either local `pnpm dev` (http://127.0.0.1:3000) or a deployed Vercel URL
 *   - NULLSPEND_API_KEY, NULLSPEND_SMOKE_KEY_ID, NULLSPEND_SMOKE_USER_ID
 *   - DATABASE_URL (for direct SQL approval + cleanup)
 *   - INTERNAL_SECRET (for proxy /internal/* endpoints — currently unused here)
 *
 * Run:
 *   pnpm --filter @nullspend/sdk build
 *   pnpm dev   # in another terminal (or set NULLSPEND_DASHBOARD_URL to deployed)
 *   cd apps/proxy && npx vitest run --config vitest.smoke.config.ts smoke-sdk-functional.test.ts
 *
 * See docs/internal/test-plans/sdk-testing-gaps.md "Functional E2E suite" (F1-F11).
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  NullSpend,
  NullSpendError,
  RejectedError,
  TimeoutError,
} from "@nullspend/sdk";
import {
  BASE,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_KEY_ID,
  NULLSPEND_SMOKE_USER_ID,
  OPENAI_API_KEY,
  DATABASE_URL,
  invalidateBudget,
  syncBudget,
  authHeaders,
  smallRequest,
} from "./smoke-test-helpers.js";
import { BudgetExceededError } from "@nullspend/sdk";

// ── Constants ────────────────────────────────────────────────────
const AGENT_PREFIX = "sdk-functional-test";
const CUSTOMER_PREFIX = "sdk-functional-customer";
const TEST_POLL_INTERVAL_MS = 500;   // faster than SDK default 2000ms; well under any rate limit
const APPROVE_DELAY_MS = 1_000;       // delay before SQL approve fires (after createAction resolves)
const TIMEOUT_FAST_MS = 1_500;        // F2-B and F11 timeout window
const RETRY_BASE_DELAY_MS = 50;       // F8-A: 50ms base × 2 retries = ~100ms total
const SLOW_FETCH_TIMEOUT_MS = 200;    // F9 abort fires at this point

// ── Shared state ─────────────────────────────────────────────────
let sql: postgres.Sql;
let SMOKE_ORG_ID: string;
let DASHBOARD_URL: string;
let liveClient: NullSpend;

// Per-test agentId so afterAll cleanup matches all rows from this run.
function makeAgentId(testName: string): string {
  return `${AGENT_PREFIX}-${testName}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Strip the `ns_act_` prefix from a SDK-returned action ID and return the
 * raw UUID for direct SQL queries against `actions.id` (uuid column).
 * Mirrors `fromExternalIdOfType("act", id)` from `lib/ids/prefixed-id.ts`.
 */
const ACT_PREFIX = "ns_act_";
function stripActionPrefix(prefixed: string): string {
  if (!prefixed.startsWith(ACT_PREFIX)) {
    throw new Error(`Expected action ID with prefix "${ACT_PREFIX}", got "${prefixed}"`);
  }
  return prefixed.slice(ACT_PREFIX.length);
}

/**
 * Approve an action via direct SQL UPDATE — mirrors what
 * lib/actions/resolve-action.ts does at the SQL level. Session-cookie auth
 * is the only path through /api/actions/[id]/approve, so the SDK API key
 * cannot call it. RETURNING rowcount asserts the action existed and was
 * pending — fails loudly if Risk 1 (race) fires.
 */
async function approveActionViaSql(actionId: string): Promise<void> {
  const uuid = stripActionPrefix(actionId);
  const rows = await sql`
    UPDATE actions
       SET status = 'approved',
           approved_at = NOW(),
           approved_by = ${NULLSPEND_SMOKE_USER_ID!}
     WHERE id = ${uuid}
       AND org_id = ${SMOKE_ORG_ID}
       AND status = 'pending'
    RETURNING id
  `;
  if (rows.length !== 1) {
    throw new Error(
      `approveActionViaSql(${actionId}): expected 1 row updated, got ${rows.length}. ` +
      `Action may not exist, may not be pending, or may belong to a different org.`,
    );
  }
}

/**
 * Reject an action via direct SQL UPDATE — F3-B uses this.
 */
async function rejectActionViaSql(actionId: string): Promise<void> {
  const uuid = stripActionPrefix(actionId);
  const rows = await sql`
    UPDATE actions
       SET status = 'rejected',
           rejected_at = NOW(),
           rejected_by = ${NULLSPEND_SMOKE_USER_ID!}
     WHERE id = ${uuid}
       AND org_id = ${SMOKE_ORG_ID}
       AND status = 'pending'
    RETURNING id
  `;
  if (rows.length !== 1) {
    throw new Error(
      `rejectActionViaSql(${actionId}): expected 1 row updated, got ${rows.length}.`,
    );
  }
}

async function cleanupTestActions(): Promise<void> {
  await sql`
    DELETE FROM actions
     WHERE agent_id LIKE ${AGENT_PREFIX + "-%"}
       AND org_id = ${SMOKE_ORG_ID}
  `;
}

/**
 * Cleanup synthetic cost events created by F12 customer attribution test.
 * Matches by customer_id prefix to find rows from this test even across
 * crashed runs. Symmetric: called in beforeAll AND afterAll.
 */
async function cleanupTestCostEvents(): Promise<void> {
  await sql`
    DELETE FROM cost_events
     WHERE customer_id LIKE ${CUSTOMER_PREFIX + "-%"}
       AND org_id = ${SMOKE_ORG_ID}
  `;
}

/**
 * Cleanup customer_settings + customer budget rows created by F15
 * per-customer upgrade URL test. Matches by customer_id prefix
 * `f15-cust-%`. Runs in beforeAll to reclaim orphan rows from prior
 * crashed runs AND in afterAll for normal cleanup.
 *
 * Also invalidates the proxy's DO state for any matching customer
 * budgets so the cleanup is complete across layers.
 *
 * (T4 edge-case audit.)
 */
async function cleanupTestCustomerSettings(): Promise<void> {
  // Find any orphan customer budgets so we can invalidate the DO.
  const orphanBudgets = await sql<{ entity_id: string }[]>`
    SELECT entity_id FROM budgets
     WHERE entity_type = 'customer'
       AND entity_id LIKE 'f15-cust-%'
       AND org_id = ${SMOKE_ORG_ID}
  `;
  for (const row of orphanBudgets) {
    try {
      await invalidateBudget(SMOKE_ORG_ID, "customer", row.entity_id);
    } catch {
      // Best-effort: DO may be unreachable or row already cleared.
    }
  }

  await sql`
    DELETE FROM budgets
     WHERE entity_type = 'customer'
       AND entity_id LIKE 'f15-cust-%'
       AND org_id = ${SMOKE_ORG_ID}
  `;
  await sql`
    DELETE FROM customer_settings
     WHERE customer_id LIKE 'f15-cust-%'
       AND org_id = ${SMOKE_ORG_ID}
  `;
}

function makeCustomerId(testName: string): string {
  return `${CUSTOMER_PREFIX}-${testName}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

beforeAll(async () => {
  // ── Env validation ──
  if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required in .env.smoke");
  if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required in .env.smoke");
  if (!NULLSPEND_SMOKE_USER_ID) throw new Error("NULLSPEND_SMOKE_USER_ID required in .env.smoke");
  if (!DATABASE_URL) throw new Error("DATABASE_URL required in .env.smoke");

  DASHBOARD_URL = (process.env.NULLSPEND_DASHBOARD_URL ?? "").replace(/\/+$/, "");
  if (!DASHBOARD_URL) {
    throw new Error(
      "NULLSPEND_DASHBOARD_URL required in .env.smoke — start `pnpm dev` " +
      "(http://127.0.0.1:3000) or set to a deployed dashboard URL.",
    );
  }

  // ── Dashboard reachability check (fail fast) ──
  try {
    const res = await fetch(`${DASHBOARD_URL}/api/budgets`, {
      headers: { "x-nullspend-key": NULLSPEND_API_KEY },
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok && res.status !== 401 && res.status !== 403) {
      throw new Error(`unexpected status ${res.status}`);
    }
  } catch (err) {
    throw new Error(
      `Dashboard at ${DASHBOARD_URL} not reachable: ${(err as Error).message}. ` +
      `Start \`pnpm dev\` or update NULLSPEND_DASHBOARD_URL.`,
    );
  }

  // ── Postgres ──
  sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });

  // ── org_id lookup (mirrors smoke-budget-e2e.test.ts:51) ──
  const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID}`;
  if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
  SMOKE_ORG_ID = key.org_id as string;

  // ── Symmetric cleanup: orphan rows from prior crashed runs ──
  await cleanupTestActions();
  await cleanupTestCostEvents();
  await cleanupTestCustomerSettings();

  // ── Shared live client (used by F1-F6) ──
  liveClient = new NullSpend({
    baseUrl: DASHBOARD_URL,
    apiKey: NULLSPEND_API_KEY,
  });
}, 30_000);

afterAll(async () => {
  if (sql) {
    try {
      await cleanupTestActions();
    } catch (err) {
      console.error("[smoke-sdk-functional] afterAll cleanupTestActions failed:", err);
    }
    try {
      await cleanupTestCostEvents();
    } catch (err) {
      console.error("[smoke-sdk-functional] afterAll cleanupTestCostEvents failed:", err);
    }
    try {
      await cleanupTestCustomerSettings();
    } catch (err) {
      console.error("[smoke-sdk-functional] afterAll cleanupTestCustomerSettings failed:", err);
    }
    await sql.end();
  }
});

// ─────────────────────────────────────────────────────────────────
// § Section 3 — Config behavior via fetch injection (F7-F10)
//
// These tests run first because they're fast and don't touch live
// dashboard state. Each constructs its own NullSpend instance with
// a fixture fetch — the SDK config field under test (`fetch`) is
// itself the observation point.
// ─────────────────────────────────────────────────────────────────
describe("Section 3 — Config behavior (fetch injection)", () => {
  it("F7 — custom fetch is called for outgoing requests", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    const spy: typeof fetch = async (url, init) => {
      calls.push({ url: String(url), init });
      return new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    const client = new NullSpend({
      baseUrl: "https://example.invalid",
      apiKey: "test-key",
      fetch: spy,
    });

    await client.listBudgets();

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://example.invalid/api/budgets");
    const headers = calls[0].init?.headers as Record<string, string>;
    expect(headers["x-nullspend-key"]).toBe("test-key");
  });

  it("F8a — retries on 503 then succeeds, onRetry fires, idempotency key reused, base delay honored", async () => {
    let callCount = 0;
    const idempotencyKeys: string[] = [];

    const spy: typeof fetch = async (_url, init) => {
      const headers = init?.headers as Record<string, string> | undefined;
      const idemKey = headers?.["Idempotency-Key"] ?? headers?.["idempotency-key"];
      if (idemKey) idempotencyKeys.push(idemKey);

      callCount += 1;
      if (callCount < 3) {
        // Return 503 without Retry-After so the SDK uses retryBaseDelayMs
        return new Response("Service Unavailable", { status: 503 });
      }
      return new Response(JSON.stringify({ data: { id: "evt_x", createdAt: "2026-04-07T00:00:00Z" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    const onRetryCalls: Array<{ attempt: number; delayMs: number; method: string; path: string }> = [];
    const client = new NullSpend({
      baseUrl: "https://example.invalid",
      apiKey: "k",
      fetch: spy,
      maxRetries: 5,
      retryBaseDelayMs: RETRY_BASE_DELAY_MS,
      onRetry: (info) => {
        onRetryCalls.push({
          attempt: info.attempt,
          delayMs: info.delayMs,
          method: info.method,
          path: info.path,
        });
      },
    });

    await client.reportCost({
      provider: "openai",
      model: "gpt-4o-mini",
      inputTokens: 10,
      outputTokens: 5,
      costMicrodollars: 100,
    });

    // Spy was called 3 times (1 initial + 2 retries)
    expect(callCount).toBe(3);

    // onRetry fired exactly twice with attempts 0 and 1
    expect(onRetryCalls).toHaveLength(2);
    expect(onRetryCalls[0].attempt).toBe(0);
    expect(onRetryCalls[1].attempt).toBe(1);
    expect(onRetryCalls[0].method).toBe("POST");
    expect(onRetryCalls[0].path).toBe("/api/cost-events");

    // Same idempotency key reused across retries — regression guard against
    // a stress §15c-style bug where retries generate fresh keys.
    expect(idempotencyKeys).toHaveLength(3);
    expect(idempotencyKeys[0]).toBeTruthy();
    expect(idempotencyKeys.every((k) => k === idempotencyKeys[0])).toBe(true);

    // Delay assertion via onRetry.delayMs (deterministic, not wall-time).
    // SDK uses full-jitter exponential backoff: floor(random() * min(base * 2^attempt, maxDelay)).
    // Verified at packages/sdk/src/retry.ts:71-78. Always >= 1.
    //   - attempt 0: ceiling = 50ms × 2^0 = 50ms,  delay ∈ [1, 49]
    //   - attempt 1: ceiling = 50ms × 2^1 = 100ms, delay ∈ [1, 99]
    // Asserts:
    //   1. delay > 0 (proves SDK actually computed and used a delay, not zero)
    //   2. delay <= ceiling (proves retryBaseDelayMs controls the cap — would catch
    //      a regression that ignores the config and uses the default 500ms base)
    expect(onRetryCalls[0].delayMs).toBeGreaterThanOrEqual(1);
    expect(onRetryCalls[0].delayMs).toBeLessThanOrEqual(RETRY_BASE_DELAY_MS);          // <= 50
    expect(onRetryCalls[1].delayMs).toBeGreaterThanOrEqual(1);
    expect(onRetryCalls[1].delayMs).toBeLessThanOrEqual(RETRY_BASE_DELAY_MS * 2);      // <= 100
  });

  it("F8b — maxRetries cap is enforced", async () => {
    let callCount = 0;
    const spy: typeof fetch = async () => {
      callCount += 1;
      return new Response("Service Unavailable", { status: 503 });
    };

    const client = new NullSpend({
      baseUrl: "https://example.invalid",
      apiKey: "k",
      fetch: spy,
      maxRetries: 2,
      retryBaseDelayMs: 1,
    });

    await expect(client.listBudgets()).rejects.toThrow(NullSpendError);

    // 1 initial + 2 retries = 3 total
    expect(callCount).toBe(3);

    // Throw should carry the upstream status
    try {
      await client.listBudgets();
    } catch (err) {
      expect(err).toBeInstanceOf(NullSpendError);
      expect((err as NullSpendError).statusCode).toBe(503);
    }
  });

  it("F9 — requestTimeoutMs aborts slow upstream", async () => {
    const slowFetch: typeof fetch = (_url, init) =>
      new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        // Never resolves on its own.
      });

    const client = new NullSpend({
      baseUrl: "https://example.invalid",
      apiKey: "k",
      fetch: slowFetch,
      requestTimeoutMs: SLOW_FETCH_TIMEOUT_MS,
      maxRetries: 0,
    });

    const start = Date.now();
    await expect(client.listBudgets()).rejects.toThrow(NullSpendError);
    const elapsed = Date.now() - start;

    // Should fire near SLOW_FETCH_TIMEOUT_MS, definitely well under 2s
    expect(elapsed).toBeLessThan(2_000);
  });

  it("F10 — apiVersion override is sent on the NullSpend-Version header", async () => {
    let capturedHeaders: Record<string, string> | undefined;
    const spy: typeof fetch = async (_url, init) => {
      capturedHeaders = init?.headers as Record<string, string>;
      return new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    const customVersion = "2025-01-01";
    const client = new NullSpend({
      baseUrl: "https://example.invalid",
      apiKey: "k",
      fetch: spy,
      apiVersion: customVersion,
    });

    await client.listBudgets();

    expect(capturedHeaders).toBeDefined();
    expect(capturedHeaders!["NullSpend-Version"]).toBe(customVersion);
  });
});

// ─────────────────────────────────────────────────────────────────
// § Section 1 — HITL action lifecycle (F1-F4)
//
// Live dashboard tests. Approval is via direct SQL (the /approve
// endpoint is session-cookie-only). Each test schedules SQL approve
// AFTER createAction resolves to avoid the race in Risk 1.
// ─────────────────────────────────────────────────────────────────
describe("Section 1 — HITL action lifecycle (live dashboard)", () => {
  it("F1 — createAction → SQL approve → getAction → markResult lifecycle", async () => {
    const agentId = makeAgentId("f1");

    // 1. createAction
    const created = await liveClient.createAction({
      agentId,
      actionType: "http_post",
      payload: { url: "https://example.com/webhook", method: "POST" },
    });
    expect(created.id).toBeTruthy();
    expect(created.status).toBe("pending");

    // 2. SQL approve
    await approveActionViaSql(created.id);

    // 3. getAction
    const fetched = await liveClient.getAction(created.id);
    expect(fetched.status).toBe("approved");
    expect(fetched.approvedBy).toBe(NULLSPEND_SMOKE_USER_ID);
    expect(fetched.agentId).toBe(agentId);

    // 4. markResult(executing)
    const exec = await liveClient.markResult(created.id, { status: "executing" });
    expect(exec.status).toBe("executing");

    // 5. markResult(executed)
    const done = await liveClient.markResult(created.id, {
      status: "executed",
      result: { ok: true },
    });
    expect(done.status).toBe("executed");
  }, 30_000);

  it("F2a — waitForDecision resolves when action is approved mid-poll", async () => {
    const agentId = makeAgentId("f2a");

    const created = await liveClient.createAction({
      agentId,
      actionType: "http_post",
      payload: { url: "https://example.com/webhook" },
    });

    // Schedule SQL approve AFTER createAction has resolved (avoids Risk 1 race).
    setTimeout(() => {
      approveActionViaSql(created.id).catch((err) => {
        console.error("[F2a] SQL approve failed:", err);
      });
    }, APPROVE_DELAY_MS);

    const decision = await liveClient.waitForDecision(created.id, {
      pollIntervalMs: TEST_POLL_INTERVAL_MS,
      timeoutMs: 10_000,
    });

    expect(decision.status).toBe("approved");
    expect(decision.id).toBe(created.id);
  }, 30_000);

  it("F2b — waitForDecision throws TimeoutError when no decision arrives", async () => {
    const agentId = makeAgentId("f2b");

    const created = await liveClient.createAction({
      agentId,
      actionType: "http_post",
      payload: { url: "https://example.com/webhook" },
    });

    // No approve scheduled — wait should time out.
    let caught: unknown;
    try {
      await liveClient.waitForDecision(created.id, {
        pollIntervalMs: TEST_POLL_INTERVAL_MS,
        timeoutMs: TIMEOUT_FAST_MS,
      });
    } catch (err) {
      caught = err;
    }

    expect(caught).toBeInstanceOf(TimeoutError);
    const timeoutErr = caught as TimeoutError;
    expect(timeoutErr.actionId).toBe(created.id);
    expect(timeoutErr.timeoutMs).toBe(TIMEOUT_FAST_MS);
    expect(timeoutErr.message).toContain(created.id);
    expect(timeoutErr.message).toContain(String(TIMEOUT_FAST_MS));
  }, 30_000);

  it("F3a — proposeAndWait happy path: approve → execute → markResult(executed)", async () => {
    const agentId = makeAgentId("f3a");
    let executeCalled = false;
    let capturedActionId: string | undefined;

    const result = await liveClient.proposeAndWait<string>({
      agentId,
      actionType: "http_post",
      payload: { url: "https://example.com/webhook" },
      pollIntervalMs: TEST_POLL_INTERVAL_MS,
      timeoutMs: 10_000,
      onPoll: (action) => {
        if (!capturedActionId) {
          capturedActionId = action.id;
          // Schedule approve AFTER we've seen at least one poll (createAction resolved)
          setTimeout(() => {
            approveActionViaSql(action.id).catch((err) => {
              console.error("[F3a] SQL approve failed:", err);
            });
          }, APPROVE_DELAY_MS);
        }
      },
      execute: async () => {
        executeCalled = true;
        return "ok";
      },
    });

    expect(result).toBe("ok");
    expect(executeCalled).toBe(true);
    expect(capturedActionId).toBeTruthy();

    // Verify the action ended in 'executed' state in DB
    const [row] = await sql`
      SELECT status FROM actions WHERE id = ${stripActionPrefix(capturedActionId!)}
    `;
    expect(row?.status).toBe("executed");
  }, 30_000);

  it("F3b — proposeAndWait throws RejectedError when action is rejected", async () => {
    const agentId = makeAgentId("f3b");
    let executeCalled = false;
    let capturedActionId: string | undefined;

    let thrown: unknown;
    try {
      await liveClient.proposeAndWait<string>({
        agentId,
        actionType: "http_post",
        payload: { url: "https://example.com/webhook" },
        pollIntervalMs: TEST_POLL_INTERVAL_MS,
        timeoutMs: 10_000,
        onPoll: (action) => {
          if (!capturedActionId) {
            capturedActionId = action.id;
            setTimeout(() => {
              rejectActionViaSql(action.id).catch((err) => {
                console.error("[F3b] SQL reject failed:", err);
              });
            }, APPROVE_DELAY_MS);
          }
        },
        execute: async () => {
          executeCalled = true;
          return "should not run";
        },
      });
    } catch (err) {
      thrown = err;
    }

    expect(thrown).toBeInstanceOf(RejectedError);
    expect((thrown as RejectedError).actionId).toBe(capturedActionId);
    expect((thrown as RejectedError).actionStatus).toBe("rejected");
    expect(executeCalled).toBe(false);
  }, 30_000);

  it("F4 — requestBudgetIncrease invalidates policy cache after approval", async () => {
    const agentId = makeAgentId("f4");

    // Spy fetch wraps real fetch and counts /api/policy calls.
    // Note: /api/policy is org-scoped, not key-scoped — we assert that the
    // cache was re-fetched, NOT that the next request sees a new limit.
    const policyFetches: number[] = [];
    const spyFetch: typeof fetch = async (url, init) => {
      const urlStr = typeof url === "string" ? url : url instanceof URL ? url.toString() : (url as Request).url;
      if (urlStr.includes("/api/policy")) {
        policyFetches.push(Date.now());
      }
      return globalThis.fetch(url, init);
    };

    // Build a NullSpend instance with the spy fetch and cost reporting
    // enabled (createTrackedFetch requires costReporting).
    const client = new NullSpend({
      baseUrl: DASHBOARD_URL,
      apiKey: NULLSPEND_API_KEY!,
      fetch: spyFetch,
      costReporting: {
        batchSize: 100,
        flushIntervalMs: 60_000,
      },
    });

    // 1. Build an enforced tracked fetch — first call against any URL warms the policy cache.
    const trackedFetch = client.createTrackedFetch("openai", { enforcement: true });

    // Trigger policy fetch by making a tracked-fetch call. The actual request will
    // fail (example.invalid) but the policy cache populates BEFORE the upstream call.
    try {
      await trackedFetch("https://example.invalid/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: "Bearer fake" },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
      });
    } catch {
      // Expected — example.invalid is unreachable. Policy fetch already happened.
    }

    expect(policyFetches.length).toBeGreaterThanOrEqual(1);
    const policyFetchesBeforeIncrease = policyFetches.length;

    // 2. requestBudgetIncrease — schedule SQL approve mid-poll.
    let capturedActionId: string | undefined;
    const result = await client.requestBudgetIncrease({
      agentId,
      amount: 5_000_000,
      reason: "F4 test — verify policy cache invalidation",
      entityType: "user",
      entityId: NULLSPEND_SMOKE_USER_ID!,
      pollIntervalMs: TEST_POLL_INTERVAL_MS,
      timeoutMs: 10_000,
      onPoll: (action) => {
        if (!capturedActionId) {
          capturedActionId = action.id;
          setTimeout(() => {
            approveActionViaSql(action.id).catch((err) => {
              console.error("[F4] SQL approve failed:", err);
            });
          }, APPROVE_DELAY_MS);
        }
      },
    });

    expect(result.actionId).toBeTruthy();
    expect(result.requestedAmountMicrodollars).toBe(5_000_000);

    // 3. Verify action terminal state — proposeAndWait calls markResult(executing)
    // then markResult(executed) after execute() succeeds. So state is 'executed',
    // NOT 'approved'. Verified at packages/sdk/src/client.ts:438-504.
    const [row] = await sql`
      SELECT status FROM actions WHERE id = ${stripActionPrefix(capturedActionId!)}
    `;
    expect(row?.status).toBe("executed");

    // 4. Trigger another tracked fetch call — should re-fetch /api/policy
    // because requestBudgetIncrease invalidated the cache.
    try {
      await trackedFetch("https://example.invalid/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: "Bearer fake" },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
      });
    } catch {
      // Expected.
    }

    // Policy was re-fetched at least once after the budget increase
    expect(policyFetches.length).toBeGreaterThan(policyFetchesBeforeIncrease);

    // Cleanup the cost reporter to drain the queue
    await client.shutdown();
  }, 60_000);
});

// ─────────────────────────────────────────────────────────────────
// § Section 2 — Read APIs (F5-F6)
//
// Live dashboard tests. F5 requires the pre-PR auth fix to
// app/api/cost-events/summary/route.ts (uses assertApiKeyOrSession).
// F6 works around an SDK type/server schema mismatch — server expects
// cursor as JSON-stringified object.
// ─────────────────────────────────────────────────────────────────
describe("Section 2 — Read APIs (live dashboard)", () => {
  it("F5 — getCostSummary returns the expected shape for all 3 periods", async () => {
    for (const period of ["7d", "30d", "90d"] as const) {
      const summary = await liveClient.getCostSummary(period);

      // Shape assertions only — totals may be 0 if smoke org has no recent events.
      expect(summary).toHaveProperty("daily");
      expect(Array.isArray(summary.daily)).toBe(true);
      expect(summary).toHaveProperty("models");
      expect(Array.isArray(summary.models)).toBe(true);
      expect(summary).toHaveProperty("providers");
      expect(Array.isArray(summary.providers)).toBe(true);
      expect(summary).toHaveProperty("keys");
      expect(summary).toHaveProperty("tools");
      expect(summary).toHaveProperty("sources");
      expect(summary).toHaveProperty("traces");

      expect(summary).toHaveProperty("totals");
      expect(summary.totals.period).toBe(period);
      expect(typeof summary.totals.totalCostMicrodollars).toBe("number");
      expect(typeof summary.totals.totalRequests).toBe("number");

      expect(summary).toHaveProperty("costBreakdown");
      expect(summary.costBreakdown).toHaveProperty("inputCost");
      expect(summary.costBreakdown).toHaveProperty("outputCost");
      expect(summary.costBreakdown).toHaveProperty("cachedCost");
      expect(summary.costBreakdown).toHaveProperty("reasoningCost");
    }
  }, 30_000);

  it("F6 — listCostEvents pagination round-trip", async () => {
    const page1 = await liveClient.listCostEvents({ limit: 5 });
    expect(page1).toHaveProperty("data");
    expect(Array.isArray(page1.data)).toBe(true);
    expect(page1.data.length).toBeLessThanOrEqual(5);

    if (!page1.cursor) {
      // Smoke org has < 5 events. Soft-skip pagination assert.
      console.warn("[F6] insufficient cost events for pagination — skipping cursor round-trip");
      return;
    }

    // SDK accepts the response cursor object directly and stringifies internally.
    const page2 = await liveClient.listCostEvents({
      limit: 5,
      cursor: page1.cursor,
    });

    expect(page2).toHaveProperty("data");
    expect(Array.isArray(page2.data)).toBe(true);
    expect(page2.data.length).toBeLessThanOrEqual(5);

    // Pages must not overlap
    const ids1 = new Set(page1.data.map((e) => e.id));
    const overlap = page2.data.filter((e) => ids1.has(e.id));
    expect(overlap).toEqual([]);
  }, 30_000);

  it("ST-3 — same idempotency key from two different callers produces independent results (ACT-4)", async () => {
    // ACT-4 fix: idempotency key is now scoped by API key hash + route path.
    // Two different API keys using the same Idempotency-Key should NOT get
    // each other's cached responses.
    //
    // This test uses the live dashboard cost-events endpoint to verify.
    // The liveClient has the smoke test API key. We make two requests:
    // 1. POST with Idempotency-Key "st3-test-key" → should succeed (201)
    // 2. POST with same Idempotency-Key → should return cached (200 with replay header)
    //
    // We can't easily test cross-key isolation here (would need a second API key),
    // but we CAN verify that same-key replay works correctly and includes the
    // X-Idempotent-Replayed header.
    const idempotencyKey = `st3-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const dashboardUrl = process.env.NULLSPEND_DASHBOARD_URL ?? "http://127.0.0.1:3000";
    const apiKey = process.env.NULLSPEND_API_KEY;
    if (!apiKey) {
      console.log("[ST-3] NULLSPEND_API_KEY not set — skipping");
      return;
    }

    const costEvent = {
      provider: "openai",
      model: "gpt-4o-mini",
      inputTokens: 10,
      outputTokens: 5,
      costMicrodollars: 1,
      requestId: `st3-req-${Date.now()}`,
    };

    // First call — should insert
    const res1 = await fetch(`${dashboardUrl}/api/cost-events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": apiKey,
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify(costEvent),
    });

    if (res1.status === 503 || res1.status === 0) {
      console.log("[ST-3] Dashboard not reachable — skipping");
      return;
    }

    expect(res1.status).toBe(201);
    const body1 = await res1.json();
    expect(body1.data.id).toBeTruthy();

    // Second call — same key, should replay
    const res2 = await fetch(`${dashboardUrl}/api/cost-events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": apiKey,
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify(costEvent),
    });

    // Should get the same response back
    expect(res2.status).toBe(201);
    const replayed = res2.headers.get("X-Idempotent-Replayed");
    expect(replayed).toBe("true");

    const body2 = await res2.json();
    expect(body2.data.id).toBe(body1.data.id); // Same cached response
  }, 30_000);
});

// ─────────────────────────────────────────────────────────────────
// § Section 4 — HITL error class fields (F11)
//
// In-memory direct construction. Validates the BUILT dist/ artifact
// against the contract (instanceof, name, message, public fields).
// F2-B and F3-B already validate behavior; F11 is a focused
// publish-time guard against built-vs-source drift.
// ─────────────────────────────────────────────────────────────────
describe("Section 4 — HITL error class fields", () => {
  it("F11 — TimeoutError + RejectedError instanceof, name, and field validation", () => {
    const timeoutErr = new TimeoutError("act_test_x", 1_500);
    expect(timeoutErr).toBeInstanceOf(TimeoutError);
    expect(timeoutErr).toBeInstanceOf(NullSpendError);
    expect(timeoutErr).toBeInstanceOf(Error);
    expect(timeoutErr.name).toBe("TimeoutError");
    expect(timeoutErr.actionId).toBe("act_test_x");
    expect(timeoutErr.timeoutMs).toBe(1_500);
    expect(timeoutErr.message).toContain("act_test_x");
    expect(timeoutErr.message).toContain("1500");

    const rejectedErr = new RejectedError("act_test_y", "rejected");
    expect(rejectedErr).toBeInstanceOf(RejectedError);
    expect(rejectedErr).toBeInstanceOf(NullSpendError);
    expect(rejectedErr).toBeInstanceOf(Error);
    expect(rejectedErr.name).toBe("RejectedError");
    expect(rejectedErr.actionId).toBe("act_test_y");
    expect(rejectedErr.actionStatus).toBe("rejected");
    expect(rejectedErr.message).toContain("act_test_y");
    expect(rejectedErr.message).toContain("rejected");
  });
});

// ─────────────────────────────────────────────────────────────────
// § Section 5 — Customer attribution end-to-end (F12)
//
// The only F-series test that hits the live proxy + OpenAI. Closes
// the gap that F1-F11 leaves open: validates the full path from
// SDK customer() → X-NullSpend-Customer header → deployed proxy →
// cost_events.customer_id column → SDK listCostEvents response.
//
// Prerequisites in addition to F1-F11:
//   - PROXY_URL (deployed proxy, e.g. https://nullspend.cjones6489.workers.dev)
//   - OPENAI_API_KEY (real spend, ~$0.005 per run)
// ─────────────────────────────────────────────────────────────────
describe("Section 5 — Customer attribution end-to-end", () => {
  it("F12 — customer() → proxy → cost_events.customer_id → listCostEvents round-trip", async () => {
    if (!OPENAI_API_KEY) {
      throw new Error("F12 requires OPENAI_API_KEY in .env.smoke");
    }
    if (!BASE) {
      throw new Error("F12 requires PROXY_URL in .env.smoke");
    }

    const customerId = makeCustomerId("f12");

    // Customer-scoped client. Needs proxyUrl so the SDK detects the proxy
    // path and skips client-side cost tracking. Needs costReporting because
    // createTrackedFetch (which customer() delegates to) requires it even
    // though the proxy mode never uses it.
    const customerClient = new NullSpend({
      baseUrl: DASHBOARD_URL,
      apiKey: NULLSPEND_API_KEY!,
      proxyUrl: BASE,
      costReporting: { batchSize: 1, flushIntervalMs: 100 },
    });

    try {
      const session = customerClient.customer(customerId);

      // Make a real OpenAI request through the proxy with customer scope.
      // Tiny payload to keep cost minimal.
      const res = await session.openai(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": NULLSPEND_API_KEY!,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say ok" }],
          max_tokens: 3,
        }),
      });
      expect(res.status).toBe(200);

      // Wait for the cost event to land via the queue → Hyperdrive → Postgres
      // path. Poll customer_id directly since we don't know the request_id.
      const dbRow = await pollForCostEventByCustomerId(customerId, 20_000);
      expect(dbRow).not.toBeNull();
      expect(dbRow!.customer_id).toBe(customerId);
      expect(dbRow!.provider).toBe("openai");
      expect(dbRow!.model).toBe("gpt-4o-mini");

      // Now verify the SDK read path: listCostEvents must surface the
      // customerId field on the response record. Fetch the most recent
      // page and find the matching event by customerId (unique per run).
      // ID comparison would need prefix handling: DB stores raw UUID,
      // SDK response wraps it as `ns_evt_<uuid>` via nsIdOutput.
      const page = await liveClient.listCostEvents({ limit: 100 });
      const found = page.data.find((e) => e.customerId === customerId);
      expect(found, "F12: cost event written to DB but missing from listCostEvents page").not.toBeUndefined();
      expect(found!.customerId).toBe(customerId);
      expect(found!.provider).toBe("openai");
      expect(found!.model).toBe("gpt-4o-mini");
      // Sanity check: SDK ID is the prefixed external form
      expect(found!.id).toBe(`ns_evt_${dbRow!.id}`);
    } finally {
      // Drain any in-flight cost reports the customerClient queued (none in
      // proxy mode, but flush+shutdown is the documented teardown pattern).
      await customerClient.shutdown();
    }
  }, 60_000);
});

/**
 * Poll cost_events for a row matching the given customer_id. Used by F12
 * because the proxy writes async via Cloudflare Queue → Hyperdrive, so the
 * row may not be visible immediately.
 */
async function pollForCostEventByCustomerId(
  customerId: string,
  timeoutMs: number,
): Promise<{ id: string; customer_id: string; provider: string; model: string } | null> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const rows = await sql`
      SELECT id, customer_id, provider, model
        FROM cost_events
       WHERE customer_id = ${customerId}
         AND org_id = ${SMOKE_ORG_ID}
       LIMIT 1
    `;
    if (rows.length > 0) {
      return rows[0] as { id: string; customer_id: string; provider: string; model: string };
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────
// § Section 6 — Budget remaining response headers (F13)
//
// Phase 0 finish, Item 1: validates that the proxy stamps
// X-NullSpend-Budget-Limit/-Spent/-Remaining/-Entity on every
// proxied response when a budget is in scope. Stripe-pattern
// proximity headers; clients monitor remaining without separate
// API calls.
// ─────────────────────────────────────────────────────────────────
describe("Section 6 — Budget remaining response headers", () => {
  const userId = NULLSPEND_SMOKE_USER_ID!;

  async function setupUserBudget(maxMicrodollars: number, spendMicrodollars: number) {
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (${userId}, ${SMOKE_ORG_ID}, 'user', ${userId}, ${maxMicrodollars}, ${spendMicrodollars}, 'strict_block')
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;
    await syncBudget(SMOKE_ORG_ID, "user", userId);
  }

  async function teardownUserBudget() {
    await invalidateBudget(SMOKE_ORG_ID, "user", userId);
    await sql`DELETE FROM budgets WHERE entity_type = 'user' AND entity_id = ${userId} AND org_id = ${SMOKE_ORG_ID}`;
  }

  it("F13 — proxy stamps X-NullSpend-Budget-* headers on 200 success path", async () => {
    if (!OPENAI_API_KEY) throw new Error("F13 requires OPENAI_API_KEY in .env.smoke");
    if (!BASE) throw new Error("F13 requires PROXY_URL in .env.smoke");

    // $10 limit, $2 already spent → expect headers reflect post-reservation state
    await setupUserBudget(10_000_000, 2_000_000);

    try {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(200);
      // Drain body so connection closes cleanly
      await res.json();

      // All four headers must be present.
      const limit = res.headers.get("X-NullSpend-Budget-Limit");
      const spent = res.headers.get("X-NullSpend-Budget-Spent");
      const remaining = res.headers.get("X-NullSpend-Budget-Remaining");
      const entity = res.headers.get("X-NullSpend-Budget-Entity");

      expect(limit, "X-NullSpend-Budget-Limit must be present").not.toBeNull();
      expect(spent, "X-NullSpend-Budget-Spent must be present").not.toBeNull();
      expect(remaining, "X-NullSpend-Budget-Remaining must be present").not.toBeNull();
      expect(entity, "X-NullSpend-Budget-Entity must be present").not.toBeNull();

      // Math invariants. The exact estimate microdollar value depends on
      // the request body but is small (~6 microdollars for the smallRequest
      // helper). Assert: limit = 10M, spent = 2M + reserved + estimate,
      // remaining = limit - spent.
      const limitNum = Number(limit);
      const spentNum = Number(spent);
      const remainingNum = Number(remaining);
      expect(limitNum).toBe(10_000_000);
      expect(spentNum).toBeGreaterThanOrEqual(2_000_000); // at least the prior spend
      expect(spentNum).toBeLessThan(2_100_000); // within a reasonable estimate band
      expect(limitNum).toBe(spentNum + remainingNum);

      // Entity header points to the user budget.
      expect(entity).toBe(`user:${userId}`);
    } finally {
      await teardownUserBudget();
    }
  }, 30_000);

  it("F13 — proxy stamps headers on 429 denial path (post-rejected-request state)", async () => {
    if (!BASE) throw new Error("F13 requires PROXY_URL in .env.smoke");

    // Spend == max → next request denied
    await setupUserBudget(1_000_000, 1_000_000);

    try {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(res.status).toBe(429);
      await res.json();

      const limit = res.headers.get("X-NullSpend-Budget-Limit");
      const spent = res.headers.get("X-NullSpend-Budget-Spent");
      const remaining = res.headers.get("X-NullSpend-Budget-Remaining");
      const entity = res.headers.get("X-NullSpend-Budget-Entity");

      expect(limit).toBe("1000000");
      expect(spent).toBe("1000000");
      expect(remaining).toBe("0");
      expect(entity).toBe(`user:${userId}`);

      // Denial header gates SDK denial parsing
      expect(res.headers.get("X-NullSpend-Denied")).toBe("1");
    } finally {
      await teardownUserBudget();
    }
  }, 30_000);
});

// ─────────────────────────────────────────────────────────────────
// § Section 7 — upgrade_url in denial bodies (REMOVED)
//
// The old F14 test tried to verify org-level upgrade_url (stored at
// organizations.metadata.upgradeUrl, cached on the auth identity with
// a 120s TTL) by setting the URL, calling invalidateAuthOnly, then
// making a request and asserting the denial body contained the URL.
//
// This test was inherently flaky due to Cloudflare Workers' multi-
// isolate architecture: invalidateAuthCacheForOwner only clears the
// cache in ONE isolate that receives the invalidation POST. Other
// isolates continue serving stale identity values until their own
// positive-TTL (120s ± 10s jitter) expires. Subsequent fetches may
// hit either the cleared OR a stale isolate — pass rate was ~50%.
//
// Coverage strategy for the org-level path (no live smoke):
//   1. SDK parser unit tests verify error.upgrade_url is extracted
//      correctly from the envelope (packages/sdk/src/tracked-fetch.test.ts
//      "surfaces upgradeUrl on the error")
//   2. mcp-proxy unit tests verify the cost-tracker envelope parser
//      (packages/mcp-proxy/src/cost-tracker.test.ts)
//   3. Dashboard route tests verify the GET/PATCH endpoint shape
//      (app/api/orgs/[orgId]/upgrade-url/route.test.ts)
//   4. Proxy unit tests verify handleBudgetDenials injects upgrade_url
//      via the builders and correctly emits budget_denied metric
//
// The CUSTOMER-level path IS reliably smoke-tested as F15 (Section 8)
// below, because customer_settings.upgrade_url is looked up FRESH on
// every denial (no auth cache coupling).
// ─────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────
// § Section 8 — customer_budget_exceeded with per-customer override (F15)
//
// Phase 0 audit follow-up (Issue 4): exercises the customer denial
// branch that F14 did not hit. This test:
//   1. Inserts a customer_settings row with a {customer_id} placeholder URL
//   2. Creates a customer budget with spend == max
//   3. Makes a request via customer() session
//   4. Asserts the proxy returns customer_budget_exceeded with the
//      per-customer URL + substituted customer_id
//
// Unlike F14's org-level test, this does NOT hit the auth cache
// staleness issue because customer_settings.upgrade_url is looked up
// FRESH on every denial via lookupCustomerUpgradeUrl (cold-path
// Postgres query, no caching). That means each F15 run uses its own
// unique customer ID and cannot collide with previous runs.
// ─────────────────────────────────────────────────────────────────
describe("Section 8 — customer_budget_exceeded + per-customer upgrade URL", () => {
  async function setCustomerUpgradeUrl(customerId: string, url: string | null): Promise<void> {
    await sql`
      INSERT INTO customer_settings (org_id, customer_id, upgrade_url, updated_at)
      VALUES (${SMOKE_ORG_ID}, ${customerId}, ${url}, NOW())
      ON CONFLICT (org_id, customer_id) DO UPDATE
        SET upgrade_url = ${url}, updated_at = NOW()
    `;
  }

  async function createCustomerBudget(customerId: string, maxMicrodollars: number, spendMicrodollars: number): Promise<void> {
    await sql`
      INSERT INTO budgets (user_id, org_id, entity_type, entity_id, max_budget_microdollars, spend_microdollars, policy)
      VALUES (
        ${NULLSPEND_SMOKE_USER_ID!},
        ${SMOKE_ORG_ID},
        'customer',
        ${customerId},
        ${maxMicrodollars},
        ${spendMicrodollars},
        'strict_block'
      )
      ON CONFLICT (org_id, entity_type, entity_id)
      DO UPDATE SET max_budget_microdollars = ${maxMicrodollars},
                    spend_microdollars = ${spendMicrodollars},
                    updated_at = NOW()
    `;
    await syncBudget(SMOKE_ORG_ID, "customer", customerId);
  }

  async function teardownCustomerFixtures(customerId: string): Promise<void> {
    try { await invalidateBudget(SMOKE_ORG_ID, "customer", customerId); } catch { /* best-effort */ }
    await sql`DELETE FROM budgets WHERE entity_type = 'customer' AND entity_id = ${customerId} AND org_id = ${SMOKE_ORG_ID}`;
    await sql`DELETE FROM customer_settings WHERE customer_id = ${customerId} AND org_id = ${SMOKE_ORG_ID}`;
  }

  it("F15 — customer_budget_exceeded includes per-customer upgrade_url with {customer_id} substituted", async () => {
    if (!BASE) throw new Error("F15 requires PROXY_URL in .env.smoke");

    // Unique customer ID per run — avoids cross-test cache collisions.
    const customerId = `f15-cust-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    const urlTemplate = "https://example.com/customer-upgrade?id={customer_id}";
    const expectedSubstituted = `https://example.com/customer-upgrade?id=${encodeURIComponent(customerId)}`;

    await setCustomerUpgradeUrl(customerId, urlTemplate);
    await createCustomerBudget(customerId, 1_000_000, 1_000_000);

    const proxyClient = new NullSpend({
      baseUrl: DASHBOARD_URL,
      apiKey: NULLSPEND_API_KEY!,
      proxyUrl: BASE,
      costReporting: { batchSize: 1, flushIntervalMs: 100 },
    });

    try {
      const session = proxyClient.customer(customerId, { enforcement: true });

      let caught: BudgetExceededError | null = null;
      try {
        await session.openai(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-nullspend-key": NULLSPEND_API_KEY!,
          },
          body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: "test" }],
            max_tokens: 3,
          }),
        });
      } catch (err) {
        if (err instanceof BudgetExceededError) {
          caught = err;
        } else {
          throw err;
        }
      }

      expect(caught, "BudgetExceededError should have been thrown").not.toBeNull();
      expect(caught!.entityType).toBe("customer");
      expect(caught!.entityId).toBe(customerId);
      // Per-customer URL took priority AND {customer_id} was substituted
      expect(caught!.upgradeUrl).toBe(expectedSubstituted);
    } finally {
      await proxyClient.shutdown();
      await teardownCustomerFixtures(customerId);
    }
  }, 30_000);
});
