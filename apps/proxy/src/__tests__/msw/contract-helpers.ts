/**
 * Shared helpers for in-process contract tests.
 *
 * Pattern rules for contract tests using this module:
 *
 * 1. Use ROUTE HANDLER invocation (import handleX from routes/x.js)
 *    unless the tested behavior happens PRE-ROUTING (body parsing,
 *    top-level auth, etc). Body-size check is the only current example
 *    of a pre-routing test — use invokeWorker() for that.
 *
 * 2. Contract tests that rely on MSW handlers (setupFiles in
 *    vitest.config.ts) must NOT overwrite globalThis.fetch with
 *    vi.fn(). Overwriting fetch bypasses MSW's interceptor silently.
 *    MSW is configured onUnhandledRequest: "bypass" so existing tests
 *    using globalThis.fetch overwrites are unaffected, but NEW contract
 *    tests that want MSW must leave globalThis.fetch alone.
 *
 * 3. When testing denial paths (budget denied, rate limited, loop
 *    detected), mock `budget-do-client.js` to return the relevant
 *    DoCheckResult shape. The DO itself is covered in
 *    user-budget-do.do.test.ts (workerd pool).
 */

import { vi } from "vitest";
import type { RequestContext } from "../../lib/context.js";

export const TEST_CONNECTION_STRING =
  "postgresql://postgres:postgres@127.0.0.1:54322/postgres";

export const TEST_TRACE_ID = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4";

export const MAX_BODY_SIZE = 1_048_576;

export function makeContractEnv(overrides: Partial<Env> = {}): Env {
  return {
    HYPERDRIVE: { connectionString: TEST_CONNECTION_STRING },
    IP_RATE_LIMITER: { limit: vi.fn().mockResolvedValue({ success: true }) },
    KEY_RATE_LIMITER: { limit: vi.fn().mockResolvedValue({ success: true }) },
    CACHE_KV: {
      get: vi.fn().mockResolvedValue(null),
      put: vi.fn(),
      delete: vi.fn(),
    },
    METRICS: { writeDataPoint: vi.fn() },
    USER_BUDGET: {
      idFromName: vi.fn().mockReturnValue("do-id"),
      get: vi.fn().mockReturnValue({}),
    },
    RECONCILE_QUEUE: { send: vi.fn() },
    COST_EVENT_QUEUE: { send: vi.fn(), sendBatch: vi.fn() },
    BODY_STORAGE: { put: vi.fn(), get: vi.fn().mockResolvedValue(null) },
    ...overrides,
  } as unknown as Env;
}

export function makeContractCtx(
  body: Record<string, unknown>,
  overrides: Partial<RequestContext> = {},
): RequestContext {
  const bodyText = JSON.stringify(body);
  return {
    body,
    bodyText,
    bodyByteLength: new TextEncoder().encode(bodyText).byteLength,
    auth: {
      userId: "user-1",
      keyId: "key-1",
      hasWebhooks: false,
      hasBudgets: false,
      orgId: "org-test",
      apiVersion: "2026-04-01",
      defaultTags: {},
    },
    ownerId: "org-test",
    connectionString: TEST_CONNECTION_STRING,
    skipDbWrites: true,
    sessionId: null,
    traceId: TEST_TRACE_ID,
    tags: {},
    customerId: null,
    customerWarning: null,
    webhookDispatcher: null,
    resolvedApiVersion: "2026-04-01",
    requestStartMs: performance.now(),
    requestLoggingEnabled: false,
    finalize: false,
    ...overrides,
  };
}

export function makeContractRequest(
  pathOrUrl: string,
  body: BodyInit | null,
  headers: Record<string, string> = {},
  init: Partial<RequestInit> = {},
): Request {
  const url = pathOrUrl.startsWith("http")
    ? pathOrUrl
    : `http://localhost${pathOrUrl}`;
  const finalInit: RequestInit = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-nullspend-key": "ns_key_test_contract",
      ...headers,
    },
    body,
    ...init,
  };
  return new Request(url, finalInit);
}

/**
 * Invoke the top-level worker fetch handler in-process.
 *
 * Use ONLY for pre-routing tests (body-size, top-level auth failures).
 * For route-handler-level behavior, import the handler directly.
 */
export async function invokeWorker(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const worker = (await import("../../index.js")).default;
  return worker.fetch(request, env, ctx);
}

/**
 * Minimal ExecutionContext mock. waitUntil swallows errors so tests
 * don't see unhandled rejections from async work kicked off inside
 * the handler.
 */
export function makeExecutionContext(): ExecutionContext {
  return {
    waitUntil: (p: Promise<unknown>) => {
      void Promise.resolve(p).catch(() => {});
    },
    passThroughOnException: () => {},
    props: {},
  } as unknown as ExecutionContext;
}
