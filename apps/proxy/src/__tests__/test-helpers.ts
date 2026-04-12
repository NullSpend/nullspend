/**
 * Shared test mock factories for proxy unit tests.
 *
 * These provide consistent mock objects for cloudflare:workers, db.js,
 * and common Env bindings. Import and spread into vi.mock() calls.
 *
 * NOTE: vi.mock() calls must be at the top level of each test file
 * (vitest hoists them). These helpers provide the factory objects,
 * not the vi.mock() calls themselves.
 */
import { vi } from "vitest";

/**
 * Mock cloudflare:workers — waitUntil executes promises and swallows errors.
 * Use with: vi.mock("cloudflare:workers", () => cloudflareWorkersMock())
 */
export function cloudflareWorkersMock() {
  return {
    waitUntil: vi.fn((p: Promise<unknown>) => { p.catch(() => {}); }),
    DurableObject: class DurableObject {
      ctx: unknown;
      env: unknown;
      constructor(ctx: unknown, env: unknown) { this.ctx = ctx; this.env = env; }
    },
  };
}

/**
 * Create a mock postgres.js tagged template function.
 * Returns an empty array by default; override with mockSql.mockResolvedValueOnce([...]).
 */
export function createMockSql() {
  return vi.fn().mockResolvedValue([]);
}

/**
 * Create a mock Drizzle DB with chainable select/from/where and transaction support.
 */
export function createMockDrizzleDb() {
  const mockUpdateWhere = vi.fn().mockResolvedValue(undefined);
  const mockUpdateSet = vi.fn().mockReturnValue({ where: mockUpdateWhere });
  const mockDb: Record<string, unknown> = {
    select: vi.fn().mockReturnThis(),
    from: vi.fn().mockReturnThis(),
    where: vi.fn().mockResolvedValue([]),
    insert: vi.fn().mockReturnValue({
      values: vi.fn().mockReturnValue({
        onConflictDoNothing: vi.fn().mockResolvedValue(undefined),
      }),
    }),
    update: vi.fn().mockReturnValue({ set: mockUpdateSet }),
    transaction: vi.fn(async (cb: (tx: unknown) => Promise<void>) => {
      await cb(mockDb);
    }),
  };
  return { mockDb, mockUpdateWhere, mockUpdateSet };
}

/**
 * Standard Env mock with all current bindings.
 * Override specific bindings as needed: makeEnv({ CACHE_KV: customMock })
 */
export function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    HYPERDRIVE: { connectionString: "postgresql://postgres:postgres@127.0.0.1:54322/postgres" },
    IP_RATE_LIMITER: { limit: vi.fn().mockResolvedValue({ success: true }) },
    KEY_RATE_LIMITER: { limit: vi.fn().mockResolvedValue({ success: true }) },
    CACHE_KV: { get: vi.fn().mockResolvedValue(null), put: vi.fn(), delete: vi.fn() },
    METRICS: { writeDataPoint: vi.fn() },
    USER_BUDGET: { idFromName: vi.fn(), get: vi.fn() },
    RECONCILE_QUEUE: { send: vi.fn() },
    COST_EVENT_QUEUE: { send: vi.fn(), sendBatch: vi.fn() },
    BODY_STORAGE: { put: vi.fn(), get: vi.fn().mockResolvedValue(null) },
    ...overrides,
  } as unknown as Env;
}
