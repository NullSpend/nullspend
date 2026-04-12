import type postgres from "postgres";

export const BASE = process.env.PROXY_URL ?? `http://127.0.0.1:${process.env.PROXY_PORT ?? "8787"}`;
export const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
export const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
export const NULLSPEND_API_KEY = process.env.NULLSPEND_API_KEY;
export const NULLSPEND_SMOKE_USER_ID = process.env.NULLSPEND_SMOKE_USER_ID;
export const NULLSPEND_SMOKE_KEY_ID = process.env.NULLSPEND_SMOKE_KEY_ID;
export const DATABASE_URL = process.env.DATABASE_URL;
export const INTERNAL_SECRET = process.env.INTERNAL_SECRET;

/**
 * Build auth headers for OpenAI proxy requests.
 * Uses x-nullspend-key (API key auth). Proxy derives userId/keyId from the key.
 */
export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${OPENAI_API_KEY}`,
    "x-nullspend-key": NULLSPEND_API_KEY!,
  };
  if (extra) Object.assign(h, extra);
  return h;
}

export function smallRequest(overrides: Record<string, unknown> = {}) {
  return JSON.stringify({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say ok" }],
    max_tokens: 3,
    ...overrides,
  });
}

/**
 * Build auth headers for Anthropic proxy requests.
 * Uses x-nullspend-key (API key auth). Proxy derives userId/keyId from the key.
 */
export function anthropicAuthHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY!,
    "x-nullspend-key": NULLSPEND_API_KEY!,
  };
  if (extra) Object.assign(h, extra);
  return h;
}

export function smallAnthropicRequest(
  overrides: Record<string, unknown> = {},
) {
  return JSON.stringify({
    model: "claude-3-haiku-20240307",
    max_tokens: 10,
    messages: [{ role: "user", content: "Say ok" }],
    ...overrides,
  });
}

export async function isServerUp(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function waitForCostEvent(
  sql: postgres.Sql,
  requestId: string,
  timeoutMs = 15_000,
  provider = "openai",
): Promise<Record<string, unknown> | null> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const rows = await sql`
      SELECT * FROM cost_events
      WHERE request_id = ${requestId} AND provider = ${provider}
      LIMIT 1
    `;
    if (rows.length > 0) return rows[0] as Record<string, unknown>;
    await new Promise((r) => setTimeout(r, 500));
  }
  return null;
}

/**
 * Poll Postgres until budget spend is > 0, or timeout.
 * Used to wait for async reconciliation (waitUntil + queue consumer)
 * to write spend back to Postgres after a proxied request completes.
 */
export async function waitForBudgetSpend(
  sql: postgres.Sql,
  entityType: string,
  entityId: string,
  timeoutMs = 15_000,
): Promise<number> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const rows = await sql`
      SELECT spend_microdollars::text as spend
      FROM budgets
      WHERE entity_type = ${entityType} AND entity_id = ${entityId}
    `;
    if (rows.length > 0 && Number(rows[0].spend) > 0) {
      return Number(rows[0].spend);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return 0;
}

/**
 * Call the proxy's internal budget invalidation endpoint to properly clean up
 * all three layers of budget state: DO SQLite, DO lookup cache, and auth cache.
 *
 * Requires INTERNAL_SECRET in .env.smoke.
 */
export async function invalidateBudget(
  ownerId: string,
  entityType: string,
  entityId: string,
  action: "remove" | "reset_spend" = "remove",
): Promise<void> {
  if (!INTERNAL_SECRET) {
    throw new Error("INTERNAL_SECRET required in .env.smoke for budget invalidation");
  }
  const res = await fetch(`${BASE}/internal/budget/invalidate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${INTERNAL_SECRET}`,
    },
    body: JSON.stringify({ action, ownerId, entityType, entityId }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Budget invalidation failed (${res.status}): ${body}`);
  }
}

/**
 * Force the proxy to sync budget state from Postgres to the Durable Object.
 * This bypasses all Worker isolate caches — queries Postgres directly and
 * calls doBudgetPopulate on the DO (which is a single global instance).
 *
 * Call this after inserting/updating budgets in Postgres to ensure the DO
 * has the latest state regardless of which Worker isolate handles the request.
 */
export async function syncBudget(
  ownerId: string,
  entityType: string,
  entityId: string,
): Promise<void> {
  if (!INTERNAL_SECRET) {
    throw new Error("INTERNAL_SECRET required in .env.smoke for budget sync");
  }
  const res = await fetch(`${BASE}/internal/budget/invalidate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${INTERNAL_SECRET}`,
    },
    body: JSON.stringify({ action: "sync", ownerId, entityType, entityId }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Budget sync failed (${res.status}): ${body}`);
  }
}

export async function countCostEventsSince(
  sql: postgres.Sql,
  since: Date,
): Promise<number> {
  const rows = await sql`
    SELECT COUNT(*)::int as count FROM cost_events
    WHERE created_at >= ${since.toISOString()}
  `;
  return rows[0].count as number;
}

/**
 * Call the proxy's auth_only invalidation. Used by metadata-only edits
 * (like organizations.metadata.upgradeUrl) where there's no budget entity
 * to sync but the auth cache must be cleared so the next request picks
 * up the new value within a single round trip.
 *
 * Requires INTERNAL_SECRET in .env.smoke.
 */
export async function invalidateAuthOnly(ownerId: string): Promise<void> {
  if (!INTERNAL_SECRET) {
    throw new Error("INTERNAL_SECRET required in .env.smoke for auth invalidation");
  }
  const res = await fetch(`${BASE}/internal/budget/invalidate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${INTERNAL_SECRET}`,
    },
    body: JSON.stringify({ action: "auth_only", ownerId }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Auth invalidation failed (${res.status}): ${body}`);
  }
}
