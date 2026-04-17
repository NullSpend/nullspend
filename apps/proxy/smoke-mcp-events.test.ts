/**
 * Nightly smoke test — MCP cost event DB round-trip (NF-1 regression).
 *
 * This file is the TRIMMED nightly-tier remnant of the original
 * apps/proxy/smoke-mcp-events.test.ts (2026-04-16). All route-level
 * coverage was moved to:
 *   - apps/proxy/src/__tests__/mcp-route.test.ts (39 unit tests —
 *     validation, auth, NF-1 orgId arg-shape regression, etc.)
 *
 * The ONLY assertion unique to live-stack integration is the full DB
 * round-trip: "after sending N MCP events, do they actually land in
 * Postgres with the correct org_id?" That's what this file preserves.
 *
 * Wave 3 Phase 3 gates this behind SMOKE_LIVE=1 for the nightly tier.
 * See `docs/internal/wave-3-phase-2-port-plan-20260416.md` D8.
 *
 * Requires live proxy + NULLSPEND_API_KEY + NULLSPEND_SMOKE_KEY_ID +
 * DATABASE_URL.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  NULLSPEND_API_KEY,
  NULLSPEND_SMOKE_KEY_ID,
  DATABASE_URL,
  isServerUp,
} from "./smoke-test-helpers.js";

describe("MCP cost event DB round-trip (NF-1) — nightly smoke", () => {
  let sql: postgres.Sql;
  let expectedOrgId: string;

  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable.");
    if (!NULLSPEND_API_KEY) throw new Error("NULLSPEND_API_KEY required.");
    if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required.");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required.");

    sql = postgres(DATABASE_URL, { max: 2, idle_timeout: 10 });

    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    expectedOrgId = key.org_id;
  });

  afterAll(async () => {
    await sql?.end();
  });

  it("accepts batch and persists cost events with correct org_id (end-to-end DB)", async () => {
    const uniqueTag = `smoke-mcp-events-${Date.now()}`;
    const events = [
      {
        toolName: uniqueTag,
        serverName: "smoke-test",
        durationMs: 123,
        costMicrodollars: 1500,
        status: "success",
      },
      {
        toolName: uniqueTag,
        serverName: "smoke-test",
        durationMs: 234,
        costMicrodollars: 2500,
        status: "success",
      },
    ];

    const res = await fetch(`${BASE}/v1/mcp/events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-key": NULLSPEND_API_KEY!,
      },
      body: JSON.stringify({ events }),
    });

    expect(res.status).toBe(200);
    const payload = await res.json();
    expect(payload.accepted).toBe(2);

    // Verify cost events landed in Postgres with correct orgId.
    // This is the end-to-end proof of NF-1 — if orgId plumbing is
    // broken, the rows would either not land or land with null orgId.
    const start = Date.now();
    let rows: Record<string, unknown>[] = [];
    while (Date.now() - start < 15_000) {
      rows = (await sql`
        SELECT org_id, provider, tool_name, cost_microdollars, tool_server
        FROM cost_events
        WHERE tool_name = ${uniqueTag} AND provider = 'mcp'
        ORDER BY created_at ASC
      `) as unknown as Record<string, unknown>[];
      if (rows.length === 2) break;
      await new Promise((r) => setTimeout(r, 500));
    }

    expect(rows.length).toBe(2);
    expect(rows[0].org_id).toBe(expectedOrgId);
    expect(rows[1].org_id).toBe(expectedOrgId);
    expect(rows[0].provider).toBe("mcp");
    expect(rows[0].tool_server).toBe("smoke-test");
    const costs = rows.map((r) => Number(r.cost_microdollars)).sort((a, b) => a - b);
    expect(costs).toEqual([1500, 2500]);
  }, 30_000);
});
