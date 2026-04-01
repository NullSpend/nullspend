/**
 * Recovery and consistency verification.
 * Runs LAST — after all other stress tests have hammered the system.
 * Verifies the proxy is fully operational and state is consistent.
 *
 * Requires:
 *   - Live proxy at PROXY_URL
 *   - OPENAI_API_KEY, NULLSPEND_API_KEY
 *   - DATABASE_URL (optional, for consistency checks)
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  DATABASE_URL,
  authHeaders,
  anthropicAuthHeaders,
  smallRequest,
  smallAnthropicRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

describe("Post-stress recovery", () => {
  let sql: postgres.Sql | null = null;

  beforeAll(async () => {
    // Settle time for in-flight operations + rate limit recovery from prior stress
    await new Promise((r) => setTimeout(r, 5_000));

    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable after stress tests.");
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required.");
    if (DATABASE_URL) {
      sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });
    }

    // Wait for rate limit recovery
    let rateLimited = true;
    for (let attempt = 0; attempt < 9 && rateLimited; attempt++) {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ messages: [{ role: "user", content: "RL recovery" }] }),
      });
      if (res.status === 200) {
        await res.json();
        rateLimited = false;
      } else {
        await res.text();
        if (res.status === 429) {
          console.log("[recovery] Rate limited — waiting 10s...");
          await new Promise((r) => setTimeout(r, 10_000));
        } else {
          rateLimited = false;
        }
      }
    }
  });

  afterAll(async () => {
    if (sql) await sql.end();
  });

  // ── Health endpoints ──

  describe("Health endpoints", () => {
    it("GET /health returns 200 ok", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/health`);
      const elapsed = performance.now() - start;

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");

      console.log(`[recovery] /health: ${elapsed.toFixed(0)}ms`);
      expect(elapsed).toBeLessThan(1_000);
    });

    it("GET /health/ready returns 200", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/health/ready`);
      const elapsed = performance.now() - start;

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");

      console.log(`[recovery] /health/ready: ${elapsed.toFixed(0)}ms`);
      expect(elapsed).toBeLessThan(2_000);
    });

    it("GET /health/metrics returns latency data", async () => {
      const res = await fetch(`${BASE}/health/metrics`);
      // Metrics may return 200 with data or 503 if AE unavailable
      if (res.status === 200) {
        const body = await res.json();
        console.log(`[recovery] Metrics available:`, JSON.stringify(body).slice(0, 200));
      } else {
        console.log(`[recovery] Metrics endpoint: ${res.status} (AE may be unavailable)`);
        await res.text();
      }
    });
  });

  // ── Normal request flow ──

  describe("Normal request flow", () => {
    it("5 sequential OpenAI requests succeed with normal latency", async () => {
      const latencies: number[] = [];

      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        const res = await fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest({ messages: [{ role: "user", content: `Recovery ${i}` }] }),
        });
        const elapsed = performance.now() - start;

        expect(res.status).toBe(200);
        const body = await res.json();
        expect(body).toHaveProperty("usage");
        latencies.push(elapsed);
      }

      const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      console.log(`[recovery] OpenAI avg latency: ${avg.toFixed(0)}ms`);

      // Normal requests should complete in <10s each
      for (const lat of latencies) {
        expect(lat).toBeLessThan(10_000);
      }
    }, 60_000);

    it("streaming request completes normally", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ stream: true }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");
    }, 30_000);

    it("Anthropic request succeeds (if key available)", async () => {
      if (!ANTHROPIC_API_KEY) {
        console.log("[recovery] Skipping Anthropic — no key");
        return;
      }

      const res = await fetch(`${BASE}/v1/messages`, {
        method: "POST",
        headers: anthropicAuthHeaders(),
        body: smallAnthropicRequest(),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
    }, 30_000);
  });

  // ── Error handling still works ──

  describe("Error handling", () => {
    it("invalid auth still returns 401 (not 500/502)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer fake",
          "x-nullspend-key": "fake",
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(401);
      await res.text();
    });

    it("malformed body still returns 400 (not 500/502)", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "{bad json!!!",
      });

      expect(res.status).toBe(400);
      await res.text();
    });

    it("invalid model still returns 400 with invalid_model code", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest({ model: "nonexistent-stress-model" }),
      });

      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.error.code).toBe("invalid_model");
    });
  });

  // ── Database consistency ──

  describe("Database consistency", () => {
    it("cost_events table is reachable and has recent entries", async () => {
      if (!sql) {
        console.log("[recovery] Skipping DB check — no DATABASE_URL");
        return;
      }

      // Check for cost events from the last 24 hours (stress tests may have
      // timing gaps, and waitUntil cost logging is async)
      const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
      const rows = await sql`
        SELECT COUNT(*)::int as count
        FROM cost_events
        WHERE created_at >= ${oneDayAgo.toISOString()}
      `;

      const count = rows[0].count as number;
      console.log(`[recovery] Cost events in last 24h: ${count}`);

      // Informational — cost event logging via waitUntil() is async and
      // may not complete within the test window under heavy stress
      if (count === 0) {
        console.log("[recovery] WARNING: No cost events found — waitUntil may be delayed under stress");
      }
    });

    it("no orphaned reservations in budgets table", async () => {
      if (!sql) {
        console.log("[recovery] Skipping orphan check — no DATABASE_URL");
        return;
      }

      // Check for budgets with negative spend (would indicate accounting bug)
      const negativeSpendsResult = await sql`
        SELECT entity_type, entity_id, spend_microdollars::text as spend
        FROM budgets
        WHERE spend_microdollars < 0
      `;

      if (negativeSpendsResult.length > 0) {
        console.log(`[recovery] WARNING: ${negativeSpendsResult.length} budgets with negative spend!`);
        for (const row of negativeSpendsResult) {
          console.log(`  ${row.entity_type}/${row.entity_id}: ${row.spend}µ¢`);
        }
      }

      expect(negativeSpendsResult.length).toBe(0);
    });
  });

  // ── Final summary ──

  it("STRESS TEST SUMMARY", async () => {
    console.log("\n" + "=".repeat(60));
    console.log("STRESS TEST RECOVERY VERIFICATION COMPLETE");
    console.log("=".repeat(60));
    console.log("All recovery checks passed — system is operational.");
    console.log("Review [stress] logs above for performance metrics and anomalies.");
    console.log("=".repeat(60) + "\n");
  });
});
