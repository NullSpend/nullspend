/**
 * Smoke tests for /health/metrics endpoint and proxy latency headers.
 * Verifies Analytics Engine pipeline works end-to-end.
 *
 * Requires:
 *   - Deployed proxy worker (or `pnpm proxy:dev`)
 *   - CF_ACCOUNT_ID and CF_API_TOKEN worker secrets set
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run --config vitest.smoke.config.ts smoke-metrics.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  BASE,
  OPENAI_API_KEY,
  isServerUp,
  authHeaders,
  smallRequest,
} from "./smoke-test-helpers.js";

describe("metrics smoke tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required for smoke tests.");
    }
  });

  describe("GET /health/metrics", () => {
    it("returns 200 with valid JSON shape", async () => {
      const res = await fetch(`${BASE}/health/metrics`);

      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("application/json");

      const body = await res.json();
      // Verify shape matches LatencyMetrics interface
      expect(body).toHaveProperty("overhead_ms");
      expect(body).toHaveProperty("upstream_ms");
      expect(body).toHaveProperty("total_ms");
      expect(body).toHaveProperty("request_count");
      expect(body).toHaveProperty("window_seconds");
      expect(body).toHaveProperty("measured_at");

      // Each percentile group has p50, p95, p99
      for (const key of ["overhead_ms", "upstream_ms", "total_ms"]) {
        expect(body[key]).toHaveProperty("p50");
        expect(body[key]).toHaveProperty("p95");
        expect(body[key]).toHaveProperty("p99");
        expect(typeof body[key].p50).toBe("number");
        expect(typeof body[key].p95).toBe("number");
        expect(typeof body[key].p99).toBe("number");
      }

      expect(typeof body.request_count).toBe("number");
      expect(body.window_seconds).toBe(300);
    });

    it("returns Prometheus format with Accept: text/plain", async () => {
      const res = await fetch(`${BASE}/health/metrics`, {
        headers: { Accept: "text/plain" },
      });

      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("text/plain");

      const text = await res.text();
      expect(text).toContain("# HELP nullspend_proxy_overhead_ms");
      expect(text).toContain("# TYPE nullspend_proxy_overhead_ms summary");
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.5"}');
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.95"}');
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.99"}');
      expect(text).toContain("nullspend_proxy_overhead_ms_count");

      expect(text).toContain("# HELP nullspend_upstream_latency_ms");
      expect(text).toContain("# HELP nullspend_total_latency_ms");
    });

    it("populates with non-zero data after traffic", async () => {
      // Generate a proxy request to ensure AE has data
      const proxyRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });
      expect(proxyRes.status).toBe(200);
      await proxyRes.text(); // consume body

      // Poll /health/metrics until request_count > 0 (AE ingestion delay)
      const deadline = Date.now() + 60_000;
      let lastBody: Record<string, unknown> | null = null;

      while (Date.now() < deadline) {
        const res = await fetch(`${BASE}/health/metrics`);
        if (res.status === 200) {
          const body = await res.json();
          lastBody = body;
          if (typeof body.request_count === "number" && body.request_count > 0) {
            // AE data is populated
            expect(body.overhead_ms.p50).toBeGreaterThanOrEqual(0);
            expect(body.upstream_ms.p50).toBeGreaterThanOrEqual(0);
            expect(body.total_ms.p50).toBeGreaterThanOrEqual(0);
            return; // test passes
          }
        } else {
          await res.text(); // consume error body
        }
        await new Promise((r) => setTimeout(r, 5_000));
      }

      // If we get here, AE ingestion didn't complete in time.
      // This may mean CF_ACCOUNT_ID / CF_API_TOKEN secrets aren't set,
      // or AE ingestion is delayed beyond 60s.
      console.warn(
        "[smoke-metrics] AE data not populated after 60s. Last response:",
        lastBody,
      );
      // Don't hard-fail — AE ingestion can be slow. Log warning and skip.
      expect.soft(lastBody?.request_count).toBeGreaterThan(0);
    }, 90_000);
  });

  describe("latency headers on proxied requests", () => {
    it("includes Server-Timing header with overhead, upstream, and total", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const serverTiming = res.headers.get("Server-Timing");
      expect(serverTiming).toBeTruthy();
      expect(serverTiming).toContain("overhead;dur=");
      expect(serverTiming).toContain("upstream;dur=");
      expect(serverTiming).toContain("total;dur=");

      await res.text(); // consume body
    }, 30_000);

    it("includes x-nullspend-overhead-ms header with numeric value", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const overhead = res.headers.get("x-nullspend-overhead-ms");
      expect(overhead).toBeTruthy();
      expect(overhead).toMatch(/^\d+$/);

      await res.text(); // consume body
    }, 30_000);
  });
});
