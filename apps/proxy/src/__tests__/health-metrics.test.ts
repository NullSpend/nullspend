import { describe, it, expect, vi, beforeEach } from "vitest";
import { handleMetrics } from "../routes/metrics.js";

const mockKvStore = new Map<string, string>();
const mockKv = {
  get: vi.fn(async (key: string) => {
    const val = mockKvStore.get(key);
    return val ? JSON.parse(val) : null;
  }),
  put: vi.fn(async (key: string, value: string) => {
    mockKvStore.set(key, value);
  }),
};

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    CACHE_KV: mockKv,
    ...overrides,
  } as unknown as Env;
}

function makeRequest(accept?: string, queryParams?: Record<string, string>): Request {
  const headers: Record<string, string> = {};
  if (accept) headers["accept"] = accept;
  const url = new URL("http://localhost/health/metrics");
  if (queryParams) {
    for (const [k, v] of Object.entries(queryParams)) {
      url.searchParams.set(k, v);
    }
  }
  return new Request(url.toString(), { headers });
}

// Matches real AE response: Float64 as numbers, UInt64 as strings
const MOCK_AE_RESPONSE = {
  data: [
    {
      p50_overhead: 5.2,
      p95_overhead: 12.8,
      p99_overhead: 23.1,
      p50_upstream: 800,
      p95_upstream: 1200,
      p99_upstream: 2500,
      p50_total: 805,
      p95_total: 1213,
      p99_total: 2523,
      request_count: "1847",
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockKvStore.clear();
  vi.stubGlobal("fetch", vi.fn());
});

describe("handleMetrics", () => {
  describe("KV cache hit", () => {
    it("returns cached metrics without querying AE", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 1847,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest(), makeEnv());

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms.p50).toBe(5);
      expect(body.request_count).toBe(1847);
      expect(vi.mocked(globalThis.fetch)).not.toHaveBeenCalled();
    });
  });

  describe("KV cache miss — AE query", () => {
    it("queries AE and returns metrics on cache miss", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "test-account",
        CF_API_TOKEN: "test-token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms.p50).toBe(5);
      expect(body.overhead_ms.p95).toBe(13);
      expect(body.overhead_ms.p99).toBe(23);
      expect(body.request_count).toBe(1847);
      expect(body.window_seconds).toBe(300);
      expect(body.measured_at).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it("caches the AE result in KV", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "test-account",
        CF_API_TOKEN: "test-token",
      });

      await handleMetrics(makeRequest(), env);

      expect(mockKv.put).toHaveBeenCalledWith(
        "metrics:proxy_latency",
        expect.any(String),
        { expirationTtl: 90 },
      );
    });

    it("sends correct Authorization header to AE SQL API", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct-123",
        CF_API_TOKEN: "my-secret-token",
      });

      await handleMetrics(makeRequest(), env);

      expect(vi.mocked(globalThis.fetch)).toHaveBeenCalledWith(
        "https://api.cloudflare.com/client/v4/accounts/acct-123/analytics_engine/sql",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer my-secret-token",
          }),
        }),
      );
    });

    it("includes FORMAT JSON in the SQL query", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct-123",
        CF_API_TOKEN: "token",
      });

      await handleMetrics(makeRequest(), env);

      const call = vi.mocked(globalThis.fetch).mock.calls[0];
      const body = call[1]?.body as string;
      expect(body).toContain("FORMAT JSON");
    });

    it("uses quantileExactWeighted in the SQL query", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct-123",
        CF_API_TOKEN: "token",
      });

      await handleMetrics(makeRequest(), env);

      const call = vi.mocked(globalThis.fetch).mock.calls[0];
      const body = call[1]?.body as string;
      expect(body).toContain("quantileExactWeighted");
      expect(body).not.toContain("quantileWeighted(");
    });
  });

  describe("AE not configured", () => {
    it("returns empty metrics when CF_ACCOUNT_ID is missing", async () => {
      const res = await handleMetrics(makeRequest(), makeEnv());

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.request_count).toBe(0);
    });
  });

  describe("AE query failure", () => {
    it("returns 503 with upstream_status when AE SQL API returns error", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Internal error", { status: 500 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(503);
      const body = await res.json();
      expect(body.error.code).toBe("metrics_unavailable");
      expect(body.error.details.upstream_status).toBe(500);
    });

    it("returns 503 with upstream_status for 401 (expired token)", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Unauthorized", { status: 401 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "expired-token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(503);
      const body = await res.json();
      expect(body.error.details.upstream_status).toBe(401);
    });

    it("returns 503 when fetch throws", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockRejectedValue(new Error("network error"));

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(503);
    });

    it("returns timeout-specific message when fetch times out", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      const timeoutErr = new DOMException("signal timed out", "TimeoutError");
      vi.mocked(globalThis.fetch).mockRejectedValue(timeoutErr);

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(503);
      const body = await res.json();
      expect(body.error.message).toContain("timed out");
    });

    it("returns 503 when AE returns malformed JSON", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("not valid json{{{", { status: 200, headers: { "Content-Type": "application/json" } }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(503);
    });
  });

  describe("negative caching", () => {
    it("caches empty metrics on AE non-200 to prevent thundering herd", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Server Error", { status: 500 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      await handleMetrics(makeRequest(), env);

      // Should have written a negative-cache sentinel with short TTL
      expect(mockKv.put).toHaveBeenCalledWith(
        "metrics:proxy_latency",
        expect.any(String),
        { expirationTtl: 30 },
      );

      // Subsequent request hits cache instead of AE
      const res2 = await handleMetrics(makeRequest(), env);
      expect(res2.status).toBe(200);
      const body = await res2.json();
      expect(body.request_count).toBe(0);
      // fetch should only have been called once (first request), not twice
      expect(vi.mocked(globalThis.fetch)).toHaveBeenCalledTimes(1);
    });

    it("does not negative-cache on 401 (auth errors stay loud)", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Unauthorized", { status: 401 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "expired-token",
      });

      await handleMetrics(makeRequest(), env);

      // KV put should NOT have been called — auth errors should not be cached
      expect(mockKv.put).not.toHaveBeenCalled();
    });

    it("does not negative-cache on 403 (permission errors stay loud)", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Forbidden", { status: 403 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "wrong-perms-token",
      });

      await handleMetrics(makeRequest(), env);

      expect(mockKv.put).not.toHaveBeenCalled();
    });

    it("still returns 503 when negative-cache write fails", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.spyOn(console, "warn").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("Server Error", { status: 500 }),
      );
      // Make the negative-cache KV write fail
      mockKv.put.mockRejectedValueOnce(new Error("KV unavailable"));

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      // Should still return 503 despite cache write failure
      expect(res.status).toBe(503);
      const body = await res.json();
      expect(body.error.details.upstream_status).toBe(500);
    });

    it("caches empty metrics on fetch timeout", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      const timeoutErr = new DOMException("signal timed out", "TimeoutError");
      vi.mocked(globalThis.fetch).mockRejectedValue(timeoutErr);

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      await handleMetrics(makeRequest(), env);

      expect(mockKv.put).toHaveBeenCalledWith(
        "metrics:proxy_latency",
        expect.any(String),
        { expirationTtl: 30 },
      );
    });
  });

  describe("KV failure resilience", () => {
    it("falls through to AE query when KV read fails", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      mockKv.get.mockRejectedValueOnce(new Error("KV unavailable"));
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.request_count).toBe(1847);
    });

    it("still returns AE result when KV write fails", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      mockKv.put.mockRejectedValueOnce(new Error("KV write failed"));
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({
        CF_ACCOUNT_ID: "acct",
        CF_API_TOKEN: "token",
      });

      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms.p50).toBe(5);
    });
  });

  describe("AE value coercion", () => {
    it("returns zeros when AE returns null values via NaN serialization (JSON.stringify(NaN) → null)", async () => {
      // NaN values become null through JSON round-trip, so this is
      // functionally equivalent to the explicit null test below.
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify({
          data: [{
            p50_overhead: NaN, p95_overhead: NaN, p99_overhead: NaN,
            p50_upstream: NaN, p95_upstream: NaN, p99_upstream: NaN,
            p50_total: NaN, p95_total: NaN, p99_total: NaN,
            request_count: "0",
          }],
        }), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.upstream_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.total_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.request_count).toBe(0);
    });

    it("returns zeros when AE returns null values (no data in window)", async () => {
      // Matches observed AE behavior: Float64 returns null, UInt64 returns "0"
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify({
          data: [{
            p50_overhead: null, p95_overhead: null, p99_overhead: null,
            p50_upstream: null, p95_upstream: null, p99_upstream: null,
            p50_total: null, p95_total: null, p99_total: null,
            request_count: "0",
          }],
        }), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.upstream_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.total_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.request_count).toBe(0);
    });

    it("returns zeros when AE returns empty data array", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify({ data: [] }), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.request_count).toBe(0);
    });

    it("returns zeros when AE returns empty string values", async () => {
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify({
          data: [{
            p50_overhead: "", p95_overhead: "", p99_overhead: "",
            p50_upstream: "", p95_upstream: "", p99_upstream: "",
            p50_total: "", p95_total: "", p99_total: "",
            request_count: "",
          }],
        }), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
      expect(body.request_count).toBe(0);
    });

    it("handles UInt64 string values from AE (request_count)", async () => {
      // AE returns UInt64 as strings and Float64 as numbers
      const aeResponse = {
        data: [
          {
            p50_overhead: 151,
            p95_overhead: 264,
            p99_overhead: 355,
            p50_upstream: 287,
            p95_upstream: 584,
            p99_upstream: 1176,
            p50_total: 461,
            p95_total: 765,
            p99_total: 1328,
            request_count: "402",
          },
        ],
      };
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(aeResponse), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms.p50).toBe(151);
      expect(body.request_count).toBe(402);
    });
  });

  describe("AE response key fallback", () => {
    it("parses metrics from 'result' key when 'data' key is absent", async () => {
      const resultResponse = {
        result: [
          {
            p50_overhead: 5.2,
            p95_overhead: 12.8,
            p99_overhead: 23.1,
            p50_upstream: 800,
            p95_upstream: 1200,
            p99_upstream: 2500,
            p50_total: 805,
            p95_total: 1213,
            p99_total: 2523,
            request_count: "1847",
          },
        ],
      };
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(resultResponse), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.overhead_ms.p50).toBe(5);
      expect(body.request_count).toBe(1847);
    });

    it("prefers 'data' key when both 'data' and 'result' are present", async () => {
      const bothKeysResponse = {
        data: [
          {
            p50_overhead: 10,
            p95_overhead: 20,
            p99_overhead: 30,
            p50_upstream: 100,
            p95_upstream: 200,
            p99_upstream: 300,
            p50_total: 110,
            p95_total: 220,
            p99_total: 330,
            request_count: "500",
          },
        ],
        result: [
          {
            p50_overhead: 99,
            p95_overhead: 99,
            p99_overhead: 99,
            p50_upstream: 99,
            p95_upstream: 99,
            p99_upstream: 99,
            p50_total: 99,
            p95_total: 99,
            p99_total: 99,
            request_count: "99",
          },
        ],
      };
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(bothKeysResponse), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      const res = await handleMetrics(makeRequest(), env);

      expect(res.status).toBe(200);
      const body = await res.json();
      // data key should win
      expect(body.overhead_ms.p50).toBe(10);
      expect(body.request_count).toBe(500);
    });
  });

  describe("content negotiation", () => {
    it("returns JSON by default", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest(), makeEnv());

      expect(res.headers.get("content-type")).toContain("application/json");
    });

    it("returns Prometheus format when Accept: text/plain", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest("text/plain"), makeEnv());

      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("text/plain");
      const text = await res.text();
      expect(text).toContain("# HELP nullspend_proxy_overhead_ms");
      expect(text).toContain("# TYPE nullspend_proxy_overhead_ms summary");
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.5"} 5');
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.95"} 12');
      expect(text).toContain('nullspend_proxy_overhead_ms{quantile="0.99"} 23');
      expect(text).toContain("nullspend_proxy_overhead_ms_count 100");
    });

    it("includes Vary: Accept header on JSON response", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest(), makeEnv());

      expect(res.headers.get("vary")).toBe("Accept");
    });

    it("includes Vary: Accept header on Prometheus response", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest("text/plain"), makeEnv());

      expect(res.headers.get("vary")).toBe("Accept");
    });
  });

  describe("metric emission", () => {
    it("emits ae_cache hit metric on KV cache hit", async () => {
      const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      await handleMetrics(makeRequest(), makeEnv());

      const metricCalls = logSpy.mock.calls
        .map(([arg]) => typeof arg === "string" ? arg : "")
        .filter((s) => s.includes('"_metric"'));
      const cacheMetric = metricCalls.find((s) => s.includes("ae_cache"));
      expect(cacheMetric).toBeTruthy();
      expect(JSON.parse(cacheMetric!).hit).toBe(true);
    });

    it("emits ae_cache miss + ae_query_success metrics on AE query", async () => {
      const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify(MOCK_AE_RESPONSE), { status: 200 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      await handleMetrics(makeRequest(), env);

      const metricCalls = logSpy.mock.calls
        .map(([arg]) => typeof arg === "string" ? arg : "")
        .filter((s) => s.includes('"_metric"'));

      const cacheMiss = metricCalls.find((s) => s.includes("ae_cache") && s.includes("false"));
      expect(cacheMiss).toBeTruthy();

      const querySuccess = metricCalls.find((s) => s.includes("ae_query_success"));
      expect(querySuccess).toBeTruthy();
      expect(JSON.parse(querySuccess!).duration_ms).toBeGreaterThanOrEqual(0);
    });

    it("emits ae_query_error metric on AE failure", async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.spyOn(console, "warn").mockImplementation(() => {});
      const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
      vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response("fail", { status: 500 }),
      );

      const env = makeEnv({ CF_ACCOUNT_ID: "acct", CF_API_TOKEN: "token" });
      await handleMetrics(makeRequest(), env);

      const metricCalls = logSpy.mock.calls
        .map(([arg]) => typeof arg === "string" ? arg : "")
        .filter((s) => s.includes('"_metric"'));

      const queryError = metricCalls.find((s) => s.includes("ae_query_error"));
      expect(queryError).toBeTruthy();
      expect(JSON.parse(queryError!).status).toBe(500);
    });
  });

  describe("TTFB metrics", () => {
    it("backfills ttfb_ms for cached entries without it", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
        // No ttfb_ms — simulates old cached data
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest("application/json"), makeEnv());

      expect(res.status).toBe(200);
      const body = await res.json() as Record<string, unknown>;
      expect(body.ttfb_ms).toEqual({ p50: 0, p95: 0, p99: 0 });
    });

    it("returns ttfb_ms from cached entry when present", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        ttfb_ms: { p50: 250, p95: 600, p99: 900 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest("application/json"), makeEnv());

      expect(res.status).toBe(200);
      const body = await res.json() as Record<string, unknown>;
      expect(body.ttfb_ms).toEqual({ p50: 250, p95: 600, p99: 900 });
    });

    it("includes TTFB in Prometheus format when ttfb_ms is present", async () => {
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        ttfb_ms: { p50: 250, p95: 600, p99: 900 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-20T14:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(makeRequest("text/plain"), makeEnv());

      const text = await res.text();
      expect(text).toContain("# HELP nullspend_ttfb_ms");
      expect(text).toContain('nullspend_ttfb_ms{quantile="0.5"} 250');
      expect(text).toContain('nullspend_ttfb_ms{quantile="0.95"} 600');
      expect(text).toContain('nullspend_ttfb_ms{quantile="0.99"} 900');
    });
  });

  describe("query param filters", () => {
    it("uses filter-specific cache key for ?provider=openai", async () => {
      const cached = {
        overhead_ms: { p50: 3, p95: 8, p99: 15 },
        upstream_ms: { p50: 600, p95: 900, p99: 1800 },
        total_ms: { p50: 603, p95: 908, p99: 1815 },
        ttfb_ms: { p50: 200, p95: 400, p99: 700 },
        request_count: 50,
        window_seconds: 300,
        measured_at: "2026-03-24T12:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency:p=openai:m=", JSON.stringify(cached));

      const res = await handleMetrics(
        makeRequest("application/json", { provider: "openai" }),
        makeEnv(),
      );

      expect(res.status).toBe(200);
      const body = await res.json() as Record<string, unknown>;
      expect(body.request_count).toBe(50);
    });

    it("uses separate cache key for ?provider=openai&model=gpt-4o", async () => {
      const cached = {
        overhead_ms: { p50: 2, p95: 5, p99: 10 },
        upstream_ms: { p50: 500, p95: 800, p99: 1500 },
        total_ms: { p50: 502, p95: 805, p99: 1510 },
        ttfb_ms: { p50: 150, p95: 300, p99: 500 },
        request_count: 25,
        window_seconds: 300,
        measured_at: "2026-03-24T12:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency:p=openai:m=gpt-4o", JSON.stringify(cached));

      const res = await handleMetrics(
        makeRequest("application/json", { provider: "openai", model: "gpt-4o" }),
        makeEnv(),
      );

      expect(res.status).toBe(200);
      const body = await res.json() as Record<string, unknown>;
      expect(body.request_count).toBe(25);
    });

    it("rejects filter values with special characters (SQL injection prevention)", async () => {
      const res = await handleMetrics(
        makeRequest("application/json", { provider: "openai'; DROP TABLE--" }),
        makeEnv(),
      );

      expect(res.status).toBe(400);
      const body = await res.json() as { error: { code: string } };
      expect(body.error.code).toBe("bad_request");
    });

    it("rejects filter values with backslash", async () => {
      const res = await handleMetrics(
        makeRequest("application/json", { provider: "openai\\" }),
        makeEnv(),
      );

      expect(res.status).toBe(400);
    });

    it("ignores empty filter values", async () => {
      // ?provider= (empty) should behave like no filter
      const cached = {
        overhead_ms: { p50: 5, p95: 12, p99: 23 },
        upstream_ms: { p50: 800, p95: 1200, p99: 2500 },
        total_ms: { p50: 805, p95: 1213, p99: 2523 },
        ttfb_ms: { p50: 0, p95: 0, p99: 0 },
        request_count: 100,
        window_seconds: 300,
        measured_at: "2026-03-24T12:00:00Z",
      };
      mockKvStore.set("metrics:proxy_latency", JSON.stringify(cached));

      const res = await handleMetrics(
        makeRequest("application/json", { provider: "" }),
        makeEnv(),
      );

      expect(res.status).toBe(200);
      const body = await res.json() as Record<string, unknown>;
      expect(body.request_count).toBe(100);
    });
  });
});
