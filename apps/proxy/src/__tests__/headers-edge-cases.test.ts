import { describe, it, expect } from "vitest";
import {
  buildUpstreamHeaders,
  buildClientHeaders,
  appendTimingHeaders,
} from "../lib/headers.js";

function makeRequest(headers: Record<string, string> = {}): Request {
  return new Request("https://proxy.example.com/v1/chat/completions", {
    method: "POST",
    headers,
    body: "{}",
  });
}

function makeResponse(
  status: number,
  headers: Record<string, string> = {},
): Response {
  return new Response(null, { status, headers });
}

describe("buildUpstreamHeaders edge cases", () => {
  it("always sets content-type: application/json even when client omits it", () => {
    const req = makeRequest({ authorization: "Bearer sk-test" });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("authorization")).toBe("Bearer sk-test");
  });

  it("overrides client content-type to application/json", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      "content-type": "text/plain",
    });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("content-type")).toBe("application/json");
  });

  it("returns minimal headers when no auth or org headers present", () => {
    const req = makeRequest({});
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("authorization")).toBeNull();
    expect(headers.get("openai-organization")).toBeNull();
    expect(headers.get("openai-project")).toBeNull();
  });

  it("forwards openai-organization and openai-project when present", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      "openai-organization": "org-123",
      "openai-project": "proj-456",
    });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("openai-organization")).toBe("org-123");
    expect(headers.get("openai-project")).toBe("proj-456");
  });

  it("forwards traceparent and tracestate to upstream", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      traceparent: "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
      tracestate: "vendor1=value1,vendor2=value2",
    });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBe(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    );
    expect(headers.get("tracestate")).toBe("vendor1=value1,vendor2=value2");
  });

  it("omits traceparent and tracestate when not present", () => {
    const req = makeRequest({ authorization: "Bearer sk-test" });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBeNull();
    expect(headers.get("tracestate")).toBeNull();
  });

  it("does NOT forward x-nullspend-key to upstream", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      "x-nullspend-key": "ns_live_sk_test0001",
    });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("x-nullspend-key")).toBeNull();
  });

  it("does NOT forward host or content-length", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      host: "proxy.example.com",
      "content-length": "42",
    });
    const headers = buildUpstreamHeaders(req);

    expect(headers.get("host")).toBeNull();
    expect(headers.get("content-length")).toBeNull();
  });
});

describe("buildClientHeaders edge cases", () => {
  it("forwards retry-after header when present (rate limiting)", () => {
    const res = makeResponse(429, {
      "content-type": "application/json",
      "retry-after": "30",
      "x-ratelimit-remaining-requests": "0",
      "x-ratelimit-reset-requests": "30s",
    });
    const headers = buildClientHeaders(res);

    expect(headers.get("retry-after")).toBe("30");
    expect(headers.get("x-ratelimit-remaining-requests")).toBe("0");
  });

  it("does NOT include retry-after when not present", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
      "x-request-id": "req-abc",
    });
    const headers = buildClientHeaders(res);

    expect(headers.get("retry-after")).toBeNull();
  });

  it("forwards all x-ratelimit-* headers", () => {
    const res = makeResponse(200, {
      "content-type": "text/event-stream",
      "x-ratelimit-limit-requests": "500",
      "x-ratelimit-limit-tokens": "30000",
      "x-ratelimit-remaining-requests": "499",
      "x-ratelimit-remaining-tokens": "29000",
      "x-ratelimit-reset-requests": "20ms",
      "x-ratelimit-reset-tokens": "100ms",
    });
    const headers = buildClientHeaders(res);

    expect(headers.get("x-ratelimit-limit-requests")).toBe("500");
    expect(headers.get("x-ratelimit-limit-tokens")).toBe("30000");
    expect(headers.get("x-ratelimit-remaining-requests")).toBe("499");
    expect(headers.get("x-ratelimit-remaining-tokens")).toBe("29000");
    expect(headers.get("x-ratelimit-reset-requests")).toBe("20ms");
    expect(headers.get("x-ratelimit-reset-tokens")).toBe("100ms");
  });

  it("does NOT forward server, date, or other non-allow-listed headers", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
      server: "cloudflare",
      date: "Sat, 07 Mar 2026 00:00:00 GMT",
      "cf-ray": "abc123",
      "set-cookie": "session=xyz",
    });
    const headers = buildClientHeaders(res);

    expect(headers.get("server")).toBeNull();
    expect(headers.get("date")).toBeNull();
    expect(headers.get("cf-ray")).toBeNull();
    expect(headers.get("set-cookie")).toBeNull();
  });

  it("forwards content-type and x-request-id", () => {
    const res = makeResponse(200, {
      "content-type": "text/event-stream",
      "x-request-id": "req-xyz-123",
    });
    const headers = buildClientHeaders(res);

    expect(headers.get("content-type")).toBe("text/event-stream");
    expect(headers.get("x-request-id")).toBe("req-xyz-123");
  });

  it("handles response with no headers gracefully", () => {
    const res = makeResponse(200, {});
    const headers = buildClientHeaders(res);

    expect(headers.get("content-type")).toBeNull();
    expect(headers.get("x-request-id")).toBeNull();
  });

  it("sets NullSpend-Version when apiVersion is provided", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildClientHeaders(res, "2026-04-01");

    expect(headers.get("NullSpend-Version")).toBe("2026-04-01");
  });

  it("does not set NullSpend-Version when apiVersion is omitted", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildClientHeaders(res);

    expect(headers.get("NullSpend-Version")).toBeNull();
  });
});

describe("appendTimingHeaders", () => {
  it("sets x-nullspend-overhead-ms with numeric value", () => {
    const headers = new Headers();
    const startMs = performance.now() - 50; // simulate 50ms ago
    appendTimingHeaders(headers, startMs, 30);

    const overhead = headers.get("x-nullspend-overhead-ms");
    expect(overhead).toMatch(/^\d+$/);
  });

  it("sets Server-Timing with all three components", () => {
    const headers = new Headers();
    const startMs = performance.now() - 100;
    appendTimingHeaders(headers, startMs, 60);

    const serverTiming = headers.get("Server-Timing")!;
    expect(serverTiming).toContain("overhead;dur=");
    expect(serverTiming).toContain('desc="Proxy overhead"');
    expect(serverTiming).toContain("upstream;dur=60");
    expect(serverTiming).toContain('desc="Provider latency"');
    expect(serverTiming).toContain("total;dur=");
  });

  it("returns totalMs and overheadMs as non-negative numbers", () => {
    const headers = new Headers();
    const startMs = performance.now() - 20;
    const { totalMs, overheadMs } = appendTimingHeaders(headers, startMs, 10);

    expect(typeof totalMs).toBe("number");
    expect(typeof overheadMs).toBe("number");
    expect(totalMs).toBeGreaterThanOrEqual(0);
    expect(overheadMs).toBeGreaterThanOrEqual(0);
  });

  it("clamps overheadMs to 0 when upstream exceeds total", () => {
    const headers = new Headers();
    // Start time very close to now but upstream duration larger
    const startMs = performance.now();
    const { overheadMs } = appendTimingHeaders(headers, startMs, 9999);

    expect(overheadMs).toBe(0);
    expect(headers.get("x-nullspend-overhead-ms")).toBe("0");
  });

  it("returns totalMs >= upstreamDurationMs in normal conditions", () => {
    const headers = new Headers();
    const startMs = performance.now() - 100;
    const { totalMs } = appendTimingHeaders(headers, startMs, 50);

    expect(totalMs).toBeGreaterThanOrEqual(50);
  });
});

