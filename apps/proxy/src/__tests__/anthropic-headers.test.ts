import { describe, it, expect } from "vitest";
import {
  buildAnthropicUpstreamHeaders,
  buildAnthropicClientHeaders,
} from "../lib/anthropic-headers.js";

function makeRequest(headers: Record<string, string> = {}): Request {
  return new Request("https://proxy.example.com/v1/messages", {
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

describe("buildAnthropicUpstreamHeaders", () => {
  it("extracts Bearer token and forwards as x-api-key", () => {
    const req = makeRequest({ authorization: "Bearer sk-ant-api03-test" });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("x-api-key")).toBe("sk-ant-api03-test");
    expect(headers.get("authorization")).toBeNull();
  });

  it("forwards x-api-key directly when no Bearer token", () => {
    const req = makeRequest({ "x-api-key": "sk-ant-api03-direct" });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("x-api-key")).toBe("sk-ant-api03-direct");
  });

  it("defaults to anthropic-version: 2023-06-01 when client omits it", () => {
    const req = makeRequest({});
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("anthropic-version")).toBe("2023-06-01");
  });

  it("forwards client-specified anthropic-version", () => {
    const req = makeRequest({ "anthropic-version": "2024-10-22" });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("anthropic-version")).toBe("2024-10-22");
  });

  it("always sets content-type: application/json", () => {
    const req = makeRequest({ "content-type": "text/plain" });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("content-type")).toBe("application/json");
  });

  it("forwards anthropic-beta when present", () => {
    const req = makeRequest({
      authorization: "Bearer sk-ant-test",
      "anthropic-beta": "extended-cache-ttl-2025-04-11",
    });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("anthropic-beta")).toBe(
      "extended-cache-ttl-2025-04-11",
    );
  });

  it("does NOT forward x-nullspend-key, host, or content-length", () => {
    const req = makeRequest({
      authorization: "Bearer sk-ant-test",
      "x-nullspend-key": "ns_live_sk_test0001",
      host: "proxy.example.com",
      "content-length": "42",
    });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("x-nullspend-key")).toBeNull();
    expect(headers.get("host")).toBeNull();
    expect(headers.get("content-length")).toBeNull();
  });

  it("sets no x-api-key when auth is missing entirely", () => {
    const req = makeRequest({});
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("x-api-key")).toBeNull();
    expect(headers.get("anthropic-version")).toBe("2023-06-01");
  });

  it("forwards traceparent and tracestate to upstream", () => {
    const req = makeRequest({
      authorization: "Bearer sk-ant-test",
      traceparent:
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
      tracestate: "vendor1=value1,vendor2=value2",
    });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBe(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    );
    expect(headers.get("tracestate")).toBe("vendor1=value1,vendor2=value2");
  });

  it("omits traceparent and tracestate when not present", () => {
    const req = makeRequest({ authorization: "Bearer sk-ant-test" });
    const headers = buildAnthropicUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBeNull();
    expect(headers.get("tracestate")).toBeNull();
  });
});

describe("buildAnthropicClientHeaders", () => {
  it("forwards content-type from upstream", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
    });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("content-type")).toBe("application/json");
  });

  it("normalizes request-id to x-request-id", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
      "request-id": "req_018EeWyXxfu5pfWkrYcMdjWG",
    });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("x-request-id")).toBe(
      "req_018EeWyXxfu5pfWkrYcMdjWG",
    );
    expect(headers.get("request-id")).toBeNull();
  });

  it("forwards all anthropic-ratelimit-* headers", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
      "anthropic-ratelimit-requests-limit": "1000",
      "anthropic-ratelimit-requests-remaining": "999",
      "anthropic-ratelimit-tokens-limit": "100000",
      "anthropic-ratelimit-tokens-remaining": "99000",
      "anthropic-ratelimit-input-tokens-limit": "80000",
      "anthropic-ratelimit-output-tokens-limit": "20000",
    });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("anthropic-ratelimit-requests-limit")).toBe("1000");
    expect(headers.get("anthropic-ratelimit-requests-remaining")).toBe("999");
    expect(headers.get("anthropic-ratelimit-tokens-limit")).toBe("100000");
    expect(headers.get("anthropic-ratelimit-tokens-remaining")).toBe("99000");
    expect(headers.get("anthropic-ratelimit-input-tokens-limit")).toBe(
      "80000",
    );
    expect(headers.get("anthropic-ratelimit-output-tokens-limit")).toBe(
      "20000",
    );
  });

  it("forwards retry-after when present", () => {
    const res = makeResponse(429, {
      "content-type": "application/json",
      "retry-after": "30",
    });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("retry-after")).toBe("30");
  });

  it("does NOT forward non-allowlisted headers", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
      server: "cloudflare",
      date: "Mon, 09 Mar 2026 00:00:00 GMT",
      "cf-ray": "abc123",
      "set-cookie": "session=xyz",
    });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("server")).toBeNull();
    expect(headers.get("date")).toBeNull();
    expect(headers.get("cf-ray")).toBeNull();
    expect(headers.get("set-cookie")).toBeNull();
  });

  it("handles empty upstream headers gracefully", () => {
    const res = makeResponse(200, {});
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("content-type")).toBeNull();
    expect(headers.get("x-request-id")).toBeNull();
    expect(headers.get("retry-after")).toBeNull();
  });

  it("sets NullSpend-Version when apiVersion is provided", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildAnthropicClientHeaders(res, "2026-04-01");

    expect(headers.get("NullSpend-Version")).toBe("2026-04-01");
  });

  it("does not set NullSpend-Version when apiVersion is omitted", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildAnthropicClientHeaders(res);

    expect(headers.get("NullSpend-Version")).toBeNull();
  });
});
