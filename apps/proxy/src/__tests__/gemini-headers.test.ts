import { describe, it, expect } from "vitest";
import {
  buildGeminiUpstreamHeaders,
  buildGeminiClientHeaders,
} from "../lib/gemini-headers.js";

function makeRequest(headers: Record<string, string> = {}): Request {
  return new Request("https://proxy.example.com/v1beta/models/gemini-2.5-pro:generateContent", {
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

describe("buildGeminiUpstreamHeaders", () => {
  it("extracts Bearer token and forwards as x-goog-api-key", () => {
    const req = makeRequest({ authorization: "Bearer sk-test" });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("x-goog-api-key")).toBe("sk-test");
    expect(headers.get("authorization")).toBeNull();
  });

  it("forwards x-goog-api-key directly when no Bearer token", () => {
    const req = makeRequest({ "x-goog-api-key": "AIzaSy-direct-key" });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("x-goog-api-key")).toBe("AIzaSy-direct-key");
  });

  it("sets no x-goog-api-key when auth is missing entirely", () => {
    const req = makeRequest({});
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("x-goog-api-key")).toBeNull();
  });

  it("always sets content-type: application/json", () => {
    const req = makeRequest({ "content-type": "text/plain" });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("content-type")).toBe("application/json");
  });

  it("forwards traceparent and tracestate to upstream", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      traceparent:
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
      tracestate: "vendor1=value1,vendor2=value2",
    });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBe(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    );
    expect(headers.get("tracestate")).toBe("vendor1=value1,vendor2=value2");
  });

  it("omits traceparent and tracestate when not present", () => {
    const req = makeRequest({ authorization: "Bearer sk-test" });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("traceparent")).toBeNull();
    expect(headers.get("tracestate")).toBeNull();
  });

  it("Bearer with empty token — forwards empty string as x-goog-api-key", () => {
    const req = makeRequest({ authorization: "Bearer " });
    const headers = buildGeminiUpstreamHeaders(req);
    // "Bearer " sliced from index 7 = empty string.
    // Google will reject this with 401, which is correct behavior.
    // The header value may be "" or null depending on runtime normalization.
    const value = headers.get("x-goog-api-key");
    expect(value === "" || value === null).toBe(true);
  });

  it("only returns expected headers — no leaking of original request headers", () => {
    const req = makeRequest({
      authorization: "Bearer sk-test",
      "x-custom-secret": "should-not-leak",
      "cookie": "session=abc",
    });
    const headers = buildGeminiUpstreamHeaders(req);
    expect(headers.get("x-custom-secret")).toBeNull();
    expect(headers.get("cookie")).toBeNull();
    expect(headers.get("authorization")).toBeNull();
    // Only expected headers present
    const headerKeys = [...headers.keys()];
    expect(headerKeys.sort()).toEqual(["content-type", "x-goog-api-key"].sort());
  });

  it("prefers Bearer token over x-goog-api-key when both present", () => {
    const req = makeRequest({
      authorization: "Bearer sk-from-bearer",
      "x-goog-api-key": "AIzaSy-from-header",
    });
    const headers = buildGeminiUpstreamHeaders(req);

    expect(headers.get("x-goog-api-key")).toBe("sk-from-bearer");
  });
});

describe("buildGeminiClientHeaders", () => {
  it("forwards content-type from upstream", () => {
    const res = makeResponse(200, {
      "content-type": "application/json",
    });
    const headers = buildGeminiClientHeaders(res);

    expect(headers.get("content-type")).toBe("application/json");
  });

  it("forwards retry-after when present", () => {
    const res = makeResponse(429, {
      "content-type": "application/json",
      "retry-after": "30",
    });
    const headers = buildGeminiClientHeaders(res);

    expect(headers.get("retry-after")).toBe("30");
  });

  it("sets NullSpend-Version when apiVersion is provided", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildGeminiClientHeaders(res, "2026-04-01");

    expect(headers.get("NullSpend-Version")).toBe("2026-04-01");
  });

  it("does not set content-type when upstream omits it", () => {
    const res = makeResponse(200, {});
    const headers = buildGeminiClientHeaders(res);

    expect(headers.get("content-type")).toBeNull();
  });

  it("does not set NullSpend-Version when apiVersion is omitted", () => {
    const res = makeResponse(200, { "content-type": "application/json" });
    const headers = buildGeminiClientHeaders(res);

    expect(headers.get("NullSpend-Version")).toBeNull();
  });
});
