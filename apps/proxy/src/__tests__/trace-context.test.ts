import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { resolveTraceId } from "../lib/trace-context.js";

function makeRequest(headers: Record<string, string> = {}): Request {
  return new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    headers,
  });
}

describe("resolveTraceId", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("extracts trace-id from valid traceparent", () => {
    const traceId = "0af7651916cd43dd8448eb211c80319c";
    const req = makeRequest({
      traceparent: `00-${traceId}-b7ad6b7169203331-01`,
    });
    expect(resolveTraceId(req)).toBe(traceId);
  });

  it("rejects all-zeros trace-id and auto-generates", () => {
    const req = makeRequest({
      traceparent: "00-00000000000000000000000000000000-b7ad6b7169203331-01",
    });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(result).not.toBe("00000000000000000000000000000000");
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects version ff and auto-generates", () => {
    const req = makeRequest({
      traceparent: "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(result).not.toBe("0af7651916cd43dd8448eb211c80319c");
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects all-zeros span-id and auto-generates", () => {
    const req = makeRequest({
      traceparent: "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01",
    });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(result).not.toBe("0af7651916cd43dd8448eb211c80319c");
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects malformed traceparent (too short) and auto-generates", () => {
    const req = makeRequest({ traceparent: "00-abc-def-01" });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects malformed traceparent (wrong format) and auto-generates", () => {
    const req = makeRequest({ traceparent: "not-a-valid-traceparent-at-all" });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects uppercase traceparent (spec requires lowercase)", () => {
    const req = makeRequest({
      traceparent: "00-0AF7651916CD43DD8448EB211C80319C-B7AD6B7169203331-01",
    });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("uses X-NullSpend-Trace-Id when no traceparent", () => {
    const traceId = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4";
    const req = makeRequest({ "x-nullspend-trace-id": traceId });
    expect(resolveTraceId(req)).toBe(traceId);
  });

  it("traceparent takes priority over custom header", () => {
    const traceparentId = "0af7651916cd43dd8448eb211c80319c";
    const customId = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4";
    const req = makeRequest({
      traceparent: `00-${traceparentId}-b7ad6b7169203331-01`,
      "x-nullspend-trace-id": customId,
    });
    expect(resolveTraceId(req)).toBe(traceparentId);
  });

  it("rejects invalid custom header and auto-generates", () => {
    const req = makeRequest({ "x-nullspend-trace-id": "not-valid" });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("rejects uppercase custom header", () => {
    const req = makeRequest({
      "x-nullspend-trace-id": "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4",
    });
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("auto-generates unique 32-char hex when no headers present", () => {
    const req = makeRequest();
    const result = resolveTraceId(req);
    expect(result).toMatch(/^[0-9a-f]{32}$/);
    expect(result).toHaveLength(32);
    // Should not have warned
    expect(console.warn).not.toHaveBeenCalled();
  });

  it("auto-generates different IDs for different requests", () => {
    const req1 = makeRequest();
    const req2 = makeRequest();
    const id1 = resolveTraceId(req1);
    const id2 = resolveTraceId(req2);
    expect(id1).not.toBe(id2);
  });

  it("falls through from invalid traceparent to valid custom header", () => {
    const customId = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4";
    const req = makeRequest({
      traceparent: "invalid",
      "x-nullspend-trace-id": customId,
    });
    expect(resolveTraceId(req)).toBe(customId);
    expect(console.warn).toHaveBeenCalledOnce();
  });
});
