/**
 * Smoke tests for W3C traceparent propagation.
 * Verifies trace IDs flow through the proxy end-to-end.
 *
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run smoke-trace.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  BASE,
  OPENAI_API_KEY,
  isServerUp,
  authHeaders,
  smallRequest,
} from "./smoke-test-helpers.js";

const TRACE_HEADER = "X-NullSpend-Trace-Id";
const VALID_TRACE_ID = "0af7651916cd43dd8448eb211c80319c";
const VALID_TRACEPARENT = `00-${VALID_TRACE_ID}-b7ad6b7169203331-01`;

describe("trace context smoke tests", () => {
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

  describe("W3C traceparent header", () => {
    it("extracts trace-id from valid traceparent and echoes it back", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ traceparent: VALID_TRACEPARENT }),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      expect(res.headers.get(TRACE_HEADER)).toBe(VALID_TRACE_ID);
    }, 30_000);

    it("auto-generates trace-id when traceparent has invalid version ff", async () => {
      const badTraceparent = `ff-${VALID_TRACE_ID}-b7ad6b7169203331-01`;
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ traceparent: badTraceparent }),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const traceId = res.headers.get(TRACE_HEADER);
      expect(traceId).toBeTruthy();
      expect(traceId).not.toBe(VALID_TRACE_ID);
      expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    }, 30_000);
  });

  describe("custom x-nullspend-trace-id header", () => {
    it("uses custom header when no traceparent is present", async () => {
      const customTraceId = "abcdef0123456789abcdef0123456789";
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ "x-nullspend-trace-id": customTraceId }),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      expect(res.headers.get(TRACE_HEADER)).toBe(customTraceId);
    }, 30_000);

    it("traceparent takes priority over custom header", async () => {
      const customTraceId = "11111111111111111111111111111111";
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({
          traceparent: VALID_TRACEPARENT,
          "x-nullspend-trace-id": customTraceId,
        }),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      expect(res.headers.get(TRACE_HEADER)).toBe(VALID_TRACE_ID);
    }, 30_000);
  });

  describe("auto-generation", () => {
    it("generates a valid 32-char hex trace-id when no trace headers sent", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: smallRequest(),
      });

      expect(res.status).toBe(200);
      const traceId = res.headers.get(TRACE_HEADER);
      expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    }, 30_000);

    it("generates unique trace-ids for concurrent requests", async () => {
      const requests = Array.from({ length: 5 }, () =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: authHeaders(),
          body: smallRequest(),
        }),
      );
      const responses = await Promise.all(requests);
      const traceIds = responses.map((r) => r.headers.get(TRACE_HEADER));

      for (const id of traceIds) {
        expect(id).toMatch(/^[0-9a-f]{32}$/);
      }
      // All should be unique
      expect(new Set(traceIds).size).toBe(traceIds.length);

      // Consume bodies to avoid connection leaks
      await Promise.all(responses.map((r) => r.text()));
    }, 60_000);
  });

  describe("error responses include trace-id", () => {
    it("401 response includes trace-id header", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "invalid-key-for-trace-test",
          traceparent: VALID_TRACEPARENT,
        },
        body: smallRequest(),
      });

      expect(res.status).toBe(401);
      // Trace ID is resolved before auth, so it should be present
      const traceId = res.headers.get(TRACE_HEADER);
      expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    });

    it("streaming response includes trace-id header", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders({ traceparent: VALID_TRACEPARENT }),
        body: smallRequest({ stream: true }),
      });

      expect(res.status).toBe(200);
      expect(res.headers.get(TRACE_HEADER)).toBe(VALID_TRACE_ID);

      // Consume body
      await res.text();
    }, 30_000);
  });
});
