import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn((p: Promise<unknown>) => { p.catch(() => {}); }),
}));

const mockEmitMetric = vi.fn();
vi.mock("../lib/metrics.js", () => ({
  emitMetric: (...args: unknown[]) => mockEmitMetric(...args),
}));

import {
  storeRequestBody,
  storeResponseBody,
  storeStreamingResponseBody,
  retrieveBodies,
} from "../lib/body-storage.js";

describe("body-storage", () => {
  let mockBucket: {
    put: ReturnType<typeof vi.fn>;
    get: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    mockBucket = {
      put: vi.fn().mockResolvedValue(undefined),
      get: vi.fn().mockResolvedValue(null),
    };
    vi.clearAllMocks();
  });

  describe("storeRequestBody", () => {
    it("stores body at correct R2 key", async () => {
      await storeRequestBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        '{"model":"gpt-4"}',
      );

      expect(mockBucket.put).toHaveBeenCalledWith(
        "org_123/req_abc/request.json",
        '{"model":"gpt-4"}',
        { httpMetadata: { contentType: "application/json" } },
      );
    });

    it("skips bodies exceeding 1MB", async () => {
      const largeBody = "x".repeat(1_048_577);
      await storeRequestBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        largeBody,
      );

      expect(mockBucket.put).not.toHaveBeenCalled();
    });

    it("does not throw on R2 error", async () => {
      mockBucket.put.mockRejectedValueOnce(new Error("R2 unavailable"));

      await expect(
        storeRequestBody(
          mockBucket as unknown as R2Bucket,
          "org_123",
          "req_abc",
          '{"model":"gpt-4"}',
        ),
      ).resolves.toBeUndefined();
    });
  });

  describe("storeResponseBody", () => {
    it("stores body at correct R2 key", async () => {
      await storeResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        '{"choices":[]}',
      );

      expect(mockBucket.put).toHaveBeenCalledWith(
        "org_123/req_abc/response.json",
        '{"choices":[]}',
        { httpMetadata: { contentType: "application/json" } },
      );
    });

    it("skips bodies exceeding 1MB", async () => {
      const largeBody = "x".repeat(1_048_577);
      await storeResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        largeBody,
      );

      expect(mockBucket.put).not.toHaveBeenCalled();
    });

    it("does not throw on R2 error", async () => {
      mockBucket.put.mockRejectedValueOnce(new Error("R2 write failed"));

      await expect(
        storeResponseBody(
          mockBucket as unknown as R2Bucket,
          "org_123",
          "req_abc",
          '{"choices":[]}',
        ),
      ).resolves.toBeUndefined();
    });
  });

  describe("storeStreamingResponseBody", () => {
    it("stores SSE body at correct R2 key with text/event-stream content type", async () => {
      const sseBody = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\n";
      await storeStreamingResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        sseBody,
      );

      expect(mockBucket.put).toHaveBeenCalledWith(
        "org_123/req_abc/response.sse",
        sseBody,
        { httpMetadata: { contentType: "text/event-stream" } },
      );
    });

    it("emits body_storage_write metric on success", async () => {
      await storeStreamingResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        "data: test\n\n",
      );

      expect(mockEmitMetric).toHaveBeenCalledWith("body_storage_write", { type: "response_sse" });
    });

    it("skips bodies exceeding 1MB and emits skipped metric", async () => {
      const largeBody = "x".repeat(1_048_577);
      await storeStreamingResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        largeBody,
      );

      expect(mockBucket.put).not.toHaveBeenCalled();
      expect(mockEmitMetric).toHaveBeenCalledWith("body_storage_skipped", { type: "response_sse", reason: "too_large" });
    });

    it("emits body_storage_error metric on R2 failure", async () => {
      mockBucket.put.mockRejectedValueOnce(new Error("R2 unavailable"));

      await storeStreamingResponseBody(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
        "data: test\n\n",
      );

      expect(mockEmitMetric).toHaveBeenCalledWith("body_storage_error", { type: "response_sse" });
    });

    it("does not throw on R2 error", async () => {
      mockBucket.put.mockRejectedValueOnce(new Error("R2 unavailable"));

      await expect(
        storeStreamingResponseBody(
          mockBucket as unknown as R2Bucket,
          "org_123",
          "req_abc",
          "data: test\n\n",
        ),
      ).resolves.toBeUndefined();
    });
  });

  describe("retrieveBodies", () => {
    it("returns JSON response body with format when present", async () => {
      mockBucket.get
        .mockResolvedValueOnce({ text: () => Promise.resolve('{"model":"gpt-4"}') })  // request.json
        .mockResolvedValueOnce({ text: () => Promise.resolve('{"choices":[]}') })      // response.json
        .mockResolvedValueOnce(null);                                                   // response.sse

      const result = await retrieveBodies(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
      );

      expect(result).toEqual({
        requestBody: '{"model":"gpt-4"}',
        responseBody: '{"choices":[]}',
        responseFormat: "json",
      });

      expect(mockBucket.get).toHaveBeenCalledWith("org_123/req_abc/request.json");
      expect(mockBucket.get).toHaveBeenCalledWith("org_123/req_abc/response.json");
      expect(mockBucket.get).toHaveBeenCalledWith("org_123/req_abc/response.sse");
    });

    it("returns SSE response body when no JSON body exists", async () => {
      const sseText = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\n";
      mockBucket.get
        .mockResolvedValueOnce({ text: () => Promise.resolve('{"model":"gpt-4"}') })  // request.json
        .mockResolvedValueOnce(null)                                                    // response.json
        .mockResolvedValueOnce({ text: () => Promise.resolve(sseText) });              // response.sse

      const result = await retrieveBodies(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
      );

      expect(result).toEqual({
        requestBody: '{"model":"gpt-4"}',
        responseBody: sseText,
        responseFormat: "sse",
      });
    });

    it("prefers JSON over SSE when both exist", async () => {
      mockBucket.get
        .mockResolvedValueOnce(null)                                                    // request.json
        .mockResolvedValueOnce({ text: () => Promise.resolve('{"choices":[]}') })      // response.json
        .mockResolvedValueOnce({ text: () => Promise.resolve("data: ...\n\n") });      // response.sse

      const result = await retrieveBodies(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
      );

      expect(result).toEqual({
        requestBody: null,
        responseBody: '{"choices":[]}',
        responseFormat: "json",
      });
    });

    it("returns null for missing bodies with null format", async () => {
      mockBucket.get
        .mockResolvedValueOnce(null)   // request.json
        .mockResolvedValueOnce(null)   // response.json
        .mockResolvedValueOnce(null);  // response.sse

      const result = await retrieveBodies(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
      );

      expect(result).toEqual({
        requestBody: null,
        responseBody: null,
        responseFormat: null,
      });
    });

    it("returns partial when only request exists", async () => {
      mockBucket.get
        .mockResolvedValueOnce({ text: () => Promise.resolve('{"model":"gpt-4"}') })
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce(null);

      const result = await retrieveBodies(
        mockBucket as unknown as R2Bucket,
        "org_123",
        "req_abc",
      );

      expect(result).toEqual({
        requestBody: '{"model":"gpt-4"}',
        responseBody: null,
        responseFormat: null,
      });
    });
  });
});
