import { describe, it, expect } from "vitest";
import { sanitizeUpstreamError } from "../lib/sanitize-upstream-error.js";

function makeResponse(body: string | object | null, status: number): Response {
  const bodyStr = body === null ? "" : typeof body === "string" ? body : JSON.stringify(body);
  return new Response(bodyStr, { status });
}

describe("sanitizeUpstreamError", () => {
  describe("401 response", () => {
    it("replaces body entirely regardless of content", async () => {
      const response = makeResponse(
        { error: { message: "Invalid API key: sk-live-abc123...", type: "authentication_error" } },
        401,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "authentication_error", message: "Upstream authentication failed" },
      });
    });

    it("strips API key fragments from 401 body", async () => {
      const response = makeResponse(
        "Incorrect API key provided: sk-proj-abc***def. You can find your API key at...",
        401,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream authentication failed");
      expect(JSON.stringify(result)).not.toContain("sk-proj");
    });

    it("handles 401 for anthropic provider the same way", async () => {
      const response = makeResponse(
        { type: "error", error: { type: "authentication_error", message: "invalid x-api-key" } },
        401,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result).toEqual({
        error: { type: "authentication_error", message: "Upstream authentication failed" },
      });
    });
  });

  describe("403 response", () => {
    it("replaces body entirely", async () => {
      const response = makeResponse(
        { error: { message: "You don't have access to this resource", type: "permission_error" } },
        403,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "permission_error", message: "Upstream permission denied" },
      });
    });

    it("strips any upstream details from 403", async () => {
      const response = makeResponse(
        { error: { message: "Organization org-abc123 is deactivated", org_id: "org-abc123" } },
        403,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream permission denied");
      expect(JSON.stringify(result)).not.toContain("org-abc123");
    });
  });

  describe("OpenAI error format", () => {
    it("extracts safe fields from OpenAI-style error", async () => {
      const response = makeResponse(
        {
          error: {
            message: "Rate limit exceeded",
            type: "rate_limit_error",
            code: "rate_limit_exceeded",
          },
        },
        429,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: {
          type: "rate_limit_error",
          message: "Rate limit exceeded",
          code: "rate_limit_exceeded",
        },
      });
    });
  });

  describe("Anthropic error format", () => {
    it("extracts safe fields from Anthropic-style error", async () => {
      const response = makeResponse(
        {
          type: "error",
          error: {
            type: "overloaded_error",
            message: "Overloaded",
          },
        },
        529,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result).toEqual({
        error: {
          type: "overloaded_error",
          message: "Overloaded",
        },
      });
    });
  });

  describe("non-string type field fallback", () => {
    it("falls back to upstream_error when type is a number", async () => {
      const response = makeResponse(
        { error: { type: 42, message: "Something went wrong" } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
      expect(result.error.message).toBe("Something went wrong");
    });

    it("falls back to upstream_error when type is an object", async () => {
      const response = makeResponse(
        { error: { type: { nested: true }, message: "Bad" } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
    });

    it("falls back to upstream_error when type is null", async () => {
      const response = makeResponse(
        { error: { type: null, message: "Error occurred" } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
    });
  });

  describe("non-string message fallback", () => {
    it("falls back to default message when message is a number", async () => {
      const response = makeResponse(
        { error: { type: "server_error", message: 500 } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream request failed");
    });

    it("falls back to default message when message is an array", async () => {
      const response = makeResponse(
        { error: { type: "server_error", message: ["err1", "err2"] } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream request failed");
    });

    it("falls back to default message when message is missing", async () => {
      const response = makeResponse(
        { error: { type: "server_error" } },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream request failed");
    });
  });

  describe("dangerous fields stripped", () => {
    it("strips api_key from error object", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "Internal error",
            api_key: "sk-live-secret123",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(JSON.stringify(result)).not.toContain("sk-live-secret123");
      expect(result.error).not.toHaveProperty("api_key");
    });

    it("strips account_id from error object", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "Internal error",
            account_id: "acct_abc123",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error).not.toHaveProperty("account_id");
    });

    it("strips org_id from error object", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "Internal error",
            org_id: "org-secret",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error).not.toHaveProperty("org_id");
    });

    it("only includes type, message, and code — nothing else", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "Error",
            code: "internal",
            param: "model",
            request_id: "req_abc",
            api_key: "sk-123",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      const keys = Object.keys(result.error);
      expect(keys).toContain("type");
      expect(keys).toContain("message");
      expect(keys).toContain("code");
      expect(keys).not.toContain("param");
      expect(keys).not.toContain("request_id");
      expect(keys).not.toContain("api_key");
    });
  });

  describe("code field handling", () => {
    it("includes code field when present as string", async () => {
      const response = makeResponse(
        {
          error: {
            type: "invalid_request_error",
            message: "Invalid model",
            code: "model_not_found",
          },
        },
        404,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.code).toBe("model_not_found");
    });

    it("omits code field when not present", async () => {
      const response = makeResponse(
        {
          error: {
            type: "invalid_request_error",
            message: "Bad request",
          },
        },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error).not.toHaveProperty("code");
    });

    it("omits code field when it is not a string", async () => {
      const response = makeResponse(
        {
          error: {
            type: "invalid_request_error",
            message: "Bad request",
            code: 123,
          },
        },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error).not.toHaveProperty("code");
    });
  });

  describe("non-JSON response body", () => {
    it("returns generic error for plain text body", async () => {
      const response = new Response("Internal Server Error", { status: 500 });
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });

    it("returns generic error for HTML body", async () => {
      const response = new Response("<html><body>Bad Gateway</body></html>", { status: 502 });
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 502" },
      });
    });
  });

  describe("empty response body", () => {
    it("returns generic error for empty body", async () => {
      const response = new Response("", { status: 500 });
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });

    it("returns generic error for null body", async () => {
      const response = new Response(null, { status: 500 });
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });
  });

  describe("unrecognized JSON structure", () => {
    it("returns generic error for JSON with no error field", async () => {
      const response = makeResponse({ status: "fail", reason: "unknown" }, 500);
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });

    it("returns generic error for JSON array", async () => {
      const response = new Response(JSON.stringify([{ error: "bad" }]), { status: 500 });
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });

    it("returns generic error when error is a string instead of object", async () => {
      const response = makeResponse({ error: "something broke" }, 500);
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result).toEqual({
        error: { type: "upstream_error", message: "Upstream returned 500" },
      });
    });
  });

  describe("429 rate limit with error body", () => {
    it("extracts safe fields from 429 response", async () => {
      const response = makeResponse(
        {
          error: {
            type: "rate_limit_error",
            message: "You have exceeded your rate limit",
            code: "rate_limit_exceeded",
          },
        },
        429,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("rate_limit_error");
      expect(result.error.message).toBe("You have exceeded your rate limit");
      expect(result.error.code).toBe("rate_limit_exceeded");
    });
  });

  describe("500 with error body", () => {
    it("extracts safe fields from 500 response", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "The server had an error processing your request",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result.error.type).toBe("server_error");
      expect(result.error.message).toBe("The server had an error processing your request");
    });

    it("extracts safe fields from 500 with Anthropic format", async () => {
      const response = makeResponse(
        {
          type: "error",
          error: {
            type: "api_error",
            message: "Internal server error",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result.error.type).toBe("api_error");
      expect(result.error.message).toBe("Internal server error");
    });
  });
});
