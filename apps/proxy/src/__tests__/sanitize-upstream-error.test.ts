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
    it("extracts safe fields from Anthropic-style error (4xx)", async () => {
      const response = makeResponse(
        {
          type: "error",
          error: {
            type: "invalid_request_error",
            message: "Max tokens exceeded",
          },
        },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      expect(result).toEqual({
        error: {
          type: "invalid_request_error",
          message: "Max tokens exceeded",
        },
      });
    });
  });

  describe("non-string type field fallback", () => {
    it("falls back to upstream_error when type is a number", async () => {
      const response = makeResponse(
        { error: { type: 42, message: "Something went wrong" } },
        422,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
      expect(result.error.message).toBe("Something went wrong");
    });

    it("falls back to upstream_error when type is an object", async () => {
      const response = makeResponse(
        { error: { type: { nested: true }, message: "Bad" } },
        422,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
    });

    it("falls back to upstream_error when type is null", async () => {
      const response = makeResponse(
        { error: { type: null, message: "Error occurred" } },
        422,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
    });
  });

  describe("non-string message fallback", () => {
    it("falls back to default message when message is a number", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error", message: 400 } },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream request failed");
    });

    it("falls back to default message when message is an array", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error", message: ["err1", "err2"] } },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream request failed");
    });

    it("falls back to default message when message is missing", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error" } },
        400,
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
            type: "invalid_request_error",
            message: "Error",
            code: "bad_param",
            param: "model",
            request_id: "req_abc",
            api_key: "sk-123",
          },
        },
        400,
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

  // PXY-12: 5xx errors now return generic message — never forward internal details
  describe("5xx responses (PXY-12)", () => {
    it("returns generic message for 500 — never forwards body", async () => {
      const response = makeResponse(
        {
          error: {
            type: "server_error",
            message: "Connection to database pool db-internal-7a3f failed: timeout after 5000ms",
          },
        },
        500,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.type).toBe("upstream_error");
      expect(result.error.message).toBe("Upstream returned 500");
    });

    it("returns generic message for 502", async () => {
      const response = makeResponse("Bad Gateway", 502);
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("Upstream returned 502");
    });

    it("returns generic message for 503", async () => {
      const response = makeResponse(
        { error: { type: "overloaded_error", message: "Overloaded — org-abc123 has too many concurrent requests" } },
        503,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "anthropic"));
      // Must NOT contain the org ID
      expect(result.error.message).toBe("Upstream returned 503");
    });
  });

  // PXY-12: 4xx message sanitization
  describe("4xx message sanitization (PXY-12)", () => {
    it("strips OpenAI org IDs from error messages", async () => {
      const response = makeResponse(
        { error: { type: "rate_limit_error", message: "Rate limit reached for org-abcDEF12345 on requests per min" } },
        429,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toContain("org-***");
      expect(result.error.message).not.toContain("org-abcDEF12345");
      expect(result.error.type).toBe("rate_limit_error");
    });

    it("strips API key fragments from error messages", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error", message: "Incorrect API key provided: sk-proj-abc123def. You can find your key at platform.openai.com" } },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toContain("sk-***");
      expect(result.error.message).not.toContain("sk-proj-abc123def");
    });

    it("strips email addresses from error messages", async () => {
      const response = makeResponse(
        { error: { type: "permission_error", message: "Account owner user@company.com must accept terms" } },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toContain("***@***");
      expect(result.error.message).not.toContain("user@company.com");
    });

    it("strips long hex IDs from error messages", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error", message: "Deployment 0123456789abcdef0123456789abcdef not found" } },
        404,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toContain("***");
      expect(result.error.message).not.toContain("0123456789abcdef0123456789abcdef");
    });

    it("preserves safe error messages unchanged", async () => {
      const response = makeResponse(
        { error: { type: "invalid_request_error", message: "This model does not support streaming", code: "unsupported_streaming" } },
        400,
      );
      const result = JSON.parse(await sanitizeUpstreamError(response, "openai"));
      expect(result.error.message).toBe("This model does not support streaming");
      expect(result.error.code).toBe("unsupported_streaming");
    });
  });
});
