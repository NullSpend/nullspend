import { describe, it, expect } from "vitest";
import {
  isTrackedRoute,
  extractModelFromBody,
  isStreamingRequest,
  isStreamingResponse,
  extractOpenAIUsageFromJSON,
  extractAnthropicUsageFromJSON,
} from "./provider-parsers.js";

// ---------------------------------------------------------------------------
// isTrackedRoute
// ---------------------------------------------------------------------------

describe("isTrackedRoute", () => {
  it("returns true for OpenAI POST /chat/completions", () => {
    expect(
      isTrackedRoute("openai", "https://api.openai.com/v1/chat/completions", "POST"),
    ).toBe(true);
  });

  it("returns true for Anthropic POST /messages", () => {
    expect(
      isTrackedRoute("anthropic", "https://api.anthropic.com/v1/messages", "POST"),
    ).toBe(true);
  });

  it("returns false for GET requests", () => {
    expect(
      isTrackedRoute("openai", "https://api.openai.com/v1/chat/completions", "GET"),
    ).toBe(false);
    expect(
      isTrackedRoute("anthropic", "https://api.anthropic.com/v1/messages", "GET"),
    ).toBe(false);
  });

  it("returns true for OpenAI embeddings", () => {
    expect(
      isTrackedRoute("openai", "https://api.openai.com/v1/embeddings", "POST"),
    ).toBe(true);
  });

  it("returns true for legacy OpenAI completions", () => {
    expect(
      isTrackedRoute("openai", "https://api.openai.com/v1/completions", "POST"),
    ).toBe(true);
  });

  it("returns false for non-matching paths (models)", () => {
    expect(
      isTrackedRoute("openai", "https://api.openai.com/v1/models", "POST"),
    ).toBe(false);
  });

  it("returns true for Azure OpenAI paths", () => {
    expect(
      isTrackedRoute(
        "openai",
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15",
        "POST",
      ),
    ).toBe(true);
  });

  it("returns false for invalid URL", () => {
    expect(isTrackedRoute("openai", "not a url", "POST")).toBe(false);
  });

  it("returns false when provider does not match path", () => {
    // Anthropic provider but OpenAI path
    expect(
      isTrackedRoute("anthropic", "https://api.openai.com/v1/chat/completions", "POST"),
    ).toBe(false);
    // OpenAI provider but Anthropic path
    expect(
      isTrackedRoute("openai", "https://api.anthropic.com/v1/messages", "POST"),
    ).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractModelFromBody
// ---------------------------------------------------------------------------

describe("extractModelFromBody", () => {
  it("extracts model from valid JSON", () => {
    expect(
      extractModelFromBody(JSON.stringify({ model: "gpt-4o", messages: [] })),
    ).toBe("gpt-4o");
  });

  it("returns null when model is missing", () => {
    expect(extractModelFromBody(JSON.stringify({ messages: [] }))).toBeNull();
  });

  it("returns null for invalid JSON", () => {
    expect(extractModelFromBody("{not json")).toBeNull();
  });

  it("returns null for empty model string", () => {
    expect(
      extractModelFromBody(JSON.stringify({ model: "" })),
    ).toBeNull();
  });

  it("returns null when model is a number", () => {
    expect(
      extractModelFromBody(JSON.stringify({ model: 42 })),
    ).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// isStreamingRequest
// ---------------------------------------------------------------------------

describe("isStreamingRequest", () => {
  it("returns true when stream is true", () => {
    expect(isStreamingRequest(JSON.stringify({ stream: true }))).toBe(true);
  });

  it("returns false when stream is false", () => {
    expect(isStreamingRequest(JSON.stringify({ stream: false }))).toBe(false);
  });

  it("returns false when no stream field", () => {
    expect(isStreamingRequest(JSON.stringify({ model: "gpt-4o" }))).toBe(false);
  });

  it("returns false for invalid JSON", () => {
    expect(isStreamingRequest("not json")).toBe(false);
  });

  it("returns false when stream is a string (not boolean true)", () => {
    expect(isStreamingRequest(JSON.stringify({ stream: "true" }))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// isStreamingResponse
// ---------------------------------------------------------------------------

describe("isStreamingResponse", () => {
  it("returns true for text/event-stream content-type", () => {
    const response = new Response("", {
      headers: { "content-type": "text/event-stream" },
    });
    expect(isStreamingResponse(response)).toBe(true);
  });

  it("returns true for text/event-stream with charset", () => {
    const response = new Response("", {
      headers: { "content-type": "text/event-stream; charset=utf-8" },
    });
    expect(isStreamingResponse(response)).toBe(true);
  });

  it("returns false for application/json", () => {
    const response = new Response("{}", {
      headers: { "content-type": "application/json" },
    });
    expect(isStreamingResponse(response)).toBe(false);
  });

  it("returns false when no content-type header", () => {
    const response = new Response("");
    expect(isStreamingResponse(response)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractOpenAIUsageFromJSON
// ---------------------------------------------------------------------------

describe("extractOpenAIUsageFromJSON", () => {
  it("extracts valid usage from response JSON", () => {
    const json = {
      id: "chatcmpl-1",
      model: "gpt-4o",
      usage: { prompt_tokens: 10, completion_tokens: 20 },
    };
    const result = extractOpenAIUsageFromJSON(json);
    expect(result).toEqual({ prompt_tokens: 10, completion_tokens: 20 });
  });

  it("preserves prompt_tokens_details and completion_tokens_details", () => {
    const json = {
      usage: {
        prompt_tokens: 100,
        completion_tokens: 50,
        prompt_tokens_details: { cached_tokens: 80 },
        completion_tokens_details: { reasoning_tokens: 30 },
      },
    };
    const result = extractOpenAIUsageFromJSON(json);
    expect(result!.prompt_tokens_details!.cached_tokens).toBe(80);
    expect(result!.completion_tokens_details!.reasoning_tokens).toBe(30);
  });

  it("returns null when usage is missing", () => {
    expect(extractOpenAIUsageFromJSON({ model: "gpt-4o" })).toBeNull();
  });

  it("returns null for non-object input", () => {
    expect(extractOpenAIUsageFromJSON(null)).toBeNull();
    expect(extractOpenAIUsageFromJSON("string")).toBeNull();
    expect(extractOpenAIUsageFromJSON(42)).toBeNull();
  });

  it("returns null when prompt_tokens is not a number", () => {
    expect(
      extractOpenAIUsageFromJSON({
        usage: { prompt_tokens: "10", completion_tokens: 20 },
      }),
    ).toBeNull();
  });

  it("returns null when completion_tokens is not a number", () => {
    expect(
      extractOpenAIUsageFromJSON({
        usage: { prompt_tokens: 10, completion_tokens: "20" },
      }),
    ).toBeNull();
  });

  it("defaults completion_tokens to 0 when missing (embeddings response)", () => {
    const json = {
      object: "list",
      model: "text-embedding-3-small",
      usage: { prompt_tokens: 5, total_tokens: 5 },
    };
    const result = extractOpenAIUsageFromJSON(json);
    expect(result).not.toBeNull();
    expect(result!.prompt_tokens).toBe(5);
    expect(result!.completion_tokens).toBe(0);
  });

  it("defaults completion_tokens to 0 when null", () => {
    const json = {
      usage: { prompt_tokens: 10, completion_tokens: null },
    };
    const result = extractOpenAIUsageFromJSON(json);
    expect(result).not.toBeNull();
    expect(result!.completion_tokens).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// extractAnthropicUsageFromJSON
// ---------------------------------------------------------------------------

describe("extractAnthropicUsageFromJSON", () => {
  it("extracts valid usage from response JSON", () => {
    const json = {
      id: "msg-1",
      model: "claude-sonnet-4-20250514",
      usage: { input_tokens: 100, output_tokens: 200 },
    };
    const result = extractAnthropicUsageFromJSON(json);
    expect(result).not.toBeNull();
    expect(result!.usage.input_tokens).toBe(100);
    expect(result!.usage.output_tokens).toBe(200);
    expect(result!.cacheDetail).toBeNull();
  });

  it("extracts cache_creation detail when present", () => {
    const json = {
      usage: {
        input_tokens: 100,
        output_tokens: 50,
        cache_creation_input_tokens: 80,
        cache_read_input_tokens: 20,
        cache_creation: {
          ephemeral_5m_input_tokens: 60,
          ephemeral_1h_input_tokens: 20,
        },
      },
    };
    const result = extractAnthropicUsageFromJSON(json);
    expect(result!.cacheDetail).toEqual({
      ephemeral_5m_input_tokens: 60,
      ephemeral_1h_input_tokens: 20,
    });
  });

  it("returns null when usage is missing", () => {
    expect(extractAnthropicUsageFromJSON({ model: "claude-sonnet-4-20250514" })).toBeNull();
  });

  it("returns null for non-object input", () => {
    expect(extractAnthropicUsageFromJSON(null)).toBeNull();
    expect(extractAnthropicUsageFromJSON(undefined)).toBeNull();
    expect(extractAnthropicUsageFromJSON(123)).toBeNull();
  });

  it("returns null when input_tokens is not a number", () => {
    expect(
      extractAnthropicUsageFromJSON({
        usage: { input_tokens: "100", output_tokens: 50 },
      }),
    ).toBeNull();
  });
});
