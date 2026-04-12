import { describe, it, expect } from "vitest";
import { ensureStreamOptions, extractModelFromBody } from "../lib/request-utils.js";

describe("ensureStreamOptions edge cases", () => {
  it("does NOT inject when stream is the string 'true' (not boolean)", () => {
    const body: Record<string, unknown> = { model: "gpt-4o", stream: "true" };
    ensureStreamOptions(body);
    expect(body.stream_options).toBeUndefined();
  });

  it("does NOT inject when stream is 1 (truthy but not boolean true)", () => {
    const body: Record<string, unknown> = { model: "gpt-4o", stream: 1 };
    ensureStreamOptions(body);
    expect(body.stream_options).toBeUndefined();
  });

  it("replaces stream_options: null with proper object", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: null,
    };
    ensureStreamOptions(body);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("replaces stream_options: [] (array) with proper object", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: [],
    };
    ensureStreamOptions(body);
    // Arrays are typeof "object" but are handled by the isArray-like truthy check
    // The function checks typeof === "object", arrays pass this.
    // This tests whether the function correctly handles array edge case
    expect((body.stream_options as any).include_usage).toBe(true);
  });

  it("replaces stream_options: 42 (number) with proper object", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: 42,
    };
    ensureStreamOptions(body);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("replaces stream_options: false with proper object", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: false,
    };
    ensureStreamOptions(body);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("preserves include_usage: true when already set correctly", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: { include_usage: true },
    };
    ensureStreamOptions(body);
    expect((body.stream_options as Record<string, unknown>).include_usage).toBe(true);
  });

  it("handles deeply nested extra fields in stream_options", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: {
        include_usage: false,
        custom_nested: { deep: { value: 42 } },
      },
    };
    ensureStreamOptions(body);
    const opts = body.stream_options as Record<string, unknown>;
    expect(opts.include_usage).toBe(true);
    expect((opts.custom_nested as any).deep.value).toBe(42);
  });
});

describe("extractModelFromBody edge cases", () => {
  it("returns 'unknown' for null model", () => {
    expect(extractModelFromBody({ model: null })).toBe("unknown");
  });

  it("returns 'unknown' for undefined model", () => {
    expect(extractModelFromBody({ model: undefined })).toBe("unknown");
  });

  it("returns 'unknown' for boolean model", () => {
    expect(extractModelFromBody({ model: true })).toBe("unknown");
  });

  it("returns 'unknown' for array model", () => {
    expect(extractModelFromBody({ model: ["gpt-4o"] })).toBe("unknown");
  });

  it("returns 'unknown' for object model", () => {
    expect(extractModelFromBody({ model: { name: "gpt-4o" } })).toBe("unknown");
  });

  it("returns empty string model as-is (it is technically a string)", () => {
    expect(extractModelFromBody({ model: "" })).toBe("");
  });

  it("extracts model with special characters", () => {
    expect(extractModelFromBody({ model: "ft:gpt-4o:my-org:custom_name:abc123" })).toBe(
      "ft:gpt-4o:my-org:custom_name:abc123",
    );
  });

  it("extracts model with unicode characters", () => {
    expect(extractModelFromBody({ model: "my-model-日本語" })).toBe("my-model-日本語");
  });
});
