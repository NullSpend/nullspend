import { describe, it, expect } from "vitest";
import { ensureStreamOptions, extractModelFromBody } from "../lib/request-utils.js";

describe("ensureStreamOptions", () => {
  it("injects stream_options when stream: true and no stream_options present", () => {
    const body: Record<string, unknown> = { model: "gpt-4o", stream: true };
    ensureStreamOptions(body);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("forces include_usage: true when stream_options exists but include_usage is false", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: { include_usage: false },
    };
    ensureStreamOptions(body);
    expect((body.stream_options as Record<string, unknown>).include_usage).toBe(true);
  });

  it("forces include_usage: true when stream_options exists but include_usage is missing", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: {},
    };
    ensureStreamOptions(body);
    expect((body.stream_options as Record<string, unknown>).include_usage).toBe(true);
  });

  it("preserves other stream_options fields when force-merging include_usage", () => {
    const body: Record<string, unknown> = {
      model: "gpt-4o",
      stream: true,
      stream_options: { include_usage: false, some_other_field: "keep_me" },
    };
    ensureStreamOptions(body);
    const opts = body.stream_options as Record<string, unknown>;
    expect(opts.include_usage).toBe(true);
    expect(opts.some_other_field).toBe("keep_me");
  });

  it("does NOT inject stream_options when stream: false", () => {
    const body: Record<string, unknown> = { model: "gpt-4o", stream: false };
    ensureStreamOptions(body);
    expect(body.stream_options).toBeUndefined();
  });

  it("does NOT inject stream_options when stream field is absent", () => {
    const body: Record<string, unknown> = { model: "gpt-4o" };
    ensureStreamOptions(body);
    expect(body.stream_options).toBeUndefined();
  });

  it("does NOT inject stream_options when stream_options is a non-object truthy value", () => {
    const body: Record<string, unknown> = { model: "gpt-4o", stream: true, stream_options: "invalid" };
    ensureStreamOptions(body);
    expect(body.stream_options).toEqual({ include_usage: true });
  });
});

describe("extractModelFromBody", () => {
  it("extracts model string from body", () => {
    expect(extractModelFromBody({ model: "gpt-4o" })).toBe("gpt-4o");
  });

  it("returns 'unknown' when model is missing", () => {
    expect(extractModelFromBody({})).toBe("unknown");
  });

  it("returns 'unknown' when model is not a string", () => {
    expect(extractModelFromBody({ model: 42 })).toBe("unknown");
  });
});
