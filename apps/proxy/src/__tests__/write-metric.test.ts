import { describe, it, expect, vi } from "vitest";
import { writeLatencyDataPoint } from "../lib/write-metric.js";

function makeEnv(metrics?: { writeDataPoint: ReturnType<typeof vi.fn> }): Env {
  return {
    ...(metrics ? { METRICS: metrics } : {}),
  } as unknown as Env;
}

describe("writeLatencyDataPoint", () => {
  it("calls writeDataPoint with correct blobs, doubles, and indexes", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o", true, 200, 8, 1200, 1208);

    expect(writeDataPoint).toHaveBeenCalledOnce();
    expect(writeDataPoint).toHaveBeenCalledWith({
      blobs: ["openai", "gpt-4o", "stream", "200", ""],
      doubles: [8, 1200, 1208, 0],
      indexes: ["openai"],
    });
  });

  it("writes 'json' blob for non-streaming requests", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "anthropic", "claude-3-haiku-20240307", false, 200, 5, 800, 805);

    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        blobs: ["anthropic", "claude-3-haiku-20240307", "json", "200", ""],
      }),
    );
  });

  it("is a no-op when METRICS binding is undefined", () => {
    const env = makeEnv(); // no METRICS binding

    // Should not throw
    expect(() => {
      writeLatencyDataPoint(env, "openai", "gpt-4o", true, 200, 8, 1200, 1208);
    }).not.toThrow();
  });

  it("swallows errors if writeDataPoint throws", () => {
    const writeDataPoint = vi.fn().mockImplementation(() => {
      throw new Error("AE binding misconfigured");
    });
    const env = makeEnv({ writeDataPoint });

    expect(() => {
      writeLatencyDataPoint(env, "openai", "gpt-4o", true, 200, 8, 1200, 1208);
    }).not.toThrow();

    expect(writeDataPoint).toHaveBeenCalledOnce();
  });

  it("handles zero values correctly", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o-mini", false, 200, 0, 0, 0);

    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        doubles: [0, 0, 0, 0],
      }),
    );
  });

  it("includes status code as string blob for error responses", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o", false, 429, 12, 0, 12);

    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        blobs: ["openai", "gpt-4o", "json", "429", ""],
      }),
    );
  });

  it("includes ttfbMs as fourth double when provided", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o", true, 200, 8, 1200, 1208, 350, "user-123");

    expect(writeDataPoint).toHaveBeenCalledWith({
      blobs: ["openai", "gpt-4o", "stream", "200", "user-123"],
      doubles: [8, 1200, 1208, 350],
      indexes: ["openai"],
    });
  });

  it("defaults userId to empty string when only ttfbMs provided", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o", true, 200, 8, 1200, 1208, 350);

    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        blobs: ["openai", "gpt-4o", "stream", "200", ""],
        doubles: [8, 1200, 1208, 350],
      }),
    );
  });

  it("defaults ttfbMs to 0 when not provided", () => {
    const writeDataPoint = vi.fn();
    const env = makeEnv({ writeDataPoint });

    writeLatencyDataPoint(env, "openai", "gpt-4o", false, 200, 5, 800, 805);

    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        doubles: [5, 800, 805, 0],
      }),
    );
  });
});
