import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { emitMetric } from "../lib/metrics.js";

describe("emitMetric", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(Date, "now").mockReturnValue(1710000000000);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("emits valid JSON with _metric name and _ts timestamp", () => {
    emitMetric("reconciliation", { status: "ok", durationMs: 42 });

    expect(console.log).toHaveBeenCalledTimes(1);
    const output = (console.log as any).mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed._metric).toBe("reconciliation");
    expect(parsed._ts).toBe(1710000000000);
    expect(parsed.status).toBe("ok");
    expect(parsed.durationMs).toBe(42);
  });

  it("handles boolean and numeric tags", () => {
    emitMetric("test_metric", { count: 5, success: true, rate: 0.95 });

    const output = (console.log as any).mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed.count).toBe(5);
    expect(parsed.success).toBe(true);
    expect(parsed.rate).toBe(0.95);
  });

  it("handles empty tags", () => {
    emitMetric("empty", {});

    const output = (console.log as any).mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed._metric).toBe("empty");
    expect(parsed._ts).toBe(1710000000000);
  });
});
