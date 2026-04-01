import { describe, it, expect, vi, afterEach } from "vitest";
import {
  isRetryableStatusCode,
  isRetryableError,
  parseRetryAfterMs,
  calculateRetryDelayMs,
  DEFAULT_MAX_RETRY_DELAY_MS,
} from "./retry.js";

// ---------------------------------------------------------------------------
// isRetryableStatusCode
// ---------------------------------------------------------------------------

describe("isRetryableStatusCode", () => {
  it.each([429, 500, 502, 503, 504])("returns true for %d", (status) => {
    expect(isRetryableStatusCode(status)).toBe(true);
  });

  it.each([400, 401, 403, 404, 409, 422])("returns false for %d", (status) => {
    expect(isRetryableStatusCode(status)).toBe(false);
  });

  it.each([200, 201, 204])("returns false for success %d", (status) => {
    expect(isRetryableStatusCode(status)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// isRetryableError
// ---------------------------------------------------------------------------

describe("isRetryableError", () => {
  it("returns true for TypeError", () => {
    expect(isRetryableError(new TypeError("fetch failed"))).toBe(true);
  });

  it("returns true for DOMException with name TimeoutError", () => {
    const err = new DOMException("signal timed out", "TimeoutError");
    expect(isRetryableError(err)).toBe(true);
  });

  it("returns false for DOMException with name AbortError", () => {
    const err = new DOMException("aborted", "AbortError");
    expect(isRetryableError(err)).toBe(false);
  });

  it("returns false for generic Error", () => {
    expect(isRetryableError(new Error("oops"))).toBe(false);
  });

  it("returns false for non-Error values", () => {
    expect(isRetryableError("string error")).toBe(false);
    expect(isRetryableError(null)).toBe(false);
    expect(isRetryableError(undefined)).toBe(false);
    expect(isRetryableError(42)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// parseRetryAfterMs
// ---------------------------------------------------------------------------

describe("parseRetryAfterMs", () => {
  it("parses numeric seconds: '30' → 30000", () => {
    expect(parseRetryAfterMs("30", 60_000)).toBe(30_000);
  });

  it("parses decimal seconds: '1.5' → 1500", () => {
    expect(parseRetryAfterMs("1.5", 60_000)).toBe(1_500);
  });

  it("returns null for null input", () => {
    expect(parseRetryAfterMs(null, 60_000)).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(parseRetryAfterMs("", 60_000)).toBeNull();
  });

  it("returns null for whitespace-only string", () => {
    expect(parseRetryAfterMs("  ", 60_000)).toBeNull();
  });

  it("returns null for non-numeric non-date string", () => {
    expect(parseRetryAfterMs("not-a-number", 60_000)).toBeNull();
  });

  it("returns null for negative value", () => {
    expect(parseRetryAfterMs("-5", 60_000)).toBeNull();
  });

  it("returns 0 for '0' (immediate retry)", () => {
    expect(parseRetryAfterMs("0", 60_000)).toBe(0);
  });

  it("parses HTTP date format (future date → positive ms)", () => {
    const futureDate = new Date(Date.now() + 5_000).toUTCString();
    const result = parseRetryAfterMs(futureDate, 60_000);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(5_100); // allow small timing variance
  });

  it("caps at maxDelayMs", () => {
    expect(parseRetryAfterMs("60", 5_000)).toBe(5_000);
  });

  it("returns null for past HTTP date", () => {
    const pastDate = new Date(Date.now() - 10_000).toUTCString();
    expect(parseRetryAfterMs(pastDate, 60_000)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// calculateRetryDelayMs
// ---------------------------------------------------------------------------

describe("calculateRetryDelayMs", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("attempt 0, base 500 → result in [1, 500]", () => {
    for (let i = 0; i < 50; i++) {
      const delay = calculateRetryDelayMs(0, 500, DEFAULT_MAX_RETRY_DELAY_MS);
      expect(delay).toBeGreaterThanOrEqual(1);
      expect(delay).toBeLessThanOrEqual(500);
    }
  });

  it("attempt 1, base 500 → result in [1, 1000]", () => {
    for (let i = 0; i < 50; i++) {
      const delay = calculateRetryDelayMs(1, 500, DEFAULT_MAX_RETRY_DELAY_MS);
      expect(delay).toBeGreaterThanOrEqual(1);
      expect(delay).toBeLessThanOrEqual(1000);
    }
  });

  it("attempt 2, base 500 → result in [1, 2000]", () => {
    for (let i = 0; i < 50; i++) {
      const delay = calculateRetryDelayMs(2, 500, DEFAULT_MAX_RETRY_DELAY_MS);
      expect(delay).toBeGreaterThanOrEqual(1);
      expect(delay).toBeLessThanOrEqual(2000);
    }
  });

  it("attempt 10, base 500, max 5000 → result in [1, 5000] (capped)", () => {
    for (let i = 0; i < 50; i++) {
      const delay = calculateRetryDelayMs(10, 500, 5000);
      expect(delay).toBeGreaterThanOrEqual(1);
      expect(delay).toBeLessThanOrEqual(5000);
    }
  });

  it("always returns >= 1 (never zero)", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    const delay = calculateRetryDelayMs(0, 500, 5000);
    expect(delay).toBeGreaterThanOrEqual(1);
  });

  it("returns deterministic value with mocked random", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    // attempt 0, base 500: floor(0.5 * min(500 * 1, 5000)) = floor(250) = 250
    expect(calculateRetryDelayMs(0, 500, 5000)).toBe(250);
    // attempt 1, base 500: floor(0.5 * min(500 * 2, 5000)) = floor(500) = 500
    expect(calculateRetryDelayMs(1, 500, 5000)).toBe(500);
  });
});
