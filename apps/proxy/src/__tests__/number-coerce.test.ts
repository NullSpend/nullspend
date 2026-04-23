import { describe, it, expect } from "vitest";
import { toFiniteNumber } from "../lib/number-coerce.js";

// Per PR-2a Decision #36 / plan-audit A6 / codex PR-2a-R3-1:
// postgres.js with `fetch_types:false` returns BIGINT columns as string or BigInt,
// NOT JS number. Without `toFiniteNumber` coercion, the `ApiKeyIdentity.subscriptionPeriodStart`
// type contract (`number | null`) is fiction and downstream `Date.UTC(...)` math produces NaN.

describe("toFiniteNumber", () => {
  // C-TFN1 — string input (most common case under fetch_types:false)
  it("coerces numeric string to number", () => {
    expect(toFiniteNumber("1744848000000")).toBe(1744848000000);
  });

  // C-TFN2 — BigInt input
  it("coerces BigInt to number", () => {
    expect(toFiniteNumber(1744848000000n)).toBe(1744848000000);
  });

  // C-TFN3 — number input (pass-through)
  it("returns number as-is", () => {
    expect(toFiniteNumber(1744848000000)).toBe(1744848000000);
  });

  // C-TFN4 — null → null
  it("returns null for null", () => {
    expect(toFiniteNumber(null)).toBeNull();
  });

  // C-TFN5 — undefined → null
  it("returns null for undefined", () => {
    expect(toFiniteNumber(undefined)).toBeNull();
  });

  // C-TFN6 — unparseable string → null
  it("returns null for non-numeric string", () => {
    expect(toFiniteNumber("not a number")).toBeNull();
  });

  // C-TFN7 — Infinity → null (via Number.isFinite guard)
  it("returns null for Infinity", () => {
    expect(toFiniteNumber(Number.POSITIVE_INFINITY)).toBeNull();
    expect(toFiniteNumber(Number.NEGATIVE_INFINITY)).toBeNull();
  });

  // C-TFN8 — NaN → null
  it("returns null for NaN", () => {
    expect(toFiniteNumber(NaN)).toBeNull();
  });

  // C-TFN9 — negative values pass through (coerce doesn't filter sign)
  it("passes through negative values", () => {
    expect(toFiniteNumber(-1)).toBe(-1);
    expect(toFiniteNumber("-1744848000000")).toBe(-1744848000000);
  });

  // Extra defensive tests — object/array/boolean coercion
  it("returns null for object", () => {
    expect(toFiniteNumber({})).toBeNull();
    expect(toFiniteNumber({ value: 123 })).toBeNull();
  });

  it("returns null for array", () => {
    // [] coerces to 0, [1] coerces to 1 via Number() — but both are defensible edge cases.
    // We expect null to fail-soft rather than accept ambiguous input.
    expect(toFiniteNumber([])).toBeNull();
    expect(toFiniteNumber([1])).toBeNull();
  });

  it("returns null for boolean", () => {
    // true → 1, false → 0 via Number() coercion — but we reject booleans
    // because they're never valid for DB BIGINT columns.
    expect(toFiniteNumber(true)).toBeNull();
    expect(toFiniteNumber(false)).toBeNull();
  });

  // Cubic post-merge review: `Number(Symbol())` throws TypeError. Without an
  // explicit guard, the "fail-soft" contract in the JSDoc would be a lie.
  it("returns null for Symbol (does NOT throw TypeError)", () => {
    expect(toFiniteNumber(Symbol("x"))).toBeNull();
    expect(toFiniteNumber(Symbol.iterator)).toBeNull();
    // Verify explicitly that this doesn't throw — the previous impl would have.
    expect(() => toFiniteNumber(Symbol("y"))).not.toThrow();
  });
});
