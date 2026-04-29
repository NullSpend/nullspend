import { describe, expect, it } from "vitest";
import { ApiVersion, isCalVer } from "./api-version";

describe("isCalVer", () => {
  it("accepts well-formed CalVer", () => {
    expect(isCalVer("2026-04-01")).toBe(true);
    expect(isCalVer("1999-12-31")).toBe(true);
  });

  it.each([
    [""],
    ["2026-4-1"],
    ["2026/04/01"],
    ["04-01-2026"],
    ["2026-04-01T00:00:00Z"],
    [" 2026-04-01"],
    ["2026-04-01 "],
    ["v1"],
    ["latest"],
  ])("rejects malformed %j", (value) => {
    expect(isCalVer(value)).toBe(false);
  });

  it.each([
    ["2026-13-01"],
    ["2026-00-15"],
    ["2026-02-30"],
    ["2026-04-31"],
    ["2026-06-31"],
    ["2026-09-31"],
    ["2026-11-31"],
    ["0000-00-00"],
  ])("rejects semantically-invalid date %j", (value) => {
    expect(isCalVer(value)).toBe(false);
  });

  it("accepts leap-year Feb 29 (2024) and rejects non-leap Feb 29 (2026)", () => {
    expect(isCalVer("2024-02-29")).toBe(true);
    expect(isCalVer("2026-02-29")).toBe(false);
  });

  it("rejects non-strings", () => {
    expect(isCalVer(null)).toBe(false);
    expect(isCalVer(undefined)).toBe(false);
    expect(isCalVer(20260401)).toBe(false);
    expect(isCalVer({})).toBe(false);
  });
});

describe("ApiVersion comparators", () => {
  const v1 = new ApiVersion("2026-04-01", 0);
  const v2 = new ApiVersion("2026-05-01", 1);
  const v1Twin = new ApiVersion("2026-04-01", 0);

  it("eq compares by index", () => {
    expect(v1.eq(v1Twin)).toBe(true);
    expect(v1.eq(v2)).toBe(false);
  });

  it("lt / lte / gt / gte", () => {
    expect(v1.lt(v2)).toBe(true);
    expect(v2.lt(v1)).toBe(false);
    expect(v1.lte(v1Twin)).toBe(true);
    expect(v2.gt(v1)).toBe(true);
    expect(v1.gte(v1Twin)).toBe(true);
    expect(v1.gte(v2)).toBe(false);
  });

  it("toString / toJSON return value", () => {
    expect(v1.toString()).toBe("2026-04-01");
    expect(JSON.stringify({ v: v1 })).toBe('{"v":"2026-04-01"}');
  });
});
