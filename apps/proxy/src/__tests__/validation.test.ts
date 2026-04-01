import { describe, it, expect } from "vitest";
import { UUID_RE, stripNsPrefix } from "../lib/validation.js";

describe("UUID_RE", () => {
  it("is exported for use in other modules", () => {
    expect(UUID_RE).toBeInstanceOf(RegExp);
    expect(UUID_RE.test("550e8400-e29b-41d4-a716-446655440000")).toBe(true);
  });

  it("rejects non-UUID strings", () => {
    expect(UUID_RE.test("not-a-uuid")).toBe(false);
    expect(UUID_RE.test("")).toBe(false);
    expect(UUID_RE.test("550e8400e29b41d4a716446655440000")).toBe(false);
  });
});

describe("stripNsPrefix", () => {
  const UUID = "550e8400-e29b-41d4-a716-446655440000";

  it("strips ns_act_ prefix and returns UUID", () => {
    expect(stripNsPrefix("ns_act_", `ns_act_${UUID}`)).toBe(UUID);
  });

  it("returns null for null input", () => {
    expect(stripNsPrefix("ns_act_", null)).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(stripNsPrefix("ns_act_", "")).toBeNull();
  });

  it("returns null for raw UUID (no prefix)", () => {
    expect(stripNsPrefix("ns_act_", UUID)).toBeNull();
  });

  it("returns null for wrong prefix", () => {
    expect(stripNsPrefix("ns_act_", `ns_bgt_${UUID}`)).toBeNull();
  });

  it("returns null when UUID after prefix is invalid", () => {
    expect(stripNsPrefix("ns_act_", "ns_act_not-a-uuid")).toBeNull();
  });

  it("returns null for SQL injection attempt", () => {
    expect(stripNsPrefix("ns_act_", "ns_act_'; DROP TABLE budgets;--")).toBeNull();
  });
});
