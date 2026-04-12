import { cloudflareWorkersMock } from "./test-helpers.js";
import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => cloudflareWorkersMock());

import { parseThresholds } from "../durable-objects/user-budget.js";

describe("parseThresholds", () => {
  it("returns default thresholds for null input", () => {
    expect(parseThresholds(null)).toEqual([50, 80, 90, 95]);
  });

  it("returns default thresholds for empty string", () => {
    expect(parseThresholds("")).toEqual([50, 80, 90, 95]);
  });

  it("parses valid JSON array of numbers", () => {
    expect(parseThresholds("[25,50,75]")).toEqual([25, 50, 75]);
  });

  it("parses single-element array", () => {
    expect(parseThresholds("[90]")).toEqual([90]);
  });

  it("parses empty array", () => {
    expect(parseThresholds("[]")).toEqual([]);
  });

  it("returns default for malformed JSON", () => {
    expect(parseThresholds("not json")).toEqual([50, 80, 90, 95]);
  });

  it("returns default for JSON object (not array)", () => {
    expect(parseThresholds('{"a":1}')).toEqual([50, 80, 90, 95]);
  });

  it("returns default for JSON string (not array)", () => {
    expect(parseThresholds('"hello"')).toEqual([50, 80, 90, 95]);
  });

  it("returns default for array containing non-numbers", () => {
    expect(parseThresholds('["a","b"]')).toEqual([50, 80, 90, 95]);
  });

  it("returns default for mixed array (numbers and strings)", () => {
    expect(parseThresholds('[50,"80",90]')).toEqual([50, 80, 90, 95]);
  });

  it("returns default for JSON null", () => {
    expect(parseThresholds("null")).toEqual([50, 80, 90, 95]);
  });

  it("returns default for JSON number (not array)", () => {
    expect(parseThresholds("42")).toEqual([50, 80, 90, 95]);
  });

  it("preserves the default value [50,80,90,95] format", () => {
    expect(parseThresholds("[50,80,90,95]")).toEqual([50, 80, 90, 95]);
  });

  it("returns a fresh array each time (no shared reference mutation)", () => {
    const a = parseThresholds(null);
    const b = parseThresholds(null);
    expect(a).toEqual(b);
    expect(a).not.toBe(b); // different references
    a.push(100);
    expect(b).toEqual([50, 80, 90, 95]); // b unaffected
  });
});
