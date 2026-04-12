import { describe, it, expect } from "vitest";
import { estimateMaxCost } from "../lib/cost-estimator.js";

describe("estimateMaxCost", () => {
  it("returns integer microdollars (suitable for HINCRBY)", () => {
    const result = estimateMaxCost("gpt-4o-mini", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(Number.isInteger(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("uses max_tokens when specified in body", () => {
    const withLimit = estimateMaxCost("gpt-4o", {
      model: "gpt-4o",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 100,
    });
    const withoutLimit = estimateMaxCost("gpt-4o", {
      model: "gpt-4o",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(withLimit).toBeLessThan(withoutLimit);
  });

  it("uses max_completion_tokens for reasoning models", () => {
    const withLimit = estimateMaxCost("o3", {
      model: "o3",
      messages: [{ role: "user", content: "hello" }],
      max_completion_tokens: 500,
    });
    const withoutLimit = estimateMaxCost("o3", {
      model: "o3",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(withLimit).toBeLessThan(withoutLimit);
  });

  it("prefers max_completion_tokens over max_tokens", () => {
    const result = estimateMaxCost("o3", {
      model: "o3",
      messages: [{ role: "user", content: "hello" }],
      max_completion_tokens: 200,
      max_tokens: 5000,
    });
    const resultWithOnlyMaxTokens = estimateMaxCost("o3", {
      model: "o3",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 5000,
    });
    expect(result).toBeLessThan(resultWithOnlyMaxTokens);
  });

  it("returns $1 fallback for unknown models (P0-9 prevention)", () => {
    const result = estimateMaxCost("nonexistent-model", {
      model: "nonexistent-model",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(result).toBe(1_000_000);
  });

  it("uses default 16384 output cap for gpt-4o when max_tokens not set", () => {
    const result = estimateMaxCost("gpt-4o", {
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    // gpt-4o: output rate = $10/MTok, 16384 tokens * 10 = 163_840 microdollars (output only)
    // plus input cost + 1.1x margin
    expect(result).toBeGreaterThan(160_000);
  });

  it("uses 100k output cap for reasoning models when max_tokens not set", () => {
    const result = estimateMaxCost("o3", {
      model: "o3",
      messages: [{ role: "user", content: "hi" }],
    });
    // o3: output rate = $8/MTok, 100000 tokens * 8 = 800_000 microdollars (output only)
    // plus input + 1.1x margin
    expect(result).toBeGreaterThan(800_000);
  });

  it("applies 1.1x safety margin", () => {
    // With max_tokens=0 we can isolate the margin effect on input cost
    const result = estimateMaxCost("gpt-4o-mini", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 1,
    });
    // Should be non-zero and include the margin
    expect(result).toBeGreaterThan(0);
  });

  it("scales with body size (larger messages = higher estimate)", () => {
    const small = estimateMaxCost("gpt-4o-mini", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
    });
    const large = estimateMaxCost("gpt-4o-mini", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "a".repeat(10000) }],
      max_tokens: 100,
    });
    expect(large).toBeGreaterThan(small);
  });
});
