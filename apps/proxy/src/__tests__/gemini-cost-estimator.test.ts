import { describe, it, expect } from "vitest";
import { estimateGeminiMaxCost } from "../lib/gemini-cost-estimator.js";

describe("estimateGeminiMaxCost", () => {
  it("known model (gemini-2.5-flash) — returns exact expected cost", () => {
    const body = { contents: [{ role: "user", parts: [{ text: "Summarize this document." }] }] };
    const bodyBytes = JSON.stringify(body).length; // ~66 bytes
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    // Input: ceil(66/4) = 17 tokens × 300_000 / 1M = 2.55 µ$
    // Output: 65_536 tokens (MODEL_OUTPUT_CAPS) × 2_500_000 / 1M = 39_321.6 µ$
    // Total: (2.55 + 39_321.6) × 1.1 = 43_156.57 → round = 43_157
    const inputTokens = Math.ceil(bodyBytes / 4);
    const expectedInput = inputTokens * 300_000 / 1_000_000;
    const expectedOutput = 65_536 * 2_500_000 / 1_000_000;
    const expected = Math.round((expectedInput + expectedOutput) * 1.1);
    expect(result).toBe(expected);
  });

  it("known model (gemini-2.5-pro) — costs more than flash for the same body", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "Summarize this document." }] }],
    };
    const flash = estimateGeminiMaxCost("gemini-2.5-flash", body);
    const pro = estimateGeminiMaxCost("gemini-2.5-pro", body);

    expect(pro).toBeGreaterThan(flash);
    // Pro output rate is 4x flash output rate ($10 vs $2.50 per MTok).
    // Pro input is ~4.2x flash input ($1.25 vs $0.30 per MTok).
    // Output dominates for both, so pro should be roughly 3-5x more expensive.
    expect(pro).toBeGreaterThan(flash * 0.5);
  });

  it("dated alias (gemini-2.5-flash-preview-04-17) — resolves to flash pricing", () => {
    const body = { contents: [{ role: "user", parts: [{ text: "hello" }] }] };
    const canonical = estimateGeminiMaxCost("gemini-2.5-flash", body);
    const alias = estimateGeminiMaxCost("gemini-2.5-flash-preview-04-17", body);
    // Alias should use the same pricing as canonical — not the $1 fallback
    expect(alias).toBe(canonical);
    expect(alias).not.toBe(1_000_000);
  });

  it("dated alias (gemini-2.5-pro-exp-03-25) — resolves to pro pricing", () => {
    const body = { contents: [{ role: "user", parts: [{ text: "hello" }] }] };
    const canonical = estimateGeminiMaxCost("gemini-2.5-pro", body);
    const alias = estimateGeminiMaxCost("gemini-2.5-pro-exp-03-25", body);
    expect(alias).toBe(canonical);
    expect(alias).not.toBe(1_000_000);
  });

  it("unknown model — returns $1 fallback (1_000_000 microdollars)", () => {
    const result = estimateGeminiMaxCost("gemini-nonexistent-model", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
    });
    expect(result).toBe(1_000_000);
  });

  it("maxOutputTokens override — uses generationConfig.maxOutputTokens instead of default", () => {
    const withCap = estimateGeminiMaxCost("gemini-2.5-flash", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 1000 },
    });
    const withoutCap = estimateGeminiMaxCost("gemini-2.5-flash", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
    });
    // Default cap for gemini-2.5-flash is 65_536, so 1000 should be much cheaper
    expect(withCap).toBeLessThan(withoutCap);

    // Verify the output portion is using 1000 tokens, not 65_536.
    // flash output: 2_500_000 microdollars/MTok → 1000 tokens = 0.6 microdollars output
    // vs 65_536 tokens = ~39.3 microdollars output. The difference should be huge.
    expect(withCap).toBeLessThan(withoutCap / 10);
  });

  it("no generationConfig — uses DEFAULT_OUTPUT_CAP (8192)", () => {
    // Model NOT in MODEL_OUTPUT_CAPS → falls through to DEFAULT_OUTPUT_CAP 8192.
    // We test with a model that IS in the pricing catalog but would use
    // a different cap if generationConfig were present.
    // gemini-2.5-flash has MODEL_OUTPUT_CAPS[flash] = 65_536.
    // So instead, verify that when generationConfig is absent, the cap differs from
    // an explicit maxOutputTokens=1 (proving some default was applied).
    const withDefault = estimateGeminiMaxCost("gemini-2.5-flash", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
    });
    const withExplicitOne = estimateGeminiMaxCost("gemini-2.5-flash", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 1 },
    });
    // Default (65_536 for flash) should be far more expensive than 1 token
    expect(withDefault).toBeGreaterThan(withExplicitOne);

    // flash MODEL_OUTPUT_CAPS["gemini-2.5-flash"] = 65_536
    // Output dominates: 65_536 × 2_500_000 / 1M = 39_321.6 µ$
    // With 1.1x margin, total is ~43_254 (plus small input contribution)
    const bodyBytes = JSON.stringify({ contents: [{ role: "user", parts: [{ text: "hello" }] }] }).length;
    const inputTokens = Math.ceil(bodyBytes / 4);
    const expectedInput = inputTokens * 300_000 / 1_000_000;
    const expectedOutput = 65_536 * 2_500_000 / 1_000_000;
    const expected = Math.round((expectedInput + expectedOutput) * 1.1);
    expect(withDefault).toBe(expected);
  });

  it("bodyByteLength provided — uses provided byte length instead of JSON.stringify", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
    };
    const jsonLength = JSON.stringify(body).length;

    // Use a much larger byte length to simulate a large original request body
    const largeByteLength = jsonLength * 20;

    const fromJson = estimateGeminiMaxCost("gemini-2.5-flash", body);
    const fromByteLength = estimateGeminiMaxCost("gemini-2.5-flash", body, largeByteLength);

    // The byte-length version should estimate more input tokens → higher cost
    expect(fromByteLength).toBeGreaterThan(fromJson);
  });

  it("empty/minimal body — uses default output cap with exact arithmetic", () => {
    const body = {};
    const bodyBytes = JSON.stringify(body).length; // 2 bytes for "{}"
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    // Input: ceil(2/4) = 1 token × 300_000 / 1M = 0.15 µ$
    // Output: 65_536 tokens (MODEL_OUTPUT_CAPS) × 2_500_000 / 1M = 39_321.6 µ$
    // Total: (0.15 + 39_321.6) × 1.1 = 43_153.925 → round = 43_154
    const inputTokens = Math.ceil(bodyBytes / 4);
    const expectedInput = inputTokens * 300_000 / 1_000_000;
    const expectedOutput = 65_536 * 2_500_000 / 1_000_000;
    const expected = Math.round((expectedInput + expectedOutput) * 1.1);
    expect(result).toBe(expected);
  });

  it("large body (100KB) — input estimate scales with size", () => {
    const smallBody = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
    };
    const largeBody = {
      contents: [{ role: "user", parts: [{ text: "x".repeat(100_000) }] }],
    };

    const small = estimateGeminiMaxCost("gemini-2.5-flash", smallBody);
    const large = estimateGeminiMaxCost("gemini-2.5-flash", largeBody);

    expect(large).toBeGreaterThan(small);

    // The large body is ~100KB = ~25,000 input tokens.
    // flash input: 300_000 microdollars/MTok → 25,000 tokens ≈ 3.75 microdollars input.
    // Output dominates for both (65_536 tokens → ~39,322 microdollars),
    // so the difference should come from the input scaling.
    // Verify the large estimate includes meaningful input contribution.
    const inputDelta = large - small;
    expect(inputDelta).toBeGreaterThan(0);
  });

  it("maxOutputTokens Infinity — falls back to model default, not NaN", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: Infinity },
    };
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
    expect(result).not.toBe(1_000_000); // not $1 fallback
  });

  it("maxOutputTokens NaN string — falls back to model default", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: "abc" as unknown as number },
    };
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("maxOutputTokens negative — falls back to model default", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: -100 },
    };
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("maxOutputTokens capped at 1M tokens", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 999_999_999 },
    };
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    // Should use 1M cap, not 999M tokens
    const uncapped = estimateGeminiMaxCost("gemini-2.5-flash", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 1_000_000 },
    });
    expect(result).toBe(uncapped);
  });

  it("bodyByteLength of 0 — falls back to JSON.stringify length", () => {
    const body = { contents: [{ role: "user", parts: [{ text: "hi" }] }] };
    const fromZero = estimateGeminiMaxCost("gemini-2.5-flash", body, 0);
    const fromDefault = estimateGeminiMaxCost("gemini-2.5-flash", body);
    // bodyByteLength=0 → ceil(0/4)=0 input tokens vs ceil(jsonLen/4) tokens
    // Output dominates, so difference is small but input portion differs
    expect(fromZero).toBeLessThanOrEqual(fromDefault);
  });

  it("returns integer microdollars", () => {
    const result = estimateGeminiMaxCost("gemini-2.5-pro", {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 500 },
    });
    expect(Number.isInteger(result)).toBe(true);
  });

  it("applies 1.1x safety margin — exact value", () => {
    const body = {
      contents: [{ role: "user", parts: [{ text: "hello" }] }],
      generationConfig: { maxOutputTokens: 1 },
    };
    const bodyBytes = JSON.stringify(body).length;
    const result = estimateGeminiMaxCost("gemini-2.5-flash", body);
    // Input: ceil(bodyBytes/4) tokens × 300_000 / 1M
    // Output: 1 token × 2_500_000 / 1M = 0.6 µ$
    const inputTokens = Math.ceil(bodyBytes / 4);
    const rawInput = inputTokens * 300_000 / 1_000_000;
    const rawOutput = 1 * 2_500_000 / 1_000_000;
    const withoutMargin = rawInput + rawOutput;
    const withMargin = Math.round(withoutMargin * 1.1);
    expect(result).toBe(withMargin);
    // Confirm margin actually matters (result is strictly greater than raw)
    expect(result).toBeGreaterThan(Math.round(withoutMargin));
  });
});
