import { describe, it, expect } from "vitest";
import * as costEngine from "./index.js";
import type { ModelPricing, CostEvent, Provider } from "./types.js";

describe("cost-engine public API surface", () => {
  it("exports getModelPricing function", () => {
    expect(typeof costEngine.getModelPricing).toBe("function");
  });

  it("exports costComponent function", () => {
    expect(typeof costEngine.costComponent).toBe("function");
  });

  it("exports isKnownModel function", () => {
    expect(typeof costEngine.isKnownModel).toBe("function");
  });

  it("exports getAllPricing function", () => {
    expect(typeof costEngine.getAllPricing).toBe("function");
  });

  it("exports exactly 5 runtime values", () => {
    const runtimeExports = Object.keys(costEngine).filter(
      (k) => typeof (costEngine as Record<string, unknown>)[k] !== "undefined",
    );
    expect(runtimeExports).toHaveLength(5);
    expect(runtimeExports.sort()).toEqual(["RATE_SCALE", "costComponent", "getAllPricing", "getModelPricing", "isKnownModel"]);
  });
});

describe("type contract stability", () => {
  it("ModelPricing has expected shape (integer µ$/MTok rates)", () => {
    const pricing: ModelPricing = {
      inputPerMTok: 1_000_000,
      cachedInputPerMTok: 500_000,
      outputPerMTok: 2_000_000,
    };
    expect(pricing.inputPerMTok).toBe(1_000_000);
  });

  it("ModelPricing with Anthropic cache fields", () => {
    const pricing: ModelPricing = {
      inputPerMTok: 3_000_000,
      cachedInputPerMTok: 300_000,
      outputPerMTok: 15_000_000,
      cacheWrite5mPerMTok: 3_750_000,
      cacheWrite1hPerMTok: 6_000_000,
    };
    expect(pricing.cacheWrite5mPerMTok).toBe(3_750_000);
  });

  it("CostEvent has expected shape", () => {
    const event: CostEvent = {
      requestId: "req-123",
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      cachedInputTokens: 0,
      reasoningTokens: 0,
      costMicrodollars: 1750,
    };
    expect(event.costMicrodollars).toBe(1750);
  });

  it("CostEvent with optional durationMs", () => {
    const event: CostEvent = {
      requestId: "req-456",
      provider: "anthropic",
      model: "claude-sonnet-4-6",
      inputTokens: 200,
      outputTokens: 100,
      cachedInputTokens: 50,
      reasoningTokens: 0,
      costMicrodollars: 2250,
      durationMs: 1500,
    };
    expect(event.durationMs).toBe(1500);
  });

  it("Provider type accepts valid values", () => {
    const providers: Provider[] = ["openai", "anthropic", "google"];
    expect(providers).toHaveLength(3);
  });
});
