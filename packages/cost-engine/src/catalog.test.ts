import { describe, it, expect } from "vitest";
import pricingData from "./pricing-data.json";
import { getModelPricing } from "./pricing.js";
import type { ModelPricing } from "./types.js";

const catalog = pricingData as Record<string, ModelPricing>;
const entries = Object.entries(catalog);

const LEGACY_ANTHROPIC_MODELS = new Set([
  "anthropic/claude-haiku-3",
  "anthropic/claude-3-haiku-20240307",
]);

describe("pricing catalog integrity", () => {
  it("has at least 10 models", () => {
    expect(entries.length).toBeGreaterThanOrEqual(10);
  });

  it("every key follows provider/model format", () => {
    for (const [key] of entries) {
      expect(key).toMatch(/^[a-z]+\/[a-z0-9._-]+$/);
    }
  });

  it("every key has a valid provider prefix", () => {
    const validProviders = new Set(["openai", "anthropic", "google"]);
    for (const [key] of entries) {
      const provider = key.split("/")[0];
      expect(validProviders.has(provider), `unknown provider: ${provider} in ${key}`).toBe(true);
    }
  });

  it("no duplicate keys (JSON parse would merge, but verify explicit structure)", () => {
    const keys = entries.map(([k]) => k);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("per-model field validation", () => {
  for (const [key, pricing] of entries) {
    describe(key, () => {
      it("has positive inputPerMTok", () => {
        expect(pricing.inputPerMTok).toBeGreaterThan(0);
        expect(Number.isFinite(pricing.inputPerMTok)).toBe(true);
      });

      it("has non-negative cachedInputPerMTok", () => {
        expect(pricing.cachedInputPerMTok).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(pricing.cachedInputPerMTok)).toBe(true);
      });

      it("has positive outputPerMTok", () => {
        expect(pricing.outputPerMTok).toBeGreaterThan(0);
        expect(Number.isFinite(pricing.outputPerMTok)).toBe(true);
      });

      it("cached input rate is <= standard input rate", () => {
        expect(pricing.cachedInputPerMTok).toBeLessThanOrEqual(pricing.inputPerMTok);
      });

      it("output rate is >= input rate (standard LLM pricing pattern)", () => {
        expect(pricing.outputPerMTok).toBeGreaterThanOrEqual(pricing.inputPerMTok);
      });

      if (key.startsWith("anthropic/")) {
        it("has cacheWrite5mPerMTok field", () => {
          expect(pricing.cacheWrite5mPerMTok).toBeDefined();
          expect(pricing.cacheWrite5mPerMTok).toBeGreaterThan(0);
        });

        it("has cacheWrite1hPerMTok field", () => {
          expect(pricing.cacheWrite1hPerMTok).toBeDefined();
          expect(pricing.cacheWrite1hPerMTok).toBeGreaterThan(0);
        });

        if (LEGACY_ANTHROPIC_MODELS.has(key)) {
          it("cache write 5m > input (legacy model — non-standard multiplier)", () => {
            expect(pricing.cacheWrite5mPerMTok!).toBeGreaterThan(pricing.inputPerMTok);
          });

          it("cache write 1h > cache write 5m (legacy model)", () => {
            expect(pricing.cacheWrite1hPerMTok!).toBeGreaterThan(pricing.cacheWrite5mPerMTok!);
          });
        } else {
          it("5m cache write rate is 1.25x base input (Anthropic pricing rule)", () => {
            expect(pricing.cacheWrite5mPerMTok).toBeCloseTo(pricing.inputPerMTok * 1.25, 10);
          });

          it("1h cache write rate is 2.0x base input (Anthropic pricing rule)", () => {
            expect(pricing.cacheWrite1hPerMTok).toBeCloseTo(pricing.inputPerMTok * 2.0, 10);
          });

          it("cache read rate is 0.1x base input (Anthropic pricing rule)", () => {
            expect(pricing.cachedInputPerMTok).toBeCloseTo(pricing.inputPerMTok * 0.1, 10);
          });
        }

        it("cache write rates follow 5m < 1h ordering", () => {
          expect(pricing.cacheWrite5mPerMTok!).toBeLessThan(pricing.cacheWrite1hPerMTok!);
        });
      }

      if (key.startsWith("openai/") || key.startsWith("google/")) {
        it("does NOT have Anthropic-specific cache write fields", () => {
          expect(pricing.cacheWrite5mPerMTok).toBeUndefined();
          expect(pricing.cacheWrite1hPerMTok).toBeUndefined();
        });
      }

      it("no unexpected fields exist", () => {
        const allowedKeys = new Set([
          "inputPerMTok",
          "cachedInputPerMTok",
          "outputPerMTok",
          "cacheWrite5mPerMTok",
          "cacheWrite1hPerMTok",
        ]);
        for (const field of Object.keys(pricing)) {
          expect(allowedKeys.has(field), `unexpected field "${field}" in ${key}`).toBe(true);
        }
      });

      it("no NaN or Infinity values", () => {
        for (const [field, value] of Object.entries(pricing)) {
          if (typeof value === "number") {
            expect(Number.isNaN(value), `NaN in ${key}.${field}`).toBe(false);
            expect(Number.isFinite(value), `Infinity in ${key}.${field}`).toBe(true);
          }
        }
      });
    });
  }
});

describe("alias consistency", () => {
  const aliasPairs: [string, string[]][] = [
    ["claude-opus-4-6", ["claude-opus-4-6-20260205"]],
    ["claude-sonnet-4-6", ["claude-sonnet-4-6-20260217"]],
    ["claude-sonnet-4-5", ["claude-sonnet-4-5-20250929"]],
    ["claude-opus-4-5", ["claude-opus-4-5-20251101"]],
    ["claude-haiku-4-5", ["claude-haiku-4-5-20251001"]],
    ["claude-opus-4-1", ["claude-opus-4-1-20250805"]],
    ["claude-opus-4", ["claude-opus-4-20250514", "claude-opus-4-0"]],
    ["claude-sonnet-4", ["claude-sonnet-4-20250514", "claude-sonnet-4-0"]],
    ["claude-haiku-3.5", ["claude-3-5-haiku-20241022"]],
    ["claude-haiku-3", ["claude-3-haiku-20240307"]],
  ];

  for (const [shortName, aliases] of aliasPairs) {
    for (const alias of aliases) {
      it(`${alias} has same pricing as ${shortName}`, () => {
        const base = getModelPricing("anthropic", shortName);
        const aliased = getModelPricing("anthropic", alias);
        expect(base, `${shortName} not found in pricing data`).not.toBeNull();
        expect(aliased, `${alias} not found in pricing data`).not.toBeNull();
        expect(aliased).toEqual(base);
      });
    }
  }
});

describe("cross-model sanity checks", () => {
  it("opus 4 ($75) is the most expensive Anthropic model per output token", () => {
    const opus4 = catalog["anthropic/claude-opus-4"];
    const opus46 = catalog["anthropic/claude-opus-4-6"];

    expect(opus4.outputPerMTok).toBeGreaterThan(opus46.outputPerMTok);
  });

  it("opus 4.6 ($25) > sonnet 4.6 ($15) on output rate", () => {
    const opus46 = catalog["anthropic/claude-opus-4-6"];
    const sonnet46 = catalog["anthropic/claude-sonnet-4-6"];

    expect(opus46.outputPerMTok).toBeGreaterThan(sonnet46.outputPerMTok);
  });

  it("sonnet 4.6 ($15) > haiku 4.5 ($5) > haiku 3.5 ($4) > haiku 3 ($1.25) on output rate", () => {
    const sonnet46 = catalog["anthropic/claude-sonnet-4-6"];
    const haiku45 = catalog["anthropic/claude-haiku-4-5"];
    const haiku35 = catalog["anthropic/claude-haiku-3.5"];
    const haiku3 = catalog["anthropic/claude-haiku-3"];

    expect(sonnet46.outputPerMTok).toBeGreaterThan(haiku45.outputPerMTok);
    expect(haiku45.outputPerMTok).toBeGreaterThan(haiku35.outputPerMTok);
    expect(haiku35.outputPerMTok).toBeGreaterThan(haiku3.outputPerMTok);
  });

  it("every Anthropic model: cacheWrite5m > input and cacheWrite1h > cacheWrite5m", () => {
    for (const [key, pricing] of entries) {
      if (!key.startsWith("anthropic/")) continue;
      expect(pricing.cacheWrite5mPerMTok!, `${key}: 5m write > input`).toBeGreaterThan(pricing.inputPerMTok);
      expect(pricing.cacheWrite1hPerMTok!, `${key}: 1h write > 5m write`).toBeGreaterThan(pricing.cacheWrite5mPerMTok!);
    }
  });

  it("gpt-4o is more expensive than gpt-4o-mini", () => {
    const full = catalog["openai/gpt-4o"];
    const mini = catalog["openai/gpt-4o-mini"];

    expect(full.inputPerMTok).toBeGreaterThan(mini.inputPerMTok);
    expect(full.outputPerMTok).toBeGreaterThan(mini.outputPerMTok);
  });

  it("gpt-4.1 is more expensive than gpt-4.1-mini", () => {
    const full = catalog["openai/gpt-4.1"];
    const mini = catalog["openai/gpt-4.1-mini"];

    expect(full.inputPerMTok).toBeGreaterThan(mini.inputPerMTok);
    expect(full.outputPerMTok).toBeGreaterThan(mini.outputPerMTok);
  });

  it("gemini-2.5-pro is more expensive than gemini-2.5-flash", () => {
    const pro = catalog["google/gemini-2.5-pro"];
    const flash = catalog["google/gemini-2.5-flash"];

    expect(pro.inputPerMTok).toBeGreaterThan(flash.inputPerMTok);
    expect(pro.outputPerMTok).toBeGreaterThan(flash.outputPerMTok);
  });

  it("every provider has at least 2 models", () => {
    const byProvider = new Map<string, number>();
    for (const [key] of entries) {
      const provider = key.split("/")[0];
      byProvider.set(provider, (byProvider.get(provider) ?? 0) + 1);
    }
    for (const [provider, count] of byProvider) {
      expect(count, `${provider} should have ≥2 models`).toBeGreaterThanOrEqual(2);
    }
  });
});
