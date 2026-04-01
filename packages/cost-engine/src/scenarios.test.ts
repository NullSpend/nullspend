import { describe, it, expect } from "vitest";
import { getModelPricing, costComponent } from "./pricing.js";

function calculateCost(components: { tokens: number; rate: number }[]): number {
  return Math.round(
    components.reduce((sum, c) => sum + costComponent(c.tokens, c.rate), 0),
  );
}

describe("realistic API call scenarios", () => {
  it("short chatbot reply: GPT-4o 500 input, 150 output", () => {
    const p = getModelPricing("openai", "gpt-4o")!;
    const cost = calculateCost([
      { tokens: 500, rate: p.inputPerMTok },
      { tokens: 150, rate: p.outputPerMTok },
    ]);
    // 500*2.5 + 150*10 = 1250 + 1500 = 2750 microdollars = $0.00275
    expect(cost).toBe(2750);
  });

  it("code generation: GPT-4.1 3000 input, 800 output", () => {
    const p = getModelPricing("openai", "gpt-4.1")!;
    const cost = calculateCost([
      { tokens: 3000, rate: p.inputPerMTok },
      { tokens: 800, rate: p.outputPerMTok },
    ]);
    // 3000*2.0 + 800*8.0 = 6000 + 6400 = 12400
    expect(cost).toBe(12400);
  });

  it("batch processing with mini model: 10 calls × (200in + 50out)", () => {
    const p = getModelPricing("openai", "gpt-4o-mini")!;
    const perCall = calculateCost([
      { tokens: 200, rate: p.inputPerMTok },
      { tokens: 50, rate: p.outputPerMTok },
    ]);
    const total = perCall * 10;
    // per call: 200*0.15 + 50*0.6 = 30 + 30 = 60 → ×10 = 600
    expect(perCall).toBe(60);
    expect(total).toBe(600);
  });

  it("Claude Sonnet with prompt caching: 80K cached + 5K fresh + 2K out", () => {
    const p = getModelPricing("anthropic", "claude-sonnet-4-6")!;
    const cost = calculateCost([
      { tokens: 80_000, rate: p.cachedInputPerMTok },
      { tokens: 5_000, rate: p.inputPerMTok },
      { tokens: 2_000, rate: p.outputPerMTok },
    ]);
    // 80000*0.30 + 5000*3.00 + 2000*15.00 = 24000 + 15000 + 30000 = 69000
    expect(cost).toBe(69_000);
    expect(cost / 1_000_000).toBeCloseTo(0.069, 3);
  });

  it("Claude Opus deep analysis: 128K input, 8K output (expensive call)", () => {
    const p = getModelPricing("anthropic", "claude-opus-4")!;
    const cost = calculateCost([
      { tokens: 128_000, rate: p.inputPerMTok },
      { tokens: 8_000, rate: p.outputPerMTok },
    ]);
    // 128000*15 + 8000*75 = 1,920,000 + 600,000 = 2,520,000
    expect(cost).toBe(2_520_000);
    expect(cost / 1_000_000).toBeCloseTo(2.52, 2);
  });

  it("Anthropic 5-minute cache write scenario", () => {
    const p = getModelPricing("anthropic", "claude-sonnet-4-6")!;
    const cost = calculateCost([
      { tokens: 10_000, rate: p.cacheWrite5mPerMTok! },
      { tokens: 5_000, rate: p.inputPerMTok },
      { tokens: 1_000, rate: p.outputPerMTok },
    ]);
    // 10000*3.75 + 5000*3.00 + 1000*15.00 = 37500 + 15000 + 15000 = 67500
    expect(cost).toBe(67_500);
  });

  it("Anthropic 1-hour extended cache write scenario", () => {
    const p = getModelPricing("anthropic", "claude-sonnet-4-6")!;
    const cost = calculateCost([
      { tokens: 10_000, rate: p.cacheWrite1hPerMTok! },
      { tokens: 5_000, rate: p.inputPerMTok },
      { tokens: 1_000, rate: p.outputPerMTok },
    ]);
    // 10000*6.00 + 5000*3.00 + 1000*15.00 = 60000 + 15000 + 15000 = 90000
    expect(cost).toBe(90_000);
  });

  it("Gemini Flash bulk: 50 calls × (1K input + 500 output)", () => {
    const p = getModelPricing("google", "gemini-2.5-flash")!;
    const perCall = calculateCost([
      { tokens: 1_000, rate: p.inputPerMTok },
      { tokens: 500, rate: p.outputPerMTok },
    ]);
    const total = perCall * 50;
    // per call: 1000*0.15 + 500*0.60 = 150 + 300 = 450 → ×50 = 22500
    expect(perCall).toBe(450);
    expect(total).toBe(22_500);
  });

  it("Gemini Pro with cached context: 50K cached + 2K fresh + 4K out", () => {
    const p = getModelPricing("google", "gemini-2.5-pro")!;
    const cost = calculateCost([
      { tokens: 50_000, rate: p.cachedInputPerMTok },
      { tokens: 2_000, rate: p.inputPerMTok },
      { tokens: 4_000, rate: p.outputPerMTok },
    ]);
    // 50000*0.3125 + 2000*1.25 + 4000*10.00 = 15625 + 2500 + 40000 = 58125
    expect(cost).toBe(58_125);
  });
});

describe("multi-provider budget tracking simulation", () => {
  it("tracks cumulative spend across providers accurately", () => {
    let totalSpend = 0;

    // Call 1: GPT-4o
    const gpt4o = getModelPricing("openai", "gpt-4o")!;
    totalSpend += calculateCost([
      { tokens: 2000, rate: gpt4o.inputPerMTok },
      { tokens: 500, rate: gpt4o.outputPerMTok },
    ]);

    // Call 2: Claude Sonnet
    const sonnet = getModelPricing("anthropic", "claude-sonnet-4-6")!;
    totalSpend += calculateCost([
      { tokens: 3000, rate: sonnet.inputPerMTok },
      { tokens: 1000, rate: sonnet.outputPerMTok },
    ]);

    // Call 3: Gemini Flash
    const flash = getModelPricing("google", "gemini-2.5-flash")!;
    totalSpend += calculateCost([
      { tokens: 5000, rate: flash.inputPerMTok },
      { tokens: 2000, rate: flash.outputPerMTok },
    ]);

    // GPT-4o: 2000*2.5 + 500*10 = 5000+5000 = 10000
    // Sonnet: 3000*3.0 + 1000*15 = 9000+15000 = 24000
    // Flash: 5000*0.15 + 2000*0.6 = 750+1200 = 1950
    expect(totalSpend).toBe(10_000 + 24_000 + 1_950);
    expect(totalSpend).toBe(35_950);
  });

  it("budget enforcement: detects when cumulative spend exceeds limit", () => {
    const budgetMicrodollars = 50_000; // $0.05 budget
    let spent = 0;
    const calls: { provider: string; model: string; input: number; output: number }[] = [
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
      { provider: "openai", model: "gpt-4o", input: 1000, output: 500 },
    ];

    let blocked = false;
    for (const call of calls) {
      const p = getModelPricing(call.provider, call.model)!;
      const callCost = calculateCost([
        { tokens: call.input, rate: p.inputPerMTok },
        { tokens: call.output, rate: p.outputPerMTok },
      ]);

      if (spent + callCost > budgetMicrodollars) {
        blocked = true;
        break;
      }
      spent += callCost;
    }

    // Each call: 1000*2.5 + 500*10 = 2500+5000 = 7500
    // 6 calls = 45000 (under 50000), 7th = 52500 (over)
    expect(blocked).toBe(true);
    expect(spent).toBe(45_000);
  });
});

describe("dollar formatting from microdollars", () => {
  it("$0.00 for zero microdollars", () => {
    expect((0 / 1_000_000).toFixed(6)).toBe("0.000000");
  });

  it("$0.03125 for 31250 microdollars", () => {
    expect((31_250 / 1_000_000).toFixed(5)).toBe("0.03125");
  });

  it("$2.52 for 2520000 microdollars", () => {
    expect((2_520_000 / 1_000_000).toFixed(2)).toBe("2.52");
  });

  it("$100.00 for 100 million microdollars", () => {
    expect((100_000_000 / 1_000_000).toFixed(2)).toBe("100.00");
  });
});
