/**
 * Loop Detection Stress Tests — TypeScript SDK
 *
 * These exercise real async behavior, timing precision, memory bounds,
 * and content hash collision rates.
 *
 * Run: cd packages/sdk && npx vitest run src/loop-detector-stress.test.ts
 */
import { describe, it, expect, vi } from "vitest";
import { LoopDetector } from "./loop-detector.js";
import { LoopDetectedError } from "./errors.js";
import { buildTrackedFetch } from "./tracked-fetch.js";
import type { DenialReason } from "./types.js";

vi.mock("@nullspend/cost-engine", () => ({
  getModelPricing: vi.fn(() => ({
    inputPerMTok: 2.5,
    outputPerMTok: 10,
    cachedInputPerMTok: 1.25,
  })),
  costComponent: vi.fn((tokens: number, rate: number) => {
    if (tokens <= 0 || rate <= 0) return 0;
    return tokens * rate;
  }),
}));

const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const PROXY_URL = "https://proxy.nullspend.dev";

function makeBody(model = "gpt-4o", content = "Hi"): string {
  return JSON.stringify({ model, messages: [{ role: "user", content }] });
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ── Per-key detection precision ──────────────────────────────────

describe("Stress — Per-Key Detection Precision", () => {
  it("denies at exactly threshold 50", () => {
    const d = new LoopDetector({ maxCalls: 50 });
    for (let i = 0; i < 49; i++) {
      expect(d.check("openai:gpt-4o", "same").isLoop).toBe(false);
    }
    expect(d.check("openai:gpt-4o", "same").isLoop).toBe(true);
    expect(d.check("openai:gpt-4o", "same").callCount).toBe(51);
  });

  it("100 unique hashes never trigger at threshold 50", () => {
    const d = new LoopDetector({ maxCalls: 50 });
    for (let i = 0; i < 100; i++) {
      expect(d.check("openai:gpt-4o", `hash_${i}`).isLoop).toBe(false);
    }
  });

  it("10 models independently tracked (unique content per model)", () => {
    const d = new LoopDetector({ maxCalls: 5, aggregateMaxKeys: 20 });
    // Use unique content per model to avoid aggregate trigger
    for (let m = 0; m < 10; m++) {
      for (let i = 0; i < 4; i++) {
        expect(d.check(`openai:model-${m}`, `hash-${m}`).isLoop).toBe(false);
      }
    }
    // Each model has 4 calls. 5th call on model-0 triggers per-key.
    expect(d.check("openai:model-0", "hash-0").isLoop).toBe(true);
    // model-1 still at 4
    expect(d.check("openai:model-1", "hash-1").isLoop).toBe(true); // now 5
  });

  it("maxCalls=0 blocks every call", () => {
    const d = new LoopDetector({ maxCalls: 0 });
    for (let i = 0; i < 10; i++) {
      expect(d.check("openai:gpt-4o", `hash_${i}`).isLoop).toBe(true);
    }
  });
});

// ── Aggregate detection precision ────────────────────────────────

describe("Stress — Aggregate Detection Precision", () => {
  it("triggers at exactly 5 qualifying keys", () => {
    const d = new LoopDetector({ maxCalls: 100, aggregateMaxKeys: 5 });
    // Build 4 qualifying keys
    for (let k = 0; k < 4; k++) {
      for (let i = 0; i < 3; i++) {
        d.check(`openai:model-${k}`, "same");
      }
    }
    // 5th key at 2 repeats — not yet qualifying
    d.check("openai:model-4", "same");
    d.check("openai:model-4", "same");
    expect(d.check("openai:model-0", "same").isLoop).toBe(false); // only 4 qualifying

    // 5th key gets 3rd repeat — now qualifies
    d.check("openai:model-4", "same");
    // model-4 now has 3 repeats, 5 keys qualify
    // But wait — the check was on model-4, count is for model-4's hash count (3)
    // Aggregate should trigger since 5 keys now qualify
    // The next check on any key should show aggregate
    const r2 = d.check("openai:model-0", "same");
    expect(r2.isLoop).toBe(true);
    expect(r2.detectionType).toBe("aggregate");
  });

  it("mixed content — only keys with 3+ same-content qualify", () => {
    const d = new LoopDetector({ maxCalls: 100, aggregateMaxKeys: 3 });
    // 5 keys, but only 2 with repeating content
    for (const model of ["m1", "m2"]) {
      for (let i = 0; i < 5; i++) {
        d.check(`openai:${model}`, "same");
      }
    }
    for (const model of ["m3", "m4", "m5"]) {
      for (let i = 0; i < 5; i++) {
        d.check(`openai:${model}`, `unique_${model}_${i}`);
      }
    }
    expect(d.check("openai:m1", "same").isLoop).toBe(false);
  });
});

// ── Window expiry ────────────────────────────────────────────────

describe("Stress — Window Expiry", () => {
  it("counter resets after window expires", async () => {
    const d = new LoopDetector({ maxCalls: 5, windowSeconds: 0.15 });
    for (let i = 0; i < 4; i++) {
      d.check("openai:gpt-4o", "same");
    }
    await new Promise((r) => setTimeout(r, 200));
    expect(d.check("openai:gpt-4o", "same").callCount).toBe(1);
  });

  it("stale keys pruned during aggregate scan", async () => {
    const d = new LoopDetector({ maxCalls: 100, windowSeconds: 0.1, aggregateMaxKeys: 3 });
    for (const model of ["m1", "m2", "m3"]) {
      for (let i = 0; i < 3; i++) d.check(`openai:${model}`, "same");
    }
    await new Promise((r) => setTimeout(r, 150));
    // m1 check: only m1 is fresh, m2 and m3 expired
    expect(d.check("openai:m1", "same").isLoop).toBe(false);
  });
});

// ── Memory bounds ────────────────────────────────────────────────

describe("Stress — Memory Bounds", () => {
  it("10K distinct keys pruned after window expires", async () => {
    const d = new LoopDetector({ maxCalls: 100, windowSeconds: 0.1 });
    for (let i = 0; i < 10_000; i++) {
      d.check(`provider:model-${i}`, `hash-${i}`);
    }
    expect(d["_callLog"].size).toBeLessThanOrEqual(10_000);

    await new Promise((r) => setTimeout(r, 150));
    // Trigger aggregate scan to prune
    d.check("provider:model-0", "new-hash");
    expect(d["_callLog"].size).toBeLessThan(100);
  });
});

// ── Content hash ─────────────────────────────────────────────────

describe("Stress — Content Hash", () => {
  it("FNV-1a collision rate < 0.5% on 100K unique bodies", () => {
    const hashes = new Set<string>();
    let collisions = 0;
    for (let i = 0; i < 100_000; i++) {
      const body = JSON.stringify({ model: "gpt-4o", messages: [{ content: `unique-${i}` }] });
      const h = LoopDetector.contentHashSync(body);
      if (hashes.has(h)) collisions++;
      hashes.add(h);
    }
    const rate = collisions / 100_000;
    expect(rate).toBeLessThan(0.005);
  });

  it("SHA-256 async matches expected format", async () => {
    const h = await LoopDetector.contentHash("hello world");
    expect(h).toHaveLength(8);
    expect(h).toMatch(/^[0-9a-f]{8}$/);
  });

  it("8KB cap: bodies differing only after 8KB produce same hash", () => {
    const prefix = "A".repeat(8192);
    expect(LoopDetector.contentHashSync(prefix + "XXX")).toBe(
      LoopDetector.contentHashSync(prefix + "YYY"),
    );
  });
});

// ── Warning precision ────────────────────────────────────────────

describe("Stress — Warning Precision", () => {
  it("warning fires at exactly 80% (call 40 of 50)", () => {
    const d = new LoopDetector({ maxCalls: 50 });
    let warningFired = false;
    for (let i = 0; i < 50; i++) {
      const r = d.check("openai:gpt-4o", "same");
      if (r.isWarning) {
        expect(i).toBe(39); // 0-indexed, 40th call
        warningFired = true;
      }
    }
    expect(warningFired).toBe(true);
  });

  it("warning fires once per composite key, resets after window", async () => {
    const d = new LoopDetector({ maxCalls: 10, windowSeconds: 0.1 });
    let warnings = 0;
    for (let i = 0; i < 9; i++) {
      if (d.check("openai:gpt-4o", "same").isWarning) warnings++;
    }
    expect(warnings).toBe(1); // fires once at 80%

    await new Promise((r) => setTimeout(r, 150));

    // Re-populate — warning should fire again
    let warned2 = false;
    for (let i = 0; i < 9; i++) {
      if (d.check("openai:gpt-4o", "same").isWarning) warned2 = true;
    }
    expect(warned2).toBe(true);
  });
});

// ── buildTrackedFetch integration stress ─────────────────────────

describe("Stress — buildTrackedFetch Integration", () => {
  it("blocks at threshold in direct mode — zero extra fetches", async () => {
    let fetchCount = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => {
      fetchCount++;
      return jsonResponse({ choices: [] });
    }) as typeof fetch;

    try {
      const body = makeBody();
      const tf = buildTrackedFetch("openai", {
        loopDetection: { maxCalls: 5, windowSeconds: 60 },
      }, vi.fn(), null);

      for (let i = 0; i < 4; i++) {
        await tf(OPENAI_URL, { method: "POST", body });
      }
      expect(fetchCount).toBe(4);

      await expect(tf(OPENAI_URL, { method: "POST", body })).rejects.toThrow(LoopDetectedError);
      expect(fetchCount).toBe(4); // No extra fetch
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("blocks at threshold in proxy mode — zero extra fetches", async () => {
    let fetchCount = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => {
      fetchCount++;
      return jsonResponse({ choices: [] });
    }) as typeof fetch;

    try {
      const body = makeBody();
      const tf = buildTrackedFetch("openai", {
        loopDetection: { maxCalls: 3, windowSeconds: 60 },
      }, vi.fn(), null, PROXY_URL);

      await tf(`${PROXY_URL}/v1/chat/completions`, { method: "POST", body });
      await tf(`${PROXY_URL}/v1/chat/completions`, { method: "POST", body });
      expect(fetchCount).toBe(2);

      await expect(
        tf(`${PROXY_URL}/v1/chat/completions`, { method: "POST", body }),
      ).rejects.toThrow(LoopDetectedError);
      expect(fetchCount).toBe(2); // Blocked before proxy call
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("concurrent Promise.all — loop fires on Nth call", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => {
      await new Promise((r) => setTimeout(r, 5));
      return jsonResponse({ choices: [] });
    }) as typeof fetch;

    try {
      const body = makeBody();
      const tf = buildTrackedFetch("openai", {
        loopDetection: { maxCalls: 5, windowSeconds: 60 },
      }, vi.fn(), null);

      const results = await Promise.allSettled(
        Array.from({ length: 10 }, () => tf(OPENAI_URL, { method: "POST", body })),
      );

      const fulfilled = results.filter((r) => r.status === "fulfilled").length;
      const rejected = results.filter((r) => r.status === "rejected").length;
      expect(fulfilled).toBe(4);
      expect(rejected).toBe(6); // calls 5-10 are loops
      for (const r of results) {
        if (r.status === "rejected") {
          expect(r.reason).toBeInstanceOf(LoopDetectedError);
        }
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("different bodies never trigger — 50 unique calls", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => jsonResponse({ choices: [] })) as typeof fetch;

    try {
      const tf = buildTrackedFetch("openai", {
        loopDetection: { maxCalls: 5, windowSeconds: 60 },
      }, vi.fn(), null);

      for (let i = 0; i < 50; i++) {
        await tf(OPENAI_URL, { method: "POST", body: makeBody("gpt-4o", `msg ${i}`) });
      }
      // All 50 passed — no false positive
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("onDenied callback receives correct fields", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () => jsonResponse({ choices: [] })) as typeof fetch;

    try {
      const body = makeBody();
      const denials: DenialReason[] = [];
      const tf = buildTrackedFetch("openai", {
        loopDetection: { maxCalls: 2, windowSeconds: 60 },
        onDenied: (r) => denials.push(r),
      }, vi.fn(), null);

      await tf(OPENAI_URL, { method: "POST", body });
      await expect(tf(OPENAI_URL, { method: "POST", body })).rejects.toThrow();

      expect(denials).toHaveLength(1);
      expect(denials[0].type).toBe("loop");
      if (denials[0].type === "loop") {
        expect(denials[0].model).toBe("gpt-4o");
        expect(denials[0].callCount).toBe(2);
        expect(denials[0].maxCalls).toBe(2);
        expect(denials[0].windowSeconds).toBe(60);
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

// ── Reset stress ─────────────────────────────────────────────────

describe("Stress — Reset", () => {
  it("reset clears all state — counter restarts from 0", () => {
    const d = new LoopDetector({ maxCalls: 5 });
    for (let i = 0; i < 4; i++) d.check("openai:gpt-4o", "same");
    d.reset();
    expect(d.check("openai:gpt-4o", "same").callCount).toBe(1);
  });

  it("reset specific key preserves other keys", () => {
    const d = new LoopDetector({ maxCalls: 5 });
    for (let i = 0; i < 4; i++) {
      d.check("openai:gpt-4o", "same");
      d.check("openai:gpt-4o-mini", "same");
    }
    d.reset("openai:gpt-4o");
    expect(d.check("openai:gpt-4o", "same").callCount).toBe(1);
    expect(d.check("openai:gpt-4o-mini", "same").callCount).toBe(5); // still accumulated
  });
});
