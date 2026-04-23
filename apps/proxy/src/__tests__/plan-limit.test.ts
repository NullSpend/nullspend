import { describe, it, expect } from "vitest";
import { checkPlanLimit } from "../lib/plan-limit.js";

// Per PR-2a tests C4-C7. Pure helper — deterministic input/output, no side effects.

describe("checkPlanLimit", () => {
  // C4 — null blockAt always approves (self-hosted / unlimited / Enterprise)
  it("approves when blockAt is null (self-hosted/unlimited)", () => {
    expect(checkPlanLimit(null, "hard", 9_999_999)).toEqual({ status: "approved" });
    expect(checkPlanLimit(null, "soft", 0)).toEqual({ status: "approved" });
  });

  // C5 — soft mode never denies regardless of count
  it("approves in soft mode regardless of count", () => {
    expect(checkPlanLimit(500_000, "soft", 500_001)).toEqual({ status: "approved" });
    expect(checkPlanLimit(500_000, "soft", 1_000_000_000)).toEqual({ status: "approved" });
  });

  // C6 — hard mode denies past threshold (blockAt + 1)
  it("denies at blockAt + 1 in hard mode", () => {
    expect(checkPlanLimit(100_000, "hard", 100_001)).toEqual({
      status: "denied",
      blockAt: 100_000,
    });
  });

  // C7 — hard mode approves AT the threshold (boundary case)
  it("approves at exactly blockAt in hard mode", () => {
    expect(checkPlanLimit(100_000, "hard", 100_000)).toEqual({ status: "approved" });
  });

  // Additional boundary: count below threshold always approves
  it("approves below threshold in hard mode", () => {
    expect(checkPlanLimit(100_000, "hard", 99_999)).toEqual({ status: "approved" });
    expect(checkPlanLimit(100_000, "hard", 0)).toEqual({ status: "approved" });
  });

  // Edge: count deeply past threshold still returns single denial shape
  it("returns stable denial shape when count far exceeds blockAt", () => {
    expect(checkPlanLimit(100_000, "hard", 1_000_000)).toEqual({
      status: "denied",
      blockAt: 100_000,
    });
  });
});
