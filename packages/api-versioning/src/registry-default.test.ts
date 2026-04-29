import { describe, expect, it } from "vitest";
import {
  CURRENT_VERSION,
  LATEST,
  LATEST_VALUE,
  NULLSPEND_REGISTRY,
  SUPPORTED_VERSIONS,
} from "./index";

describe("NULLSPEND_REGISTRY (Phase 0 baseline)", () => {
  it("has the baseline version registered", () => {
    expect(NULLSPEND_REGISTRY.has(LATEST_VALUE)).toBe(true);
  });

  it("LATEST is the latest registered version", () => {
    expect(LATEST.value).toBe(LATEST_VALUE);
    expect(NULLSPEND_REGISTRY.latest().value).toBe(LATEST_VALUE);
  });

  it("legacy CURRENT_VERSION + SUPPORTED_VERSIONS aliases", () => {
    expect(CURRENT_VERSION).toBe(LATEST_VALUE);
    expect(SUPPORTED_VERSIONS).toContain(LATEST_VALUE);
  });

  it("Phase 0 has zero registered transforms", () => {
    for (const resource of ["cost-events", "budgets", "keys", "webhooks"]) {
      expect(NULLSPEND_REGISTRY.getChanges(resource)).toHaveLength(0);
    }
  });
});
