import { describe, it, expect } from "vitest";
import {
  resolveApiVersion,
  SUPPORTED_VERSIONS,
  CURRENT_VERSION,
} from "../lib/api-version.js";

describe("resolveApiVersion", () => {
  it("returns header when header is a supported version", () => {
    expect(resolveApiVersion("2026-04-01", "2026-04-01")).toBe("2026-04-01");
  });

  it("returns key version when header is null", () => {
    expect(resolveApiVersion(null, "2026-04-01")).toBe("2026-04-01");
  });

  it("returns key version when header is unsupported", () => {
    expect(resolveApiVersion("2099-01-01", "2026-04-01")).toBe("2026-04-01");
  });

  it("returns CURRENT_VERSION when both header and key are unsupported", () => {
    expect(resolveApiVersion("2099-01-01", "2099-01-01")).toBe(CURRENT_VERSION);
  });

  it("returns CURRENT_VERSION when header is empty string and key is unsupported", () => {
    expect(resolveApiVersion("", "bad")).toBe(CURRENT_VERSION);
  });

  it("returns CURRENT_VERSION when header is null and key is empty string", () => {
    expect(resolveApiVersion(null, "")).toBe(CURRENT_VERSION);
  });

  it("header takes priority over key version", () => {
    expect(resolveApiVersion("2026-04-01", "2026-04-01")).toBe("2026-04-01");
  });

  it("falls through all three tiers correctly", () => {
    // Tier 1: valid header
    expect(resolveApiVersion("2026-04-01", "bad")).toBe("2026-04-01");
    // Tier 2: invalid header, valid key
    expect(resolveApiVersion("bad", "2026-04-01")).toBe("2026-04-01");
    // Tier 3: both invalid
    expect(resolveApiVersion("bad", "bad")).toBe(CURRENT_VERSION);
  });
});

describe("SUPPORTED_VERSIONS", () => {
  it("contains the current version", () => {
    expect((SUPPORTED_VERSIONS as readonly string[]).includes(CURRENT_VERSION)).toBe(true);
  });

  it("is a non-empty array", () => {
    expect(SUPPORTED_VERSIONS.length).toBeGreaterThan(0);
  });
});
