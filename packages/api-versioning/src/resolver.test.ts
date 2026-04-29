import { describe, expect, it } from "vitest";
import {
  InvalidVersionError,
  parseVersionHeader,
  resolveApiVersion,
  UnknownVersionError,
  VersionRegistry,
} from "./index";

function buildRegistry(): VersionRegistry {
  const r = new VersionRegistry();
  r.registerVersion("2026-04-01");
  r.registerVersion("2026-05-01");
  return r;
}

describe("parseVersionHeader", () => {
  it("returns null for null / empty / whitespace-only", () => {
    const r = buildRegistry();
    expect(parseVersionHeader(null, r)).toBeNull();
    expect(parseVersionHeader("", r)).toBeNull();
    expect(parseVersionHeader("   ", r)).toBeNull();
  });

  it("returns the registered ApiVersion for a known value", () => {
    const r = buildRegistry();
    const v = parseVersionHeader("2026-04-01", r);
    expect(v?.value).toBe("2026-04-01");
  });

  it("trims whitespace before lookup", () => {
    const r = buildRegistry();
    expect(parseVersionHeader("  2026-04-01  ", r)?.value).toBe("2026-04-01");
  });

  it("throws InvalidVersionError on malformed CalVer", () => {
    const r = buildRegistry();
    expect(() => parseVersionHeader("v1", r)).toThrow(InvalidVersionError);
    expect(() => parseVersionHeader("latest", r)).toThrow(InvalidVersionError);
    expect(() => parseVersionHeader("2026/04/01", r)).toThrow(InvalidVersionError);
  });

  it("throws UnknownVersionError on well-formed but unregistered", () => {
    const r = buildRegistry();
    expect(() => parseVersionHeader("2099-01-01", r)).toThrow(UnknownVersionError);
  });

  it("hostile-input fuzz: rejects null bytes / unicode / oversized", () => {
    const r = buildRegistry();
    expect(() => parseVersionHeader("2026-04-01\u0000", r)).toThrow(InvalidVersionError);
    expect(() => parseVersionHeader("\u202E2026-04-01", r)).toThrow(InvalidVersionError);
    expect(() => parseVersionHeader("<script>alert(1)</script>", r)).toThrow(InvalidVersionError);
    expect(() => parseVersionHeader("a".repeat(1024 * 1024), r)).toThrow(InvalidVersionError);
  });

  it("InvalidVersionError carries supported list", () => {
    const r = buildRegistry();
    try {
      parseVersionHeader("v1", r);
      expect.fail("expected throw");
    } catch (err) {
      expect(err).toBeInstanceOf(InvalidVersionError);
      expect((err as InvalidVersionError).supported).toEqual(["2026-04-01", "2026-05-01"]);
      expect((err as InvalidVersionError).code).toBe("invalid_version");
    }
  });
});

describe("resolveApiVersion fallback chain", () => {
  it("header valid → returns header version", () => {
    const r = buildRegistry();
    const v = resolveApiVersion({ headerValue: "2026-04-01", registry: r });
    expect(v.value).toBe("2026-04-01");
  });

  it("header malformed → throws InvalidVersionError", () => {
    const r = buildRegistry();
    expect(() => resolveApiVersion({ headerValue: "v1", registry: r })).toThrow(InvalidVersionError);
  });

  it("header unknown → throws UnknownVersionError", () => {
    const r = buildRegistry();
    expect(() => resolveApiVersion({ headerValue: "2099-01-01", registry: r })).toThrow(UnknownVersionError);
  });

  it("header missing + keyVersion present → returns keyVersion", () => {
    const r = buildRegistry();
    const v = resolveApiVersion({
      headerValue: null,
      keyVersion: "2026-04-01",
      registry: r,
    });
    expect(v.value).toBe("2026-04-01");
  });

  it("header missing + keyVersion stale + orgDefault present → returns orgDefault", () => {
    const r = buildRegistry();
    const v = resolveApiVersion({
      headerValue: null,
      keyVersion: "2099-01-01",
      orgDefault: "2026-05-01",
      registry: r,
    });
    expect(v.value).toBe("2026-05-01");
  });

  it("all defaults missing → returns LATEST from registry", () => {
    const r = buildRegistry();
    const v = resolveApiVersion({ headerValue: null, registry: r });
    expect(v.value).toBe("2026-05-01");
  });

  it("stale keyVersion silently falls through to LATEST", () => {
    const r = buildRegistry();
    const v = resolveApiVersion({
      headerValue: null,
      keyVersion: "2099-01-01",
      registry: r,
    });
    expect(v.value).toBe("2026-05-01");
  });
});
