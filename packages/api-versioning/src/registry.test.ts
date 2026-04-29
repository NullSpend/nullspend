import { describe, expect, it } from "vitest";
import { RegistryAssertionError, VersionRegistry } from "./index";

describe("VersionRegistry.registerVersion", () => {
  it("registers and assigns ascending indexes", () => {
    const r = new VersionRegistry();
    const v1 = r.registerVersion("2026-04-01");
    const v2 = r.registerVersion("2026-05-01");
    expect(v1.index).toBe(0);
    expect(v2.index).toBe(1);
    expect(r.list().length).toBe(2);
    expect(r.latest().value).toBe("2026-05-01");
  });

  it("rejects malformed CalVer", () => {
    const r = new VersionRegistry();
    expect(() => r.registerVersion("v1")).toThrow(RegistryAssertionError);
  });

  it("rejects semantically-invalid dates that pass the positional regex", () => {
    const r = new VersionRegistry();
    expect(() => r.registerVersion("2026-13-01")).toThrow(RegistryAssertionError);
    expect(() => r.registerVersion("2026-02-30")).toThrow(RegistryAssertionError);
    expect(() => r.registerVersion("2026-04-31")).toThrow(RegistryAssertionError);
  });

  it("rejects duplicates", () => {
    const r = new VersionRegistry();
    r.registerVersion("2026-04-01");
    expect(() => r.registerVersion("2026-04-01")).toThrow(/already registered/);
  });

  it("rejects out-of-order", () => {
    const r = new VersionRegistry();
    r.registerVersion("2026-05-01");
    expect(() => r.registerVersion("2026-04-01")).toThrow(/out-of-order/);
  });

  it("latest() throws on empty registry", () => {
    const r = new VersionRegistry();
    expect(() => r.latest()).toThrow(/empty registry/);
  });
});

describe("VersionRegistry.registerChange", () => {
  it("registers an adjacent change", () => {
    const r = new VersionRegistry();
    r.registerVersion("2026-04-01");
    r.registerVersion("2026-05-01");
    r.registerChange({
      resource: "cost-events",
      oldVersion: "2026-04-01",
      newVersion: "2026-05-01",
    });
    expect(r.getChanges("cost-events")).toHaveLength(1);
  });

  it("rejects gap (non-adjacent versions)", () => {
    const r = new VersionRegistry();
    r.registerVersion("2026-04-01");
    r.registerVersion("2026-05-01");
    r.registerVersion("2026-06-01");
    expect(() =>
      r.registerChange({
        resource: "cost-events",
        oldVersion: "2026-04-01",
        newVersion: "2026-06-01",
      }),
    ).toThrow(/gap between/);
  });

  it("rejects unregistered version", () => {
    const r = new VersionRegistry();
    r.registerVersion("2026-04-01");
    expect(() =>
      r.registerChange({
        resource: "cost-events",
        oldVersion: "2026-04-01",
        newVersion: "2026-05-01",
      }),
    ).toThrow(/not registered/);
  });

});
