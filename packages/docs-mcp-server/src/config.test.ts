import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { loadConfig, ConfigError } from "./config.js";

describe("loadConfig", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("returns defaults with no env vars set", () => {
    delete process.env.NULLSPEND_DOCS_SERVER_NAME;

    const config = loadConfig();
    expect(config.serverName).toBe("nullspend-docs");
    expect(config.version).toBe("0.1.0");
  });

  it("does not throw — zero required env vars", () => {
    expect(() => loadConfig()).not.toThrow();
  });

  it("respects NULLSPEND_DOCS_SERVER_NAME override", () => {
    process.env.NULLSPEND_DOCS_SERVER_NAME = "my-custom-docs";

    const config = loadConfig();
    expect(config.serverName).toBe("my-custom-docs");
  });
});

describe("ConfigError", () => {
  it("has correct name property", () => {
    const err = new ConfigError("test");
    expect(err.name).toBe("ConfigError");
    expect(err.message).toBe("test");
    expect(err).toBeInstanceOf(Error);
  });
});
