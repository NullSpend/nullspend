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

  it("returns config when all required vars are set", () => {
    process.env.NULLSPEND_URL = "http://localhost:3000";
    process.env.NULLSPEND_API_KEY = "ns_live_sk_test0001";

    const config = loadConfig();
    expect(config.nullspendUrl).toBe("http://localhost:3000");
    expect(config.nullspendApiKey).toBe("ns_live_sk_test0001");
    expect(config.agentId).toBe("mcp-agent");
  });

  it("uses custom NULLSPEND_AGENT_ID when set", () => {
    process.env.NULLSPEND_URL = "http://localhost:3000";
    process.env.NULLSPEND_API_KEY = "ns_live_sk_test0001";
    process.env.NULLSPEND_AGENT_ID = "my-custom-agent";

    const config = loadConfig();
    expect(config.agentId).toBe("my-custom-agent");
  });

  it("throws ConfigError when NULLSPEND_URL is missing", () => {
    process.env.NULLSPEND_API_KEY = "ns_live_sk_test0001";
    delete process.env.NULLSPEND_URL;

    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("NULLSPEND_URL");
  });

  it("throws ConfigError when NULLSPEND_API_KEY is missing", () => {
    process.env.NULLSPEND_URL = "http://localhost:3000";
    delete process.env.NULLSPEND_API_KEY;

    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("NULLSPEND_API_KEY");
  });

  it("throws ConfigError listing all missing vars", () => {
    delete process.env.NULLSPEND_URL;
    delete process.env.NULLSPEND_API_KEY;

    try {
      loadConfig();
      expect.fail("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ConfigError);
      expect((err as ConfigError).message).toContain("NULLSPEND_URL");
      expect((err as ConfigError).message).toContain("NULLSPEND_API_KEY");
    }
  });
});
