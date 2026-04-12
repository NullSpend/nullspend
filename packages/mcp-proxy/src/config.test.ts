import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { loadConfig, ConfigError } from "./config.js";

const REQUIRED_ENV = {
  NULLSPEND_URL: "http://127.0.0.1:3000",
  NULLSPEND_API_KEY: "ns_live_sk_test0001",
  UPSTREAM_COMMAND: "node",
};

describe("loadConfig", () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    for (const key of Object.keys(process.env)) {
      if (
        key.startsWith("NULLSPEND_") ||
        key.startsWith("UPSTREAM_") ||
        key === "GATED_TOOLS" ||
        key === "PASSTHROUGH_TOOLS" ||
        key === "APPROVAL_TIMEOUT_SECONDS"
      ) {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("throws ConfigError when required vars are missing", () => {
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("NULLSPEND_URL");
  });

  it("throws ConfigError listing all missing vars", () => {
    process.env.NULLSPEND_URL = "http://test.com";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("NULLSPEND_API_KEY");
    expect(() => loadConfig()).toThrow("UPSTREAM_COMMAND");
  });

  it("returns config with defaults when only required vars are set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();

    expect(config.nullspendUrl).toBe("http://127.0.0.1:3000");
    expect(config.nullspendApiKey).toBe("ns_live_sk_test0001");
    expect(config.upstreamCommand).toBe("node");
    expect(config.upstreamArgs).toEqual([]);
    expect(config.upstreamEnv).toEqual({});
    expect(config.gatedTools).toBe("*");
    expect(config.passthroughTools).toEqual(new Set());
    expect(config.agentId).toBe("mcp-proxy");
    expect(config.approvalTimeoutSeconds).toBe(300);
  });

  it("parses UPSTREAM_ARGS as JSON array", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ARGS = '["server.js", "--port", "8080"]';
    const config = loadConfig();
    expect(config.upstreamArgs).toEqual(["server.js", "--port", "8080"]);
  });

  it("defaults UPSTREAM_ARGS to [] when not set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.upstreamArgs).toEqual([]);
  });

  it("throws ConfigError when UPSTREAM_ARGS is invalid JSON", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ARGS = "not-json";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("UPSTREAM_ARGS is not valid JSON");
  });

  it("throws ConfigError when UPSTREAM_ARGS is not an array", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ARGS = '{"key": "value"}';
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("JSON array");
  });

  it("parses UPSTREAM_ENV as JSON object", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ENV = '{"DB_HOST": "localhost", "DB_PORT": "5432"}';
    const config = loadConfig();
    expect(config.upstreamEnv).toEqual({ DB_HOST: "localhost", DB_PORT: "5432" });
  });

  it("stores raw UPSTREAM_ENV without merging process.env", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ENV = '{"CUSTOM_VAR": "value"}';
    const config = loadConfig();
    expect(config.upstreamEnv).toEqual({ CUSTOM_VAR: "value" });
    expect(config.upstreamEnv).not.toHaveProperty("PATH");
  });

  it("throws ConfigError when UPSTREAM_ENV is not an object", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.UPSTREAM_ENV = '["not", "object"]';
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("JSON object");
  });

  it("parses GATED_TOOLS as specific tool set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.GATED_TOOLS = "run_query, delete_file, send_email";
    const config = loadConfig();
    expect(config.gatedTools).toEqual(new Set(["run_query", "delete_file", "send_email"]));
  });

  it("treats GATED_TOOLS=* as wildcard", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.GATED_TOOLS = "*";
    const config = loadConfig();
    expect(config.gatedTools).toBe("*");
  });

  it("defaults GATED_TOOLS to * when not set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.gatedTools).toBe("*");
  });

  it("parses PASSTHROUGH_TOOLS into a set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.PASSTHROUGH_TOOLS = "read_file, list_directory";
    const config = loadConfig();
    expect(config.passthroughTools).toEqual(new Set(["read_file", "list_directory"]));
  });

  it("parses custom NULLSPEND_AGENT_ID", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_AGENT_ID = "my-proxy";
    const config = loadConfig();
    expect(config.agentId).toBe("my-proxy");
  });

  it("parses custom APPROVAL_TIMEOUT_SECONDS", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.APPROVAL_TIMEOUT_SECONDS = "60";
    const config = loadConfig();
    expect(config.approvalTimeoutSeconds).toBe(60);
  });

  it("throws ConfigError when APPROVAL_TIMEOUT_SECONDS is invalid", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.APPROVAL_TIMEOUT_SECONDS = "abc";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("positive number");
  });

  it("throws ConfigError when APPROVAL_TIMEOUT_SECONDS is zero", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.APPROVAL_TIMEOUT_SECONDS = "0";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("positive number");
  });

  it("throws ConfigError when APPROVAL_TIMEOUT_SECONDS is negative", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.APPROVAL_TIMEOUT_SECONDS = "-10";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("positive number");
  });

  it("treats GATED_TOOLS='' (empty string) as gate nothing", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.GATED_TOOLS = "";
    const config = loadConfig();
    expect(config.gatedTools).toEqual(new Set());
  });

  it("treats GATED_TOOLS with only whitespace as gate nothing", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.GATED_TOOLS = "  ";
    const config = loadConfig();
    expect(config.gatedTools).toEqual(new Set());
  });

  // --- Cost tracking config ---

  it("enables cost tracking by default", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.costTrackingEnabled).toBe(true);
    expect(config.budgetEnforcementEnabled).toBe(true);
  });

  it("disables cost tracking when NULLSPEND_COST_TRACKING=false", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_COST_TRACKING = "false";
    const config = loadConfig();
    expect(config.costTrackingEnabled).toBe(false);
  });

  it("disables budget enforcement when NULLSPEND_BUDGET_ENFORCEMENT=false", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_BUDGET_ENFORCEMENT = "false";
    const config = loadConfig();
    expect(config.budgetEnforcementEnabled).toBe(false);
  });

  it("backendUrl defaults to nullspendUrl", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.costTrackingEnabled).toBe(true);
    expect(config.backendUrl).toBe("http://127.0.0.1:3000");
  });

  it("ignores legacy env vars silently", () => {
    Object.assign(process.env, {
      ...REQUIRED_ENV,
      NULLSPEND_BACKEND_URL: "http://localhost:8787",
      NULLSPEND_PLATFORM_KEY: "pk-test",
      NULLSPEND_USER_ID: "user-1",
      NULLSPEND_KEY_ID: "key-1",
    });
    const config = loadConfig();
    expect(config).not.toHaveProperty("platformKey");
    expect(config).not.toHaveProperty("userId");
    expect(config).not.toHaveProperty("keyId");
    expect(config).not.toHaveProperty("authMode");
    expect(config.backendUrl).toBe("http://127.0.0.1:3000");
  });

  it("does not require cost tracking vars when NULLSPEND_COST_TRACKING=false", () => {
    Object.assign(process.env, {
      NULLSPEND_URL: "http://127.0.0.1:3000",
      NULLSPEND_API_KEY: "ns_live_sk_test0001",
      UPSTREAM_COMMAND: "node",
      NULLSPEND_COST_TRACKING: "false",
    });
    const config = loadConfig();
    expect(config.costTrackingEnabled).toBe(false);
  });

  it("defaults serverName to UPSTREAM_COMMAND", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.serverName).toBe("node");
  });

  it("uses NULLSPEND_SERVER_NAME when set", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_SERVER_NAME = "my-server";
    const config = loadConfig();
    expect(config.serverName).toBe("my-server");
  });

  it("throws ConfigError when NULLSPEND_SERVER_NAME contains '/'", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_SERVER_NAME = "server/name";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("/");
  });

  it("trims whitespace from NULLSPEND_SERVER_NAME", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_SERVER_NAME = "  my-server  ";
    const config = loadConfig();
    expect(config.serverName).toBe("my-server");
  });

  it("throws ConfigError when NULLSPEND_SERVER_NAME is whitespace-only", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_SERVER_NAME = "   ";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("empty");
  });

  it("defaults toolCostOverrides to empty object", () => {
    Object.assign(process.env, REQUIRED_ENV);
    const config = loadConfig();
    expect(config.toolCostOverrides).toEqual({});
  });

  it("parses NULLSPEND_TOOL_COSTS as JSON object", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_TOOL_COSTS = '{"run_query": 50000, "list_files": 0}';
    const config = loadConfig();
    expect(config.toolCostOverrides).toEqual({ run_query: 50000, list_files: 0 });
  });

  it("throws ConfigError when NULLSPEND_TOOL_COSTS is invalid JSON", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_TOOL_COSTS = "not-json";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("NULLSPEND_TOOL_COSTS is not valid JSON");
  });

  it("throws ConfigError when NULLSPEND_TOOL_COSTS is an array", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_TOOL_COSTS = "[1, 2]";
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("JSON object");
  });

  it("throws ConfigError when NULLSPEND_TOOL_COSTS has negative value", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_TOOL_COSTS = '{"tool": -1}';
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("non-negative number");
  });

  it("throws ConfigError when NULLSPEND_TOOL_COSTS has non-numeric value", () => {
    Object.assign(process.env, REQUIRED_ENV);
    process.env.NULLSPEND_TOOL_COSTS = '{"tool": "expensive"}';
    expect(() => loadConfig()).toThrow(ConfigError);
    expect(() => loadConfig()).toThrow("non-negative number");
  });

});
