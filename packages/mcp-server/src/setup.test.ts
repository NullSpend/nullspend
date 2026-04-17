import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runSetup } from "./setup.js";

// ── Test helpers ─────────────────────────────────────────────────

function makeStreams() {
  let stdout = "";
  let stderr = "";
  return {
    streams: {
      stdout: { write: (s: string) => { stdout += s; return true; } } as unknown as NodeJS.WriteStream,
      stderr: { write: (s: string) => { stderr += s; return true; } } as unknown as NodeJS.WriteStream,
    },
    get stdout() { return stdout; },
    get stderr() { return stderr; },
  };
}

describe("runSetup — print mode", () => {
  it("prints a snippet with the placeholder API key when no --api-key is given", () => {
    const t = makeStreams();
    const code = runSetup([], t.streams);
    expect(code).toBe(0);
    expect(t.stdout).toContain("mcpServers");
    expect(t.stdout).toContain("nullspend");
    expect(t.stdout).toContain("ns_live_sk_REPLACE_ME");
  });

  it("substitutes --api-key into NULLSPEND_API_KEY", () => {
    const t = makeStreams();
    runSetup(["--api-key", "ns_live_sk_real"], t.streams);
    expect(t.stdout).toContain("ns_live_sk_real");
    expect(t.stdout).not.toContain("REPLACE_ME");
  });

  it("uses default URL when --url is not given", () => {
    const t = makeStreams();
    runSetup([], t.streams);
    expect(t.stdout).toContain("https://nullspend.dev");
  });

  it("substitutes --url override", () => {
    const t = makeStreams();
    runSetup(["--url", "http://localhost:3000"], t.streams);
    expect(t.stdout).toContain("http://localhost:3000");
  });

  it("includes NULLSPEND_AGENT_ID only when --agent-id differs from default", () => {
    const t1 = makeStreams();
    runSetup([], t1.streams);
    expect(t1.stdout).not.toContain("NULLSPEND_AGENT_ID");

    const t2 = makeStreams();
    runSetup(["--agent-id", "custom-agent-1"], t2.streams);
    expect(t2.stdout).toContain("NULLSPEND_AGENT_ID");
    expect(t2.stdout).toContain("custom-agent-1");
  });

  it("--help prints usage and returns 0", () => {
    const t = makeStreams();
    const code = runSetup(["--help"], t.streams);
    expect(code).toBe(0);
    expect(t.stdout).toContain("Usage:");
    expect(t.stdout).toContain("--api-key");
  });
});

describe("runSetup — write mode", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "nullspend-mcp-setup-"));
    // Override the platform-specific path resolution by spying on the
    // env-var fallbacks the helper uses on Windows / Linux / macOS.
    vi.stubEnv("APPDATA", tmpDir);
    vi.stubEnv("HOME", tmpDir);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("refuses to overwrite an existing nullspend entry", () => {
    const existing = {
      mcpServers: { nullspend: { command: "different", args: [] } },
    };
    const json = JSON.stringify(existing);
    // Pre-seed all three platform candidates so the test works regardless of
    // which OS the test runs on.
    mkdirSync(join(tmpDir, "Claude"), { recursive: true });
    mkdirSync(join(tmpDir, "Library", "Application Support", "Claude"), { recursive: true });
    mkdirSync(join(tmpDir, ".config", "Claude"), { recursive: true });
    writeFileSync(join(tmpDir, "Claude", "claude_desktop_config.json"), json);
    writeFileSync(
      join(tmpDir, "Library", "Application Support", "Claude", "claude_desktop_config.json"),
      json,
    );
    writeFileSync(join(tmpDir, ".config", "Claude", "claude_desktop_config.json"), json);

    const t = makeStreams();
    const code = runSetup(["--write", "--api-key", "ns_live_sk_x"], t.streams);
    expect(code).toBe(1);
    expect(t.stderr).toMatch(/refusing to overwrite/);
  });
});
