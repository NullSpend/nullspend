import { describe, it, expect } from "vitest";
import { isAllowedUpstream } from "../lib/upstream-allowlist.js";

describe("isAllowedUpstream", () => {
  it("returns true for each allowlisted URL", () => {
    const allowed = [
      "https://api.openai.com",
      "https://api.groq.com/openai",
      "https://api.together.xyz",
      "https://api.fireworks.ai/inference",
      "https://api.mistral.ai",
      "https://openrouter.ai/api",
    ];
    for (const url of allowed) {
      expect(isAllowedUpstream(url)).toBe(true);
    }
  });

  it("returns true with trailing slash", () => {
    expect(isAllowedUpstream("https://api.groq.com/openai/")).toBe(true);
  });

  it("returns true with multiple trailing slashes", () => {
    expect(isAllowedUpstream("https://api.groq.com/openai///")).toBe(true);
  });

  it("returns false for unknown URL", () => {
    expect(isAllowedUpstream("https://evil.example.com")).toBe(false);
  });

  it("returns false for empty string", () => {
    expect(isAllowedUpstream("")).toBe(false);
  });

  it("returns false for partial match", () => {
    // api.groq.com without /openai path is not in the allowlist
    expect(isAllowedUpstream("https://api.groq.com")).toBe(false);
  });

  it("returns false for internal/private IPs", () => {
    expect(isAllowedUpstream("http://169.254.169.254")).toBe(false);
    expect(isAllowedUpstream("http://localhost")).toBe(false);
  });

  it("returns false for URL with query string", () => {
    expect(isAllowedUpstream("https://api.groq.com/openai?redirect=evil")).toBe(false);
  });

  it("returns false for http:// variant (protocol downgrade)", () => {
    expect(isAllowedUpstream("http://api.groq.com/openai")).toBe(false);
  });

  it("returns true for uppercase domain (case-insensitive, DNS is case-insensitive)", () => {
    expect(isAllowedUpstream("https://API.GROQ.COM/openai")).toBe(true);
    expect(isAllowedUpstream("https://Api.OpenAI.Com")).toBe(true);
  });
});
