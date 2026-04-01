import { describe, it, expect, vi, beforeEach } from "vitest";
import { parseTags, mergeTags } from "../lib/tags.js";

describe("parseTags", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("returns empty object for null header", () => {
    expect(parseTags(null)).toEqual({});
  });

  it("returns empty object for empty string", () => {
    expect(parseTags("")).toEqual({});
  });

  it("parses valid JSON object", () => {
    const result = parseTags('{"project":"alpha","env":"prod"}');
    expect(result).toEqual({ project: "alpha", env: "prod" });
  });

  it("returns empty object and warns for invalid JSON", () => {
    expect(parseTags("{not json}")).toEqual({});
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("Malformed"));
  });

  it("returns empty object and warns for JSON array", () => {
    expect(parseTags('["a","b"]')).toEqual({});
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("not a JSON object"));
  });

  it("returns empty object for JSON primitive", () => {
    expect(parseTags('"hello"')).toEqual({});
    expect(parseTags("42")).toEqual({});
    expect(parseTags("true")).toEqual({});
    expect(parseTags("null")).toEqual({});
  });

  it("truncates to first 10 keys when exceeding limit", () => {
    const tags: Record<string, string> = {};
    for (let i = 0; i < 15; i++) {
      tags[`key${i}`] = `value${i}`;
    }
    const result = parseTags(JSON.stringify(tags));
    expect(Object.keys(result).length).toBe(10);
  });

  it("warns when entries are dropped", () => {
    const result = parseTags('{"valid":"ok","invalid key":"no"}');
    expect(result).toEqual({ valid: "ok" });
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("Dropped 1"));
  });

  it("drops keys with invalid format", () => {
    const result = parseTags('{"valid_key":"ok","invalid key":"no","also.invalid":"no","good-key":"ok"}');
    expect(result).toEqual({ valid_key: "ok", "good-key": "ok" });
  });

  it("drops keys exceeding 64 characters", () => {
    const longKey = "a".repeat(65);
    const okKey = "b".repeat(64);
    const result = parseTags(JSON.stringify({ [longKey]: "v1", [okKey]: "v2" }));
    expect(result).toEqual({ [okKey]: "v2" });
  });

  it("drops empty keys", () => {
    const result = parseTags('{"":"value","ok":"value2"}');
    expect(result).toEqual({ ok: "value2" });
  });

  it("drops values exceeding 256 characters", () => {
    const longValue = "x".repeat(257);
    const okValue = "y".repeat(256);
    const result = parseTags(JSON.stringify({ a: longValue, b: okValue }));
    expect(result).toEqual({ b: okValue });
  });

  it("drops non-string values", () => {
    const result = parseTags('{"str":"ok","num":42,"bool":true,"obj":{},"arr":[]}');
    expect(result).toEqual({ str: "ok" });
  });

  it("drops values containing null bytes", () => {
    const result = parseTags(JSON.stringify({ ok: "fine", bad: "val\u0000ue" }));
    expect(result).toEqual({ ok: "fine" });
  });

  it("allows empty string values", () => {
    const result = parseTags('{"key":""}');
    expect(result).toEqual({ key: "" });
  });

  it("allows alphanumeric, underscore, and hyphen in keys", () => {
    const result = parseTags('{"abc-123_DEF":"val"}');
    expect(result).toEqual({ "abc-123_DEF": "val" });
  });

  it("preserves __proto__ as a tag key via null-prototype object", () => {
    const result = parseTags('{"__proto__":"team-alpha","other":"val"}');
    expect(Object.keys(result)).toContain("__proto__");
    expect(result["__proto__"]).toBe("team-alpha");
    expect(result.other).toBe("val");
  });

  it("drops keys with reserved _ns_ prefix", () => {
    const result = parseTags('{"_ns_estimated":"true","_ns_cancelled":"true","env":"prod"}');
    expect(result).toEqual({ env: "prod" });
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining("Dropped 2"));
  });

  it("allows keys starting with _n or _ns but not _ns_", () => {
    const result = parseTags('{"_n":"ok","_ns":"ok","_ns_bad":"no","ns_ok":"ok"}');
    expect(result).toEqual({ _n: "ok", _ns: "ok", ns_ok: "ok" });
  });
});

describe("mergeTags", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("returns request tags only when defaults are empty", () => {
    const result = mergeTags({}, '{"env":"prod"}');
    expect(result).toEqual({ env: "prod" });
  });

  it("returns defaults only when request header is null", () => {
    const result = mergeTags({ project: "alpha" }, null);
    expect(result).toEqual({ project: "alpha" });
  });

  it("returns defaults only when request header is empty string", () => {
    const result = mergeTags({ project: "alpha" }, "");
    expect(result).toEqual({ project: "alpha" });
  });

  it("returns empty object when both are empty", () => {
    const result = mergeTags({}, null);
    expect(result).toEqual({});
  });

  it("request tags override defaults for the same key", () => {
    const result = mergeTags(
      { project: "openclaw", team: "backend" },
      '{"project":"other"}',
    );
    expect(result).toEqual({ project: "other", team: "backend" });
  });

  it("non-conflicting keys are unioned", () => {
    const result = mergeTags(
      { team: "backend" },
      '{"env":"prod"}',
    );
    expect(result).toEqual({ team: "backend", env: "prod" });
  });

  it("merged result capped at 10 keys (request tags win)", () => {
    // 8 defaults + 5 request tags — some overlap, some unique
    const defaults: Record<string, string> = {};
    for (let i = 0; i < 8; i++) {
      defaults[`def${i}`] = `v${i}`;
    }
    const requestTags: Record<string, string> = {};
    for (let i = 0; i < 5; i++) {
      requestTags[`req${i}`] = `v${i}`;
    }
    // Total unique: 13, should be capped at 10
    const result = mergeTags(defaults, JSON.stringify(requestTags));
    expect(Object.keys(result).length).toBe(10);
    // All 5 request tags must be present (they win)
    for (let i = 0; i < 5; i++) {
      expect(result[`req${i}`]).toBe(`v${i}`);
    }
  });

  it("defaults with >10 keys (malformed DB data) are capped at 10", () => {
    const defaults: Record<string, string> = {};
    for (let i = 0; i < 15; i++) {
      defaults[`key${i}`] = `v${i}`;
    }
    const result = mergeTags(defaults, null);
    expect(Object.keys(result).length).toBe(10);
  });

  it("full merge example: key defaults + request override + union", () => {
    const result = mergeTags(
      { project: "openclaw", team: "backend" },
      '{"project":"other","env":"prod"}',
    );
    expect(result).toEqual({
      project: "other",   // request wins
      team: "backend",    // from defaults
      env: "prod",        // from request
    });
  });

  it("returns a shallow copy — mutating result does not corrupt original defaults", () => {
    const defaults = { project: "openclaw", team: "backend" };
    const result = mergeTags(defaults, null);
    // Mutate the returned object
    result.injected = "bad";
    // Original defaults must be unaffected
    expect(defaults).toEqual({ project: "openclaw", team: "backend" });
    expect("injected" in defaults).toBe(false);
  });

  it("filters _ns_ prefix keys from defaults (defense-in-depth)", () => {
    const result = mergeTags(
      { _ns_estimated: "true", _ns_cancelled: "true", env: "prod" },
      null,
    );
    expect(result).toEqual({ env: "prod" });
    expect(result._ns_estimated).toBeUndefined();
    expect(result._ns_cancelled).toBeUndefined();
  });

  it("filters _ns_ prefix keys from defaults during merge", () => {
    const result = mergeTags(
      { _ns_internal: "bad", team: "backend" },
      '{"env":"prod"}',
    );
    expect(result).toEqual({ env: "prod", team: "backend" });
    expect(result._ns_internal).toBeUndefined();
  });

  it("clearing defaults with empty object returns empty from merge", () => {
    const result = mergeTags({}, null);
    expect(result).toEqual({});
  });

  it("does not drop default keys that collide with Object.prototype names during merge", () => {
    const result = mergeTags(
      { constructor: "team-a", toString: "team-b", project: "alpha" },
      '{"env":"prod"}',
    );
    expect(result.constructor).toBe("team-a");
    expect(result.toString).toBe("team-b");
    expect(result.project).toBe("alpha");
    expect(result.env).toBe("prod");
  });
});
