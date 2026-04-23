/**
 * PR-2d C45 + C46: GET /health/feature-flags.
 *
 * Shadow-mode alert signal source (plan Decision #34). The response shape is a
 * public contract with the GitHub Action poller — exact string values matter.
 */
import { describe, it, expect } from "vitest";
import { handleFeatureFlags } from "../routes/health.js";
import { CACHE_SCHEMA_VERSION } from "../lib/api-key-auth.js";

function makeEnv(overrides: Record<string, unknown> = {}): Env {
  return { ...overrides } as unknown as Env;
}

describe("C45: GET /health/feature-flags response shape", () => {
  it("returns PLAN_COUNTER_ENABLED='true' when env var is 'true'", async () => {
    const res = handleFeatureFlags(
      makeEnv({ PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true", BUILD_SHA: "abc123" }),
    );
    expect(res.status).toBe(200);
    const json = (await res.json()) as Record<string, unknown>;
    expect(json).toEqual({
      PLAN_COUNTER_ENABLED: "true",
      NULLSPEND_CLOUD: "true",
      CACHE_SCHEMA_VERSION,
      build_sha: "abc123",
    });
  });

  it("returns PLAN_COUNTER_ENABLED='false' when env var is 'false'", async () => {
    const res = handleFeatureFlags(makeEnv({ PLAN_COUNTER_ENABLED: "false", BUILD_SHA: "def456" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.PLAN_COUNTER_ENABLED).toBe("false");
    expect(json.build_sha).toBe("def456");
  });

  it("defaults PLAN_COUNTER_ENABLED to 'false' when env var is absent", async () => {
    const res = handleFeatureFlags(makeEnv({ BUILD_SHA: "sha1" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.PLAN_COUNTER_ENABLED).toBe("false");
  });

  it("defaults build_sha to 'unknown' when env var is absent", async () => {
    const res = handleFeatureFlags(makeEnv({ PLAN_COUNTER_ENABLED: "true" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.build_sha).toBe("unknown");
  });

  it("always returns the current CACHE_SCHEMA_VERSION constant", async () => {
    const res = handleFeatureFlags(makeEnv());
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.CACHE_SCHEMA_VERSION).toBe(CACHE_SCHEMA_VERSION);
  });

  it("returns exactly the four contract fields (guards against field drift)", async () => {
    const res = handleFeatureFlags(
      makeEnv({ PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true", BUILD_SHA: "s" }),
    );
    const json = (await res.json()) as Record<string, unknown>;
    expect(Object.keys(json).sort()).toEqual([
      "CACHE_SCHEMA_VERSION",
      "NULLSPEND_CLOUD",
      "PLAN_COUNTER_ENABLED",
      "build_sha",
    ]);
  });

  // PR-2e post-flip review: NULLSPEND_CLOUD is exposed so the launch-watcher
  // can detect the 2026-04-20 incident class (flag=true but cloud!=true →
  // silent no-op enforcement). Same contract rigor as PLAN_COUNTER_ENABLED.
  it("returns NULLSPEND_CLOUD='true' when env var is 'true'", async () => {
    const res = handleFeatureFlags(makeEnv({ NULLSPEND_CLOUD: "true" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.NULLSPEND_CLOUD).toBe("true");
  });

  it("returns NULLSPEND_CLOUD='false' when env var is 'false'", async () => {
    const res = handleFeatureFlags(makeEnv({ NULLSPEND_CLOUD: "false" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.NULLSPEND_CLOUD).toBe("false");
  });

  it("defaults NULLSPEND_CLOUD to 'false' when env var is absent", async () => {
    // Absence is the self-hosted default. The watcher must see 'false'
    // (not 'missing' or undefined) so its string-equality check is stable.
    const res = handleFeatureFlags(makeEnv({ PLAN_COUNTER_ENABLED: "true" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.NULLSPEND_CLOUD).toBe("false");
  });

  it("returns NULLSPEND_CLOUD as raw string value (does not coerce to boolean)", async () => {
    // Same rigor as PLAN_COUNTER_ENABLED: string value, not boolean. The
    // watcher compares ==="true" and any typo/casing drift should be
    // observable as a string mismatch, not silently coerced.
    const res = handleFeatureFlags(makeEnv({ NULLSPEND_CLOUD: "True" }));
    const json = (await res.json()) as Record<string, unknown>;
    expect(json.NULLSPEND_CLOUD).toBe("True");
    expect(json.NULLSPEND_CLOUD).not.toBe(true);
  });

  it("sets Cache-Control: no-store so the alert reads fresh state", async () => {
    const res = handleFeatureFlags(makeEnv());
    expect(res.headers.get("Cache-Control")).toBe("no-store");
  });

  it("returns application/json content-type", async () => {
    const res = handleFeatureFlags(makeEnv());
    expect(res.headers.get("content-type")).toMatch(/application\/json/);
  });
});

describe("C46: no auth required", () => {
  it("succeeds without any Authorization header", async () => {
    // handleFeatureFlags doesn't take a Request — there's no path where
    // authentication can be checked. The absence of any Request parameter
    // is the structural proof. Assert 200 unconditionally with only env.
    const res = handleFeatureFlags(makeEnv());
    expect(res.status).toBe(200);
  });

  it("succeeds even when INTERNAL_SECRET env is unset", async () => {
    // Unlike internal routes, this endpoint must not 500 when INTERNAL_SECRET
    // is missing — ops should be able to hit it without deploy-secret
    // configuration.
    const res = handleFeatureFlags(makeEnv({}));
    expect(res.status).toBe(200);
  });
});
