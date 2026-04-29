import { describe, expect, it } from "vitest";
import {
  NULLSPEND_REGISTRY,
  resolveApiVersion,
  InvalidVersionError,
  UnknownVersionError,
} from "@nullspend/api-versioning";

/**
 * Contract tests for the proxy's api-versioning integration. Validates that
 * the proxy and the dashboard share the same registry singleton and surface
 * identical resolution behavior. The proxy hot path (`apps/proxy/src/index.ts`)
 * resolves the same way; this file pins the contract.
 *
 * Phase 0: registry has 1 version, 0 transforms. Future Phase-1 transforms
 * will surface here as additional shape assertions.
 */
describe("proxy api-versioning contract", () => {
  it("missing header falls through to LATEST", () => {
    const v = resolveApiVersion({
      headerValue: null,
      registry: NULLSPEND_REGISTRY,
    });
    expect(v.value).toBe(NULLSPEND_REGISTRY.latest().value);
  });

  it("valid CalVer in header is honored", () => {
    const v = resolveApiVersion({
      headerValue: "2026-04-01",
      registry: NULLSPEND_REGISTRY,
    });
    expect(v.value).toBe("2026-04-01");
  });

  it("malformed header throws InvalidVersionError carrying supported list", () => {
    try {
      resolveApiVersion({
        headerValue: "latest",
        registry: NULLSPEND_REGISTRY,
      });
      expect.fail("expected throw");
    } catch (err) {
      expect(err).toBeInstanceOf(InvalidVersionError);
      expect((err as InvalidVersionError).supported).toContain("2026-04-01");
    }
  });

  it("well-formed but unknown CalVer throws UnknownVersionError", () => {
    try {
      resolveApiVersion({
        headerValue: "2099-01-01",
        registry: NULLSPEND_REGISTRY,
      });
      expect.fail("expected throw");
    } catch (err) {
      expect(err).toBeInstanceOf(UnknownVersionError);
    }
  });

  it("keyVersion fallback is consulted when header missing", () => {
    const v = resolveApiVersion({
      headerValue: null,
      keyVersion: "2026-04-01",
      registry: NULLSPEND_REGISTRY,
    });
    expect(v.value).toBe("2026-04-01");
  });
});
