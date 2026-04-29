import { ApiVersion, isCalVer } from "./api-version";
import { InvalidVersionError, UnknownVersionError } from "./errors";
import type { VersionRegistry } from "./registry";

/**
 * Hard cap on header length to defend against pathological inputs (1MB strings,
 * etc.). CalVer is exactly 10 chars; allow some slack but reject anything wild.
 */
const MAX_HEADER_LENGTH = 64;

export interface ResolveApiVersionOptions {
  headerValue: string | null;
  /** Version pinned to the API key (legacy fallback). */
  keyVersion?: string | null;
  /** Version pinned to the org (Phase 2 future). */
  orgDefault?: string | null;
  registry: VersionRegistry;
}

/**
 * Parse a raw header value into a registered ApiVersion. Returns null when
 * the header is absent. Throws InvalidVersionError on malformed input,
 * UnknownVersionError on well-formed-but-unregistered input.
 */
export function parseVersionHeader(
  raw: string | null,
  registry: VersionRegistry,
): ApiVersion | null {
  if (raw === null || raw === undefined) return null;
  const value = raw.trim();
  if (value.length === 0) return null;
  if (value.length > MAX_HEADER_LENGTH) {
    throw new InvalidVersionError(value.slice(0, MAX_HEADER_LENGTH), registry.listValues());
  }
  if (!isCalVer(value)) {
    throw new InvalidVersionError(value, registry.listValues());
  }
  const v = registry.get(value);
  if (!v) {
    throw new UnknownVersionError(value, registry.listValues());
  }
  return v;
}

/**
 * Resolution order: header → keyVersion (legacy) → orgDefault (Phase 2) → LATEST.
 *
 * Throws on header that is malformed (Invalid) or well-formed-but-unregistered
 * (Unknown). keyVersion / orgDefault that fail to resolve fall through to the
 * next tier silently — they are server-side defaults, not user-supplied input,
 * so a stale value should not 400 the caller.
 */
export function resolveApiVersion(opts: ResolveApiVersionOptions): ApiVersion {
  const fromHeader = parseVersionHeader(opts.headerValue, opts.registry);
  if (fromHeader) return fromHeader;

  if (opts.keyVersion) {
    const v = opts.registry.get(opts.keyVersion);
    if (v) return v;
  }
  if (opts.orgDefault) {
    const v = opts.registry.get(opts.orgDefault);
    if (v) return v;
  }
  return opts.registry.latest();
}
