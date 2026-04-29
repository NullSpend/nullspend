import { VersionRegistry } from "./registry";

/**
 * The single shared registry for NullSpend. Both proxy and dashboard import
 * this. Phase 0: register a single baseline version with zero transforms;
 * the framework is in-path on every API response from day one so the first
 * real transform (Phase 1) can be added with confidence.
 *
 * SDKs ship with `nullspend-version` request header pinned to LATEST. When
 * the first breaking change ships, register a new dated version + transform
 * here, bump SDK pin in lockstep, and existing pinned installs continue
 * receiving the old shape via transformResponse.
 */
export const NULLSPEND_REGISTRY = new VersionRegistry();

export const LATEST_VALUE = "2026-04-01";

NULLSPEND_REGISTRY.registerVersion(LATEST_VALUE);

/** Class instance form of LATEST. */
export const LATEST = NULLSPEND_REGISTRY.latest();

/** Legacy compat: string form. Pre-existing dashboard call sites import this. */
export const CURRENT_VERSION: string = LATEST_VALUE;

/** Legacy compat: array of registered version strings. */
export const SUPPORTED_VERSIONS: readonly string[] = NULLSPEND_REGISTRY.listValues();
