import type { VersionRegistry } from "./registry";

/**
 * Shape returned by `GET /api/_versions` (dashboard) and `GET /v1/_versions`
 * (proxy). Both surfaces serve byte-identical bodies because they read the
 * same registry via this single helper. Used by SDK codegen, Postman, and any
 * future CLI to enumerate supported versions.
 *
 * Status enum:
 * - `current` — the registry's LATEST
 * - `supported` — registered, not the latest (Phase 1+ when multiple versions
 *   coexist)
 *
 * Deprecation status will be re-introduced when Phase 1 ships the first
 * deprecated version, with the corresponding RFC 9745 + 8594 header machinery
 * driven by an explicit `sunsetAt` field on `VersionChange`.
 */
export type VersionStatus = "current" | "supported";

export interface VersionDescriptor {
  version: string;
  status: VersionStatus;
}

export interface VersionsResponse {
  versions: VersionDescriptor[];
  default: string;
}

export function buildVersionsResponse(registry: VersionRegistry): VersionsResponse {
  const latest = registry.latest();
  return {
    versions: registry.list().map((v) => ({
      version: v.value,
      status: v.eq(latest) ? ("current" as const) : ("supported" as const),
    })),
    default: latest.value,
  };
}
