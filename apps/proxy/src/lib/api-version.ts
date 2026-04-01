// SYNC: Dashboard copy at lib/api-version.ts must match
export const SUPPORTED_VERSIONS = ["2026-04-01"] as const;
export const CURRENT_VERSION = "2026-04-01";
export type ApiVersion = (typeof SUPPORTED_VERSIONS)[number];

export function resolveApiVersion(
  header: string | null,
  keyVersion: string,
): ApiVersion {
  if (header && (SUPPORTED_VERSIONS as readonly string[]).includes(header))
    return header as ApiVersion;
  if ((SUPPORTED_VERSIONS as readonly string[]).includes(keyVersion))
    return keyVersion as ApiVersion;
  return CURRENT_VERSION;
}
