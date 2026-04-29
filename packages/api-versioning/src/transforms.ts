import { ApiVersion } from "./api-version";
import type { VersionRegistry } from "./registry";

/**
 * Walk response transforms from `fromVersion` (newest, what the handler returned)
 * back to `toVersion` (what the client wants). Each VersionChange.transformResponse
 * projects the NEW shape into the OLD shape.
 *
 * Order: walk descending. If registry has changes [v1→v2, v2→v3] and we need
 * to go from v3 (handler) back to v1 (client), apply v2→v3.transformResponse
 * (gives v2 shape), then v1→v2.transformResponse (gives v1 shape).
 */
export function applyResponseChanges(
  payload: unknown,
  registry: VersionRegistry,
  resource: string,
  fromVersion: ApiVersion,
  toVersion: ApiVersion,
): unknown {
  if (fromVersion.eq(toVersion) || fromVersion.lt(toVersion)) return payload;
  const changes = registry.getChanges(resource);
  let current = payload;
  for (let i = changes.length - 1; i >= 0; i--) {
    const change = changes[i];
    const newV = registry.get(change.newVersion);
    const oldV = registry.get(change.oldVersion);
    if (!newV || !oldV) continue;
    if (newV.lte(toVersion)) break;
    if (newV.gt(fromVersion)) continue;
    if (change.transformResponse) {
      current = change.transformResponse(current);
    }
  }
  return current;
}
