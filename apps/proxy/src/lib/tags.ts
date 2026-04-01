export const MAX_TAGS = 10;
const MAX_KEYS = MAX_TAGS;
const MAX_KEY_LENGTH = 64;
const MAX_VALUE_LENGTH = 256;
const KEY_PATTERN = /^[a-zA-Z0-9_-]+$/;
const RESERVED_PREFIX = "_ns_";

/**
 * Parse and validate tags from the X-NullSpend-Tags header.
 * Returns `{}` on any failure — tags are supplementary metadata,
 * never a reason to reject a request.
 */
export function parseTags(header: string | null): Record<string, string> {
  if (!header) return {};

  let parsed: unknown;
  try {
    parsed = JSON.parse(header);
  } catch {
    console.warn("[tags] Malformed X-NullSpend-Tags header — ignoring");
    return {};
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    console.warn("[tags] X-NullSpend-Tags header is not a JSON object — ignoring");
    return {};
  }

  // Object.create(null) avoids __proto__ setter swallowing keys
  const result = Object.create(null) as Record<string, string>;
  let count = 0;
  let dropped = 0;

  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (count >= MAX_KEYS) {
      dropped++;
      continue;
    }

    if (
      typeof key !== "string" ||
      key.length < 1 ||
      key.length > MAX_KEY_LENGTH ||
      !KEY_PATTERN.test(key) ||
      key.startsWith(RESERVED_PREFIX)
    ) {
      dropped++;
      continue;
    }

    if (typeof value !== "string" || value.length > MAX_VALUE_LENGTH || value.includes("\0")) {
      dropped++;
      continue;
    }

    result[key] = value;
    count++;
  }

  if (dropped > 0) {
    console.warn(`[tags] Dropped ${dropped} invalid tag entries`);
  }

  return result;
}

/**
 * Merge API key default tags with request-level tags.
 *
 * Semantics: request tags override defaults for the same key,
 * non-conflicting keys are unioned. Merged result is capped at MAX_TAGS.
 */
export function mergeTags(
  defaults: Record<string, string>,
  requestHeader: string | null,
): Record<string, string> {
  const requestTags = parseTags(requestHeader);
  if (Object.keys(defaults).length === 0) return requestTags;
  if (Object.keys(requestTags).length === 0) {
    // Defaults alone — shallow copy to avoid mutating the cached identity object.
    // Also cap at MAX_TAGS and filter reserved _ns_ prefix (defense-in-depth against malformed DB data).
    const result: Record<string, string> = {};
    let count = 0;
    for (const [k, v] of Object.entries(defaults)) {
      if (count >= MAX_TAGS) break;
      if (k.startsWith(RESERVED_PREFIX)) continue;
      result[k] = v;
      count++;
    }
    return result;
  }
  // Merge: request tags win, then fill remaining slots from defaults up to MAX_TAGS.
  // Filter _ns_ prefix from defaults (defense-in-depth against malformed DB data).
  const merged: Record<string, string> = { ...requestTags };
  for (const [k, v] of Object.entries(defaults)) {
    if (Object.keys(merged).length >= MAX_TAGS) break;
    if (Object.hasOwn(merged, k)) continue;
    if (k.startsWith(RESERVED_PREFIX)) continue;
    merged[k] = v;
  }
  return merged;
}
