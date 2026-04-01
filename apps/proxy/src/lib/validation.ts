const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export { UUID_RE };

/**
 * Strips a NullSpend prefixed ID (e.g. "ns_act_<uuid>") and returns the raw UUID.
 * Returns null if the value is null, missing the expected prefix, or has an invalid UUID.
 */
export function stripNsPrefix(prefix: string, value: string | null): string | null {
  if (!value) return null;

  if (value.startsWith(prefix)) {
    const uuid = value.slice(prefix.length);
    return UUID_RE.test(uuid) ? uuid : null;
  }

  return null;
}
