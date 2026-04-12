// ---------------------------------------------------------------------------
// Retry helpers — pure functions, no SDK imports
// ---------------------------------------------------------------------------

export const RETRYABLE_STATUS_CODES: ReadonlySet<number> = new Set([
  429, 500, 502, 503, 504,
]);

export const DEFAULT_MAX_RETRIES = 2;
export const DEFAULT_RETRY_BASE_DELAY_MS = 500;
export const DEFAULT_MAX_RETRY_DELAY_MS = 5_000;

export function isRetryableStatusCode(status: number): boolean {
  return RETRYABLE_STATUS_CODES.has(status);
}

/**
 * Returns true for transient fetch errors that are safe to retry:
 * - `TypeError` → network failures (all runtimes)
 * - `DOMException` with name `"TimeoutError"` → AbortSignal.timeout() fired
 *
 * NOT retryable:
 * - `DOMException` with name `"AbortError"` → user-initiated cancel
 */
export function isRetryableError(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  if (
    typeof DOMException !== "undefined" &&
    err instanceof DOMException &&
    err.name === "TimeoutError"
  ) {
    return true;
  }
  return false;
}

/**
 * Parse an HTTP `Retry-After` header value into milliseconds.
 *
 * Handles:
 * - Numeric seconds: `"2"` → 2000
 * - HTTP dates (RFC 9110 §10.2.3): `"Sun, 16 Mar 2026 12:00:00 GMT"` → delta ms
 *
 * Returns `null` on unparseable/negative input. Caps at `maxMs`.
 */
export function parseRetryAfterMs(
  value: string | null,
  maxMs: number,
): number | null {
  if (value === null || value.trim() === "") return null;

  const numeric = Number(value);
  if (!Number.isNaN(numeric)) {
    if (numeric < 0) return null;
    return Math.min(Math.round(numeric * 1_000), maxMs);
  }

  // Try HTTP date
  const date = Date.parse(value);
  if (Number.isNaN(date)) return null;

  const deltaMs = date - Date.now();
  if (deltaMs < 0) return null;
  return Math.min(deltaMs, maxMs);
}

/**
 * Full-jitter exponential backoff: `floor(random() * min(base * 2^attempt, maxDelay))`.
 * Always returns >= 1 (never zero).
 */
export function calculateRetryDelayMs(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
): number {
  const ceiling = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
  return Math.max(1, Math.floor(Math.random() * ceiling));
}
