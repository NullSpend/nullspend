import { NullSpendError } from "./errors.js";

/**
 * Maximum length of a customer ID. Mirrors the proxy's validation
 * in apps/proxy/src/lib/customer.ts to fail fast at session creation
 * instead of silently dropping attribution at the proxy.
 */
export const MAX_CUSTOMER_ID_LENGTH = 256;

/**
 * Valid customer ID characters. Mirrors the proxy's `CUSTOMER_ID_PATTERN`.
 * Intentionally narrow so that malformed IDs fail at the SDK boundary.
 */
export const CUSTOMER_ID_PATTERN = /^[a-zA-Z0-9._:-]+$/;

/**
 * Validate and normalize a customer ID. Throws NullSpendError with a clear
 * message on any failure (empty, whitespace-only, too long, bad characters).
 *
 * Returns the trimmed, validated customer ID.
 *
 * Used by both `NullSpend.customer()` and `createTrackedFetch()`'s
 * `options.customer` so direct and indirect callers get identical fail-fast
 * behavior.
 */
export function validateCustomerId(customerId: unknown): string {
  if (typeof customerId !== "string") {
    throw new NullSpendError(
      `customer must be a string, got ${customerId === null ? "null" : typeof customerId}`,
    );
  }
  if (!customerId || !customerId.trim()) {
    throw new NullSpendError("customer requires a non-empty customerId");
  }
  const trimmed = customerId.trim();
  if (trimmed.length > MAX_CUSTOMER_ID_LENGTH) {
    throw new NullSpendError(
      `customer customerId exceeds ${MAX_CUSTOMER_ID_LENGTH} characters`,
    );
  }
  if (!CUSTOMER_ID_PATTERN.test(trimmed)) {
    throw new NullSpendError(
      "customer customerId must contain only letters, digits, and . _ : - characters",
    );
  }
  return trimmed;
}
