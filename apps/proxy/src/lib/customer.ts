import { emitMetric } from "./metrics.js";

const MAX_CUSTOMER_ID_LENGTH = 256;
const CUSTOMER_ID_PATTERN = /^[a-zA-Z0-9._:-]+$/;

/**
 * Parse and validate the X-NullSpend-Customer header.
 * Returns the customer ID if valid, null otherwise.
 * Sets a warning flag when the header is present but invalid.
 */
export function parseCustomerHeader(
  header: string | null,
): { customerId: string | null; warning: string | null } {
  if (!header) return { customerId: null, warning: null };

  const trimmed = header.trim();
  if (trimmed.length === 0) return { customerId: null, warning: null };

  if (trimmed.length > MAX_CUSTOMER_ID_LENGTH) {
    console.warn(
      `[customer] X-NullSpend-Customer exceeds ${MAX_CUSTOMER_ID_LENGTH} chars — ignoring`,
    );
    emitMetric("customer_header_invalid", { reason: "too_long" });
    return { customerId: null, warning: "invalid_customer" };
  }

  if (!CUSTOMER_ID_PATTERN.test(trimmed)) {
    console.warn("[customer] X-NullSpend-Customer contains invalid characters — ignoring");
    emitMetric("customer_header_invalid", { reason: "bad_chars" });
    return { customerId: null, warning: "invalid_customer" };
  }

  return { customerId: trimmed, warning: null };
}

/**
 * Resolve the effective customer ID from header and tags.
 * Priority: X-NullSpend-Customer header > tags["customer"].
 * When resolved, auto-injects "customer" into the tags map
 * so tag-based budget matching continues to work.
 */
export function resolveCustomerId(
  headerResult: { customerId: string | null; warning: string | null },
  tags: Record<string, string>,
): string | null {
  const customerId = headerResult.customerId ?? tags["customer"] ?? null;

  // Auto-inject into tags for tag-budget compatibility
  if (customerId && tags["customer"] !== customerId) {
    tags["customer"] = customerId;
  }

  return customerId;
}
