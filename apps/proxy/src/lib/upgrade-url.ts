/**
 * Resolve the upgrade URL for a denial response.
 *
 * Priority order (for customer denials):
 *   1. customer-level URL (customer_settings.upgrade_url)
 *   2. org-level URL (organizations.metadata.upgradeUrl)
 *   3. null (no upgrade_url field in the response)
 *
 * For non-customer denials (generic budget_exceeded where the denying
 * entity is user/api_key/default/org), only the org-level URL is used.
 *
 * The caller is responsible for determining whether the denial code
 * warrants an upgrade URL at all — this helper is agnostic to denial
 * type. Per decision 5 of the plan, only `budget_exceeded` and
 * `customer_budget_exceeded` should call this helper.
 *
 * `{customer_id}` placeholder substitution: if the resolved URL contains
 * the literal string `{customer_id}`, it is replaced with the actual
 * customer ID (URL-encoded). If no customer ID is in scope, the
 * placeholder is left untouched — the caller gets a URL that a dev
 * viewing the raw response can fix, rather than a silently-null field.
 *
 * @param orgUrl        organizations.metadata.upgradeUrl, or null
 * @param customerUrl   customer_settings.upgrade_url, or null
 * @param customerId    current request's customer ID, or null/undefined
 * @returns             resolved URL string, or null if neither URL is set
 */
export function resolveUpgradeUrl(
  orgUrl: string | null,
  customerUrl: string | null,
  customerId: string | null | undefined,
): string | null {
  const raw = customerUrl ?? orgUrl;
  if (!raw) return null;

  if (!raw.includes("{customer_id}")) return raw;

  if (customerId) {
    return raw.replace(/\{customer_id\}/g, encodeURIComponent(customerId));
  }

  // Placeholder present but no customer in scope — leave untouched.
  // Debugging signal: dev viewing the raw response sees the placeholder
  // and knows the customer attribution pipeline didn't reach this denial.
  return raw;
}
