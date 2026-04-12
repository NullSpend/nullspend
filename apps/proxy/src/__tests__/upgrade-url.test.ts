import { describe, it, expect } from "vitest";
import { resolveUpgradeUrl } from "../lib/upgrade-url.js";

describe("resolveUpgradeUrl", () => {
  it("returns null when both org and customer URLs are null", () => {
    expect(resolveUpgradeUrl(null, null, null)).toBeNull();
    expect(resolveUpgradeUrl(null, null, "cust_abc")).toBeNull();
  });

  it("returns org URL when only org URL is set", () => {
    expect(resolveUpgradeUrl("https://acme.com/billing", null, null)).toBe(
      "https://acme.com/billing",
    );
  });

  it("returns customer URL when only customer URL is set", () => {
    expect(resolveUpgradeUrl(null, "https://acme.com/cust", "cust_abc")).toBe(
      "https://acme.com/cust",
    );
  });

  it("customer URL takes priority over org URL", () => {
    expect(
      resolveUpgradeUrl(
        "https://acme.com/org-default",
        "https://acme.com/customer-specific",
        "cust_abc",
      ),
    ).toBe("https://acme.com/customer-specific");
  });

  it("substitutes {customer_id} placeholder with URL-encoded ID", () => {
    expect(
      resolveUpgradeUrl("https://acme.com/upgrade?customer={customer_id}", null, "cust_abc"),
    ).toBe("https://acme.com/upgrade?customer=cust_abc");
  });

  it("URL-encodes customer IDs with special characters", () => {
    expect(
      resolveUpgradeUrl(
        "https://acme.com/u?c={customer_id}",
        null,
        "cust with spaces",
      ),
    ).toBe("https://acme.com/u?c=cust%20with%20spaces");
  });

  it("substitutes multiple {customer_id} placeholders", () => {
    expect(
      resolveUpgradeUrl(
        "https://acme.com/{customer_id}/path?ref={customer_id}",
        null,
        "cust_abc",
      ),
    ).toBe("https://acme.com/cust_abc/path?ref=cust_abc");
  });

  it("leaves placeholder untouched when customer ID is missing", () => {
    // Debugging signal: a dev seeing the literal {customer_id} in a raw
    // response body knows the attribution pipeline didn't reach this denial.
    expect(
      resolveUpgradeUrl("https://acme.com/upgrade?customer={customer_id}", null, null),
    ).toBe("https://acme.com/upgrade?customer={customer_id}");
  });

  it("leaves placeholder untouched when customer ID is empty string", () => {
    // Same debugging intent as null/undefined
    expect(
      resolveUpgradeUrl("https://acme.com/upgrade?c={customer_id}", null, ""),
    ).toBe("https://acme.com/upgrade?c={customer_id}");
  });

  it("handles customer URL with placeholder", () => {
    expect(
      resolveUpgradeUrl(
        "https://acme.com/org",
        "https://acme.com/cust?id={customer_id}",
        "cust_abc",
      ),
    ).toBe("https://acme.com/cust?id=cust_abc");
  });

  it("does not add a placeholder if the resolved URL does not contain one", () => {
    // Sanity: a URL without the placeholder is returned unchanged
    expect(
      resolveUpgradeUrl("https://acme.com/static", null, "cust_abc"),
    ).toBe("https://acme.com/static");
  });
});
