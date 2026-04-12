import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({
  waitUntil: vi.fn(),
  DurableObject: class {},
}));

import { parseCustomerHeader, resolveCustomerId } from "../lib/customer.js";

describe("parseCustomerHeader", () => {
  it("returns null for null header", () => {
    const result = parseCustomerHeader(null);
    expect(result).toEqual({ customerId: null, warning: null });
  });

  it("returns null for empty string", () => {
    const result = parseCustomerHeader("");
    expect(result).toEqual({ customerId: null, warning: null });
  });

  it("returns null for whitespace-only string", () => {
    const result = parseCustomerHeader("   ");
    expect(result).toEqual({ customerId: null, warning: null });
  });

  it("parses valid alphanumeric customer ID", () => {
    const result = parseCustomerHeader("acme-corp");
    expect(result).toEqual({ customerId: "acme-corp", warning: null });
  });

  it("trims whitespace from valid customer ID", () => {
    const result = parseCustomerHeader("  acme-corp  ");
    expect(result).toEqual({ customerId: "acme-corp", warning: null });
  });

  it("allows dots in customer ID", () => {
    const result = parseCustomerHeader("acme.corp");
    expect(result).toEqual({ customerId: "acme.corp", warning: null });
  });

  it("allows colons in customer ID", () => {
    const result = parseCustomerHeader("org:acme");
    expect(result).toEqual({ customerId: "org:acme", warning: null });
  });

  it("allows underscores and hyphens", () => {
    const result = parseCustomerHeader("acme_corp-123");
    expect(result).toEqual({ customerId: "acme_corp-123", warning: null });
  });

  it("rejects customer ID with spaces", () => {
    const result = parseCustomerHeader("acme corp");
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });

  it("rejects customer ID with special characters", () => {
    const result = parseCustomerHeader("acme@corp");
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });

  it("rejects customer ID with slashes", () => {
    const result = parseCustomerHeader("acme/corp");
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });

  it("accepts customer ID at exactly 256 chars", () => {
    const id = "a".repeat(256);
    const result = parseCustomerHeader(id);
    expect(result).toEqual({ customerId: id, warning: null });
  });

  it("rejects customer ID exceeding 256 chars", () => {
    const id = "a".repeat(257);
    const result = parseCustomerHeader(id);
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });

  it("rejects newline injection", () => {
    const result = parseCustomerHeader("acme\ncorp");
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });

  it("rejects null byte injection", () => {
    const result = parseCustomerHeader("acme\0corp");
    expect(result).toEqual({ customerId: null, warning: "invalid_customer" });
  });
});

describe("resolveCustomerId", () => {
  it("returns header customer ID when present", () => {
    const tags: Record<string, string> = {};
    const result = resolveCustomerId({ customerId: "acme", warning: null }, tags);
    expect(result).toBe("acme");
  });

  it("falls back to tags['customer'] when header is null", () => {
    const tags: Record<string, string> = { customer: "from-tag" };
    const result = resolveCustomerId({ customerId: null, warning: null }, tags);
    expect(result).toBe("from-tag");
  });

  it("returns null when neither header nor tag present", () => {
    const tags: Record<string, string> = {};
    const result = resolveCustomerId({ customerId: null, warning: null }, tags);
    expect(result).toBeNull();
  });

  it("header takes precedence over tag", () => {
    const tags: Record<string, string> = { customer: "from-tag" };
    const result = resolveCustomerId({ customerId: "from-header", warning: null }, tags);
    expect(result).toBe("from-header");
  });

  it("auto-injects customer into tags when resolved from header", () => {
    const tags: Record<string, string> = {};
    resolveCustomerId({ customerId: "acme", warning: null }, tags);
    expect(tags["customer"]).toBe("acme");
  });

  it("overwrites tag when header differs", () => {
    const tags: Record<string, string> = { customer: "old-value" };
    resolveCustomerId({ customerId: "new-value", warning: null }, tags);
    expect(tags["customer"]).toBe("new-value");
  });

  it("does not modify tags when resolved from tag (same value)", () => {
    const tags: Record<string, string> = { customer: "acme" };
    resolveCustomerId({ customerId: null, warning: null }, tags);
    expect(tags["customer"]).toBe("acme");
  });

  it("does not inject into tags when customer is null", () => {
    const tags: Record<string, string> = { env: "prod" };
    resolveCustomerId({ customerId: null, warning: null }, tags);
    expect(tags["customer"]).toBeUndefined();
  });

  it("preserves other tags when injecting customer", () => {
    const tags: Record<string, string> = { env: "prod", team: "backend" };
    resolveCustomerId({ customerId: "acme", warning: null }, tags);
    expect(tags).toEqual({ env: "prod", team: "backend", customer: "acme" });
  });
});
