// Regression: ISSUE-034 — SDK consumers using Authorization: Bearer against
// the proxy got the same generic 401 as no-key requests, with no hint that
// they should use x-nullspend-key. Same gap as the dashboard's ISSUE-008,
// but on the more critical proxy surface.
// Found by /qa R8 on 2026-04-27
// Report: .gstack/qa-reports/qa-report-local-2026-04-27.md

import { describe, expect, it } from "vitest";

import { unauthorizedResponse } from "./errors.js";

describe("unauthorizedResponse (ISSUE-034)", () => {
  it("returns generic 401 when no auth headers are present", async () => {
    const req = new Request("http://localhost/v1/chat/completions", { method: "POST" });
    const resp = unauthorizedResponse(req);
    const body = (await resp.json()) as { error: { code: string; message: string } };

    expect(resp.status).toBe(401);
    expect(body.error.code).toBe("unauthorized");
    expect(body.error.message).toBe("Invalid or missing authentication header");
  });

  it("returns Bearer-hint message when caller sends Authorization: Bearer", async () => {
    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: "Bearer ns_live_sk_test" },
    });
    const resp = unauthorizedResponse(req);
    const body = (await resp.json()) as { error: { code: string; message: string } };

    expect(resp.status).toBe(401);
    expect(body.error.code).toBe("unauthorized");
    expect(body.error.message).toContain("x-nullspend-key");
    expect(body.error.message).toContain("Authorization: Bearer");
  });

  it("keeps generic 401 when caller sends both x-nullspend-key (invalid) and Authorization", async () => {
    // If a caller sends BOTH headers, we use x-nullspend-key as the source of truth
    // and treat this as a normal invalid-key 401, not a Bearer hint.
    const req = new Request("http://localhost/v1/chat/completions", {
      method: "POST",
      headers: {
        "x-nullspend-key": "ns_live_sk_invalid",
        Authorization: "Bearer ignored",
      },
    });
    const resp = unauthorizedResponse(req);
    const body = (await resp.json()) as { error: { code: string; message: string } };

    expect(resp.status).toBe(401);
    expect(body.error.message).toBe("Invalid or missing authentication header");
  });
});
