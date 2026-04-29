import { describe, expect, it } from "vitest";
import { applyResponseChanges, VersionRegistry } from "./index";

function buildRegistry(): VersionRegistry {
  const r = new VersionRegistry();
  r.registerVersion("2026-04-01");
  r.registerVersion("2026-05-01");
  r.registerVersion("2026-06-01");
  // v2 renamed `id` → `eventId`
  r.registerChange<{ id: string }, { eventId: string }>({
    resource: "cost-events",
    oldVersion: "2026-04-01",
    newVersion: "2026-05-01",
    transformResponse: (next) => ({ id: next.eventId }),
  });
  // v3 added required `provider` field; transformResponse drops it back out
  r.registerChange<{ eventId: string }, { eventId: string; provider: string }>({
    resource: "cost-events",
    oldVersion: "2026-05-01",
    newVersion: "2026-06-01",
    transformResponse: (next) => ({ eventId: next.eventId }),
  });
  return r;
}

describe("applyResponseChanges", () => {
  it("no-op when fromVersion equals toVersion", () => {
    const r = buildRegistry();
    const v3 = r.get("2026-06-01")!;
    const payload = { eventId: "evt_1", provider: "openai" };
    expect(applyResponseChanges(payload, r, "cost-events", v3, v3)).toEqual(payload);
  });

  it("applies one step backwards", () => {
    const r = buildRegistry();
    const v3 = r.get("2026-06-01")!;
    const v2 = r.get("2026-05-01")!;
    const out = applyResponseChanges(
      { eventId: "evt_1", provider: "openai" },
      r,
      "cost-events",
      v3,
      v2,
    );
    expect(out).toEqual({ eventId: "evt_1" });
  });

  it("chains transforms across multiple steps", () => {
    const r = buildRegistry();
    const v3 = r.get("2026-06-01")!;
    const v1 = r.get("2026-04-01")!;
    const out = applyResponseChanges(
      { eventId: "evt_1", provider: "openai" },
      r,
      "cost-events",
      v3,
      v1,
    );
    expect(out).toEqual({ id: "evt_1" });
  });

  it("unknown resource is a no-op (no transforms registered)", () => {
    const r = buildRegistry();
    const v3 = r.get("2026-06-01")!;
    const v1 = r.get("2026-04-01")!;
    const payload = { foo: "bar" };
    expect(applyResponseChanges(payload, r, "unknown-resource", v3, v1)).toEqual(payload);
  });
});
