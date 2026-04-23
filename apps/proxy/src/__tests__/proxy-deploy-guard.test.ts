/**
 * Unit tests for the pre-deploy guard logic.
 *
 * The entrypoint (proxy-deploy-guard.ts) shells out to wrangler; this test
 * file targets the pure decision functions in proxy-deploy-guard-lib.ts so
 * the guard's correctness can be verified without spawning processes.
 *
 * Coverage objective: every failure path of evaluateGuard() + every
 * detection branch of argvHasVarFlagFor() and secretListContains(). Per
 * PR-2e codex R5 lesson #1: monitoring/safety code that fails silently is
 * worse than no safety code, so the guard's logic must be exhaustively tested.
 */

import { describe, it, expect } from "vitest";

import {
  CLOUD_FLAG_NAME,
  FLAG_NAME,
  argvHasVarFlagFor,
  decideDeployExit,
  evaluateCloudInvariant,
  evaluateGuard,
  evaluateMigrationParity,
  parseDrizzleJournal,
  parseWranglerVars,
  secretListContains,
} from "../../scripts/proxy-deploy-guard-lib.js";

// =====================================================================
// PR-6b Sub-lane 5: migration-parity check
// =====================================================================

describe("parseDrizzleJournal", () => {
  it("parses a well-formed journal into normalized entries", () => {
    const content = JSON.stringify({
      version: "7",
      entries: [
        { idx: 0, version: "7", when: 1, tag: "0000_initial" },
        { idx: 1, version: "7", when: 2, tag: "0001_add_check" },
      ],
    });
    const journal = parseDrizzleJournal(content);
    expect(journal).not.toBeNull();
    expect(journal!.entries).toHaveLength(2);
    expect(journal!.entries[0].tag).toBe("0000_initial");
    expect(journal!.entries[1].idx).toBe(1);
  });

  it("filters out entries missing required fields", () => {
    const content = JSON.stringify({
      entries: [
        { idx: 0, tag: "0000_ok" },
        { idx: "not-a-number", tag: "0001_bad" },
        { idx: 2 }, // missing tag
        { idx: 3, tag: "" }, // empty tag
        { idx: 4, tag: "0004_ok" },
      ],
    });
    const journal = parseDrizzleJournal(content);
    expect(journal!.entries.map((e) => e.tag)).toEqual(["0000_ok", "0004_ok"]);
  });

  it("returns null on malformed JSON", () => {
    expect(parseDrizzleJournal("{ broken")).toBeNull();
  });

  it("returns null when entries field is missing", () => {
    expect(parseDrizzleJournal(JSON.stringify({ version: "7" }))).toBeNull();
  });

  it("returns null when entries is not an array", () => {
    expect(parseDrizzleJournal(JSON.stringify({ entries: {} }))).toBeNull();
  });
});

describe("evaluateMigrationParity (PR-6b Sub-lane 5)", () => {
  const journal = {
    entries: [
      { idx: 0, tag: "0000_initial" },
      { idx: 1, tag: "0001_add_check" },
      { idx: 2, tag: "0069_opu_disposition_expand" },
    ],
  };

  it("returns null when every journal entry has a matching .sql file", () => {
    const files = [
      "0000_initial.sql",
      "0001_add_check.sql",
      "0069_opu_disposition_expand.sql",
      "unrelated_extra.sql", // extra files are fine; only journal entries are required
    ];
    expect(evaluateMigrationParity(journal, files)).toBeNull();
  });

  it("returns migration_parity failure when one file is missing", () => {
    const files = ["0000_initial.sql", "0001_add_check.sql"]; // 0069 missing
    const result = evaluateMigrationParity(journal, files);
    expect(result?.reason).toBe("migration_parity");
    expect(result?.message).toContain("0069_opu_disposition_expand.sql");
    expect(result?.message).toContain("REFUSED");
  });

  it("lists ALL missing files in a single failure message", () => {
    const files: string[] = [];
    const result = evaluateMigrationParity(journal, files);
    expect(result?.message).toContain("0000_initial.sql");
    expect(result?.message).toContain("0001_add_check.sql");
    expect(result?.message).toContain("0069_opu_disposition_expand.sql");
    expect(result?.message).toContain("3 migration file(s)");
  });

  it("returns null when journal is null (skippable, not deploy-blocking)", () => {
    expect(evaluateMigrationParity(null, ["anything.sql"])).toBeNull();
  });

  it("returns null when journal has zero entries", () => {
    expect(evaluateMigrationParity({ entries: [] }, [])).toBeNull();
  });

  it("is order-independent on file list (uses Set lookup)", () => {
    const files1 = ["0000_initial.sql", "0001_add_check.sql", "0069_opu_disposition_expand.sql"];
    const files2 = [...files1].reverse();
    expect(evaluateMigrationParity(journal, files1)).toBeNull();
    expect(evaluateMigrationParity(journal, files2)).toBeNull();
  });

  it("case-normalizes both sides for cross-platform parity (codex P0-1)", () => {
    // Windows/macOS default to case-INSENSITIVE filesystems; Linux is
    // case-SENSITIVE. A file committed as `0070_Overage.sql` on Windows
    // would pass parity locally (filesystem matches case-insensitively)
    // but the journal entry's tag is canonical-lowercase from drizzle-kit.
    // Without normalization, the Linux deploy would fail to find
    // `0070_overage.sql` if the file was `0070_Overage.sql` on disk.
    const upperJournal = {
      entries: [{ idx: 0, tag: "0070_Overage_Cron_Runs" }],
    };
    const lowerFiles = ["0070_overage_cron_runs.sql"];
    expect(evaluateMigrationParity(upperJournal, lowerFiles)).toBeNull();

    const lowerJournal = {
      entries: [{ idx: 0, tag: "0070_overage_cron_runs" }],
    };
    const upperFiles = ["0070_Overage_Cron_Runs.SQL"];
    expect(evaluateMigrationParity(lowerJournal, upperFiles)).toBeNull();
  });
});

describe("evaluateGuard with migration-parity", () => {
  it("fails on missing migration file BEFORE running cloud-invariant or secret check", () => {
    const journal = { entries: [{ idx: 0, tag: "0070_overage_cron_runs" }] };
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "", stderr: "" },
      // Even with a broken cloud invariant set, migration-parity wins.
      { PLAN_COUNTER_ENABLED: "true" },
      { journal, drizzleSqlFiles: [] },
    );
    expect(result?.reason).toBe("migration_parity");
  });

  it("passes through to other checks when migration-parity is satisfied", () => {
    const journal = { entries: [{ idx: 0, tag: "0070_overage_cron_runs" }] };
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true" }, // breaks cloud invariant
      { journal, drizzleSqlFiles: ["0070_overage_cron_runs.sql"] },
    );
    expect(result?.reason).toBe("cloud_invariant");
  });

  it("skips parity check entirely when caller omits the parameter (back-compat)", () => {
    // The pre-PR-6b call sites pass only 3 args. The 4th-arg-optional
    // contract is load-bearing for backward compatibility.
    const result = evaluateGuard([], { status: 0, stdout: "", stderr: "" }, {});
    expect(result).toBeNull();
  });
});

describe("FLAG_NAME constant", () => {
  it("is PLAN_COUNTER_ENABLED (per PR-2e Decision #41)", () => {
    expect(FLAG_NAME).toBe("PLAN_COUNTER_ENABLED");
  });
});

describe("CLOUD_FLAG_NAME constant", () => {
  it("is NULLSPEND_CLOUD (per PR-2e post-launch hotfix on 2026-04-20)", () => {
    expect(CLOUD_FLAG_NAME).toBe("NULLSPEND_CLOUD");
  });
});

describe("parseWranglerVars", () => {
  it("parses vars from JSONC with line comments stripped", () => {
    const content = `{
      // PR-2e launch gate
      "name": "nullspend-proxy",
      "vars": {
        "PLAN_COUNTER_ENABLED": "true", // enforcement flag
        "NULLSPEND_CLOUD": "true"
      }
    }`;
    const vars = parseWranglerVars(content);
    expect(vars.PLAN_COUNTER_ENABLED).toBe("true");
    expect(vars.NULLSPEND_CLOUD).toBe("true");
  });

  it("parses vars from JSONC with block comments stripped", () => {
    const content = `{
      /* top-level config
         multi-line block */
      "vars": {
        "PLAN_COUNTER_ENABLED": "true"
      }
    }`;
    const vars = parseWranglerVars(content);
    expect(vars.PLAN_COUNTER_ENABLED).toBe("true");
  });

  it("parses plain JSON (no comments) correctly", () => {
    const content = JSON.stringify({
      vars: { PLAN_COUNTER_ENABLED: "false", NULLSPEND_CLOUD: "false" },
    });
    const vars = parseWranglerVars(content);
    expect(vars.PLAN_COUNTER_ENABLED).toBe("false");
    expect(vars.NULLSPEND_CLOUD).toBe("false");
  });

  it("returns empty object when JSON is malformed (skippable, not deploy-blocking)", () => {
    expect(parseWranglerVars("{ not valid json")).toEqual({});
  });

  it("returns empty object when vars field is missing", () => {
    const content = JSON.stringify({ name: "nullspend-proxy" });
    expect(parseWranglerVars(content)).toEqual({});
  });

  it("returns empty object when vars is not an object (null)", () => {
    const content = JSON.stringify({ vars: null });
    expect(parseWranglerVars(content)).toEqual({});
  });

  it("filters out non-string values from vars", () => {
    // wrangler.jsonc supports vars with bool/number/object types; the
    // guard only cares about string values so it coerces the surface.
    const content = JSON.stringify({
      vars: {
        PLAN_COUNTER_ENABLED: "true",
        NUMERIC_FLAG: 42,
        BOOL_FLAG: true,
        NESTED: { sub: "value" },
      },
    });
    const vars = parseWranglerVars(content);
    expect(vars.PLAN_COUNTER_ENABLED).toBe("true");
    expect(vars.NUMERIC_FLAG).toBeUndefined();
    expect(vars.BOOL_FLAG).toBeUndefined();
    expect(vars.NESTED).toBeUndefined();
  });

  it("returns empty object for an empty input string", () => {
    expect(parseWranglerVars("")).toEqual({});
  });
});

describe("evaluateCloudInvariant (PR-2e post-launch hotfix — 2026-04-20)", () => {
  it("returns cloud_invariant failure when PLAN_COUNTER_ENABLED=true but NULLSPEND_CLOUD is unset", () => {
    const result = evaluateCloudInvariant({ PLAN_COUNTER_ENABLED: "true" });
    expect(result?.reason).toBe("cloud_invariant");
    expect(result?.message).toContain("PLAN_COUNTER_ENABLED");
    expect(result?.message).toContain("NULLSPEND_CLOUD");
    expect(result?.message).toContain("SILENT no-op");
  });

  it("returns cloud_invariant failure when PLAN_COUNTER_ENABLED=true and NULLSPEND_CLOUD=\"false\"", () => {
    const result = evaluateCloudInvariant({
      PLAN_COUNTER_ENABLED: "true",
      NULLSPEND_CLOUD: "false",
    });
    expect(result?.reason).toBe("cloud_invariant");
  });

  it("returns null when both are set to \"true\" (the launch-ready state)", () => {
    const result = evaluateCloudInvariant({
      PLAN_COUNTER_ENABLED: "true",
      NULLSPEND_CLOUD: "true",
    });
    expect(result).toBeNull();
  });

  it("returns null when PLAN_COUNTER_ENABLED is not \"true\" (invariant only applies when flag is on)", () => {
    // Pre-flip state: flag off, cloud flag absent → guard must not trip.
    expect(evaluateCloudInvariant({})).toBeNull();
    expect(evaluateCloudInvariant({ PLAN_COUNTER_ENABLED: "false" })).toBeNull();
  });

  it("does NOT trip on truthy-but-non-\"true\" values (string match is exact)", () => {
    // The auth layer's check is `env.NULLSPEND_CLOUD !== \"true\"` — only the
    // exact string \"true\" flips identity to cloud. Anything else keeps
    // self-hosted, so the guard must not accept \"True\" or \"1\" as passing.
    const result = evaluateCloudInvariant({
      PLAN_COUNTER_ENABLED: "true",
      NULLSPEND_CLOUD: "True",
    });
    expect(result?.reason).toBe("cloud_invariant");
  });
});

describe("argvHasVarFlagFor", () => {
  it("matches --var KEY:VALUE (two-arg form)", () => {
    expect(
      argvHasVarFlagFor(["--var", "PLAN_COUNTER_ENABLED:true"], FLAG_NAME),
    ).toBe(true);
  });

  it("matches --var=KEY:VALUE (single-arg form)", () => {
    expect(
      argvHasVarFlagFor(["--var=PLAN_COUNTER_ENABLED:true"], FLAG_NAME),
    ).toBe(true);
  });

  it("matches when --var appears alongside other args", () => {
    expect(
      argvHasVarFlagFor(
        ["--env", "production", "--var", "PLAN_COUNTER_ENABLED:true"],
        FLAG_NAME,
      ),
    ).toBe(true);
  });

  it("does NOT match an unrelated --var", () => {
    expect(
      argvHasVarFlagFor(["--var", "OTHER_VAR:value"], FLAG_NAME),
    ).toBe(false);
  });

  it("does NOT match an empty argv array", () => {
    expect(argvHasVarFlagFor([], FLAG_NAME)).toBe(false);
  });

  it("does NOT match --var without a trailing value", () => {
    expect(argvHasVarFlagFor(["--var"], FLAG_NAME)).toBe(false);
  });

  it("does NOT match --var followed by an unrelated key", () => {
    expect(
      argvHasVarFlagFor(["--var", "OTHER:value", "--var", "FOO:bar"], FLAG_NAME),
    ).toBe(false);
  });

  it("does NOT match a prefix-similar key (--var PLAN_COUNTER_ENABLED_NEW:...)", () => {
    // The colon-suffix check `${name}:` enforces an exact match before the colon.
    expect(
      argvHasVarFlagFor(
        ["--var", "PLAN_COUNTER_ENABLED_NEW:true"],
        FLAG_NAME,
      ),
    ).toBe(false);
  });

  it("matches --var=KEY:VALUE with whitespace-free single-arg form", () => {
    expect(
      argvHasVarFlagFor(
        ["--var=PLAN_COUNTER_ENABLED:true", "--env=production"],
        FLAG_NAME,
      ),
    ).toBe(true);
  });
});

describe("secretListContains", () => {
  it("matches the secret name in plain-text output (one per line)", () => {
    const stdout = "FOO_SECRET\nPLAN_COUNTER_ENABLED\nBAR_SECRET\n";
    expect(secretListContains(stdout, FLAG_NAME)).toBe(true);
  });

  it("matches the secret name in JSON output", () => {
    const stdout = `[{"name":"PLAN_COUNTER_ENABLED","type":"secret"}]`;
    expect(secretListContains(stdout, FLAG_NAME)).toBe(true);
  });

  it("does NOT match an empty output", () => {
    expect(secretListContains("", FLAG_NAME)).toBe(false);
  });

  it("does NOT match a no-secrets message", () => {
    expect(secretListContains("No secrets defined.\n", FLAG_NAME)).toBe(false);
  });

  it("does NOT match a substring-only similar name (FOO_PLAN_COUNTER_ENABLED_X)", () => {
    // Word-boundary regex enforces exact match. PLAN_COUNTER_ENABLED appears as
    // a substring of FOO_PLAN_COUNTER_ENABLED_X, but the \b anchors prevent a
    // false positive.
    expect(
      secretListContains("FOO_PLAN_COUNTER_ENABLED_X\n", FLAG_NAME),
    ).toBe(false);
  });

  it("matches when the secret name has surrounding non-word chars (quotes, commas)", () => {
    const json = `{"secrets":["OTHER","PLAN_COUNTER_ENABLED","MORE"]}`;
    expect(secretListContains(json, FLAG_NAME)).toBe(true);
  });
});

describe("evaluateGuard", () => {
  it("returns null when both checks pass", () => {
    const result = evaluateGuard(
      ["--env", "production"],
      { status: 0, stdout: "OTHER_SECRET\n", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true" },
    );
    expect(result).toBeNull();
  });

  it("returns null when wranglerVars arg is omitted (backwards-compatible default)", () => {
    // Old callers (tests, legacy scripts) should still be able to call
    // evaluateGuard with only two args — the third defaults to {}.
    const result = evaluateGuard(
      ["--env", "production"],
      { status: 0, stdout: "OTHER_SECRET\n", stderr: "" },
    );
    expect(result).toBeNull();
  });

  it("returns cloud_invariant failure when flag is on but NULLSPEND_CLOUD is missing", () => {
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true" },
    );
    expect(result?.reason).toBe("cloud_invariant");
  });

  it("prioritizes cloud_invariant over every other check (short-circuits before wrangler call)", () => {
    // All three failure conditions are present. cloud_invariant runs FIRST
    // because it's a local-only misconfiguration — no network cost to detect.
    const result = evaluateGuard(
      ["--var", "PLAN_COUNTER_ENABLED:true"],
      { status: 1, stdout: "PLAN_COUNTER_ENABLED\n", stderr: "auth fail" },
      { PLAN_COUNTER_ENABLED: "true" },
    );
    expect(result?.reason).toBe("cloud_invariant");
  });

  it("does NOT trip cloud_invariant when flag is off (pre-flip state)", () => {
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "OTHER_SECRET\n", stderr: "" },
      { PLAN_COUNTER_ENABLED: "false" },
    );
    expect(result).toBeNull();
  });

  it("returns var_flag failure when --var KEY:VALUE is in argv", () => {
    const result = evaluateGuard(
      ["--var", "PLAN_COUNTER_ENABLED:true"],
      { status: 0, stdout: "", stderr: "" },
    );
    expect(result?.reason).toBe("var_flag");
    expect(result?.message).toContain(FLAG_NAME);
    expect(result?.message).toContain("wrangler.jsonc::vars");
  });

  it("returns wrangler_failed when secret list exits non-zero", () => {
    const result = evaluateGuard(
      [],
      { status: 1, stdout: "", stderr: "Authentication required." },
    );
    expect(result?.reason).toBe("wrangler_failed");
    expect(result?.message).toContain("exit 1");
    expect(result?.message).toContain("Authentication required");
  });

  it("returns secret_collision when secret is defined in CF", () => {
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "PLAN_COUNTER_ENABLED\n", stderr: "" },
    );
    expect(result?.reason).toBe("secret_collision");
    expect(result?.message).toContain(FLAG_NAME);
    expect(result?.message).toContain("wrangler secret delete");
  });

  // PR-2e post-flip review: the CLOUD_FLAG_NAME secret-collision vector was
  // missed by the original guard, re-introducing the exact 90-minute outage
  // class that motivated it. Same rigor as PLAN_COUNTER_ENABLED.
  it("returns secret_collision when NULLSPEND_CLOUD is defined as a secret", () => {
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "NULLSPEND_CLOUD\n", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true" },
    );
    expect(result?.reason).toBe("secret_collision");
    expect(result?.message).toContain(CLOUD_FLAG_NAME);
    expect(result?.message).toContain("wrangler secret delete");
  });

  it("returns secret_collision for NULLSPEND_CLOUD even when PLAN_COUNTER_ENABLED secret is absent", () => {
    // Isolation test: NULLSPEND_CLOUD secret collision is its own failure
    // path, not a follow-on effect of the FLAG_NAME check.
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "NULLSPEND_CLOUD\nOTHER_SECRET\n", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true" },
    );
    expect(result?.reason).toBe("secret_collision");
    expect(result?.message).toContain("NULLSPEND_CLOUD");
    // Must NOT surface PLAN_COUNTER_ENABLED since that secret isn't present.
    expect(result?.message).not.toContain(
      "PLAN_COUNTER_ENABLED is defined as a Cloudflare Worker secret",
    );
  });

  it("prioritizes PLAN_COUNTER_ENABLED secret_collision over NULLSPEND_CLOUD when both present", () => {
    // Stable ordering: PLAN_COUNTER_ENABLED is checked first, so its
    // message wins when both secrets exist. Prevents flaky message-match
    // tests.
    const result = evaluateGuard(
      [],
      { status: 0, stdout: "PLAN_COUNTER_ENABLED\nNULLSPEND_CLOUD\n", stderr: "" },
      { PLAN_COUNTER_ENABLED: "true", NULLSPEND_CLOUD: "true" },
    );
    expect(result?.reason).toBe("secret_collision");
    expect(result?.message).toContain(FLAG_NAME);
    expect(result?.message).toContain("wrangler secret delete PLAN_COUNTER_ENABLED");
  });

  it("prioritizes var_flag over secret_collision when both are present", () => {
    // CLI arg check happens before secret-list check; the deploy invocation
    // itself is wrong, regardless of secret state.
    const result = evaluateGuard(
      ["--var", "PLAN_COUNTER_ENABLED:true"],
      { status: 0, stdout: "PLAN_COUNTER_ENABLED\n", stderr: "" },
    );
    expect(result?.reason).toBe("var_flag");
  });

  it("prioritizes wrangler_failed over secret_collision when both are present", () => {
    // wrangler exit code != 0 means we can't trust stdout to be authoritative;
    // surface the wrangler failure first so the operator fixes auth.
    const result = evaluateGuard(
      [],
      { status: 1, stdout: "PLAN_COUNTER_ENABLED\n", stderr: "oops" },
    );
    expect(result?.reason).toBe("wrangler_failed");
  });

  it("returns null when --var is for an unrelated key AND no secret collision", () => {
    const result = evaluateGuard(
      ["--var", "DEBUG:true"],
      { status: 0, stdout: "OTHER_SECRET\n", stderr: "" },
    );
    expect(result).toBeNull();
  });
});

describe("decideDeployExit (codex impl-diff fix — spawn-error / signal / null-status)", () => {
  it("exits with the wrangler status when status is 0 (success)", () => {
    const decision = decideDeployExit({ status: 0 });
    expect(decision.exitCode).toBe(0);
    expect(decision.errorMessage).toBeNull();
  });

  it("exits with the wrangler status when status is non-zero (preserves wrangler exit code)", () => {
    const decision = decideDeployExit({ status: 1 });
    expect(decision.exitCode).toBe(1);
    expect(decision.errorMessage).toBeNull();
  });

  it("exits 1 with error message when spawn error is set (npx not found, etc.)", () => {
    const decision = decideDeployExit({
      status: null,
      error: new Error("spawn npx ENOENT"),
    });
    expect(decision.exitCode).toBe(1);
    expect(decision.errorMessage).toContain("spawn npx ENOENT");
    expect(decision.errorMessage).toContain("failed to launch");
  });

  it("exits 1 with error message when signal kill is set (SIGTERM, Ctrl-C, etc.)", () => {
    const decision = decideDeployExit({
      status: null,
      signal: "SIGTERM",
    });
    expect(decision.exitCode).toBe(1);
    expect(decision.errorMessage).toContain("SIGTERM");
    expect(decision.errorMessage).toContain("killed by signal");
  });

  it("exits 1 (default-deny) when status is null with no error/signal (defensive)", () => {
    const decision = decideDeployExit({ status: null });
    expect(decision.exitCode).toBe(1);
    expect(decision.errorMessage).toContain("null status");
    expect(decision.errorMessage).toContain("default-deny");
  });

  it("prioritizes error over signal when both are set (caller cares which failure mode)", () => {
    const decision = decideDeployExit({
      status: null,
      error: new Error("the actual cause"),
      signal: "SIGTERM", // signal sometimes set as side-effect of error
    });
    expect(decision.exitCode).toBe(1);
    expect(decision.errorMessage).toContain("the actual cause");
    // Don't double-report: the error message takes precedence.
    expect(decision.errorMessage).not.toContain("SIGTERM");
  });

  it("NEVER exits 0 when wrangler did not successfully complete (anti-mask)", () => {
    // Lock-in the anti-mask invariant: any failure mode → status >= 1.
    // The previous bug was `process.exit(deploy.status ?? 0)` which exited
    // 0 on null status. This test prevents regressions.
    expect(decideDeployExit({ status: null }).exitCode).toBe(1);
    expect(decideDeployExit({ status: null, error: new Error("x") }).exitCode).toBe(1);
    expect(decideDeployExit({ status: null, signal: "SIGKILL" }).exitCode).toBe(1);
  });
});
