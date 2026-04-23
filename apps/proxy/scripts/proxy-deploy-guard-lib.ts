/**
 * Pure helpers for the pre-deploy guard. Co-located with the entrypoint
 * (proxy-deploy-guard.ts) but separated for unit testability — the
 * entrypoint shells out to wrangler, this module has no side effects.
 *
 * PR-2e Decision #41 + codex R4 #1 / R4 #2: PLAN_COUNTER_ENABLED MUST live
 * in `wrangler.jsonc::vars` only — never via `wrangler secret put` or
 * `wrangler deploy --var KEY:VALUE`. Both vectors silently override
 * config-file vars and would cause split-brain between the deployed
 * worker behavior and the repo source of truth.
 *
 * Intentional limitation: direct `wrangler deploy` invocations (bypassing
 * `pnpm proxy:deploy`) are NOT guarded — we can't intercept what we don't
 * wrap. CLAUDE.md documents this as the convention. Future safety: a CF
 * GitHub Action wrapper or pre-receive hook would close the gap.
 */

export const FLAG_NAME = "PLAN_COUNTER_ENABLED";
export const CLOUD_FLAG_NAME = "NULLSPEND_CLOUD";

/**
 * PR-6b Sub-lane 5: migration-parity check.
 *
 * Static guard against a specific incident class: a committed migration
 * file exists in `drizzle/meta/_journal.json` but the corresponding
 * `drizzle/*.sql` file is missing from the checkout. Causes: stale
 * `git checkout`, accidental `git rm` that missed the meta update, bad
 * merge conflict resolution. The deploy would ship proxy code referencing
 * a schema column that the Worker can't guarantee exists, then either
 * 500s on first use OR silently writes to a missing column.
 *
 * The full prod-parity check (is each migration APPLIED in Supabase?)
 * requires network access to the DB and is documented as a manual
 * `mcp__supabase__list_migrations` step in CLAUDE.md. This is the
 * local-file half of that invariant — cheap + always-on.
 *
 * Real incident (2026-04-13 → 2026-04-16): loop-detection feature shipped
 * with `drizzle/0060_loop_detection.sql` but the migration was never
 * applied to prod. The file was present; the manual Supabase step was
 * skipped. This static check would NOT have caught that (file was
 * present). But a similar class where someone pulls + deletes the file
 * locally before re-deploy WOULD be caught.
 */
export interface DrizzleJournalEntry {
  readonly idx: number;
  readonly tag: string;
}

export interface DrizzleJournal {
  readonly entries: readonly DrizzleJournalEntry[];
}

/**
 * Parse drizzle's `_journal.json` into the minimal shape the guard
 * needs. Returns `null` on any parse failure — caller treats that as
 * "skip the check" (skippable, not deploy-blocking, per the same philosophy
 * as `parseWranglerVars`).
 */
export function parseDrizzleJournal(content: string): DrizzleJournal | null {
  try {
    const parsed = JSON.parse(content) as { entries?: unknown };
    const entries = parsed?.entries;
    if (!Array.isArray(entries)) return null;
    const normalized: DrizzleJournalEntry[] = [];
    for (const raw of entries) {
      if (typeof raw !== "object" || raw === null) continue;
      const entry = raw as Record<string, unknown>;
      const idx = entry.idx;
      const tag = entry.tag;
      if (typeof idx === "number" && typeof tag === "string" && tag.length > 0) {
        normalized.push({ idx, tag });
      }
    }
    return { entries: normalized };
  } catch {
    return null;
  }
}

/**
 * Given a journal + the set of .sql filenames present in `drizzle/`,
 * return a `GuardFailure` iff any journal entry has no matching file.
 *
 * Matching rule: journal entry `tag` must map to a file named `${tag}.sql`
 * in the drizzle/ directory. drizzle-kit uses the tag as the filename stem.
 *
 * Caller reads the journal + lists drizzle/*.sql files and passes both.
 * The function is pure; no filesystem access.
 */
export function evaluateMigrationParity(
  journal: DrizzleJournal | null,
  drizzleSqlFiles: readonly string[],
): GuardFailure | null {
  if (journal === null) {
    // Journal unreadable or malformed — skip the check (skippable, not
    // deploy-blocking). A broken journal is already a louder signal; the
    // deploy will fail downstream with a clearer error.
    return null;
  }
  // Codex P0-1: case-normalize both sides. Windows + macOS ship with
  // case-INSENSITIVE filesystems by default; Linux is case-SENSITIVE.
  // Without normalization, a filename committed as `0070_Overage.sql` on
  // Windows passes parity locally but fails on the Linux deploy. Normalize
  // to lowercase — drizzle-kit emits all-lowercase tags, so this matches
  // the canonical convention.
  const fileSet = new Set(drizzleSqlFiles.map((f) => f.toLowerCase()));
  const missing: string[] = [];
  for (const entry of journal.entries) {
    const expected = `${entry.tag.toLowerCase()}.sql`;
    if (!fileSet.has(expected)) missing.push(`${entry.tag}.sql`);
  }
  if (missing.length === 0) return null;
  return {
    reason: "migration_parity",
    message:
      `REFUSED: ${missing.length} migration file(s) listed in ` +
      `drizzle/meta/_journal.json are missing from drizzle/.\n` +
      `Missing: ${missing.join(", ")}\n\n` +
      `Causes: stale checkout, accidental 'git rm', bad merge resolution.\n` +
      `Fix: 'git status drizzle/' to see what's gone, then restore the files\n` +
      `  ('git checkout HEAD -- drizzle/<file>.sql' or pull the branch fresh).\n\n` +
      `If the intent is to REMOVE a migration, also remove its entry from\n` +
      `drizzle/meta/_journal.json — drizzle-kit rejects orphaned entries.\n\n` +
      `Note: this is the LOCAL-file half of the migration invariant. The\n` +
      `PROD-applied half (is each migration applied to Supabase?) is a\n` +
      `manual 'mcp__supabase__list_migrations' step documented in CLAUDE.md.`,
  };
}

/**
 * Parse `wrangler.jsonc::vars` from raw file content.
 *
 * Caller passes the full file content (as read via fs.readFileSync). This
 * function strips JSONC comments (line + block) and returns the `vars`
 * object as a plain `Record<string, string>` for invariant checks.
 *
 * Keeps the dependency surface zero — no `jsonc-parser` package. Comment
 * stripping is a best-effort regex; it handles the canonical comment
 * forms we use in wrangler.jsonc but is NOT a full JSONC parser. If the
 * file has comments inside string literals (rare for our config), the
 * regex will corrupt them — JSON.parse will then throw, which the caller
 * treats as a parse failure (returns empty vars, skipping the check).
 *
 * Returns `{}` on any parse failure so the invariant check is skippable
 * rather than deploy-blocking on a benign config issue. The guard's job
 * is to catch KNOWN misconfigurations, not to gate deploys on JSONC
 * parsing quirks.
 */
export function parseWranglerVars(fileContent: string): Record<string, string> {
  try {
    const stripped = fileContent
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/[^\n]*/g, "");
    const parsed = JSON.parse(stripped) as { vars?: unknown };
    const vars = parsed?.vars;
    if (typeof vars !== "object" || vars === null) return {};
    const result: Record<string, string> = {};
    for (const [k, v] of Object.entries(vars as Record<string, unknown>)) {
      if (typeof v === "string") result[k] = v;
    }
    return result;
  } catch {
    return {};
  }
}

/**
 * PR-2e post-launch: cloud-invariant check.
 *
 * Caught on 2026-04-20 during live-stack T2 testing: the Sub-PR 20 flag
 * flip shipped `PLAN_COUNTER_ENABLED=true` to production, but enforcement
 * remained silently inactive because `NULLSPEND_CLOUD` was never set. The
 * auth layer (api-key-auth.ts::lookupKeyInDb) short-circuits to the
 * self-hosted identity when `env.NULLSPEND_CLOUD !== "true"`, returning
 * `planLimitBlockAt: null`. That causes `checkBudget` to skip the
 * plan-counter increment entirely, making PLAN_COUNTER_ENABLED a no-op.
 *
 * The launch was inactive for ~90 min before testing caught it.
 *
 * This invariant fails the deploy if PLAN_COUNTER_ENABLED=true is set in
 * vars without NULLSPEND_CLOUD=true alongside it — closing the gap the
 * original guard's PLAN_COUNTER_ENABLED-only scope missed.
 */
export function evaluateCloudInvariant(
  vars: Record<string, string>,
): GuardFailure | null {
  const planCounterEnabled = vars[FLAG_NAME] === "true";
  const nullspendCloud = vars[CLOUD_FLAG_NAME] === "true";
  if (planCounterEnabled && !nullspendCloud) {
    return {
      reason: "cloud_invariant",
      message:
        `REFUSED: ${FLAG_NAME}="true" but ${CLOUD_FLAG_NAME} is not "true" ` +
        `in apps/proxy/wrangler.jsonc::vars.\n` +
        `Without ${CLOUD_FLAG_NAME}="true", api-key-auth.ts short-circuits to the ` +
        `self-hosted identity with planLimitBlockAt:null, which makes ${FLAG_NAME} ` +
        `a SILENT no-op — proxy/MCP paths skip incrementPlanCounter entirely.\n` +
        `Add "${CLOUD_FLAG_NAME}": "true" to wrangler.jsonc::vars and re-deploy.\n` +
        `See commit 5920f00 for the post-launch hotfix that closed this gap on 2026-04-20.`,
    };
  }
  return null;
}

/**
 * Detect whether the deploy CLI args contain a --var override for the
 * given flag name. Catches both `--var KEY:VALUE` (two-arg) and
 * `--var=KEY:VALUE` (single-arg) forms per wrangler 4.x docs.
 *
 * Word-boundary on the colon prevents false positives on prefix-named
 * vars (e.g., `--var PLAN_COUNTER_ENABLED_NEW:foo` is NOT a match for
 * PLAN_COUNTER_ENABLED).
 */
export function argvHasVarFlagFor(argv: readonly string[], name: string): boolean {
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    // Two-arg form: --var KEY:VALUE
    if (arg === "--var" && argv[i + 1]?.startsWith(`${name}:`)) return true;
    // Single-arg form: --var=KEY:VALUE
    if (arg.startsWith(`--var=${name}:`)) return true;
  }
  return false;
}

/**
 * Detect whether the wrangler-secret-list output contains the named
 * secret as a whole word. Robust to plain-text and JSON output formats.
 *
 * Word-boundary regex prevents substring false positives. PLAN_COUNTER_ENABLED
 * is unique enough that this is belt-and-suspenders, but the helper is
 * generic so future callers (other guarded flags) get the same protection.
 */
export function secretListContains(stdout: string, name: string): boolean {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`\\b${escaped}\\b`);
  return re.test(stdout);
}

export interface GuardFailure {
  readonly reason:
    | "var_flag"
    | "secret_collision"
    | "wrangler_failed"
    | "cloud_invariant"
    | "migration_parity";
  readonly message: string;
}

export interface SpawnResult {
  readonly status: number | null;
  readonly error?: Error;
  readonly signal?: string | null;
}

export interface DeployExitDecision {
  readonly exitCode: number;
  readonly errorMessage: string | null;
}

/**
 * Decide the exit code + error message for the wrapper based on the
 * spawnSync result of `npx wrangler deploy`.
 *
 * Codex impl-diff review caught: the previous version exited with
 * `process.exit(deploy.status ?? 0)`. If spawnSync FAILED before exec
 * (e.g., npx not found, ENOENT), `status` is null and `error` is set,
 * but the wrapper would exit 0 and silently mask the failed deploy.
 *
 * Fixed contract:
 *   - spawn error (npx missing, etc.)        → exit 1 + error message
 *   - signal kill (Ctrl-C, SIGTERM, etc.)    → exit 1 + signal name
 *   - status === 0                           → exit 0 (success)
 *   - status non-zero                        → exit that status (preserves wrangler exit code)
 *   - status null with no error/signal       → exit 1 (defensive default-deny)
 */
export function decideDeployExit(result: SpawnResult): DeployExitDecision {
  if (result.error) {
    return {
      exitCode: 1,
      errorMessage: `wrangler deploy failed to launch: ${result.error.message}`,
    };
  }
  if (result.signal) {
    return {
      exitCode: 1,
      errorMessage: `wrangler deploy killed by signal ${result.signal}`,
    };
  }
  if (result.status === null) {
    return {
      exitCode: 1,
      errorMessage:
        "wrangler deploy exited with null status and no error/signal. Treating as failure (default-deny).",
    };
  }
  return { exitCode: result.status, errorMessage: null };
}

/**
 * Pure decision function: given CLI args + wrangler secret-list result,
 * decide whether the deploy should proceed.
 *
 * Returns null on pass; GuardFailure with a human-readable message on fail.
 * The entrypoint script translates GuardFailure into stderr + exit 1.
 */
export function evaluateGuard(
  passthroughArgv: readonly string[],
  wranglerSecretList: { status: number; stdout: string; stderr: string },
  wranglerVars: Record<string, string> = {},
  migrationParity?: {
    journal: DrizzleJournal | null;
    drizzleSqlFiles: readonly string[];
  },
): GuardFailure | null {
  // PR-6b Sub-lane 5: migration-parity check runs FIRST. A missing SQL
  // file is a local-only issue that must fail before any network call.
  // Caller omits `migrationParity` in tests that don't want to exercise
  // this path — the function treats that as skip, matching the null-journal
  // behavior.
  if (migrationParity !== undefined) {
    const parityFailure = evaluateMigrationParity(
      migrationParity.journal,
      migrationParity.drizzleSqlFiles,
    );
    if (parityFailure !== null) return parityFailure;
  }

  // PR-2e post-launch hotfix: cloud-invariant check runs second so it
  // short-circuits before the wrangler-secret-list network call. A missing
  // NULLSPEND_CLOUD is a local-only misconfiguration that should fail fast.
  const cloudInvariant = evaluateCloudInvariant(wranglerVars);
  if (cloudInvariant !== null) return cloudInvariant;

  if (argvHasVarFlagFor(passthroughArgv, FLAG_NAME)) {
    return {
      reason: "var_flag",
      message:
        `REFUSED: --var ${FLAG_NAME}:... in CLI args.\n` +
        `${FLAG_NAME} must live in apps/proxy/wrangler.jsonc::vars only ` +
        `(PR-2e Decision #41 + codex R4 #1).\n` +
        `Edit the file and re-run pnpm proxy:deploy.`,
    };
  }

  if (wranglerSecretList.status !== 0) {
    return {
      reason: "wrangler_failed",
      message:
        `REFUSED: 'wrangler secret list' failed (exit ${wranglerSecretList.status}).\n` +
        `Cannot verify ${FLAG_NAME} secret-collision-free state.\n` +
        `Fix wrangler auth or environment, then re-run.\n` +
        `--- wrangler stderr ---\n${wranglerSecretList.stderr}`,
    };
  }

  if (secretListContains(wranglerSecretList.stdout, FLAG_NAME)) {
    return {
      reason: "secret_collision",
      message:
        `REFUSED: ${FLAG_NAME} is defined as a Cloudflare Worker secret.\n` +
        `${FLAG_NAME} must live in apps/proxy/wrangler.jsonc::vars only ` +
        `(PR-2e Decision #41 + codex R4 #1).\n` +
        `Run: npx wrangler secret delete ${FLAG_NAME}\n` +
        `Then edit apps/proxy/wrangler.jsonc and re-run pnpm proxy:deploy.`,
    };
  }

  // PR-2e post-flip review: same secret-override vector for NULLSPEND_CLOUD.
  // Cloudflare Worker secrets override config-file vars at runtime. If an
  // operator runs `wrangler secret put NULLSPEND_CLOUD` with ANY value
  // (even "true"), the config-file NULLSPEND_CLOUD="true" is masked. If
  // the secret value is anything other than exactly "true" (e.g., "false",
  // "0", or an accidental typo), the auth layer short-circuits to the
  // self-hosted identity with planLimitBlockAt=null — making
  // PLAN_COUNTER_ENABLED a silent no-op. Same failure class as the
  // 2026-04-20 90-min enforcement outage, different vector. Refuse the
  // deploy if the secret exists at all, same as FLAG_NAME.
  if (secretListContains(wranglerSecretList.stdout, CLOUD_FLAG_NAME)) {
    return {
      reason: "secret_collision",
      message:
        `REFUSED: ${CLOUD_FLAG_NAME} is defined as a Cloudflare Worker secret.\n` +
        `${CLOUD_FLAG_NAME} must live in apps/proxy/wrangler.jsonc::vars only — ` +
        `a secret with any value (including "true") silently overrides the ` +
        `config-file var, and any value that's not exactly "true" makes ` +
        `${FLAG_NAME} a silent no-op (2026-04-20 incident class).\n` +
        `Run: npx wrangler secret delete ${CLOUD_FLAG_NAME}\n` +
        `Then verify apps/proxy/wrangler.jsonc has ${CLOUD_FLAG_NAME}="true" and re-run pnpm proxy:deploy.`,
    };
  }

  return null;
}
