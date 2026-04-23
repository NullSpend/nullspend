/**
 * Pre-deploy guard entrypoint for `pnpm proxy:deploy`.
 *
 * See proxy-deploy-guard-lib.ts for the pure decision logic + reasoning.
 *
 * Flow:
 *   1. Forward extra args from `pnpm proxy:deploy <args>` as deploy passthrough.
 *   2. Run `npx wrangler secret list` to enumerate currently-defined secrets.
 *   3. Pass both into evaluateGuard() to decide pass/fail.
 *   4. On fail: write reason to stderr, exit 1.
 *   5. On pass: spawn `npx wrangler deploy <passthrough>` with inherited stdio.
 */

import { spawnSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  decideDeployExit,
  evaluateGuard,
  parseDrizzleJournal,
  parseWranglerVars,
} from "./proxy-deploy-guard-lib.js";

function main(): void {
  const passthrough = process.argv.slice(2);

  // PR-2e post-launch hotfix: read wrangler.jsonc vars so the guard can
  // enforce the PLAN_COUNTER_ENABLED ↔ NULLSPEND_CLOUD invariant.
  // Script runs from apps/proxy/; wrangler.jsonc is a sibling of scripts/.
  let wranglerVars: Record<string, string> = {};
  try {
    const wranglerPath = resolve(process.cwd(), "wrangler.jsonc");
    wranglerVars = parseWranglerVars(readFileSync(wranglerPath, "utf-8"));
  } catch {
    // If wrangler.jsonc is unreadable, the invariant check is skipped.
    // wrangler itself will fail the deploy in that case with a clearer error.
  }

  // PR-6b Sub-lane 5: read drizzle/meta/_journal.json + the list of
  // .sql files in drizzle/ so the guard can detect a stale checkout
  // where a migration file was deleted but the journal entry stayed.
  // Script runs from apps/proxy/; repo root is two levels up.
  const repoRoot = resolve(process.cwd(), "..", "..");
  let migrationParity: Parameters<typeof evaluateGuard>[3];
  try {
    const journalPath = resolve(repoRoot, "drizzle", "meta", "_journal.json");
    const drizzleDir = resolve(repoRoot, "drizzle");
    const journal = parseDrizzleJournal(readFileSync(journalPath, "utf-8"));
    const drizzleSqlFiles = readdirSync(drizzleDir).filter((f) =>
      f.endsWith(".sql"),
    );
    migrationParity = { journal, drizzleSqlFiles };
  } catch {
    // Journal or drizzle/ unreadable — skip the check. Matches the
    // wrangler.jsonc skip philosophy: guard catches KNOWN misconfigurations,
    // not filesystem quirks.
    migrationParity = undefined;
  }

  const list = spawnSync("npx", ["wrangler", "secret", "list"], {
    encoding: "utf8",
    shell: true,
  });

  const failure = evaluateGuard(
    passthrough,
    {
      status: list.status ?? -1,
      stdout: list.stdout ?? "",
      stderr: list.stderr ?? "",
    },
    wranglerVars,
    migrationParity,
  );

  if (failure !== null) {
    process.stderr.write(`\n[proxy-deploy-guard] ${failure.message}\n\n`);
    process.exit(1);
  }

  // Guard cleared. Exec wrangler deploy.
  const deploy = spawnSync("npx", ["wrangler", "deploy", ...passthrough], {
    stdio: "inherit",
    shell: true,
  });

  // Codex impl-diff review caught: the previous version exited with
  // `process.exit(deploy.status ?? 0)`. If spawnSync failed before exec
  // (npx ENOENT, signal kill, etc.) the wrapper would exit 0 and silently
  // mask the failed deploy. decideDeployExit() handles all 4 outcomes
  // (error, signal, null status, real status) with explicit exit codes.
  const decision = decideDeployExit({
    status: deploy.status,
    error: deploy.error,
    signal: deploy.signal,
  });
  if (decision.errorMessage !== null) {
    process.stderr.write(`\n[proxy-deploy-guard] ${decision.errorMessage}\n\n`);
  }
  process.exit(decision.exitCode);
}

main();
