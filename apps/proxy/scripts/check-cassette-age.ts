#!/usr/bin/env tsx
/**
 * Fail if any MSW cassette in src/__tests__/fixtures/cassettes/ was last
 * committed more than MAX_CASSETTE_AGE_DAYS ago. Guards against silent
 * provider shape drift — cassettes are frozen-in-time request/response
 * snapshots, so if they go stale the contract tests can pass against a
 * shape that prod no longer emits.
 *
 * Uses `git log -1 --format=%ct` for the file's last-commit time rather
 * than `fs.stat` mtime. `actions/checkout@v4` sets mtimes to checkout
 * time, so stat would always show "fresh" in CI and this check would be
 * useless there.
 *
 * Remediation when this fails: `cd apps/proxy && pnpm smoke:record`.
 * Requires real OpenAI + Anthropic API keys (~$0.0001 total).
 */
import { execSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { join, relative } from "node:path";

const MAX_CASSETTE_AGE_DAYS = 90;
const CASSETTE_DIR = join(__dirname, "..", "src", "__tests__", "fixtures", "cassettes");
const REPO_ROOT = join(__dirname, "..", "..", "..");

function lastCommitEpoch(absPath: string): number | null {
  const rel = relative(REPO_ROOT, absPath).replace(/\\/g, "/");
  const out = execSync(`git log -1 --format=%ct -- "${rel}"`, {
    cwd: REPO_ROOT,
    encoding: "utf8",
  }).trim();
  if (!out) return null;
  const epoch = Number(out);
  return Number.isFinite(epoch) ? epoch : null;
}

function main(): void {
  const files = readdirSync(CASSETTE_DIR).filter((f) => f.endsWith(".json"));
  if (files.length === 0) {
    console.error(`::error::No cassette files found in ${CASSETTE_DIR}`);
    process.exit(1);
  }

  const nowEpoch = Math.floor(Date.now() / 1000);
  const stale: Array<{ file: string; ageDays: number }> = [];
  const report: Array<{ file: string; ageDays: number }> = [];

  for (const file of files) {
    const abs = join(CASSETTE_DIR, file);
    const epoch = lastCommitEpoch(abs);
    if (epoch === null) {
      console.error(`::error::Could not determine last-commit time for ${file}. Is it committed?`);
      process.exit(1);
    }
    const ageDays = Math.floor((nowEpoch - epoch) / 86400);
    report.push({ file, ageDays });
    if (ageDays > MAX_CASSETTE_AGE_DAYS) stale.push({ file, ageDays });
  }

  // Always print the status so humans can see headroom before the threshold
  const width = Math.max(...report.map((r) => r.file.length));
  console.log(`Cassette freshness (threshold: ${MAX_CASSETTE_AGE_DAYS} days)`);
  for (const { file, ageDays } of report) {
    const marker = ageDays > MAX_CASSETTE_AGE_DAYS ? "STALE" : "ok";
    console.log(`  ${file.padEnd(width)}  ${String(ageDays).padStart(4)}d  ${marker}`);
  }

  if (stale.length > 0) {
    const names = stale.map((s) => `${s.file} (${s.ageDays}d)`).join(", ");
    console.error(
      `::error::${stale.length} cassette(s) older than ${MAX_CASSETTE_AGE_DAYS} days: ${names}. ` +
        `Refresh with: cd apps/proxy && pnpm smoke:record`,
    );
    process.exit(1);
  }
}

main();
