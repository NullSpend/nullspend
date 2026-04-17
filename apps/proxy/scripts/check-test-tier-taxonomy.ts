#!/usr/bin/env tsx
/**
 * Enforces the test tier taxonomy for smoke files. Scans
 * apps/proxy/smoke-*.test.ts for patterns that belong in a stress or
 * contract tier and fails CI when any are found.
 *
 * Rules (scoped to smoke files only — stress files are exempt):
 *   1. describe/it names containing "concurrent" (rule: concurrent-name)
 *   2. `[load]` console tags (rule: load-tag)
 *   3. Latency distribution assertions (rule: latency-assertion)
 *   4. Promise.all with 5+ fetch() calls (rule: promise-all-fetches)
 *
 * Remediation when this fails: move the offending test to
 * stress-*.test.ts. See docs/internal/test-tier-taxonomy.md.
 *
 * Escape hatch: add `// tier-check-allow: <rule>` on the same line or
 * the line immediately above a flagged construct to grandfather it.
 * Use sparingly — each escape should reference why the rule doesn't
 * apply to that specific test (e.g. isolation test at low concurrency
 * where the "concurrent" keyword is descriptive of the technique, not
 * the scale).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

const PROXY_DIR = join(__dirname, "..");
const REPO_ROOT = join(__dirname, "..", "..", "..");

interface Violation {
  file: string;
  line: number;
  rule: string;
  snippet: string;
  suggestion: string;
}

function findSmokeFiles(): string[] {
  return readdirSync(PROXY_DIR)
    .filter((f) => /^smoke-.*\.test\.ts$/.test(f))
    .map((f) => join(PROXY_DIR, f));
}

/** True when the reported line or the line above carries a matching
 *  `// tier-check-allow: <rule>` escape-hatch comment. Allowlist tokens
 *  are comma-separated: `// tier-check-allow: concurrent-name, latency-assertion`. */
function hasAllowComment(lines: string[], lineIdx: number, rule: string): boolean {
  const re = /\/\/\s*tier-check-allow\s*:\s*([^\n]+)/;
  const candidates = [lines[lineIdx], lines[lineIdx - 1] ?? ""];
  for (const candidate of candidates) {
    const match = candidate.match(re);
    if (!match) continue;
    const tokens = match[1].split(/[,\s]+/).map((t) => t.trim()).filter(Boolean);
    if (tokens.includes(rule) || tokens.includes("*")) return true;
  }
  return false;
}

function checkConcurrentNames(file: string, lines: string[]): Violation[] {
  const rule = "concurrent-name";
  const violations: Violation[] = [];
  const re = /(?:describe|it|test)\s*\(\s*["'`][^"'`]*concurrent/i;
  for (let i = 0; i < lines.length; i++) {
    if (!re.test(lines[i])) continue;
    if (hasAllowComment(lines, i, rule)) continue;
    violations.push({
      file,
      line: i + 1,
      rule,
      snippet: lines[i].trim(),
      suggestion: "Move to stress-*.test.ts — 'concurrent' in test name signals stress tier",
    });
  }
  return violations;
}

function checkLoadTag(file: string, lines: string[]): Violation[] {
  const rule = "load-tag";
  const violations: Violation[] = [];
  const re = /console\.\w+\s*\([^)]*\[load\]/;
  for (let i = 0; i < lines.length; i++) {
    if (!re.test(lines[i])) continue;
    if (hasAllowComment(lines, i, rule)) continue;
    violations.push({
      file,
      line: i + 1,
      rule,
      snippet: lines[i].trim(),
      suggestion: "Move to stress-*.test.ts — [load] console tag is a load-test smell",
    });
  }
  return violations;
}

function checkLatencyAssertions(file: string, lines: string[]): Violation[] {
  const rule = "latency-assertion";
  const violations: Violation[] = [];
  // Same-line combo: p50/p95/p99 variable + toBeLessThan. Common forms:
  //   expect(p95).toBeLessThan(500)
  //   expect(latencies.p99).toBeLessThanOrEqual(1000)
  const re = /\bp(?:50|95|99)\b[\s\S]{0,80}?\.toBeLessThan(?:OrEqual)?\s*\(/;
  for (let i = 0; i < lines.length; i++) {
    if (!re.test(lines[i])) continue;
    if (hasAllowComment(lines, i, rule)) continue;
    violations.push({
      file,
      line: i + 1,
      rule,
      snippet: lines[i].trim(),
      suggestion: "Move to stress-*.test.ts — latency distribution assertions belong in performance tier",
    });
  }
  return violations;
}

function checkPromiseAllFetches(file: string, lines: string[]): Violation[] {
  const rule = "promise-all-fetches";
  const violations: Violation[] = [];
  const CONCURRENT_FETCH_THRESHOLD = 5;
  for (let i = 0; i < lines.length; i++) {
    if (!/Promise\.all\s*\(\s*\[/.test(lines[i])) continue;
    // Scan forward until we see the closing `])` or a blank-separator that
    // probably indicates we've exited the array literal. Count `fetch(`.
    let fetchCount = 0;
    let j = i;
    let closed = false;
    for (; j < lines.length && j - i < 60; j++) {
      // Count each fetch( occurrence (could be >1 per line but rare in practice)
      const matches = lines[j].match(/\bfetch\s*\(/g);
      if (matches) fetchCount += matches.length;
      if (/\]\s*\)/.test(lines[j]) && j > i) {
        closed = true;
        break;
      }
    }
    if (!closed || fetchCount < CONCURRENT_FETCH_THRESHOLD) continue;
    if (hasAllowComment(lines, i, rule)) continue;
    violations.push({
      file,
      line: i + 1,
      rule,
      snippet: `Promise.all with ${fetchCount} fetch calls (lines ${i + 1}-${j + 1})`,
      suggestion: "Move to stress-*.test.ts — 5+ concurrent fetches saturate tier-1 rate limits",
    });
  }
  return violations;
}

function main(): void {
  const files = findSmokeFiles();
  if (files.length === 0) {
    console.error("::error::No smoke files found — did the directory layout change?");
    process.exit(1);
  }

  const allViolations: Violation[] = [];
  for (const abs of files) {
    const lines = readFileSync(abs, "utf-8").split(/\r?\n/);
    allViolations.push(
      ...checkConcurrentNames(abs, lines),
      ...checkLoadTag(abs, lines),
      ...checkLatencyAssertions(abs, lines),
      ...checkPromiseAllFetches(abs, lines),
    );
  }

  console.log(`Test tier taxonomy check — scanned ${files.length} smoke file(s)`);
  if (allViolations.length === 0) {
    console.log("  All smoke files pass.");
    return;
  }

  const grouped = new Map<string, Violation[]>();
  for (const v of allViolations) {
    const key = relative(REPO_ROOT, v.file).replace(/\\/g, "/");
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(v);
  }

  for (const [file, vs] of grouped) {
    for (const v of vs) {
      console.error(`::error file=${file},line=${v.line}::[${v.rule}] ${v.snippet} — ${v.suggestion}`);
    }
  }
  console.error(
    `\n${allViolations.length} tier taxonomy violation(s) across ${grouped.size} file(s). ` +
      `See docs/internal/test-tier-taxonomy.md.`,
  );
  process.exit(1);
}

main();
