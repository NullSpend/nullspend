/**
 * Wave 3 Phase 3 smoke gate wrapper.
 *
 * Checks SMOKE_LIVE=1 before invoking vitest against the smoke
 * config. Without the env var, prints a help message to stderr and
 * exits 0 (skipped tier, not a failure).
 *
 * Defense-in-depth: vitest.smoke.config.ts also returns include: []
 * when the env is missing, so direct `npx vitest` invocations that
 * bypass this wrapper still short-circuit cleanly.
 *
 * See docs/internal/wave-3-phase-3-smoke-gating-plan-20260416.md D1.
 */
import { spawnSync } from "node:child_process";

if (process.env.SMOKE_LIVE !== "1") {
  const msg = [
    "",
    "Smoke suite is manual/nightly only (Wave 3 Phase 3).",
    "",
    "To run: SMOKE_LIVE=1 pnpm proxy:test:smoke",
    "",
    "Smoke tests hit live OpenAI/Anthropic APIs, cost money, and",
    "can flake under tier-1 rate limits. Contract tests in",
    "apps/proxy/src/__tests__/contract-*.test.ts cover PR-tier",
    "assertions without hitting live providers.",
    "",
    "See TESTING.md section 'Live Smoke (manual/nightly)' for",
    "details and the rate-limit retry config that applies when the",
    "gate is open.",
    "",
  ].join("\n");
  process.stderr.write(msg);
  process.exit(0);
}

// Gate is open. Invoke vitest with the smoke config.
// Forward any extra args passed to `pnpm test:smoke <args>`.
const extraArgs = process.argv.slice(2);
const res = spawnSync(
  "npx",
  ["vitest", "run", "--config", "vitest.smoke.config.ts", ...extraArgs],
  {
    stdio: "inherit",
    shell: true,
  },
);
process.exit(res.status ?? 0);
