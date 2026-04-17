import { readFileSync, existsSync } from "fs";
import { defineConfig } from "vitest/config";

// Wave 3 Phase 3: Live smoke is opt-in manual/nightly tier. Require
// SMOKE_LIVE=1 to actually run tests. Without it, `include: []`
// produces zero matching tests and vitest exits 0 with "no tests
// found." The `pnpm test:smoke` wrapper ALSO prints a help message
// and exits 0 before invocation — this config-level gate is
// defense-in-depth for direct `npx vitest --config` invocations.
// See docs/internal/wave-3-phase-3-smoke-gating-plan-20260416.md D1.
const smokeGateOpen = process.env.SMOKE_LIVE === "1";

// Load .env.smoke into process.env before tests run
const envPath = ".env.smoke";
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      const val = trimmed.slice(eqIdx + 1).trim();
      if (!process.env[key]) process.env[key] = val;
    }
  }
}

export default defineConfig({
  test: {
    // Empty include when gate is closed — vitest finds no tests, exits 0.
    // passWithNoTests ensures exit code 0 when include is empty
    // (default is exit 1 on "no tests found", which would misreport the
    // closed-gate case as a failure).
    include: smokeGateOpen ? ["smoke-*.test.ts"] : [],
    passWithNoTests: true,
    testTimeout: 60_000,
    // Run smoke test files sequentially — budget tests share
    // NULLSPEND_SMOKE_USER_ID and interfere if run in parallel.
    fileParallelism: false,
    // Only attach globalSetup when the gate is open, so a closed-gate
    // invocation doesn't try to boot the smoke global environment.
    globalSetup: smokeGateOpen ? ["./smoke-global-setup.ts"] : [],
    // Retry rate-limit and transient-network flake up to 3 times with
    // a 1s delay. Condition matches against error.message. Patterns:
    //   - 429: any "expected 429 to be X" assertion failure
    //   - 502 / 503 / 504 / bad gateway / service unavailable /
    //     gateway timeout: transient upstream edge failures (OpenAI /
    //     Anthropic / Cloudflare intermediaries). Added 2026-04-17
    //     after post-Wave-3 SMOKE_LIVE run permanently failed
    //     smoke-cost-e2e.test.ts:196 on a transient 502. Persistent
    //     5xx still surfaces because all 3 retries will also see it.
    //   - ECONNRESET / ETIMEDOUT / "socket hang up": undici transport
    //     errors from tier-1 rate limiting or brief upstream blips
    //   - rate[_-]?limit: body-text-forwarded rate limit keywords
    //     (when a test throws with response body included)
    // See D2 in the Phase 3 plan doc. Config-level condition must be
    // RegExp (Vitest serializes config to workers; functions only work
    // in test files).
    retry: {
      count: 3,
      delay: 1000,
      condition: /429|50[234]|bad[_ -]?gateway|service[_ -]?unavailable|gateway[_ -]?timeout|ECONNRESET|ETIMEDOUT|socket hang up|rate[_-]?limit/i,
    },
  },
});
