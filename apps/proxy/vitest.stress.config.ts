import { readFileSync, existsSync } from "fs";
import { defineConfig } from "vitest/config";

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
    include: ["stress-*.test.ts"],
    testTimeout: 300_000,
    // Run stress test files sequentially — they share
    // NULLSPEND_SMOKE_USER_ID budget state and can interfere.
    fileParallelism: false,
    globalSetup: ["./smoke-global-setup.ts"],
  },
});
