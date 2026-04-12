import { defineConfig } from "vitest/config";
import { cloudflarePool, cloudflareTest } from "@cloudflare/vitest-pool-workers";

// Hyperdrive requires a local connection string even when tests don't use it.
// Set the env var wrangler expects so the pool starts without a real Postgres.
process.env.CLOUDFLARE_HYPERDRIVE_LOCAL_CONNECTION_STRING_HYPERDRIVE ??=
  "postgresql://user:pass@localhost:5432/test";

const poolOptions = {
  wrangler: { configPath: "./wrangler.jsonc" },
};

export default defineConfig({
  plugins: [cloudflareTest(poolOptions)],
  test: {
    include: ["src/**/*.do.test.ts"],
    pool: cloudflarePool(poolOptions),
  },
});
