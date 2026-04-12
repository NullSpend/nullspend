/**
 * One-shot data repair for cost_events JSONB columns affected by the
 * cost-logger.ts double-encoding bug (regression introduced in commit
 * 3012a56 — "refactor: migrate cost-logger from Drizzle to raw SQL").
 *
 * Bug: cost-logger.ts called `JSON.stringify(value)` before passing to
 * postgres.js for JSONB columns, which caused postgres.js to send the
 * value as a STRING parameter, which Postgres stored as a JSONB string
 * scalar (e.g., `'"{\"customer\":\"acme\"}"'`) instead of a JSONB object.
 *
 * Affected columns:
 *   - tags
 *   - cost_breakdown
 *   - tool_calls_requested
 *
 * Affected readers (silently broken until fixed):
 *   - lib/cost-events/aggregate-cost-events.ts (tag filtering, group-by, key extraction)
 *   - lib/cost-events/list-cost-events.ts (tag containment filter)
 *   - lib/margins/auto-match.ts (customer extraction from tags)
 *
 * This script unwraps the broken JSONB string scalars back into objects
 * by parsing them via `(col#>>'{}')::jsonb`. Idempotent — running it
 * multiple times is safe (it only touches rows where jsonb_typeof = 'string').
 *
 * Usage:
 *   pnpm jsonb:repair                        # dry-run (count only)
 *   CLEANUP_CONFIRM=yes pnpm jsonb:repair    # actually update
 *
 * Run AFTER deploying the proxy fix to cost-logger.ts. Without the fix,
 * the proxy will keep writing broken rows alongside the repaired ones.
 */
import { existsSync, readFileSync } from "node:fs";
import postgres from "postgres";

function loadEnvSmoke(): void {
  const path = ".env.smoke";
  if (!existsSync(path)) return;
  for (const line of readFileSync(path, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      let val = trimmed.slice(eqIdx + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (!process.env[key]) process.env[key] = val;
    }
  }
}

async function main(): Promise<void> {
  loadEnvSmoke();
  const dbUrl = process.env.DATABASE_URL;
  if (!dbUrl) {
    console.error("DATABASE_URL not set (check .env.smoke or process.env).");
    process.exit(1);
  }

  const dryRun = process.env.CLEANUP_CONFIRM !== "yes";
  if (dryRun) {
    console.log("DRY RUN MODE — counts only, no updates.");
    console.log("Pass CLEANUP_CONFIRM=yes to actually repair rows.\n");
  } else {
    console.log("REPAIR MODE — broken JSONB string scalars will be unwrapped to objects.\n");
  }

  const sql = postgres(dbUrl, { max: 1, idle_timeout: 10 });
  try {
    // 1. Count broken rows in each column. Scoped to the two known callers
    // of the buggy cost-logger.ts: 'proxy' (apps/proxy/src/routes/*.ts) and
    // 'mcp' (apps/proxy/src/routes/mcp.ts). SDK direct ingest uses
    // source='api' via the dashboard's Drizzle path which serializes jsonb
    // correctly and is unaffected. Scoping prevents accidentally mutating
    // any non-cost-logger row that legitimately stores a JSON string.
    const [tagsBroken] = await sql<{ count: string }[]>`
      SELECT COUNT(*)::text AS count FROM cost_events
      WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(tags) = 'string'
    `;
    const [cbBroken] = await sql<{ count: string }[]>`
      SELECT COUNT(*)::text AS count FROM cost_events
      WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(cost_breakdown) = 'string'
    `;
    const [tcrBroken] = await sql<{ count: string }[]>`
      SELECT COUNT(*)::text AS count FROM cost_events
      WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(tool_calls_requested) = 'string'
    `;

    console.log(`Broken rows by column (source in proxy/mcp):`);
    console.log(`  tags:                 ${tagsBroken.count}`);
    console.log(`  cost_breakdown:       ${cbBroken.count}`);
    console.log(`  tool_calls_requested: ${tcrBroken.count}`);
    console.log();

    if (dryRun) {
      console.log("Dry run complete. Re-run with CLEANUP_CONFIRM=yes to repair.");
      return;
    }

    // 2. Repair each column inside a single transaction. If any UPDATE
    // fails (e.g., a string is not valid JSON and `(col#>>'{}')::jsonb`
    // raises), the entire batch rolls back so the dataset stays consistent.
    // The `(col#>>'{}')::jsonb` pattern unwraps a JSONB string scalar by
    // extracting its underlying text and re-parsing as JSONB, producing
    // the original object.
    await sql.begin(async (tx) => {
      const tagsRepair = await tx`
        UPDATE cost_events
        SET tags = (tags#>>'{}')::jsonb
        WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(tags) = 'string'
      `;
      console.log(`tags repaired: ${tagsRepair.count} rows`);

      const cbRepair = await tx`
        UPDATE cost_events
        SET cost_breakdown = (cost_breakdown#>>'{}')::jsonb
        WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(cost_breakdown) = 'string'
      `;
      console.log(`cost_breakdown repaired: ${cbRepair.count} rows`);

      const tcrRepair = await tx`
        UPDATE cost_events
        SET tool_calls_requested = (tool_calls_requested#>>'{}')::jsonb
        WHERE source IN ('proxy', 'mcp') AND jsonb_typeof(tool_calls_requested) = 'string'
      `;
      console.log(`tool_calls_requested repaired: ${tcrRepair.count} rows`);
    });

    console.log("\nRepair complete (committed atomically). Verify with:");
    console.log("  SELECT jsonb_typeof(tags), COUNT(*) FROM cost_events WHERE source IN ('proxy','mcp') GROUP BY 1;");
  } finally {
    await sql.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
