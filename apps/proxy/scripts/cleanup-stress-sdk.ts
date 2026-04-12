/**
 * Crash-recovery cleanup for `stress-sdk-features.test.ts`.
 *
 * Deletes any leftover stress-test fixtures across `cost_events`, `budgets`,
 * and `api_keys` so a future run starts from a clean slate. Matches rows
 * attributed to the stress-test isolation user pattern (`stress-sdk-%`).
 *
 * Safety:
 *   - Defaults to DRY-RUN: prints counts, does nothing. Pass CLEANUP_CONFIRM=yes
 *     to actually delete rows.
 *   - Blast radius: the `stress-sdk-` prefix is a convention. If any non-test
 *     data uses this prefix, it WILL be deleted. Only run against DBs where
 *     the stress test is the sole user of that prefix.
 *
 * Usage:
 *   pnpm stress:cleanup                     # dry-run (count only, no delete)
 *   CLEANUP_CONFIRM=yes pnpm stress:cleanup # actually delete
 *
 * Reads DATABASE_URL from .env.smoke (or process.env). Requires postgres.js.
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
      // Strip paired surrounding quotes so `DATABASE_URL="postgres://..."`
      // doesn't leak a leading `"` into the postgres.js connection string.
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
  const smokeKeyId = process.env.NULLSPEND_SMOKE_KEY_ID;
  if (!smokeKeyId) {
    console.error("NULLSPEND_SMOKE_KEY_ID not set — required to scope cleanup to the smoke org.");
    process.exit(1);
  }

  const dryRun = process.env.CLEANUP_CONFIRM !== "yes";
  if (dryRun) {
    console.log("DRY RUN MODE — counts only, no deletes.");
    console.log("Pass CLEANUP_CONFIRM=yes to actually delete rows.\n");
  } else {
    console.log("DESTRUCTIVE MODE — rows matching stress-sdk-% in the smoke org will be deleted.\n");
  }

  const sql = postgres(dbUrl, { max: 1, idle_timeout: 10 });
  try {
    // Resolve the smoke org_id to scope all DELETE/COUNT operations.
    // Without this scoping, the script would touch stress-sdk-% rows in
    // every org in the database — unbounded blast radius.
    const [keyRow] = await sql<{ org_id: string | null }[]>`
      SELECT org_id FROM api_keys WHERE id = ${smokeKeyId}
    `;
    if (!keyRow?.org_id) {
      console.error(`Smoke key ${smokeKeyId} not found or missing org_id. Aborting.`);
      process.exit(1);
    }
    const orgId = keyRow.org_id;
    console.log(`Scoping cleanup to org_id=${orgId}\n`);

    // 1. cost_events — both attribution paths, scoped by org_id.
    if (dryRun) {
      const [ebu] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM cost_events
        WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      const [ebc] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM cost_events
        WHERE org_id = ${orgId} AND customer_id LIKE 'stress-sdk-%'
      `;
      console.log(`cost_events WOULD delete: ${ebu.count} by user_id, ${ebc.count} by customer_id`);
    } else {
      const eventsByUser = await sql`
        DELETE FROM cost_events
        WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      const eventsByCustomer = await sql`
        DELETE FROM cost_events
        WHERE org_id = ${orgId} AND customer_id LIKE 'stress-sdk-%'
      `;
      console.log(`cost_events deleted: ${eventsByUser.count} by user_id, ${eventsByCustomer.count} by customer_id`);
    }

    // 2. budgets — narrow patterns to the EXACT shapes the test creates:
    //   - user/customer/api_key entities: entity_id starts with 'stress-sdk-'
    //   - tag entities: entity_id starts with '<key>=stress-sdk-' (e.g., 'plan=stress-sdk-...')
    // The previous '%stress-sdk-%' substring was too greedy and could match
    // legitimate non-test rows that happened to contain the substring.
    if (dryRun) {
      const [bbe] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM budgets
        WHERE org_id = ${orgId}
          AND (entity_id LIKE 'stress-sdk-%' OR entity_id LIKE '%=stress-sdk-%')
      `;
      const [bbu] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM budgets
        WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      console.log(`budgets WOULD delete: ${bbe.count} by entity_id, ${bbu.count} by user_id`);
    } else {
      const budgetsByEntity = await sql`
        DELETE FROM budgets
        WHERE org_id = ${orgId}
          AND (entity_id LIKE 'stress-sdk-%' OR entity_id LIKE '%=stress-sdk-%')
      `;
      const budgetsByUser = await sql`
        DELETE FROM budgets
        WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      console.log(`budgets deleted: ${budgetsByEntity.count} by entity_id, ${budgetsByUser.count} by user_id`);
    }

    // 3. api_keys — name prefix and user_id prefix both attribute back to us.
    if (dryRun) {
      const [kbn] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM api_keys
        WHERE org_id = ${orgId} AND name LIKE 'stress-sdk-%'
      `;
      const [kbu] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM api_keys
        WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      console.log(`api_keys WOULD delete: ${kbn.count} by name, ${kbu.count} by user_id`);
    } else {
      const keysByName = await sql`
        DELETE FROM api_keys WHERE org_id = ${orgId} AND name LIKE 'stress-sdk-%'
      `;
      const keysByUser = await sql`
        DELETE FROM api_keys WHERE org_id = ${orgId} AND user_id LIKE 'stress-sdk-%'
      `;
      console.log(`api_keys deleted: ${keysByName.count} by name, ${keysByUser.count} by user_id`);
    }

    console.log(dryRun ? "\nDry run complete. Re-run with CLEANUP_CONFIRM=yes to execute." : "\nCleanup complete.");
  } finally {
    await sql.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
