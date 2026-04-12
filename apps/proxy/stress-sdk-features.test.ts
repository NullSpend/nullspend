/**
 * SDK Stress Test — production validation suite.
 *
 * Exercises every NullSpend SDK feature against the deployed proxy + dashboard
 * under concurrent load with real OpenAI and Anthropic API calls. Creates test
 * data fixtures, stresses them, mutates them mid-test, verifies final state,
 * and cleans up all artifacts.
 *
 * CRITICAL: this test mutates live production data by design (proxy writes
 * cost events through Cloudflare Queue + Hyperdrive + DO state). The test
 * isolates ENTITY-level data (dedicated user + api_key created in beforeAll,
 * deleted in afterAll) but infrastructure-level state is real. Manual runs
 * only — NEVER wire into CI.
 *
 * Requires:
 *   - Deployed proxy at PROXY_URL
 *   - NULLSPEND_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
 *   - INTERNAL_SECRET, DATABASE_URL
 *   - NULLSPEND_SMOKE_KEY_ID (used only to look up org_id)
 *   - NULLSPEND_DASHBOARD_URL (optional — skips direct-mode tests if unset)
 *
 * Run: cd apps/proxy && STRESS_INTENSITY=light pnpm test:stress stress-sdk-features.test.ts
 * Intensity: STRESS_INTENSITY={light|medium|heavy} (default medium)
 *
 * See docs/internal/test-plans/sdk-stress-test-plan.md §15, §15a, §15b for
 * the design corrections this file implements.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import * as crypto from "node:crypto";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { NullSpend } from "@nullspend/sdk";
import {
  BudgetExceededError,
  MandateViolationError,
  NullSpendError,
} from "@nullspend/sdk";
import type {
  CostEventInput,
  DenialReason,
} from "@nullspend/sdk";
import {
  BASE,
  OPENAI_API_KEY,
  ANTHROPIC_API_KEY,
  NULLSPEND_SMOKE_KEY_ID,
  INTERNAL_SECRET,
  DATABASE_URL,
  smallRequest,
  smallAnthropicRequest,
  isServerUp,
  invalidateBudget,
  syncBudget,
} from "./smoke-test-helpers.js";

// ── Intensity scaling ─────────────────────────────────────────────
const INTENSITY = (process.env.STRESS_INTENSITY ?? "medium") as "light" | "medium" | "heavy";
const CUSTOMER_COUNT    = { light: 5,  medium: 15, heavy: 30  }[INTENSITY];
const CONCURRENT_REQS   = { light: 10, medium: 25, heavy: 50  }[INTENSITY];
const RACE_REQS         = { light: 15, medium: 30, heavy: 60  }[INTENSITY];
const BATCH_EVENTS      = { light: 20, medium: 50, heavy: 100 }[INTENSITY];
// Fixture math baseline per §15a-5.
// OpenAI gpt-4o-mini max_tokens=3: ~3-5 µ¢/request actual upstream cost.
// Client-side estimate (used for budget pre-checks) is larger (~3 µ¢ for a
// short prompt) because the estimator assumes 1000 input tokens.
// Anthropic claude-3-haiku max_tokens=10: ~7 µ¢/request.

// ── Test run isolation ────────────────────────────────────────────
const TEST_RUN_ID = Date.now().toString(36);
const PREFIX = `stress-sdk-${TEST_RUN_ID}`;
const DASHBOARD_URL = process.env.NULLSPEND_DASHBOARD_URL;

// ── Shared state ──────────────────────────────────────────────────
let sql: postgres.Sql;
let SMOKE_ORG_ID: string;

// Isolated test user + api_key (§15b-3) — absorbs all attribution-level
// mutations so the shared smoke user/key budget state is never touched.
let STRESS_USER_ID: string;
let STRESS_KEY_RAW: string;
let STRESS_KEY_ID: string;
let STRESS_KEY_RAW_MANDATE: string;
let STRESS_KEY_ID_MANDATE: string;

// Phase 0 pass/fail gate — downstream phases skip if foundation is broken.
let phase0Passed = false;

// Track whether dashboard direct-mode is reachable. Evaluated in beforeAll.
let dashboardReachable = false;

// ── Findings log ──────────────────────────────────────────────────
interface Finding {
  phase: string;
  finding: string;
  severity: "info" | "warn" | "bug";
  timestamp: string;
}
const findings: Finding[] = [];

function logFinding(phase: string, finding: string, severity: "info" | "warn" | "bug" = "info") {
  const entry: Finding = { phase, finding, severity, timestamp: new Date().toISOString() };
  findings.push(entry);
  console.log(`[${severity.toUpperCase()}] [${phase}] ${finding}`);
}

// Default tags injected on EVERY proxy request so cost_events written by the
// proxy always carry stress_run_id attribution. Critical for cleanup scoping
// and to keep webhook/alert listeners from seeing untagged stress traffic.
//
// Reserved keys (stress_run_id) are FORCED to TEST_RUN_ID after merging user
// tags — a caller passing `{stress_run_id: "other"}` cannot accidentally
// rebind events to a wrong run id, which would silently bypass cleanup.
function defaultStressTagsHeader(extra?: Record<string, string>): string {
  const tags: Record<string, string> = {};
  // Merge caller-supplied tags first (e.g., session, plan).
  if (extra && extra["X-NullSpend-Tags"]) {
    try {
      const parsed = JSON.parse(extra["X-NullSpend-Tags"]);
      if (parsed && typeof parsed === "object") Object.assign(tags, parsed);
    } catch { /* ignore malformed */ }
  }
  // Force the reserved attribution key LAST so caller cannot override it.
  tags.stress_run_id = TEST_RUN_ID;
  return JSON.stringify(tags);
}

// ── Helper: build proxy headers using the STRESS key (not smoke key) ──
function stressAuthHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${OPENAI_API_KEY}`,
    "x-nullspend-key": STRESS_KEY_RAW,
    "X-NullSpend-Tags": defaultStressTagsHeader(extra),
  };
  if (extra) {
    Object.assign(h, extra);
    // Re-apply the merged tags so per-call extras don't clobber stress_run_id.
    h["X-NullSpend-Tags"] = defaultStressTagsHeader(extra);
  }
  return h;
}

function stressAnthropicAuthHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY!,
    "x-nullspend-key": STRESS_KEY_RAW,
    "X-NullSpend-Tags": defaultStressTagsHeader(extra),
  };
  if (extra) {
    Object.assign(h, extra);
    h["X-NullSpend-Tags"] = defaultStressTagsHeader(extra);
  }
  return h;
}

// ── Helper: poll-until-stable cost event queue drain (§15a-8, §15a-9) ──
async function waitForQueueDrain(
  runId: string,
  opts: { maxWaitMs: number; pollIntervalMs: number; stableForSamples: number } = {
    maxWaitMs: 60_000,
    pollIntervalMs: 2_000,
    stableForSamples: 3,
  },
): Promise<number> {
  const deadline = Date.now() + opts.maxWaitMs;
  let lastCount = -1;
  let stableCount = 0;
  while (Date.now() < deadline) {
    // Use @> containment (not ->>) so the GIN index on tags (jsonb_ops)
    // can serve the query. Also match user_id since probe requests flow
    // through the stress key and land on STRESS_USER_ID without our tag.
    const rows = await sql<{ count: string }[]>`
      SELECT COUNT(*)::text AS count FROM cost_events
      WHERE tags @> ${sql.json({ stress_run_id: runId })}
         OR user_id = ${STRESS_USER_ID}
    `;
    const count = Number(rows[0]?.count ?? 0);
    if (count === lastCount) {
      stableCount++;
      if (stableCount >= opts.stableForSamples) return count;
    } else {
      stableCount = 0;
      lastCount = count;
    }
    await new Promise((r) => setTimeout(r, opts.pollIntervalMs));
  }
  return lastCount;
}

/**
 * Active DO sync barrier (§15a-10). Sends a probe request with the stress
 * key and verifies the proxy either approves (DO has budget + room) or denies
 * with the SPECIFIC code that proves THIS entity under test is live in the
 * DO — not a different entity that happens to be denying first.
 *
 * Without the entity-specific check, an unrelated org-wide denial (e.g.,
 * user budget exhausted, velocity tripped on another customer) would satisfy
 * the barrier, and downstream tests would run against stale DO state.
 */
async function waitForBudgetLive(
  entityId: string,
  maxWaitMs = 10_000,
  expectDeny: "customer_budget_exceeded" | "tag_budget_exceeded" | "none" = "customer_budget_exceeded",
): Promise<void> {
  const deadline = Date.now() + maxWaitMs;
  let lastStatus = 0;
  let lastCode: string | undefined;
  while (Date.now() < deadline) {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: stressAuthHeaders({ "X-NullSpend-Customer": entityId }),
      body: smallRequest({ messages: [{ role: "user", content: "probe" }] }),
    });
    lastStatus = res.status;
    const text = await res.text();
    let json: unknown = null;
    try { json = JSON.parse(text); } catch { /* ignore */ }
    lastCode = (json as { error?: { code?: string } } | null)?.error?.code;
    // 200 always proves the DO sees the entity (request was approved against
    // the customer budget). For tight budgets, only the entity-specific code
    // proves liveness — other 429s (e.g., user budget, velocity from another
    // entity) mean a different bottleneck fired first.
    if (res.status === 200) return;
    if (expectDeny !== "none" && res.status === 429 && lastCode === expectDeny) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(
    `Budget ${entityId} not live in DO after ${maxWaitMs}ms (last status=${lastStatus}, code=${lastCode ?? "n/a"}, expectDeny=${expectDeny})`,
  );
}

// ── Helper: create a budget row and sync to DO ──
async function createBudget(
  entityType: "customer" | "user" | "api_key" | "tag",
  entityId: string,
  maxBudgetMicrodollars: number,
  opts: {
    spend?: number;
    velocityLimitMicrodollars?: number;
    velocityWindowSeconds?: number;
    velocityCooldownSeconds?: number;
    sessionLimitMicrodollars?: number;
    policy?: "strict_block" | "soft_block" | "warn";
  } = {},
): Promise<void> {
  const spend = opts.spend ?? 0;
  const policy = opts.policy ?? "strict_block";
  await sql`
    INSERT INTO budgets (
      user_id, org_id, entity_type, entity_id,
      max_budget_microdollars, spend_microdollars, policy,
      velocity_limit_microdollars, velocity_window_seconds, velocity_cooldown_seconds,
      session_limit_microdollars
    )
    VALUES (
      ${STRESS_USER_ID}, ${SMOKE_ORG_ID}, ${entityType}, ${entityId},
      ${maxBudgetMicrodollars}, ${spend}, ${policy},
      ${opts.velocityLimitMicrodollars ?? null},
      ${opts.velocityWindowSeconds ?? null},
      ${opts.velocityCooldownSeconds ?? null},
      ${opts.sessionLimitMicrodollars ?? null}
    )
    ON CONFLICT (org_id, entity_type, entity_id)
    DO UPDATE SET
      max_budget_microdollars = EXCLUDED.max_budget_microdollars,
      spend_microdollars = EXCLUDED.spend_microdollars,
      policy = EXCLUDED.policy,
      velocity_limit_microdollars = EXCLUDED.velocity_limit_microdollars,
      velocity_window_seconds = EXCLUDED.velocity_window_seconds,
      velocity_cooldown_seconds = EXCLUDED.velocity_cooldown_seconds,
      session_limit_microdollars = EXCLUDED.session_limit_microdollars,
      updated_at = NOW()
  `;
  await syncBudget(SMOKE_ORG_ID, entityType, entityId);
}

/** Build the standard test run tags that must be present on every cost event we write. */
function runTags(extra?: Record<string, string>): Record<string, string> {
  return { stress_run_id: TEST_RUN_ID, ...extra };
}

// ── Helper: construct a fresh NullSpend instance for per-phase isolation ──
// The long-lived ns instance leaks policy caches, session counters, and
// CostReporter queue state across phases (§15a-4). Each phase builds its own.
function makeStressNs(config: { dashboardUrl?: string; phase: string; maxQueueSize?: number } = { phase: "generic" }): NullSpend {
  const baseUrl = config.dashboardUrl ?? DASHBOARD_URL ?? "http://127.0.0.1:3000";
  return new NullSpend({
    baseUrl,
    apiKey: STRESS_KEY_RAW,
    proxyUrl: BASE,
    costReporting: {
      batchSize: 10,
      flushIntervalMs: 500,
      maxQueueSize: config.maxQueueSize ?? 1_000,
      onDropped: (count) => logFinding(config.phase, `CostReporter dropped ${count} events`, "warn"),
      onFlushError: (err) => logFinding(config.phase, `CostReporter flush error: ${err.message}`, "bug"),
    },
  });
}

/**
 * Generate a synthetic cost event with the stress run tag so cleanup can find
 * it via @> containment. All direct-mode ingest tests funnel through this.
 */
function buildSyntheticCostEvent(opts: {
  customer: string;
  idempotencyKey: string;
  provider?: string;
  model?: string;
  cost?: number;
  extraTags?: Record<string, string>;
}): CostEventInput & { idempotencyKey: string } {
  return {
    provider: opts.provider ?? "openai",
    model: opts.model ?? "gpt-4o-mini",
    inputTokens: 100,
    outputTokens: 5,
    costMicrodollars: opts.cost ?? 5,
    eventType: "custom",
    customer: opts.customer,
    tags: runTags(opts.extraTags),
    idempotencyKey: opts.idempotencyKey,
  };
}

// ═══════════════════════════════════════════════════════════════════
// TOP-LEVEL SUITE
// ═══════════════════════════════════════════════════════════════════
describe(`SDK stress test — production validation [${INTENSITY}]`, () => {
  beforeAll(async () => {
    // ── Environment sanity ────────────────────────────────────────
    const up = await isServerUp();
    if (!up) throw new Error("Proxy not reachable at " + BASE);
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required in .env.smoke");
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY required in .env.smoke");
    if (!NULLSPEND_SMOKE_KEY_ID) throw new Error("NULLSPEND_SMOKE_KEY_ID required in .env.smoke");
    if (!INTERNAL_SECRET) throw new Error("INTERNAL_SECRET required in .env.smoke");
    if (!DATABASE_URL) throw new Error("DATABASE_URL required in .env.smoke");

    sql = postgres(DATABASE_URL, { max: 3, idle_timeout: 10 });

    // ── Resolve SMOKE_ORG_ID from the smoke key (same org for all stress rows) ──
    const [key] = await sql<{ org_id: string }[]>`
      SELECT org_id FROM api_keys WHERE id = ${NULLSPEND_SMOKE_KEY_ID}
    `;
    if (!key?.org_id) throw new Error("Smoke key not found or missing org_id");
    SMOKE_ORG_ID = key.org_id;

    // ── Create the isolated stress user + api_key (§15b-3) ──
    STRESS_USER_ID = `${PREFIX}-user`;
    STRESS_KEY_RAW = `ns_live_sk_${crypto.randomBytes(16).toString("hex")}`;
    const stressKeyHash = hashKeySha256(STRESS_KEY_RAW);
    const stressKeyPrefix = STRESS_KEY_RAW.slice(0, 19);
    const [keyRow] = await sql<{ id: string }[]>`
      INSERT INTO api_keys (user_id, org_id, name, key_hash, key_prefix, allowed_models, allowed_providers)
      VALUES (
        ${STRESS_USER_ID}, ${SMOKE_ORG_ID}, ${`${PREFIX}-key`},
        ${stressKeyHash}, ${stressKeyPrefix}, NULL, NULL
      )
      RETURNING id
    `;
    STRESS_KEY_ID = keyRow.id;

    // Second mandate-only key — allowed_models = ['gpt-4o-mini'] for §6.6
    STRESS_KEY_RAW_MANDATE = `ns_live_sk_${crypto.randomBytes(16).toString("hex")}`;
    const mandateHash = hashKeySha256(STRESS_KEY_RAW_MANDATE);
    const mandatePrefix = STRESS_KEY_RAW_MANDATE.slice(0, 19);
    const [mandateRow] = await sql<{ id: string }[]>`
      INSERT INTO api_keys (user_id, org_id, name, key_hash, key_prefix, allowed_models, allowed_providers)
      VALUES (
        ${STRESS_USER_ID}, ${SMOKE_ORG_ID}, ${`${PREFIX}-mandate-key`},
        ${mandateHash}, ${mandatePrefix}, ARRAY['gpt-4o-mini']::text[], NULL
      )
      RETURNING id
    `;
    STRESS_KEY_ID_MANDATE = mandateRow.id;

    // ── Dashboard reachability (direct-mode tests skip if unreachable) ──
    if (DASHBOARD_URL) {
      try {
        const res = await fetch(`${DASHBOARD_URL}/api/health`, { signal: AbortSignal.timeout(2_000) }).catch(() => null);
        // /api/health may not exist; treat any response (even 404) as reachable,
        // fetch throwing is the only non-reachable signal.
        dashboardReachable = res !== null;
      } catch {
        dashboardReachable = false;
      }
      if (!dashboardReachable) {
        logFinding("setup", `Dashboard at ${DASHBOARD_URL} not reachable — direct-mode tests will skip`, "warn");
      }
    } else {
      logFinding("setup", "NULLSPEND_DASHBOARD_URL not set — direct-mode tests will skip", "info");
    }

    console.log(
      `[stress-sdk] TEST_RUN_ID=${TEST_RUN_ID} STRESS_USER_ID=${STRESS_USER_ID} ` +
        `STRESS_KEY_ID=${STRESS_KEY_ID} INTENSITY=${INTENSITY}`,
    );
  }, 60_000);

  afterAll(async () => {
    // ── Teardown per §5.6 with §15a-8/9 corrections ──
    console.log("\n[stress-sdk] Starting teardown — draining cost event queue…");

    // (1) Drain: poll-until-stable (§15a-9). Fixed waits are unreliable under
    // Cloudflare Queue lag.
    // (1) Initial drain — wait for events from the test body to settle.
    let preCleanupCount = 0;
    try {
      preCleanupCount = await waitForQueueDrain(TEST_RUN_ID);
      console.log(`[stress-sdk] Initial drain: ${preCleanupCount} events`);
    } catch (err) {
      logFinding("teardown", `waitForQueueDrain failed: ${(err as Error).message}`, "bug");
    }

    // (2) Delete stress api_keys FIRST so the proxy stops accepting new
    // requests under our credentials. Any cached auth state in the proxy
    // expires within ~30s. This shrinks the window in which inflight
    // post-test requests can write new cost_events rows. (FK from
    // cost_events.api_key_id is ON DELETE SET NULL — no cascade.)
    try {
      await sql`DELETE FROM api_keys WHERE id = ${STRESS_KEY_ID}`;
      await sql`DELETE FROM api_keys WHERE id = ${STRESS_KEY_ID_MANDATE}`;
    } catch (err) {
      logFinding("teardown", `Delete api_keys failed: ${(err as Error).message}`, "bug");
    }

    // (3) Second drain — catch cost events that landed AFTER the initial
    // drain (e.g., post-test fire-and-forget waitUntil writes, queue retries).
    try {
      const post = await waitForQueueDrain(TEST_RUN_ID);
      console.log(`[stress-sdk] Second drain after key delete: ${post} events`);
    } catch (err) {
      logFinding("teardown", `Second waitForQueueDrain failed: ${(err as Error).message}`, "warn");
    }

    // (4) Delete cost events via containment (§15a-8) plus any probe rows
    // attributed to our stress user (catches probe requests that bypassed tag
    // injection — they won't have stress_run_id but they WILL have user_id).
    try {
      await sql`
        DELETE FROM cost_events
        WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID })}
      `;
      await sql`
        DELETE FROM cost_events
        WHERE user_id = ${STRESS_USER_ID}
      `;
    } catch (err) {
      logFinding("teardown", `Delete cost_events failed: ${(err as Error).message}`, "bug");
    }

    // (5) Snapshot stress budgets (for best-effort DO invalidation), then
    // DELETE them from Postgres FIRST so the test always cleans up its rows
    // even if the DO invalidation step is slow or fails.
    let toInvalidate: { entity_type: string; entity_id: string }[] = [];
    try {
      // Narrow patterns to EXACT shapes this test creates:
      //  - user/customer/api_key entities: entity_id starts with PREFIX
      //  - tag entities: entity_id is `<key>=PREFIX-...` (e.g., `plan=stress-sdk-...`)
      // The previous '%${PREFIX}-%' substring was too greedy.
      toInvalidate = await sql<{ entity_type: string; entity_id: string }[]>`
        SELECT entity_type, entity_id FROM budgets
        WHERE org_id = ${SMOKE_ORG_ID}
          AND (entity_id LIKE ${`${PREFIX}-%`}
            OR entity_id = ${STRESS_USER_ID}
            OR entity_id = ${STRESS_KEY_ID}
            OR entity_id LIKE ${`%=${PREFIX}-%`})
      `;
      const delResult = await sql`
        DELETE FROM budgets
        WHERE org_id = ${SMOKE_ORG_ID}
          AND (entity_id LIKE ${`${PREFIX}-%`}
            OR entity_id = ${STRESS_USER_ID}
            OR entity_id = ${STRESS_KEY_ID}
            OR entity_id LIKE ${`%=${PREFIX}-%`})
      `;
      console.log(`[stress-sdk] Deleted ${delResult.count} budget rows`);
    } catch (err) {
      logFinding("teardown", `Delete budgets failed: ${(err as Error).message}`, "bug");
    }

    // Best-effort DO invalidation for the budgets we just deleted. Each call
    // has its own timeout via Promise.race so the loop can't hang the suite.
    for (const b of toInvalidate) {
      try {
        await Promise.race([
          invalidateBudget(SMOKE_ORG_ID, b.entity_type, b.entity_id, "remove"),
          new Promise((_, rej) => setTimeout(() => rej(new Error("invalidate timeout")), 5000)),
        ]);
      } catch (err) {
        logFinding("teardown", `invalidateBudget(${b.entity_type}, ${b.entity_id}) failed: ${(err as Error).message}`, "warn");
      }
    }

    // (6) Final cost_events sweep — catches any rows that landed during the
    // budget delete + DO invalidation loop above.
    try {
      await sql`
        DELETE FROM cost_events
        WHERE user_id = ${STRESS_USER_ID}
      `;
    } catch (err) {
      logFinding("teardown", `Final cost_events sweep failed: ${(err as Error).message}`, "warn");
    }

    // (5) Post-teardown orphan check (§9.5, inside afterAll per §15a-15).
    // Defer the assertion until AFTER findings-write + sql.end() so a
    // failed orphan check still flushes diagnostics and releases the pool.
    let orphanAssertionError: Error | undefined;
    let eventCount = 0;
    let budgetCount = 0;
    let keyCount = 0;
    try {
      const orphanEvents = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM cost_events
        WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID })}
           OR user_id = ${STRESS_USER_ID}
      `;
      // Cover BOTH leading-prefix (user/customer/api_key) and tag-key-anchored
      // matches (e.g., 'plan=stress-sdk-...') so the tag budget is included.
      // Use '%=PREFIX-%' rather than '%PREFIX-%' to avoid greedy substring
      // matches against unrelated rows.
      const orphanBudgetRows = await sql<{ entity_type: string; entity_id: string }[]>`
        SELECT entity_type, entity_id FROM budgets
        WHERE org_id = ${SMOKE_ORG_ID}
          AND (entity_id LIKE ${`${PREFIX}-%`} OR entity_id LIKE ${`%=${PREFIX}-%`})
      `;
      if (orphanBudgetRows.length > 0) {
        console.log(`[stress-sdk] Orphan budgets:`, orphanBudgetRows);
      }
      const orphanKeys = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM api_keys
        WHERE id IN (${STRESS_KEY_ID}, ${STRESS_KEY_ID_MANDATE})
      `;
      eventCount = Number(orphanEvents[0]?.count ?? 0);
      budgetCount = orphanBudgetRows.length;
      keyCount = Number(orphanKeys[0]?.count ?? 0);
      if (eventCount > 0 || budgetCount > 0 || keyCount > 0) {
        logFinding(
          "phase4.5",
          `Orphan data after teardown: ${eventCount} events, ${budgetCount} budgets, ${keyCount} keys`,
          "bug",
        );
        orphanAssertionError = new Error(
          `Orphan data after teardown: ${eventCount} events, ${budgetCount} budgets, ${keyCount} keys`,
        );
      } else {
        logFinding("phase4.5", "Teardown clean — no orphan data", "info");
      }
    } catch (err) {
      logFinding("phase4.5", `Orphan verification failed: ${(err as Error).message}`, "bug");
      orphanAssertionError = err as Error;
    }

    // (6) Write findings JSON file (§15a-16) and close the pool.
    // These run regardless of orphan-check outcome so diagnostics always
    // flush and the DB pool always closes. Write to the test file's
    // directory (apps/proxy/) regardless of the caller's cwd so the
    // apps/proxy/.gitignore rule actually applies.
    try {
      const __dirname = dirname(fileURLToPath(import.meta.url));
      const path = join(__dirname, `stress-sdk-findings-${TEST_RUN_ID}.json`);
      writeFileSync(
        path,
        JSON.stringify(
          { testRunId: TEST_RUN_ID, intensity: INTENSITY, phase0Passed, findings },
          null,
          2,
        ),
      );
      console.log(`[stress-sdk] Findings written to ${path}`);
    } catch (err) {
      console.warn("[stress-sdk] Could not write findings file:", err);
      // Last-resort fallback: dump JSON to stdout so CI logs recover the data.
      try {
        console.warn(
          "[stress-sdk] FINDINGS FALLBACK (stdout):\n" +
            JSON.stringify({ testRunId: TEST_RUN_ID, intensity: INTENSITY, phase0Passed, findings }),
        );
      } catch { /* ignore */ }
    }

    console.log("\n=== FINDINGS REPORT ===");
    for (const f of findings) {
      console.log(`[${f.severity.toUpperCase()}] [${f.phase}] ${f.finding}`);
    }

    try { await sql?.end(); } catch { /* ignore */ }

    // Re-raise the orphan error now that findings are persisted and the pool
    // is closed. If no orphans were detected, run the formal expectations so
    // vitest records a clean afterAll pass.
    if (orphanAssertionError) throw orphanAssertionError;
    expect(eventCount).toBe(0);
    expect(budgetCount).toBe(0);
    expect(keyCount).toBe(0);
  }, 180_000);

  // ═════════════════════════════════════════════════════════════════
  // PHASE 0 — TRANSPORT MATRIX (§15a-2)
  // Run before ANY feature test. If any of the 4 cases fails, the
  // foundation is broken and subsequent phases skip.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 0: Transport Matrix", () => {
    let p0Ns: NullSpend;
    // Per-test pass flags. phase0Passed is only true iff EVERY non-skipped
    // Phase 0 test actually succeeds. Setting this in a single test's body
    // (as the skeleton did) is a false positive when earlier tests fail.
    let p0_1Passed = false;
    let p0_2Passed = false;
    let p0_3Passed = false;
    let p0_4Passed = false;
    let p0_5Passed = false;

    beforeAll(async () => {
      p0Ns = makeStressNs({ phase: "phase0" });

      // p0 customer fixture (generous) for direct-ingest + proxied passthrough.
      // Distinct customer id from mutation fixture so waitForBudgetLive probes
      // don't pollute the row-count assertions below.
      await createBudget("customer", `${PREFIX}-p0-customer-01`, 1_000_000);
      // p0 tight fixture for budget mutation test starting point.
      // 1 µ¢ is too small to fit ANY gpt-4o-mini estimate (~3-4 µ¢ for "Say ok").
      await createBudget("customer", `${PREFIX}-p0-mutation-01`, 1);
      // Active probe on the mutation fixture: returns 429 immediately, no probe
      // row written. We use a dedicated probe customer for the generous one so
      // its probe cost event is fully accounted for before the 0.2 test starts.
      const probeCustomer = `${PREFIX}-p0-probe`;
      await createBudget("customer", probeCustomer, 1_000_000);
      await waitForBudgetLive(probeCustomer, 15_000);
      await waitForBudgetLive(`${PREFIX}-p0-mutation-01`, 15_000);
    }, 60_000);

    afterAll(async () => {
      try { await p0Ns.shutdown(); } catch { /* ignore */ }
      // Compute the gate. If any probe was skipped (dashboard unreachable),
      // treat that test as not required — match the skipIf logic in each test.
      const requires = {
        "0.1": dashboardReachable,
        "0.2": true,
        "0.3": dashboardReachable,
        "0.4": true,
        "0.5": true,
      } as const;
      const ok =
        (!requires["0.1"] || p0_1Passed) &&
        (!requires["0.2"] || p0_2Passed) &&
        (!requires["0.3"] || p0_3Passed) &&
        (!requires["0.4"] || p0_4Passed) &&
        (!requires["0.5"] || p0_5Passed);
      if (!ok) {
        logFinding(
          "phase0.gate",
          `Phase 0 gate FAILED: 0.1=${p0_1Passed} 0.2=${p0_2Passed} 0.3=${p0_3Passed} 0.4=${p0_4Passed} 0.5=${p0_5Passed}`,
          "bug",
        );
      }
      phase0Passed = ok;
    });

    // ── 0.1 Direct ingest: SDK → dashboard cost-events endpoint ──
    it(
      "0.1 direct ingest writes cost_events row with customer_id populated",
      async (ctx) => {
        if (!dashboardReachable) ctx.skip();
        const customer = `${PREFIX}-p0-customer-01`;
        const idem = `${PREFIX}-p0-direct-01`;
        const event = buildSyntheticCostEvent({
          customer,
          idempotencyKey: idem,
          extraTags: { stress_phase: "p0.1" },
        });

        const resp = await p0Ns.reportCost(event);
        expect(resp.id).toBeTruthy();

        // Server returns a prefixed id like `ns_evt_${uuid}` — strip prefix
        // to match the raw UUID stored in the cost_events table.
        const rawId = resp.id.replace(/^ns_evt_/, "");
        const rows = await sql<{ id: string; customer_id: string | null; request_id: string }[]>`
          SELECT id::text, customer_id, request_id FROM cost_events
          WHERE id = ${rawId}::uuid
          LIMIT 1
        `;
        expect(rows.length).toBe(1);
        expect(rows[0].customer_id).toBe(customer);
        logFinding("phase0.1", "direct ingest OK — customer_id populated", "info");
        p0_1Passed = true;
      },
      60_000,
    );

    // ── 0.2 Proxied pass-through: proxy writes the cost event ──
    it("0.2 proxied pass-through → proxy writes cost_events with customer_id, SDK does not double-count", async () => {
      const customer = `${PREFIX}-p0-customer-01`;
      // Fresh session so tracked fetch has a clean session counter.
      const session = p0Ns.customer(customer);

      const res = await session.openai(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders(), // includes x-nullspend-key → proxyUrl origin match → bailout path
        body: smallRequest({ messages: [{ role: "user", content: "p0.2" }] }),
      });
      expect(res.status).toBe(200);
      const body = await res.json() as { id?: string };
      const requestId = res.headers.get("x-request-id") ?? body.id;
      expect(requestId).toBeTruthy();

      // Poll for the specific row keyed by request_id — no reliance on count deltas.
      let row: { customer_id: string | null; source: string; request_id: string } | undefined;
      const deadline = Date.now() + 20_000;
      while (Date.now() < deadline && !row) {
        const rows = await sql<{ customer_id: string | null; source: string; request_id: string }[]>`
          SELECT customer_id, source, request_id FROM cost_events
          WHERE request_id = ${requestId!} AND provider = 'openai'
          LIMIT 1
        `;
        row = rows[0];
        if (!row) await new Promise((r) => setTimeout(r, 1_000));
      }
      expect(row, `cost_events row for request_id=${requestId} never appeared`).toBeDefined();
      expect(row!.customer_id).toBe(customer);
      // Proof this came from the proxy path (not SDK-ingested): source must be "proxy"
      expect(row!.source).toBe("proxy");

      // Tag it with the run id so teardown finds it via @> containment.
      // CASE-WHEN unwraps the string-encoded jsonb (proxy bug §6.9.4 finding):
      // proxy writes tags as a JSON string in a jsonb column, so direct `||`
      // concatenation produces a malformed result. We re-parse first.
      await sql`
        UPDATE cost_events
        SET tags = (CASE
          WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb
          ELSE COALESCE(tags, '{}'::jsonb)
        END) || ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p0.2" })}
        WHERE request_id = ${requestId!} AND provider = 'openai'
      `;

      logFinding("phase0.2", "proxy pass-through wrote 1 cost event with customer_id", "info");
      p0_2Passed = true;
    }, 60_000);

    // ── 0.3 Direct provider: SDK-only accounting (no proxy) ──
    it(
      "0.3 direct provider → SDK tracks cost event with customer_id via ingest path",
      async (ctx) => {
        if (!dashboardReachable) ctx.skip();
        const customer = `${PREFIX}-p0-customer-01`;
        // A fresh NullSpend instance with NO proxyUrl set. That forces isProxied()
        // to fall back on header detection — and since we don't attach
        // x-nullspend-key, the SDK takes the tracked path.
        const directNs = new NullSpend({
          baseUrl: DASHBOARD_URL!,
          apiKey: STRESS_KEY_RAW,
          costReporting: {
            batchSize: 1, // flush on every event so the assertion window is tight
            flushIntervalMs: 200,
            onFlushError: (err) => logFinding("phase0.3", `direct flush error: ${err.message}`, "bug"),
          },
        });

        try {
          const session = directNs.customer(customer, { tags: { stress_run_id: TEST_RUN_ID, stress_phase: "p0.3" } });

          const res = await session.openai("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${OPENAI_API_KEY}`,
            },
            body: smallRequest({ messages: [{ role: "user", content: "p0.3" }] }),
          });
          expect(res.status).toBe(200);
          await res.json();

          // Drain the batch queue so the ingest call definitely ran.
          await directNs.flush();
          await directNs.shutdown();

          // The SDK's direct ingest writes a row via /api/cost-events. The
          // resolved request_id is `sdk_${uuid}` (auto-generated) — we match by
          // tag + customer + source instead.
          const rows = await sql<{ count: string }[]>`
            SELECT COUNT(*)::text AS count FROM cost_events
            WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p0.3" })}
              AND customer_id = ${customer}
              AND source = 'api'
          `;
          const count = Number(rows[0]?.count ?? 0);
          expect(count).toBe(1);
          logFinding("phase0.3", `direct provider OK — 1 SDK-ingested row with customer=${customer}`, "info");
          p0_3Passed = true;
        } finally {
          try { await directNs.shutdown(); } catch { /* ignore */ }
        }
      },
      60_000,
    );

    // ── 0.5 Anthropic pass-through sanity (smoke the second provider) ──
    it("0.5 anthropic pass-through → proxy writes cost_events with provider=anthropic", async () => {
      const customer = `${PREFIX}-p0-customer-01`;
      const session = p0Ns.customer(customer);

      const res = await session.anthropic(`${BASE}/v1/messages`, {
        method: "POST",
        headers: stressAnthropicAuthHeaders({ "anthropic-version": "2023-06-01" }),
        body: smallAnthropicRequest({ messages: [{ role: "user", content: "p0.5" }] }),
      });
      expect(res.status).toBe(200);
      const body = await res.json() as { id?: string };
      const requestId = res.headers.get("x-request-id") ?? body.id;
      expect(requestId).toBeTruthy();

      let row: { provider: string; customer_id: string | null } | undefined;
      const deadline = Date.now() + 20_000;
      while (Date.now() < deadline && !row) {
        const rows = await sql<{ provider: string; customer_id: string | null }[]>`
          SELECT provider, customer_id FROM cost_events
          WHERE request_id = ${requestId!} AND provider = 'anthropic'
          LIMIT 1
        `;
        row = rows[0];
        if (!row) await new Promise((r) => setTimeout(r, 1_000));
      }
      expect(row, `anthropic cost_events row for ${requestId} never appeared`).toBeDefined();
      expect(row!.provider).toBe("anthropic");
      expect(row!.customer_id).toBe(customer);
      // CASE-WHEN unwraps proxy's string-encoded jsonb tags before merging.
      await sql`
        UPDATE cost_events
        SET tags = (CASE
          WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb
          ELSE COALESCE(tags, '{}'::jsonb)
        END) || ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p0.5" })}
        WHERE request_id = ${requestId!} AND provider = 'anthropic'
      `;
      logFinding("phase0.5", "anthropic pass-through OK", "info");
      p0_5Passed = true;
    }, 60_000);

    // ── 0.4 Budget mutation sanity check ──
    it("0.4 mutation sanity: tight budget denies → SQL update → syncBudget → next request succeeds", async () => {
      const customer = `${PREFIX}-p0-mutation-01`;

      // Initial state: max=1 µ¢. Any request's estimate (~3-5 µ¢) overflows.
      const res1 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({ "X-NullSpend-Customer": customer }),
        body: smallRequest({ messages: [{ role: "user", content: "p0.4a" }] }),
      });
      expect(res1.status).toBe(429);
      const body1 = await res1.json().catch(() => null);
      expect((body1 as { error?: { code?: string } })?.error?.code).toBe("customer_budget_exceeded");

      // Mutation: enlarge the budget.
      await sql`
        UPDATE budgets
        SET max_budget_microdollars = 1000000, spend_microdollars = 0, updated_at = NOW()
        WHERE entity_type = 'customer' AND entity_id = ${customer} AND org_id = ${SMOKE_ORG_ID}
      `;
      await syncBudget(SMOKE_ORG_ID, "customer", customer);
      await waitForBudgetLive(customer, 15_000);

      const res2 = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({ "X-NullSpend-Customer": customer }),
        body: smallRequest({ messages: [{ role: "user", content: "p0.4b" }] }),
      });
      expect(res2.status).toBe(200);
      const body2 = await res2.json() as { id?: string };
      const requestId = res2.headers.get("x-request-id") ?? body2.id;

      if (requestId) {
        // Tag just the specific success row via request_id.
        // CASE-WHEN unwraps proxy's string-encoded jsonb tags before merging.
        await sql`
          UPDATE cost_events
          SET tags = (CASE
            WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb
            ELSE COALESCE(tags, '{}'::jsonb)
          END) || ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p0.4" })}
          WHERE request_id = ${requestId} AND provider = 'openai'
        `;
      }

      logFinding("phase0.4", "budget mutation barrier works: deny → update → sync → 200", "info");
      p0_4Passed = true;
    }, 90_000);
  });

  // ═════════════════════════════════════════════════════════════════
  // PHASE 1 — FUNCTIONAL TESTS
  // Per §15a-3 we drop §6.1.1/1.2/1.3/1.6 (covered by SDK unit tests).
  // Per §15a-14 we add §6.10/6.11/6.12/6.13 coverage bundles.
  // Per §15b-2 we replace §6.7 with user-level §6.7b.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 1: Functional tests", () => {
    let p1Ns: NullSpend;
    let p1NsMandate: NullSpend;

    beforeAll(async () => {
      // Skip Phase 1 entirely if Phase 0 didn't pass — the foundation must
      // be stable before exercising features.
      if (!phase0Passed) {
        logFinding("phase1", "Phase 0 failed — Phase 1 fixtures NOT created", "warn");
        return;
      }

      p1Ns = makeStressNs({ phase: "phase1" });
      p1NsMandate = new NullSpend({
        baseUrl: DASHBOARD_URL ?? "http://127.0.0.1:3000",
        apiKey: STRESS_KEY_RAW_MANDATE,
        proxyUrl: BASE,
        costReporting: { batchSize: 1, flushIntervalMs: 200 },
      });

      // ── Phase 1 fixtures ──
      // ORDER MATTERS: customer fixtures created + probed FIRST so the
      // generous-customer probe gets a 200 (proving DO sees the entity).
      // The exhausted user budget is created LAST — once it exists, every
      // subsequent probe via the stress key denies at the user level with
      // generic budget_exceeded (NOT customer_budget_exceeded), which would
      // confuse the strict liveness probe.

      // 6.9.1 tight customer budget → proxy 429:customer_budget_exceeded
      await createBudget("customer", `${PREFIX}-p1-tight-01`, 1);
      // 6.9.2 velocity-limited customer → proxy 429:velocity_exceeded
      await createBudget("customer", `${PREFIX}-p1-velocity-01`, 1_000_000, {
        velocityLimitMicrodollars: 10,
        velocityWindowSeconds: 60,
        velocityCooldownSeconds: 60,
      });
      // 6.9.3 session-limited customer → proxy 429:session_limit_exceeded
      await createBudget("customer", `${PREFIX}-p1-session-01`, 1_000_000, {
        sessionLimitMicrodollars: 10,
      });
      // 6.9.4 tag budget — entity_id format is "tagKey=tagValue"
      await createBudget("tag", `plan=${PREFIX}-p1-plan-tight`, 1);
      // 6.10/6.11/6.12/6.13 generous customer for streaming + read APIs + batch
      await createBudget("customer", `${PREFIX}-p1-generous-01`, 1_000_000);

      // Active probe BEFORE creating the exhausted user budget. The generous
      // customer should return 200 here.
      await waitForBudgetLive(`${PREFIX}-p1-generous-01`, 15_000);

      // 6.7b user-level budget exhausted → policy cache denies client-side
      // Created LAST so the probe above doesn't see a user-level denial.
      await createBudget("user", STRESS_USER_ID, 5, { spend: 5 });
    }, 60_000);

    afterAll(async () => {
      try { await p1Ns?.shutdown(); } catch { /* ignore */ }
      try { await p1NsMandate?.shutdown(); } catch { /* ignore */ }
    });

    // ── §6.4 Direct-mode cost event ingest ──
    describe("6.4 direct-mode ingest", () => {
      it("4.1 reportCost single event with customer field", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const event = buildSyntheticCostEvent({
          customer: `${PREFIX}-p1-generous-01`,
          idempotencyKey: `${PREFIX}-p1-4.1`,
          extraTags: { stress_phase: "p1.4.1" },
        });
        const resp = await p1Ns.reportCost(event);
        expect(resp.id).toBeTruthy();
      }, 30_000);

      it("4.2 reportCostBatch dual events", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const events: (CostEventInput & { idempotencyKey: string })[] = [
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-4.2-a`,
            extraTags: { stress_phase: "p1.4.2" },
          }),
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-4.2-b`,
            extraTags: { stress_phase: "p1.4.2" },
          }),
        ];
        const resp = await p1Ns.reportCostBatch(events);
        expect(resp.inserted).toBe(2);
        expect(resp.ids.length).toBe(2);
      }, 30_000);

      it("4.3 queueCost + flush eventually writes row", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const idem = `${PREFIX}-p1-4.3`;
        p1Ns.queueCost(buildSyntheticCostEvent({
          customer: `${PREFIX}-p1-generous-01`,
          idempotencyKey: idem,
          extraTags: { stress_phase: "p1.4.3" },
        }));
        await p1Ns.flush();

        // Wait for the row by tag (the SDK's POST overrides Idempotency-Key
        // header so we can't search by `idem` directly).
        let count = 0;
        const deadline = Date.now() + 10_000;
        while (Date.now() < deadline && count === 0) {
          const rows = await sql<{ count: string }[]>`
            SELECT COUNT(*)::text AS count FROM cost_events
            WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p1.4.3" })}
          `;
          count = Number(rows[0]?.count ?? 0);
          if (count === 0) await new Promise((r) => setTimeout(r, 500));
        }
        expect(count).toBeGreaterThanOrEqual(1);
      }, 30_000);

      it("4.5 dual-provider batch (openai + anthropic) both persist", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const events: (CostEventInput & { idempotencyKey: string })[] = [
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-4.5-oai`,
            provider: "openai",
            model: "gpt-4o-mini",
            extraTags: { stress_phase: "p1.4.5" },
          }),
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-4.5-ant`,
            provider: "anthropic",
            model: "claude-3-haiku-20240307",
            extraTags: { stress_phase: "p1.4.5" },
          }),
        ];
        const resp = await p1Ns.reportCostBatch(events);
        expect(resp.inserted).toBe(2);
      }, 30_000);
    });

    // ── §6.6 Mandate violation ──
    describe("6.6 enforcement: mandate violation", () => {
      it("rejects model='gpt-4' when allowed_models=['gpt-4o-mini']", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        let denialReason: DenialReason | undefined;
        const tracked = p1NsMandate.createTrackedFetch("openai", {
          enforcement: true,
          customer: `${PREFIX}-p1-generous-01`,
          onDenied: (r) => { denialReason = r; },
        });
        // Point at api.openai.com directly (no proxyUrl match) so the mandate
        // check fires client-side BEFORE any network call.
        await expect(
          tracked("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
            body: smallRequest({ model: "gpt-4" }),
          }),
        ).rejects.toBeInstanceOf(MandateViolationError);
        expect(denialReason?.type).toBe("mandate");
        if (denialReason && denialReason.type === "mandate") {
          expect(denialReason.requested).toBe("gpt-4");
        }
      }, 30_000);
    });

    // ── §6.7b User-level budget denial (replaces §6.7 per §15b-2) ──
    describe("6.7b enforcement: user-level client-side budget denial", () => {
      it("exhausted user budget → policy cache denies before network", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        let denialReason: DenialReason | undefined;
        const tracked = p1Ns.createTrackedFetch("openai", {
          enforcement: true,
          customer: `${PREFIX}-p1-generous-01`,
          onDenied: (r) => { denialReason = r; },
        });
        // The user budget for STRESS_USER_ID is max=5 spend=5 (remaining=0),
        // which should be the most restrictive in the org (or at least a
        // restrictive enough one that the policy endpoint returns it). The
        // SDK's policy cache then trips on checkBudget.
        await expect(
          tracked("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
            body: smallRequest(),
          }),
        ).rejects.toBeInstanceOf(BudgetExceededError);
        expect(denialReason?.type).toBe("budget");
        // Soft-verify the denial was attributed to our stress user budget
        // (the intended target). The org-scoped policy endpoint returns the
        // most-restrictive budget org-wide, so if another test or fixture
        // has a more-restrictive budget, the denial fires on THAT entity and
        // this test passes "for the wrong reason". We log a finding rather
        // than hard-fail because env-dependent most-restrictive selection
        // would make this test flaky across test runs.
        if (denialReason && denialReason.type === "budget") {
          if (denialReason.entityType !== "user" || denialReason.entityId !== STRESS_USER_ID) {
            logFinding(
              "phase1.6.7b",
              `User budget denial fired on a DIFFERENT entity (${denialReason.entityType}:${denialReason.entityId}) — most-restrictive-org-wide scoping shadowed the stress user budget. Assertion passes but the intended path wasn't exercised.`,
              "warn",
            );
          }
        }
      }, 30_000);
    });

    // ── §6.8 REMOVED ──
    // The client-side session limit fail-open test was deleted on 2026-04-08
    // because it was a known no-op AND wasted ~$0.005 of OpenAI calls per
    // stress run. The SDK's session counter only advances on successful
    // tracked responses, but the test forced policy fetch to throw — which
    // means the counter never advanced and the fail-open check never tripped.
    // It made up to 15 real requests trying to exercise a path that was
    // structurally unreachable.
    //
    // Coverage: client-side session limit fail-open is tested ONLY by unit
    // tests in packages/sdk/src/tracked-fetch.test.ts (mocked clock, mocked
    // fetch, deterministic). There is no live-stack coverage today.
    //
    // To restore live-stack coverage someone would need to either (a) rewrite
    // against a mock upstream so no real spend is incurred and the counter
    // can be advanced deterministically, or (b) coordinate with the policy
    // endpoint to return a permissive policy specifically for the stress
    // test org. Tracked under TODOS.md "§6.8 fail-open session limit test
    // burns OpenAI calls per run" → option (a) is the recommended fix.

    // ── §6.9 proxy 429 interception (verifies fix from §15c-1) ──
    // Previously a KNOWN GAP: the SDK's `isProxied()` bailout returned BEFORE
    // the 429 interception code could run, making the typed-error API surface
    // dead code in proxy mode. Fixed in §15c-1 — the proxied path now calls
    // parseDenialPayload + dispatchDenialCode after the fetch but before
    // returning, so callers get typed errors (BudgetExceededError, etc.)
    // for proxy denials with enforcement: true.
    //
    // This test verifies the fix end-to-end against the live deployed proxy:
    // a customer-budget denial through the SDK's tracked fetch should throw
    // BudgetExceededError and fire onDenied — NOT return a raw 429.
    describe("6.9 proxy 429 interception", () => {
      it("proxy 429 customer_budget_exceeded throws BudgetExceededError + fires onDenied", async (ctx) => {
        if (!phase0Passed) ctx.skip();
        // Lift the user budget so the proxy denies on the customer entity
        // (not the user budget, which would emit generic budget_exceeded).
        await sql`
          UPDATE budgets
          SET max_budget_microdollars = 1000000000, spend_microdollars = 0, updated_at = NOW()
          WHERE entity_type = 'user' AND entity_id = ${STRESS_USER_ID} AND org_id = ${SMOKE_ORG_ID}
        `;
        await syncBudget(SMOKE_ORG_ID, "user", STRESS_USER_ID);

        try {
          let onDeniedFired = false;
          const tracked = p1Ns.createTrackedFetch("openai", {
            enforcement: true,
            customer: `${PREFIX}-p1-tight-01`,
            onDenied: () => { onDeniedFired = true; },
          });
          // proxyUrl matches AND x-nullspend-key is in headers — both proxy
          // detection paths fire. The fixed proxied branch runs interception
          // after the fetch and converts the 429 into a typed BudgetExceededError.
          await expect(
            tracked(`${BASE}/v1/chat/completions`, {
              method: "POST",
              headers: stressAuthHeaders({ "X-NullSpend-Customer": `${PREFIX}-p1-tight-01` }),
              body: smallRequest({ messages: [{ role: "user", content: "p1.9.fix" }] }),
            }),
          ).rejects.toBeInstanceOf(BudgetExceededError);
          // onDenied fired before the throw (safeDenied → throw sequence)
          expect(onDeniedFired).toBe(true);
          logFinding(
            "phase1.6.9",
            "Proxy 429 interception works in proxied mode (verifies fix from §15c-1). SDK now throws typed BudgetExceededError for customer-budget denials and fires onDenied callback.",
            "info",
          );
        } finally {
          // Restore exhausted state for §6.7b symmetry.
          await sql`
            UPDATE budgets
            SET max_budget_microdollars = 5, spend_microdollars = 5, updated_at = NOW()
            WHERE entity_type = 'user' AND entity_id = ${STRESS_USER_ID} AND org_id = ${SMOKE_ORG_ID}
          `;
          await syncBudget(SMOKE_ORG_ID, "user", STRESS_USER_ID);
        }
      }, 60_000);

      // Note: the helper functions (parseDenialPayload, dispatchDenialCode)
      // are exercised in detail by unit tests in
      // packages/sdk/src/tracked-fetch.test.ts (the "proxy 429 interception"
      // describe with 23 tests across "via proxyUrl", "via x-nullspend-key
      // header", and "direct mode (defensive)" sub-blocks). This stress test
      // is the live-stack verification — proves the fix actually fires
      // against a real Cloudflare-Workers proxy with a real denial body shape.
    });

    // ── §6.10 Streaming response tracking ──
    describe("6.10 streaming response tracking", () => {
      it("openai streaming → cost event with non-zero tokens (direct-provider path)", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        // Use direct provider so the SDK is the only thing tracking cost.
        const directNs = new NullSpend({
          baseUrl: DASHBOARD_URL!,
          apiKey: STRESS_KEY_RAW,
          costReporting: { batchSize: 1, flushIntervalMs: 200 },
        });
        try {
          const session = directNs.customer(`${PREFIX}-p1-generous-01`, {
            tags: { stress_run_id: TEST_RUN_ID, stress_phase: "p1.10.openai" },
          });
          const res = await session.openai("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
            body: smallRequest({ stream: true }),
          });
          expect(res.status).toBe(200);
          // Drain the SSE so the SDK extracts usage.
          const reader = res.body!.getReader();
          while (true) {
            const { done } = await reader.read();
            if (done) break;
          }
          await directNs.flush();
          await directNs.shutdown();

          const rows = await sql<{ input_tokens: number; output_tokens: number }[]>`
            SELECT input_tokens, output_tokens FROM cost_events
            WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p1.10.openai" })}
            ORDER BY created_at DESC
            LIMIT 1
          `;
          expect(rows.length).toBe(1);
          expect(rows[0].input_tokens).toBeGreaterThan(0);
          expect(rows[0].output_tokens).toBeGreaterThan(0);
        } finally {
          try { await directNs.shutdown(); } catch { /* ignore */ }
        }
      }, 60_000);

      it("anthropic streaming → cost event with non-zero tokens (direct-provider path)", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const directNs = new NullSpend({
          baseUrl: DASHBOARD_URL!,
          apiKey: STRESS_KEY_RAW,
          costReporting: { batchSize: 1, flushIntervalMs: 200 },
        });
        try {
          const session = directNs.customer(`${PREFIX}-p1-generous-01`, {
            tags: { stress_run_id: TEST_RUN_ID, stress_phase: "p1.10.anthropic" },
          });
          const res = await session.anthropic("https://api.anthropic.com/v1/messages", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "x-api-key": ANTHROPIC_API_KEY!,
              "anthropic-version": "2023-06-01",
            },
            body: smallAnthropicRequest({ stream: true }),
          });
          expect(res.status).toBe(200);
          const reader = res.body!.getReader();
          while (true) {
            const { done } = await reader.read();
            if (done) break;
          }
          await directNs.flush();
          await directNs.shutdown();

          const rows = await sql<{ input_tokens: number; output_tokens: number }[]>`
            SELECT input_tokens, output_tokens FROM cost_events
            WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p1.10.anthropic" })}
            ORDER BY created_at DESC
            LIMIT 1
          `;
          expect(rows.length).toBe(1);
          expect(rows[0].input_tokens).toBeGreaterThan(0);
          expect(rows[0].output_tokens).toBeGreaterThan(0);
        } finally {
          try { await directNs.shutdown(); } catch { /* ignore */ }
        }
      }, 60_000);
    });

    // ── §6.11 Read APIs ──
    describe("6.11 read APIs", () => {
      it("checkBudget returns BudgetStatus shape", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const status = await p1Ns.checkBudget();
        expect(status.entities).toBeDefined();
        expect(Array.isArray(status.entities)).toBe(true);
      }, 30_000);

      it("listBudgets returns budgets array (post PR #5 dual-auth)", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const resp = await p1Ns.listBudgets();
        expect(resp.data).toBeDefined();
        expect(Array.isArray(resp.data)).toBe(true);
        // The dashboard's response schema prefixes user/api_key entityIds via
        // toExternalId (e.g., user → "ns_usr_${id}"). Match by suffix instead
        // so the test is robust against the prefix transform.
        const userBudget = resp.data.find(
          (b) => b.entityType === "user" && b.entityId.endsWith(STRESS_USER_ID),
        );
        expect(userBudget).toBeDefined();
      }, 30_000);

      it("listCostEvents returns events array (post PR #5 dual-auth)", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const resp = await p1Ns.listCostEvents({ limit: 5 });
        expect(resp.data).toBeDefined();
        expect(Array.isArray(resp.data)).toBe(true);
      }, 30_000);
    });

    // ── §6.12 Batch size boundaries ──
    describe("6.12 batch size boundaries", () => {
      it("100 events accepted", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const events: (CostEventInput & { idempotencyKey: string })[] = Array.from({ length: 100 }, (_, i) =>
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-12.100-${i}`,
            extraTags: { stress_phase: "p1.12.100" },
          }),
        );
        const resp = await p1Ns.reportCostBatch(events);
        expect(resp.inserted).toBe(100);
      }, 60_000);

      it("101 events rejected with 400 validation error", async (ctx) => {
        if (!dashboardReachable || !phase0Passed) ctx.skip();
        const events: (CostEventInput & { idempotencyKey: string })[] = Array.from({ length: 101 }, (_, i) =>
          buildSyntheticCostEvent({
            customer: `${PREFIX}-p1-generous-01`,
            idempotencyKey: `${PREFIX}-p1-12.101-${i}`,
            extraTags: { stress_phase: "p1.12.101" },
          }),
        );
        // Assert the specific failure mode: HTTP 400 from Zod schema rejection.
        // A generic NullSpendError would also pass on network timeouts / 502s,
        // masking real infra issues as "validation worked".
        let thrown: unknown;
        try {
          await p1Ns.reportCostBatch(events);
        } catch (err) {
          thrown = err;
        }
        expect(thrown).toBeInstanceOf(NullSpendError);
        expect((thrown as NullSpendError).statusCode).toBe(400);
      }, 30_000);
    });

    // ── §6.13 Shutdown idempotency ──
    describe("6.13 shutdown idempotency", () => {
      it("shutdown twice → no throw", async (ctx) => {
        if (!phase0Passed) ctx.skip();
        const ns = makeStressNs({ phase: "phase1.13" });
        await ns.shutdown();
        await expect(ns.shutdown()).resolves.toBeUndefined();
      }, 30_000);
    });
  });

  // ═════════════════════════════════════════════════════════════════
  // PHASE 2 — CONCURRENT STRESS SCENARIOS
  // Each test exercises a feature under load. Per-phase entity IDs and
  // per-phase NullSpend instance per §15a-4.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 2: Concurrent stress", () => {
    let p2Ns: NullSpend;

    beforeAll(async () => {
      if (!phase0Passed) {
        logFinding("phase2", "Phase 0 failed — Phase 2 fixtures NOT created", "warn");
        return;
      }
      p2Ns = makeStressNs({ phase: "phase2" });

      // Generous customer fixtures for concurrent happy-path tests.
      for (let i = 1; i <= CUSTOMER_COUNT; i++) {
        const id = `${PREFIX}-p2-customer-${String(i).padStart(2, "0")}`;
        await createBudget("customer", id, 1_000_000);
      }
      // Tight customer for race-against-tight-budget test.
      await createBudget("customer", `${PREFIX}-p2-tight-01`, 1);
      // Lift the user budget for Phase 2 — many concurrent calls would hit
      // the exhausted user budget otherwise.
      await sql`
        UPDATE budgets
        SET max_budget_microdollars = 1000000000, spend_microdollars = 0, updated_at = NOW()
        WHERE entity_type = 'user' AND entity_id = ${STRESS_USER_ID} AND org_id = ${SMOKE_ORG_ID}
      `;
      await syncBudget(SMOKE_ORG_ID, "user", STRESS_USER_ID);
      await waitForBudgetLive(`${PREFIX}-p2-customer-01`, 15_000);
    }, 60_000);

    afterAll(async () => {
      try { await p2Ns?.shutdown(); } catch { /* ignore */ }
    });

    // ── §7.1 Concurrent customer budget races (generous) ──
    it("7.1 concurrent requests against generous customer budget all succeed", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p2-customer-01`;
      const concurrency = CONCURRENT_REQS;

      const requests = Array.from({ length: concurrency }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p2.7.1" }),
          }),
          body: smallRequest({ messages: [{ role: "user", content: `p2.7.1.${i}` }] }),
        }),
      );
      const results = await Promise.all(requests);
      const statuses = results.map((r) => r.status);
      const successes = statuses.filter((s) => s === 200).length;
      const denied = statuses.filter((s) => s === 429).length;
      const others = statuses.filter((s) => s !== 200 && s !== 429).length;
      // Drain bodies
      for (const r of results) await r.text();

      console.log(`[stress-sdk] 7.1 concurrent generous: ${successes} ok, ${denied} 429, ${others} other`);
      expect(others).toBe(0);
      expect(successes).toBe(concurrency);
    }, 180_000);

    // ── §7.2 Concurrent customer budget races (tight) ──
    it("7.2 concurrent requests against tight customer budget all denied", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p2-tight-01`;
      await waitForBudgetLive(customer, 15_000);
      const concurrency = RACE_REQS;

      const requests = Array.from({ length: concurrency }, (_, i) =>
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p2.7.2" }),
          }),
          body: smallRequest({ messages: [{ role: "user", content: `p2.7.2.${i}` }] }),
        }),
      );
      const results = await Promise.all(requests);
      const statuses = results.map((r) => r.status);
      const denied = statuses.filter((s) => s === 429).length;
      const successes = statuses.filter((s) => s === 200).length;
      for (const r of results) await r.text();

      console.log(`[stress-sdk] 7.2 concurrent tight: ${denied}/${concurrency} denied, ${successes} leaked`);
      expect(successes).toBe(0);
      expect(denied).toBe(concurrency);
    }, 180_000);

    // ── §7.3 Rapid customer switching ──
    it("7.3 rapid customer switching across many sessions", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customers = Array.from({ length: CUSTOMER_COUNT }, (_, i) =>
        `${PREFIX}-p2-customer-${String(i + 1).padStart(2, "0")}`,
      );

      // 3 requests per customer, all in flight at once.
      const requests: Promise<Response>[] = [];
      for (const customer of customers) {
        for (let i = 0; i < 3; i++) {
          requests.push(
            fetch(`${BASE}/v1/chat/completions`, {
              method: "POST",
              headers: stressAuthHeaders({
                "X-NullSpend-Customer": customer,
                "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p2.7.3" }),
              }),
              body: smallRequest({ messages: [{ role: "user", content: `p2.7.3.${customer}.${i}` }] }),
            }),
          );
        }
      }
      const results = await Promise.all(requests);
      const statuses = results.map((r) => r.status);
      const successes = statuses.filter((s) => s === 200).length;
      for (const r of results) await r.text();

      console.log(`[stress-sdk] 7.3 rapid switch (${customers.length} customers, 3 each): ${successes}/${requests.length} ok`);
      expect(successes).toBe(requests.length);
    }, 180_000);

    // ── §7.4 Mixed provider on same customer ──
    it("7.4 mixed openai+anthropic on same customer", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p2-customer-02`;
      const requests: Promise<Response>[] = [];
      for (let i = 0; i < 3; i++) {
        requests.push(
          fetch(`${BASE}/v1/chat/completions`, {
            method: "POST",
            headers: stressAuthHeaders({
              "X-NullSpend-Customer": customer,
              "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p2.7.4" }),
            }),
            body: smallRequest({ messages: [{ role: "user", content: `p2.7.4.oai.${i}` }] }),
          }),
        );
        requests.push(
          fetch(`${BASE}/v1/messages`, {
            method: "POST",
            headers: stressAnthropicAuthHeaders({
              "anthropic-version": "2023-06-01",
              "X-NullSpend-Customer": customer,
              "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p2.7.4" }),
            }),
            body: smallAnthropicRequest({ messages: [{ role: "user", content: `p2.7.4.ant.${i}` }] }),
          }),
        );
      }
      const results = await Promise.all(requests);
      const successes = results.filter((r) => r.status === 200).length;
      for (const r of results) await r.text();
      expect(successes).toBe(requests.length);
    }, 180_000);

    // ── §7.6 Direct SDK ingest under load (with explicit idempotency keys) ──
    it("7.6 batch ingest under load with explicit idempotency keys is dedup-safe", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p2-customer-01`;
      const events: (CostEventInput & { idempotencyKey: string })[] = Array.from({ length: BATCH_EVENTS }, (_, i) =>
        buildSyntheticCostEvent({
          customer,
          idempotencyKey: `${PREFIX}-p2-7.6-${i}`,
          extraTags: { stress_phase: "p2.7.6" },
        }),
      );

      // First dispatch — should insert all.
      const first = await p2Ns.reportCostBatch(events);
      expect(first.inserted).toBe(BATCH_EVENTS);
      expect(first.ids.length).toBe(BATCH_EVENTS);

      // Second dispatch with the same idempotency keys — must dedup to 0.
      const second = await p2Ns.reportCostBatch(events);
      expect(second.inserted).toBe(0);
      expect(second.ids.length).toBe(0);

      // Ground-truth verification: DB should contain EXACTLY BATCH_EVENTS
      // rows for this phase, not 2×BATCH_EVENTS. This catches the
      // wrong-test-right-answer where `inserted: 0` came from an error path
      // instead of dedup.
      const [{ count }] = await sql<{ count: string }[]>`
        SELECT COUNT(*)::text AS count FROM cost_events
        WHERE customer_id = ${customer}
          AND source = 'api'
          AND tags @> ${sql.json({ stress_phase: "p2.7.6" })}
      `;
      expect(Number(count)).toBe(BATCH_EVENTS);
    }, 120_000);

    // ── §7.6b Direct SDK queue overflow → onDropped fires (§15a-12) ──
    it("7.6b queue overflow triggers onDropped callback with exact drop count", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      let droppedCount = 0;
      // Tight queue so the flood overflows. flushIntervalMs high so the queue
      // fills before any flush happens. No proxyUrl — direct ingest only.
      const overflowNs = new NullSpend({
        baseUrl: DASHBOARD_URL!,
        apiKey: STRESS_KEY_RAW,
        costReporting: {
          batchSize: 1000,           // never auto-flush
          flushIntervalMs: 100_000,  // never timer-flush
          maxQueueSize: 20,
          onDropped: (count) => { droppedCount += count; },
        },
      });
      try {
        for (let i = 0; i < 100; i++) {
          overflowNs.queueCost(buildSyntheticCostEvent({
            customer: `${PREFIX}-p2-customer-01`,
            idempotencyKey: `${PREFIX}-p2-7.6b-${i}`,
            extraTags: { stress_phase: "p2.7.6b" },
          }));
        }
        // 100 enqueued, maxQueueSize=20 → exactly 80 dropped.
        // Exact assertion catches SDK bugs that drop too few OR too many.
        expect(droppedCount).toBe(80);
      } finally {
        try { await overflowNs.shutdown(); } catch { /* ignore */ }
      }
    }, 30_000);

    // ── §7.7 Policy cache staleness (no 60s wait per §15a-13) ──
    it.skip("7.7 policy cache staleness — covered by §6.9 (interception now works)", async () => {
      // The full staleness path (mutate budget mid-loop → expect 429 interception)
      // is now covered by §6.9 above, which exercises proxy 429 interception
      // end-to-end after the §15c-1 fix landed.
    });
  });

  // ═════════════════════════════════════════════════════════════════
  // PHASE 3 — MID-TEST DATA MUTATION
  // Tests the create/modify/delete lifecycle against the DO sync path.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 3: Data mutation", () => {
    beforeAll(async () => {
      if (!phase0Passed) {
        logFinding("phase3", "Phase 0 failed — Phase 3 fixtures NOT created", "warn");
        return;
      }
      // §8.1 budget increase mid-stream — start tight
      await createBudget("customer", `${PREFIX}-p3-increase`, 1);
      // §8.3 budget reset mid-stream — start generous
      await createBudget("customer", `${PREFIX}-p3-reset`, 1_000_000);
      // §8.4 customer ID collision — single shared customer
      await createBudget("customer", `${PREFIX}-p3-collision`, 1_000_000);
      // §8.5 plan tag — single shared customer
      await createBudget("customer", `${PREFIX}-p3-plan-tag`, 1_000_000);
      await waitForBudgetLive(`${PREFIX}-p3-collision`, 15_000);
    }, 60_000);

    // ── §8.1 Budget increase mid-stream ──
    it("8.1 budget increase mid-stream: deny → enlarge → success", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p3-increase`;
      await waitForBudgetLive(customer, 15_000);

      // Phase A: tight budget denies.
      const denyRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({
          "X-NullSpend-Customer": customer,
          "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.1" }),
        }),
        body: smallRequest({ messages: [{ role: "user", content: "p3.8.1.deny" }] }),
      });
      expect(denyRes.status).toBe(429);
      await denyRes.text();

      // Mutate: enlarge.
      await sql`
        UPDATE budgets
        SET max_budget_microdollars = 1000000, spend_microdollars = 0, updated_at = NOW()
        WHERE entity_type = 'customer' AND entity_id = ${customer} AND org_id = ${SMOKE_ORG_ID}
      `;
      await syncBudget(SMOKE_ORG_ID, "customer", customer);
      await waitForBudgetLive(customer, 15_000);

      // Phase B: success.
      const okRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({
          "X-NullSpend-Customer": customer,
          "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.1" }),
        }),
        body: smallRequest({ messages: [{ role: "user", content: "p3.8.1.ok" }] }),
      });
      expect(okRes.status).toBe(200);
      await okRes.json();
    }, 120_000);

    // ── §8.3 Budget spend reset mid-stream ──
    // The DO's `populateIfEmpty` is misnamed — it upserts max_budget but
    // PRESERVES current spend on conflict. To force a spend mutation we must
    // first REMOVE the entity from the DO, then re-sync from the new PG state.
    it("8.3 reset_spend mid-stream allows new requests after exhaustion", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p3-reset`;

      // Pre-exhaust: SET PG spend high, then REMOVE + RE-SYNC to force DO
      // to repopulate with the new spend value.
      await sql`
        UPDATE budgets
        SET spend_microdollars = 999999, updated_at = NOW()
        WHERE entity_type = 'customer' AND entity_id = ${customer} AND org_id = ${SMOKE_ORG_ID}
      `;
      await invalidateBudget(SMOKE_ORG_ID, "customer", customer, "remove");
      await syncBudget(SMOKE_ORG_ID, "customer", customer);
      // Active barrier: probe until DO returns customer_budget_exceeded for
      // THIS specific entity (not just any 429). Replaces the unreliable
      // fixed 1s sleep that codex flagged as a race.
      await waitForBudgetLive(customer, 15_000, "customer_budget_exceeded");

      // Probe — should deny (max=1_000_000, spend=999_999, remaining=1, est~3 → deny)
      const deny = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({
          "X-NullSpend-Customer": customer,
          "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.3" }),
        }),
        body: smallRequest({ messages: [{ role: "user", content: "p3.8.3.deny" }] }),
      });
      expect(deny.status).toBe(429);
      const denyBody = await deny.json().catch(() => null);
      expect((denyBody as { error?: { code?: string } } | null)?.error?.code).toBe("customer_budget_exceeded");

      // Reset: zero PG spend, REMOVE from DO, RE-SYNC.
      await sql`
        UPDATE budgets
        SET spend_microdollars = 0, updated_at = NOW()
        WHERE entity_type = 'customer' AND entity_id = ${customer} AND org_id = ${SMOKE_ORG_ID}
      `;
      await invalidateBudget(SMOKE_ORG_ID, "customer", customer, "remove");
      await syncBudget(SMOKE_ORG_ID, "customer", customer);
      // Active barrier: probe until DO returns 200 for THIS specific entity.
      // The default expectDeny="customer_budget_exceeded" is fine — once spend
      // is reset, the probe returns 200 and the function returns immediately.
      await waitForBudgetLive(customer, 15_000);

      const ok = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({
          "X-NullSpend-Customer": customer,
          "X-NullSpend-Tags": JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.3" }),
        }),
        body: smallRequest({ messages: [{ role: "user", content: "p3.8.3.ok" }] }),
      });
      expect(ok.status).toBe(200);
      await ok.json();
    }, 120_000);

    // ── §8.4 Customer ID collision (concurrent same customer, no leak) ──
    it("8.4 two concurrent sessions on same customer aggregate correctly", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p3-collision`;
      const tagA = JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.4", session: "A" });
      const tagB = JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.4", session: "B" });

      const [resA, resB] = await Promise.all([
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": tagA,
          }),
          body: smallRequest({ messages: [{ role: "user", content: "p3.8.4.A" }] }),
        }),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": tagB,
          }),
          body: smallRequest({ messages: [{ role: "user", content: "p3.8.4.B" }] }),
        }),
      ]);
      const idA = resA.headers.get("x-request-id");
      const idB = resB.headers.get("x-request-id");
      await resA.json();
      await resB.json();
      expect(resA.status).toBe(200);
      expect(resB.status).toBe(200);

      // Wait for both rows by request_id. The CASE WHEN unwrap is defensive
      // — handles both the new (object) and legacy (string-encoded) tag
      // shapes in case this test runs against a proxy build that predates
      // the cost-logger.ts:58 fix in this PR.
      let rows: { session: string | null; customer_id: string | null }[] = [];
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline && rows.length < 2) {
        rows = await sql<{ session: string | null; customer_id: string | null }[]>`
          SELECT
            (CASE WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb ELSE tags END)->>'session' AS session,
            customer_id
          FROM cost_events
          WHERE request_id IN (${idA!}, ${idB!}) AND provider = 'openai'
        `;
        if (rows.length < 2) await new Promise((r) => setTimeout(r, 1500));
      }
      const sessions = rows.map((r) => r.session).filter((s): s is string => !!s).sort();
      expect(sessions).toEqual(["A", "B"]);
    }, 120_000);

    // ── §8.5 Plan tag modification per session ──
    it("8.5 different plan tags per session do not cross-contaminate", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p3-plan-tag`;
      const tagFree = JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.5", plan: "free" });
      const tagPro = JSON.stringify({ stress_run_id: TEST_RUN_ID, stress_phase: "p3.8.5", plan: "pro" });

      await Promise.all([
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": tagFree,
          }),
          body: smallRequest({ messages: [{ role: "user", content: "p3.8.5.free" }] }),
        }).then((r) => r.text()),
        fetch(`${BASE}/v1/chat/completions`, {
          method: "POST",
          headers: stressAuthHeaders({
            "X-NullSpend-Customer": customer,
            "X-NullSpend-Tags": tagPro,
          }),
          body: smallRequest({ messages: [{ role: "user", content: "p3.8.5.pro" }] }),
        }).then((r) => r.text()),
      ]);

      // Same workaround as 8.4 (proxy stores tags as JSON-encoded string).
      let rows: { plan: string | null }[] = [];
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline && rows.length < 2) {
        rows = await sql<{ plan: string | null }[]>`
          SELECT (tags#>>'{}')::jsonb->>'plan' AS plan FROM cost_events
          WHERE customer_id = ${customer}
            AND (tags#>>'{}')::jsonb->>'stress_phase' = 'p3.8.5'
        `;
        if (rows.length < 2) await new Promise((r) => setTimeout(r, 1500));
      }
      const plans = rows.map((r) => r.plan).filter((p): p is string => !!p).sort();
      expect(plans).toEqual(["free", "pro"]);
    }, 120_000);
  });

  // ═════════════════════════════════════════════════════════════════
  // PHASE 5 — SDK SURFACE-AREA COVERAGE
  // Per docs/internal/test-plans/sdk-testing-gaps.md §"Stress test
  // Phase 5". Focused on stress-relevant gaps: attribution propagation
  // under load, customer session interface, tool events, error fields,
  // CostReporter callback firing, shutdown race. Pure functional tests
  // (HITL, read APIs, retry config) are deferred to the Functional E2E
  // suite.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 5: SDK surface area coverage", () => {
    let p5Ns: NullSpend;

    beforeAll(async () => {
      if (!phase0Passed) {
        logFinding("phase5", "Phase 0 failed — Phase 5 fixtures NOT created", "warn");
        return;
      }
      p5Ns = makeStressNs({ phase: "phase5" });
      // Single generous customer fixture for all Phase 5 tests.
      await createBudget("customer", `${PREFIX}-p5-customer-01`, 1_000_000);
      await waitForBudgetLive(`${PREFIX}-p5-customer-01`, 15_000);
    }, 60_000);

    afterAll(async () => {
      try { await p5Ns?.shutdown(); } catch { /* ignore */ }
    });

    // ── §5.1 Tracked fetch field propagation through proxy → cost_events ──
    it("5.1 sessionId / traceId / tags propagate through proxy to cost_events columns", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p5-customer-01`;
      const sessionId = `${PREFIX}-p5-1-session`;
      // W3C traceparent format: 32-char hex
      const traceId = "deadbeef".repeat(4);

      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: stressAuthHeaders({
          "X-NullSpend-Customer": customer,
          "x-nullspend-session": sessionId,
          "x-nullspend-trace-id": traceId,
          "X-NullSpend-Tags": JSON.stringify({ propagation: "yes", phase: "5.1" }),
        }),
        body: smallRequest({ messages: [{ role: "user", content: "p5.1" }] }),
      });
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-request-id");
      await res.json();
      expect(requestId).toBeTruthy();

      // Wait for the row to land.
      let row: { session_id: string | null; trace_id: string | null; tags_session: string | null; tags_propagation: string | null } | undefined;
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline && !row) {
        const rows = await sql<{ session_id: string | null; trace_id: string | null; tags_session: string | null; tags_propagation: string | null }[]>`
          SELECT
            session_id,
            trace_id,
            (CASE WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb ELSE tags END)->>'session' AS tags_session,
            (CASE WHEN jsonb_typeof(tags) = 'string' THEN (tags#>>'{}')::jsonb ELSE tags END)->>'propagation' AS tags_propagation
          FROM cost_events
          WHERE request_id = ${requestId!} AND provider = 'openai'
          LIMIT 1
        `;
        row = rows[0];
        if (!row) await new Promise((r) => setTimeout(r, 1500));
      }
      expect(row, "cost_events row never appeared").toBeDefined();
      expect(row!.session_id).toBe(sessionId);
      expect(row!.trace_id).toBe(traceId);
      expect(row!.tags_propagation).toBe("yes");
    }, 60_000);

    // ── §5.2 Same propagation through SDK direct ingest ──
    it("5.2 fields round-trip through SDK direct ingest path", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p5-customer-01`;
      const idem = `${PREFIX}-p5-2`;
      const sessionId = `${PREFIX}-p5-2-session`;
      const traceId = "abcd1234".repeat(4);

      const event = {
        provider: "openai",
        model: "gpt-4o-mini",
        inputTokens: 100,
        outputTokens: 5,
        costMicrodollars: 5,
        eventType: "custom" as const,
        customer,
        sessionId,
        traceId,
        tags: runTags({ stress_phase: "p5.2", propagation: "direct" }),
        idempotencyKey: idem,
      };
      const resp = await p5Ns.reportCost(event);
      expect(resp.id).toBeTruthy();

      const rawId = resp.id.replace(/^ns_evt_/, "");
      const rows = await sql<{ session_id: string | null; trace_id: string | null; customer_id: string | null; tags: object }[]>`
        SELECT session_id, trace_id, customer_id, tags FROM cost_events
        WHERE id = ${rawId}::uuid
        LIMIT 1
      `;
      expect(rows.length).toBe(1);
      expect(rows[0].session_id).toBe(sessionId);
      expect(rows[0].trace_id).toBe(traceId);
      expect(rows[0].customer_id).toBe(customer);
      // Direct ingest stores tags as proper jsonb object (not the proxy's
      // double-encoded string), so direct -> 'propagation' should work.
      const tagsObj = rows[0].tags as Record<string, string>;
      expect(tagsObj.propagation).toBe("direct");
    }, 30_000);

    // ── §5.3 CustomerSession interface — plan / sessionId / tags via session() ──
    it("5.3 customer().openai with plan + tags routes through session interface", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p5-customer-01`;

      // Use the actual customer() session interface, not raw stressAuthHeaders.
      // Direct mode (no proxyUrl) so the SDK takes the tracked path and
      // populates cost events with the session's tags via metadata.
      const directNs = new NullSpend({
        baseUrl: DASHBOARD_URL!,
        apiKey: STRESS_KEY_RAW,
        costReporting: { batchSize: 1, flushIntervalMs: 200 },
      });
      try {
        const session = directNs.customer(customer, {
          plan: "pro",
          tags: { stress_run_id: TEST_RUN_ID, stress_phase: "p5.3", source: "session-interface" },
        });
        // Verify customer session contract
        expect(session.customerId).toBe(customer);
        // Memoization: session.openai should be the same reference twice
        expect(session.openai).toBe(session.openai);
        expect(session.fetch("openai")).toBe(session.openai);

        const res = await session.openai("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
          body: smallRequest({ messages: [{ role: "user", content: "p5.3" }] }),
        });
        expect(res.status).toBe(200);
        await res.json();
        await directNs.flush();
        await directNs.shutdown();

        // Verify the cost event was tagged with plan + source via the session.
        const rows = await sql<{ tags: Record<string, string>; customer_id: string | null }[]>`
          SELECT tags, customer_id FROM cost_events
          WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: "p5.3" })}
            AND customer_id = ${customer}
            AND source = 'api'
          LIMIT 1
        `;
        expect(rows.length).toBe(1);
        expect(rows[0].tags.plan).toBe("pro");
        expect(rows[0].tags.source).toBe("session-interface");
        expect(rows[0].customer_id).toBe(customer);
      } finally {
        try { await directNs.shutdown(); } catch { /* ignore */ }
      }
    }, 60_000);

    // ── §5.4 Tool event tracking via direct ingest ──
    it("5.4 reportCost with eventType=tool persists toolName / toolServer columns", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p5-customer-01`;
      const event = {
        provider: "mcp",
        model: "tool-call",
        inputTokens: 0,
        outputTokens: 0,
        costMicrodollars: 100, // 0.0001 USD
        eventType: "tool" as const,
        toolName: "search_web",
        toolServer: "exa-mcp",
        customer,
        tags: runTags({ stress_phase: "p5.4" }),
        idempotencyKey: `${PREFIX}-p5-4`,
      };
      const resp = await p5Ns.reportCost(event);
      const rawId = resp.id.replace(/^ns_evt_/, "");
      const rows = await sql<{ event_type: string; tool_name: string | null; tool_server: string | null }[]>`
        SELECT event_type, tool_name, tool_server FROM cost_events
        WHERE id = ${rawId}::uuid
        LIMIT 1
      `;
      expect(rows.length).toBe(1);
      expect(rows[0].event_type).toBe("tool");
      expect(rows[0].tool_name).toBe("search_web");
      expect(rows[0].tool_server).toBe("exa-mcp");
    }, 30_000);

    // ── §5.5 CostEventInput field round-trip ──
    it("5.5 cachedInputTokens / reasoningTokens / costBreakdown / durationMs round-trip", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const customer = `${PREFIX}-p5-customer-01`;
      const event = {
        provider: "openai",
        model: "gpt-4o-mini",
        inputTokens: 1000,
        outputTokens: 50,
        cachedInputTokens: 200,
        reasoningTokens: 30,
        costMicrodollars: 250,
        costBreakdown: {
          input: 100,
          output: 80,
          cached: 20,
          reasoning: 50,
        },
        durationMs: 1500,
        eventType: "llm" as const,
        customer,
        tags: runTags({ stress_phase: "p5.5" }),
        idempotencyKey: `${PREFIX}-p5-5`,
      };
      const resp = await p5Ns.reportCost(event);
      const rawId = resp.id.replace(/^ns_evt_/, "");
      const rows = await sql<{
        input_tokens: number;
        output_tokens: number;
        cached_input_tokens: number;
        reasoning_tokens: number;
        cost_microdollars: string;
        duration_ms: number | null;
        cost_breakdown: { input?: number; output?: number; cached?: number; reasoning?: number } | null;
        event_type: string;
      }[]>`
        SELECT input_tokens, output_tokens, cached_input_tokens, reasoning_tokens,
               cost_microdollars::text, duration_ms, cost_breakdown, event_type
        FROM cost_events WHERE id = ${rawId}::uuid LIMIT 1
      `;
      expect(rows.length).toBe(1);
      const r = rows[0];
      expect(r.input_tokens).toBe(1000);
      expect(r.output_tokens).toBe(50);
      expect(r.cached_input_tokens).toBe(200);
      expect(r.reasoning_tokens).toBe(30);
      expect(Number(r.cost_microdollars)).toBe(250);
      expect(r.duration_ms).toBe(1500);
      expect(r.event_type).toBe("llm");
      expect(r.cost_breakdown).toBeDefined();
      expect(r.cost_breakdown!.input).toBe(100);
      expect(r.cost_breakdown!.output).toBe(80);
      expect(r.cost_breakdown!.cached).toBe(20);
      expect(r.cost_breakdown!.reasoning).toBe(50);
    }, 30_000);

    // ── §5.6 Error class field validation ──
    it("5.6 BudgetExceededError carries entityType / entityId / limit / spend fields", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      // Use a fresh tight customer for this test so the field assertions
      // are deterministic. Reuse the §6.7b user-budget exhausted state.
      let caught: BudgetExceededError | undefined;
      const tracked = p5Ns.createTrackedFetch("openai", {
        enforcement: true,
        customer: `${PREFIX}-p5-customer-01`,
      });
      try {
        await tracked("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
          body: smallRequest(),
        });
      } catch (err) {
        if (err instanceof BudgetExceededError) caught = err;
        else throw err;
      }
      expect(caught).toBeDefined();
      // Field-level assertions: a regression that nulled any of these would
      // slip past the instanceof check that other tests rely on.
      expect(caught!.entityType).toBeDefined();
      expect(caught!.entityId).toBeDefined();
      expect(caught!.limitMicrodollars).toBeDefined();
      expect(caught!.spendMicrodollars).toBeDefined();
      expect(typeof caught!.remainingMicrodollars).toBe("number");
      expect(caught!.remainingMicrodollars).toBeGreaterThanOrEqual(0);
    }, 30_000);

    it("5.6b MandateViolationError carries mandate / requested / allowed fields", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const mandateNs = new NullSpend({
        baseUrl: DASHBOARD_URL!,
        apiKey: STRESS_KEY_RAW_MANDATE,
        proxyUrl: BASE,
        costReporting: { batchSize: 1, flushIntervalMs: 200 },
      });
      try {
        let caught: MandateViolationError | undefined;
        const tracked = mandateNs.createTrackedFetch("openai", {
          enforcement: true,
          customer: `${PREFIX}-p5-customer-01`,
        });
        try {
          await tracked("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
            body: smallRequest({ model: "gpt-4-turbo" }),
          });
        } catch (err) {
          if (err instanceof MandateViolationError) caught = err;
          else throw err;
        }
        expect(caught).toBeDefined();
        expect(caught!.mandate).toBeTruthy();
        expect(caught!.requested).toBe("gpt-4-turbo");
        expect(Array.isArray(caught!.allowed)).toBe(true);
        expect(caught!.allowed).toContain("gpt-4o-mini");
      } finally {
        try { await mandateNs.shutdown(); } catch { /* ignore */ }
      }
    }, 30_000);

    // ── §5.7 onCostError firing on cost reporting failure ──
    it("5.7 onCostError fires when cost ingest hits a hard error", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      // CostReporter pointed at an unreachable baseUrl. Tracked fetch
      // generates a cost event, queue tries to flush, hits a network error,
      // onFlushError fires (the actual flush callback). onCostError on
      // tracked-fetch fires for in-loop tracking errors (e.g., body extract
      // failures), not flush errors — the more reliable assertion here is
      // onFlushError.
      let flushErrors = 0;
      let lastError: Error | undefined;
      const failNs = new NullSpend({
        baseUrl: "http://127.0.0.1:1", // ECONNREFUSED
        apiKey: STRESS_KEY_RAW,
        requestTimeoutMs: 1000,
        maxRetries: 0,
        costReporting: {
          batchSize: 1,
          flushIntervalMs: 100,
          onFlushError: (err) => {
            flushErrors++;
            lastError = err;
          },
        },
      });
      try {
        // Queue an event and force a flush. Should fail and call onFlushError.
        failNs.queueCost({
          provider: "openai",
          model: "gpt-4o-mini",
          inputTokens: 100,
          outputTokens: 5,
          costMicrodollars: 5,
          eventType: "custom",
          customer: `${PREFIX}-p5-customer-01`,
          tags: runTags({ stress_phase: "p5.7" }),
        });
        await failNs.flush().catch(() => { /* expected */ });
        // Give the callback a moment to fire if it's async.
        await new Promise((r) => setTimeout(r, 200));
      } finally {
        try { await failNs.shutdown(); } catch { /* expected */ }
      }
      expect(flushErrors).toBeGreaterThan(0);
      expect(lastError).toBeDefined();
      expect(lastError!.message.length).toBeGreaterThan(0);
    }, 30_000);

    // ── §5.9 shutdown contract: drains pending events before returning ──
    // This test verifies the BASIC shutdown contract: queueCost N events,
    // then shutdown(). All N events should land in PG.
    //
    // Configured with high batchSize (so auto-flush doesn't fire) and
    // long flushIntervalMs (so timer doesn't fire). The only flush is
    // the implicit one in shutdown().
    it("5.9 shutdown() drains all queued events before returning", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const drainNs = new NullSpend({
        baseUrl: DASHBOARD_URL!,
        apiKey: STRESS_KEY_RAW,
        costReporting: {
          batchSize: 100,           // never auto-flush
          flushIntervalMs: 100_000, // never timer-flush
        },
      });
      // Use unique phase tag for this test so the count is deterministic.
      const phaseTag = "p5.9";
      const N = 20;
      try {
        for (let i = 0; i < N; i++) {
          drainNs.queueCost({
            provider: "openai",
            model: "gpt-4o-mini",
            inputTokens: 100,
            outputTokens: 5,
            costMicrodollars: 5,
            eventType: "custom",
            customer: `${PREFIX}-p5-customer-01`,
            tags: runTags({ stress_phase: phaseTag }),
            // Explicit idempotency keys so we can count exact rows.
            // (The SDK accepts this as an extra field.)
            idempotencyKey: `${PREFIX}-p5-9-${i}`,
          } as CostEventInput & { idempotencyKey: string });
        }
        // Single shutdown call. No racing flush(). The basic contract:
        // shutdown() must NOT return until all queued events are flushed.
        await drainNs.shutdown();
      } catch (err) {
        expect.fail(`shutdown threw: ${(err as Error).message}`);
      }

      // After shutdown returns, all N events should be in PG. Allow brief
      // settle for the dashboard insert to commit.
      let count = 0;
      const deadline = Date.now() + 10_000;
      while (Date.now() < deadline && count < N) {
        const rows = await sql<{ count: string }[]>`
          SELECT COUNT(*)::text AS count FROM cost_events
          WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: phaseTag })}
        `;
        count = Number(rows[0]?.count ?? 0);
        if (count < N) await new Promise((r) => setTimeout(r, 500));
      }
      expect(count).toBe(N);
    }, 60_000);

    // ── §5.10 shutdown race regression test (post-15c-18 fix) ──
    // Pre-fix: queueCost calls during an in-flight flush were dropped when
    // shutdown() ran before the queue emptied. CostReporter.shutdown() has
    // since been fixed (drain loop) — this test verifies the fix.
    //
    // Setup: low batchSize (5) so the first 5 events trigger an auto-flush,
    // then 15 more get enqueued during the in-flight flush, then shutdown()
    // races against that flush. ALL 20 events must land in PG.
    it("5.10 shutdown race: drains queue after fix (15c-18)", async (ctx) => {
      if (!dashboardReachable || !phase0Passed) ctx.skip();
      const raceNs = new NullSpend({
        baseUrl: DASHBOARD_URL!,
        apiKey: STRESS_KEY_RAW,
        costReporting: {
          batchSize: 5,        // auto-flush after 5 — triggers the race
          flushIntervalMs: 50,
        },
      });
      const phaseTag = "p5.10";
      const N = 20;
      try {
        for (let i = 0; i < N; i++) {
          raceNs.queueCost({
            provider: "openai",
            model: "gpt-4o-mini",
            inputTokens: 100,
            outputTokens: 5,
            costMicrodollars: 5,
            eventType: "custom",
            customer: `${PREFIX}-p5-customer-01`,
            tags: runTags({ stress_phase: phaseTag }),
            idempotencyKey: `${PREFIX}-p5-10-${i}`,
          } as CostEventInput & { idempotencyKey: string });
        }
        // Race: explicit flush() + shutdown() in parallel. With the
        // 15c-18 fix, shutdown's drain loop catches events queued
        // during the in-flight flush.
        await Promise.all([raceNs.flush(), raceNs.shutdown()]);
      } catch (err) {
        expect.fail(`shutdown race threw: ${(err as Error).message}`);
      }

      // After shutdown, ALL N events should have landed.
      let count = 0;
      const deadline = Date.now() + 15_000;
      while (Date.now() < deadline && count < N) {
        const rows = await sql<{ count: string }[]>`
          SELECT COUNT(*)::text AS count FROM cost_events
          WHERE tags @> ${sql.json({ stress_run_id: TEST_RUN_ID, stress_phase: phaseTag })}
        `;
        count = Number(rows[0]?.count ?? 0);
        if (count < N) await new Promise((r) => setTimeout(r, 500));
      }
      expect(count).toBe(N);
    }, 60_000);
  });

  // ═════════════════════════════════════════════════════════════════
  // PHASE 4 — VERIFICATION
  // §9.1 cost events integrity, §9.2 budget spend accuracy. §9.5 orphan
  // check is inside afterAll per §15a-15.
  // ═════════════════════════════════════════════════════════════════
  describe("Phase 4: Verification", () => {
    it("9.1 cost events integrity: customer_id always populated, no duplicates", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      // Wait for any in-flight events to settle.
      await waitForQueueDrain(TEST_RUN_ID, { maxWaitMs: 30_000, pollIntervalMs: 2_000, stableForSamples: 2 });

      // The proxy stores cost_events.tags as a JSON-encoded STRING (real bug
      // logged in §6.9.4 / 8.4). We scope by user_id (defensive, catches both
      // attribution paths) and assert tighter properties than just "rows
      // exist": every event MUST have a populated customer_id (every
      // proxy/SDK call in this suite carries a customer), and no
      // (request_id, provider) duplicates.
      const counts = await sql<{ customer_id: string | null; cnt: string }[]>`
        SELECT customer_id, COUNT(*)::text AS cnt FROM cost_events
        WHERE user_id = ${STRESS_USER_ID}
        GROUP BY customer_id
        ORDER BY customer_id
      `;
      console.log(`[stress-sdk] 9.1 events by customer:`, counts);
      expect(counts.length).toBeGreaterThan(0);

      // No event from this run should have a null customer_id. Every test
      // path explicitly sets X-NullSpend-Customer or `customer:` on direct
      // ingest. A null here means the proxy / SDK silently dropped attribution.
      const nullCustomer = counts.find((c) => c.customer_id === null);
      if (nullCustomer) {
        logFinding(
          "phase4.9.1",
          `${nullCustomer.cnt} cost_events rows with NULL customer_id from STRESS_USER_ID — proxy/SDK dropped attribution`,
          "bug",
        );
      }
      expect(nullCustomer, `Expected no NULL customer_id rows, got ${nullCustomer?.cnt ?? 0}`).toBeUndefined();

      const dupes = await sql<{ request_id: string; provider: string; cnt: string }[]>`
        SELECT request_id, provider, COUNT(*)::text AS cnt FROM cost_events
        WHERE user_id = ${STRESS_USER_ID}
        GROUP BY request_id, provider
        HAVING COUNT(*) > 1
      `;
      expect(dupes.length).toBe(0);
    }, 60_000);

    it("9.2 budget spend accuracy: p2 generous customer fixtures match sum of cost events", async (ctx) => {
      if (!phase0Passed) ctx.skip();
      // Scope to ONLY the p2 generous customer fixtures where Phase 2's happy
      // path writes known cost events. Mutation fixtures (p3-*) have
      // intentionally-drifted spend and would produce noise. Other fixtures
      // had 0 traffic so both sides are 0 and the assertion is a no-op.
      const rows = await sql<{ entity_id: string; spend: string; computed_sum: string }[]>`
        SELECT
          b.entity_id,
          b.spend_microdollars::text AS spend,
          COALESCE(SUM(c.cost_microdollars)::text, '0') AS computed_sum
        FROM budgets b
        LEFT JOIN cost_events c ON c.customer_id = b.entity_id
          AND c.user_id = ${STRESS_USER_ID}
        WHERE b.org_id = ${SMOKE_ORG_ID}
          AND b.entity_type = 'customer'
          AND b.entity_id LIKE ${`${PREFIX}-p2-customer-%`}
        GROUP BY b.entity_id, b.spend_microdollars
      `;

      // Require that fixtures exist — catches silent "no rows" false pass.
      expect(rows.length).toBeGreaterThan(0);

      let driftCount = 0;
      for (const r of rows) {
        const spend = Number(r.spend);
        const sum = Number(r.computed_sum);
        const delta = Math.abs(spend - sum);
        // Tolerance: 50% relative OR 100 µ¢ absolute floor, whichever is larger.
        // Queue lag + estimator variance is typically <20 µ¢ per event.
        const tolerance = Math.max(100, sum * 0.5);
        if (sum > 0 && delta > tolerance) {
          driftCount++;
          logFinding(
            "phase4.9.2",
            `Spend drift on ${r.entity_id}: budget.spend=${spend}, sum(events)=${sum}, delta=${delta}, tolerance=${tolerance}`,
            "warn",
          );
        }
      }
      // Hard-fail if ANY customer drifts outside tolerance.
      expect(driftCount).toBe(0);
    }, 60_000);
  });
});

// ───────────────────────────────────────────────────────────────────
// Local helpers not exported — placed at file bottom.
// ───────────────────────────────────────────────────────────────────

/**
 * Hash an API key using SHA-256 — matches lib/auth/api-key.ts:32-34
 * exactly so the stress-inserted key is accepted by the dashboard.
 */
function hashKeySha256(rawKey: string): string {
  return crypto.createHash("sha256").update(rawKey).digest("hex");
}
