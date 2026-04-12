/**
 * ST-7: Margin sync + proxy interaction smoke test.
 *
 * Full production-parity test:
 * 1. Creates Stripe test fixtures (customer + paid $50 invoice)
 * 2. Encrypts Stripe key with real STRIPE_ENCRYPTION_KEY (same as production)
 * 3. Stores connection in stripe_connections via SQL
 * 4. Triggers sync via GET /api/stripe/revenue-sync with CRON_SECRET (same as Vercel Cron)
 * 5. Verifies revenue data in DB
 * 6. Verifies margins API returns data
 * 7. Confirms proxy enforcement is unaffected
 *
 * Prerequisites (.env.smoke):
 *   - STRIPE_SECRET_KEY or STRIPE_TEST_KEY (sk_test_...)
 *   - STRIPE_ENCRYPTION_KEY (from Vercel — same key production uses)
 *   - CRON_SECRET (from Vercel — triggers revenue sync)
 *   - DATABASE_URL
 *   - PROXY_URL, OPENAI_API_KEY, NULLSPEND_API_KEY, NULLSPEND_SMOKE_KEY_ID
 *   - NULLSPEND_DASHBOARD_URL (deployed or local dashboard)
 *
 * Manual-only — never CI.
 */
import { createCipheriv, randomBytes } from "node:crypto";
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import postgres from "postgres";
import Stripe from "stripe";
import {
  BASE,
  OPENAI_API_KEY,
  NULLSPEND_API_KEY,
  authHeaders,
  smallRequest,
  isServerUp,
} from "./smoke-test-helpers.js";

const DASHBOARD_URL = process.env.NULLSPEND_DASHBOARD_URL ?? "http://127.0.0.1:3000";
const STRIPE_TEST_KEY = process.env.STRIPE_TEST_KEY ?? process.env.STRIPE_SECRET_KEY;
const STRIPE_ENCRYPTION_KEY = process.env.STRIPE_ENCRYPTION_KEY;
const CRON_SECRET = process.env.CRON_SECRET;

function encryptStripeKey(plaintext: string, orgId: string): string {
  if (!STRIPE_ENCRYPTION_KEY) throw new Error("STRIPE_ENCRYPTION_KEY required");
  const key = Buffer.from(STRIPE_ENCRYPTION_KEY, "base64");
  if (key.length !== 32) throw new Error("STRIPE_ENCRYPTION_KEY must be 32 bytes base64");
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  cipher.setAAD(Buffer.from(orgId, "utf8"));
  const encrypted = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, encrypted, tag]).toString("base64");
}

describe("ST-7: Margin sync + proxy interaction", () => {
  let sql: postgres.Sql;
  let orgId: string;
  let stripe: Stripe;
  let stripeCustomerId: string;
  let stripeProductId: string;
  let hadExistingConnection = false;
  let skip = false;

  beforeAll(async () => {
    // Prerequisite checks
    const proxyUp = await isServerUp();
    if (!proxyUp) throw new Error("Proxy not reachable at " + BASE);

    const missing: string[] = [];
    if (!OPENAI_API_KEY) missing.push("OPENAI_API_KEY");
    if (!NULLSPEND_API_KEY) missing.push("NULLSPEND_API_KEY");
    if (!STRIPE_TEST_KEY) missing.push("STRIPE_SECRET_KEY or STRIPE_TEST_KEY");
    if (!STRIPE_ENCRYPTION_KEY) missing.push("STRIPE_ENCRYPTION_KEY");
    if (!CRON_SECRET) missing.push("CRON_SECRET");
    if (!process.env.DATABASE_URL) missing.push("DATABASE_URL");
    if (!process.env.NULLSPEND_SMOKE_KEY_ID) missing.push("NULLSPEND_SMOKE_KEY_ID");

    if (missing.length > 0) {
      console.log(`[ST-7] Missing env vars — skipping: ${missing.join(", ")}`);
      skip = true;
      return;
    }

    // Check dashboard reachable
    try {
      await fetch(`${DASHBOARD_URL}/api/health`, { signal: AbortSignal.timeout(3000) });
    } catch {
      throw new Error(`Dashboard not reachable at ${DASHBOARD_URL}`);
    }

    sql = postgres(process.env.DATABASE_URL!, { max: 3, idle_timeout: 10 });

    const [key] = await sql`SELECT org_id FROM api_keys WHERE id = ${process.env.NULLSPEND_SMOKE_KEY_ID!}`;
    if (!key?.org_id) throw new Error("Smoke test API key has no org_id");
    orgId = key.org_id;

    const [existing] = await sql`SELECT id FROM stripe_connections WHERE org_id = ${orgId}`;
    hadExistingConnection = !!existing;

    // Create Stripe test fixtures
    stripe = new Stripe(STRIPE_TEST_KEY!, { apiVersion: "2024-12-18.acacia" as Stripe.LatestApiVersion });

    const customer = await stripe.customers.create({
      name: "ST-7 Smoke Test Customer",
      email: "st7-smoke@nullspend.dev",
      metadata: { nullspend_customer: "st7-smoke-customer" },
    });
    stripeCustomerId = customer.id;

    const product = await stripe.products.create({ name: "ST-7 Test API Usage" });
    stripeProductId = product.id;

    const price = await stripe.prices.create({
      product: product.id,
      unit_amount: 5000, // $50.00
      currency: "usd",
    });

    // Attach test card for auto-pay
    const pm = await stripe.paymentMethods.create({ type: "card", card: { token: "tok_visa" } });
    await stripe.paymentMethods.attach(pm.id, { customer: customer.id });
    await stripe.customers.update(customer.id, {
      invoice_settings: { default_payment_method: pm.id },
    });

    const invoice = await stripe.invoices.create({
      customer: customer.id,
      collection_method: "charge_automatically",
      auto_advance: false,
    });
    await stripe.invoiceItems.create({ customer: customer.id, invoice: invoice.id, price: price.id });
    const finalized = await stripe.invoices.finalizeInvoice(invoice.id);
    const paid = await stripe.invoices.pay(finalized.id);
    expect(paid.status).toBe("paid");

    console.log(`[ST-7] Stripe fixtures created: customer=${stripeCustomerId}, invoice=${paid.id}, amount=$50.00`);
  }, 60_000);

  afterAll(async () => {
    if (stripe) {
      try {
        if (stripeCustomerId) await stripe.customers.del(stripeCustomerId);
        if (stripeProductId) await stripe.products.update(stripeProductId, { active: false });
      } catch (err) {
        console.log("[ST-7] Stripe cleanup:", err);
      }
    }
    if (sql) {
      await sql`DELETE FROM customer_revenue WHERE stripe_customer_id = ${stripeCustomerId ?? ""}`.catch(() => {});
      await sql`DELETE FROM customer_mappings WHERE tag_value = 'st7-smoke-customer' AND org_id = ${orgId}`.catch(() => {});
      if (!hadExistingConnection) {
        await sql`DELETE FROM stripe_connections WHERE org_id = ${orgId}`.catch(() => {});
      }
      await sql.end();
    }
  });

  it("encrypts and stores Stripe connection (production-parity encryption)", async () => {
    if (skip) return;

    if (hadExistingConnection) {
      console.log("[ST-7] Existing Stripe connection found — using it");
      return;
    }

    const encryptedKey = encryptStripeKey(STRIPE_TEST_KEY!, orgId);
    const keyPrefix = STRIPE_TEST_KEY!.slice(0, 12) + "..." + STRIPE_TEST_KEY!.slice(-4);

    await sql`
      INSERT INTO stripe_connections (org_id, encrypted_key, key_prefix, status)
      VALUES (${orgId}, ${encryptedKey}, ${keyPrefix}, 'active')
      ON CONFLICT (org_id) DO UPDATE SET
        encrypted_key = ${encryptedKey}, key_prefix = ${keyPrefix},
        status = 'active', updated_at = NOW()
    `;

    // Verify it was stored
    const [conn] = await sql`SELECT status, key_prefix FROM stripe_connections WHERE org_id = ${orgId}`;
    expect(conn.status).toBe("active");
    console.log(`[ST-7] Stripe connection stored: ${conn.key_prefix} (encrypted with production key)`);
  }, 30_000);

  it("revenue sync via CRON_SECRET fetches the paid invoice", async () => {
    if (skip) return;

    // Trigger sync the same way Vercel Cron does
    const res = await fetch(`${DASHBOARD_URL}/api/stripe/revenue-sync`, {
      method: "GET",
      headers: { Authorization: `Bearer ${CRON_SECRET}` },
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    console.log("[ST-7] Sync response:", JSON.stringify(body.data).slice(0, 300));

    // Wait for async processing
    await new Promise((r) => setTimeout(r, 3_000));

    // Verify revenue in DB
    const rows = await sql`
      SELECT amount_microdollars::text as amount, customer_name
      FROM customer_revenue
      WHERE org_id = ${orgId} AND stripe_customer_id = ${stripeCustomerId}
    `;

    expect(rows.length).toBeGreaterThan(0);
    const amount = Number(rows[0].amount);
    // $50.00 = 5000 cents = 50,000,000 microdollars
    expect(amount).toBe(50_000_000);
    console.log(`[ST-7] Revenue verified: ${rows[0].customer_name} = $${(amount / 1_000_000).toFixed(2)}`);
  }, 60_000);

  it("margins API responds and customer_revenue has synced data", async () => {
    if (skip) return;

    // Verify revenue exists in DB (sync was successful)
    const revenueRows = await sql`
      SELECT COUNT(*)::int as count, COALESCE(SUM(amount_microdollars), 0)::text as total
      FROM customer_revenue
      WHERE org_id = ${orgId}
    `;
    const revenueCount = revenueRows[0].count as number;
    const revenueTotal = Number(revenueRows[0].total);
    console.log(`[ST-7] customer_revenue: ${revenueCount} rows, total=${revenueTotal}µ¢`);
    expect(revenueCount).toBeGreaterThan(0);

    // Check if auto-match created a mapping
    const mappingRows = await sql`
      SELECT tag_value, stripe_customer_id, match_type
      FROM customer_mappings
      WHERE org_id = ${orgId} AND tag_value = 'st7-smoke-customer'
    `;
    console.log(`[ST-7] Auto-match mappings: ${mappingRows.length}`);

    // Margins API should respond (may show 0 if no mappings exist yet)
    const res = await fetch(`${DASHBOARD_URL}/api/margins`, {
      headers: { "x-nullspend-key": NULLSPEND_API_KEY! },
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    const data = body.data;
    console.log(`[ST-7] Margins API: revenue=${data.summary.totalRevenueMicrodollars}µ¢, customers=${data.customers.length}, syncStatus=${data.summary.syncStatus}`);

    // If mappings exist, revenue should be > 0. If not, that's expected —
    // auto-match requires cost events with matching customer tags.
    if (mappingRows.length > 0) {
      expect(data.summary.totalRevenueMicrodollars).toBeGreaterThan(0);
    } else {
      console.log("[ST-7] No customer mappings — margins show 0 (expected: auto-match needs cost events with customer tag)");
    }
  }, 30_000);

  it("proxy budget enforcement is unaffected during margin sync", async () => {
    if (skip) return;

    const res = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: authHeaders(),
      body: smallRequest(),
    });

    expect([200, 429]).toContain(res.status);
    await res.text();
    console.log(`[ST-7] Proxy after sync: ${res.status}`);
  }, 30_000);
});
