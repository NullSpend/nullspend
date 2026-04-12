/**
 * Demo: Propose a "send_email" action and wait for approval.
 *
 * Usage:
 *   1. Start the NullSpend app:        pnpm dev
 *   2. Run this script in another terminal:
 *      pnpm tsx packages/sdk/examples/demo-send-email.ts
 *   3. Open http://localhost:3000/app/inbox and approve the action.
 *   4. Watch this terminal — the "email" will send after approval.
 *
 * Environment variables:
 *   NULLSPEND_URL     Base URL of your NullSpend instance (default: http://localhost:3000)
 *   NULLSPEND_API_KEY Your API key from the Settings page (or NULLSPEND_API_KEY env var)
 */

import { NullSpend, RejectedError, TimeoutError } from "../src/index.js";

const baseUrl = process.env.NULLSPEND_URL ?? "http://localhost:3000";
const apiKey = process.env.NULLSPEND_API_KEY ?? "";

if (!apiKey) {
  console.error(
    "Set NULLSPEND_API_KEY to a valid key from your Settings page.",
  );
  process.exit(1);
}

const seam = new NullSpend({ baseUrl, apiKey });

async function fakeSendEmail(to: string, subject: string, body: string) {
  console.log(`  [email] To: ${to}`);
  console.log(`  [email] Subject: ${subject}`);
  console.log(`  [email] Body: ${body}`);
  return { messageId: "msg_demo_123", to, subject };
}

async function main() {
  console.log("Proposing send_email action...");
  console.log(`  Dashboard: ${baseUrl}/app/inbox`);
  console.log("  Waiting for approval (polls every 3s, timeout 5min)...\n");

  try {
    const result = await seam.proposeAndWait({
      agentId: "demo-agent",
      actionType: "send_email",
      payload: {
        to: "sarah@example.com",
        subject: "Q1 Report Follow-up",
        body: "Hi Sarah, just following up on the Q1 report. Let me know if you need anything.",
      },
      execute: () =>
        fakeSendEmail(
          "sarah@example.com",
          "Q1 Report Follow-up",
          "Hi Sarah, just following up on the Q1 report.",
        ),
      pollIntervalMs: 3_000,
      timeoutMs: 5 * 60 * 1_000,
      onPoll: (action) => {
        console.log(`  [poll] status: ${action.status}`);
      },
    });

    console.log("\nAction approved and executed!");
    console.log("Result:", result);
  } catch (err) {
    if (err instanceof RejectedError) {
      console.log(`\nAction was ${err.actionStatus}. Email NOT sent.`);
    } else if (err instanceof TimeoutError) {
      console.log("\nTimed out waiting for a decision. Email NOT sent.");
    } else {
      throw err;
    }
  }
}

main();
