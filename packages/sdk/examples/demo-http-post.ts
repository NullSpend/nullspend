/**
 * Demo: Propose an "http_post" action and wait for approval.
 *
 * Simulates an agent that wants to POST data to an external API
 * (e.g. sending a lead to a CRM or posting a message to Slack).
 * Uses jsonplaceholder.typicode.com as a safe, public test endpoint.
 *
 * Usage:
 *   1. Start the NullSpend app:        pnpm dev
 *   2. Run this script in another terminal:
 *      pnpm tsx packages/sdk/examples/demo-http-post.ts
 *   3. Open http://localhost:3000/app/inbox and approve the action.
 *   4. Watch this terminal — the HTTP POST will fire after approval.
 *
 * Environment variables:
 *   NULLSPEND_URL     Base URL of your NullSpend instance (default: http://localhost:3000)
 *   NULLSPEND_API_KEY Your API key from the Settings page
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

const TARGET_URL = "https://jsonplaceholder.typicode.com/posts";

const requestBody = {
  title: "New lead from AI sales agent",
  body: JSON.stringify({
    contact: "Sarah Chen",
    email: "sarah.chen@example.com",
    company: "Acme Corp",
    source: "ai-sales-agent",
    score: 87,
  }),
  userId: 1,
};

async function executeHttpPost() {
  console.log(`  [http] POST ${TARGET_URL}`);
  console.log(`  [http] Body: ${JSON.stringify(requestBody, null, 2)}`);

  const response = await fetch(TARGET_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const result = await response.json();
  console.log(`  [http] Response: ${JSON.stringify(result)}`);
  return result as Record<string, unknown>;
}

async function main() {
  console.log("Proposing http_post action...");
  console.log(`  Dashboard: ${baseUrl}/app/inbox`);
  console.log("  Waiting for approval (polls every 3s, timeout 5min)...\n");

  try {
    const result = await seam.proposeAndWait({
      agentId: "demo-agent",
      actionType: "http_post",
      payload: {
        url: TARGET_URL,
        method: "POST",
        body: requestBody,
        description: "Post new CRM lead for Sarah Chen (Acme Corp)",
      },
      execute: executeHttpPost,
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
      console.log(`\nAction was ${err.actionStatus}. HTTP POST NOT sent.`);
    } else if (err instanceof TimeoutError) {
      console.log("\nTimed out waiting for a decision. HTTP POST NOT sent.");
    } else {
      throw err;
    }
  }
}

main();
