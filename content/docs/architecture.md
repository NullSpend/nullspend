---
title: "Architecture"
description: "How NullSpend enforces budgets in-path with sub-20ms overhead, attributes cost across four surfaces, and stays open source from end to end."
---

NullSpend has four control surfaces, one shared budget engine, and a thin reporting plane. Every piece is Apache-2.0 and runs on your infrastructure if you want it to.

## The four surfaces

The same budget enforcement applies regardless of how a request enters NullSpend.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  YOUR APP / AGENT                                                   │
  └──────┬──────────────┬─────────────────┬──────────────────┬────────┘
         │              │                 │                  │
     [PROXY]         [SDK]         [CLAUDE AGENT]        [MCP]
     in-path,        per-call         adapter          server + proxy
     drop-in env     control       (Anthropic SDK)     agent-native
         │              │                 │                  │
         └──────┬───────┴────────┬────────┴────────┬─────────┘
                ▼                ▼                 ▼
              ┌─────────────────────────┐
              │   BUDGET ENGINE         │
              │   check → reserve →     │
              │   spend → reconcile     │
              └────────────┬────────────┘
                           ▼
               OpenAI · Anthropic · Gemini
```

### Proxy — in-path enforcement

A Cloudflare Workers proxy wraps OpenAI, Anthropic, and Google endpoints. One environment variable swaps you in:

```bash
OPENAI_BASE_URL=https://proxy.nullspend.dev/v1
```

Every request goes through `check-and-reserve` against the budget engine before reaching the upstream provider. If the budget is exhausted, the proxy returns a `429` with a structured envelope **before** any provider tokens are spent. Sub-20ms overhead at p50.

Unbypassable. Your application cannot route around it without rewriting the base URL, which is a config change, not a code change.

### SDK — per-call control

When the proxy can't see a call (embedded inference, custom transport, third-party agent framework), the SDK gives you the same accounting in TypeScript or Python:

```ts
import { NullSpend } from "@nullspend/sdk";

const ns = new NullSpend({ apiKey });
const fetch = ns.createTrackedFetch("openai");
```

`createTrackedFetch` wraps the standard `fetch`, calculates cost from response token counts, and reports to the same budget engine the proxy uses. Same enforcement guarantees, applied client-side.

### Claude Agent adapter

`@nullspend/claude-agent` wraps the Claude Agent SDK so every model call routes through the NullSpend proxy with no other code changes. Same accounting and enforcement as the proxy, surfaced through the agent runtime your code already uses.

```ts
import { withNullSpend } from "@nullspend/claude-agent";
import { ClaudeAgent } from "@anthropic-ai/claude-agent-sdk";

const agent = withNullSpend(new ClaudeAgent({ /* ... */ }), { apiKey });
```

### MCP — server and proxy

For autonomous agents that need to ask for their own budget, `@nullspend/mcp-server` exposes three tools any MCP client can call:

- `check_budget` — am I allowed to spend $X?
- `request_budget` — increase my budget, route to a human if needed
- `get_cost` — what have I spent so far this session?

`@nullspend/mcp-proxy` complements it by gating *upstream* MCP tool calls through budget enforcement and HITL approval — so an agent invoking an external MCP tool is governed the same way as one invoking an LLM.

Works with Claude, Cursor, or any MCP client. The agent self-governs against the same budget state the proxy, SDK, and Claude Agent adapter enforce.

## Budget engine

The core invariant: a single state owner per `(org, key)` pair. State lives in a Cloudflare Durable Object so writes serialize without distributed locks.

The lifecycle of every request:

1. **Check** — does the budget have headroom for the estimated cost?
2. **Reserve** — atomically subtract the estimated cost from available budget
3. **Spend** — provider returns; compute exact cost from real token counts
4. **Reconcile** — release the reservation, charge the actual cost, emit a cost event

Reservations expire if the request hangs, so a stuck call cannot permanently lock budget. Reconciliation is idempotent (idempotent: replays don't double-charge), so retries and partial failures don't corrupt accounting.

## Cost engine

Pricing data is a JSON-checked-into-the-repo (`packages/cost-engine/src/pricing-data.json`). 56+ models across OpenAI, Anthropic, Google. Input tokens, output tokens, cached tokens, reasoning tokens — each priced at the model's published rate. No third-party feeds. No estimates. The arithmetic is in source.

When a new model ships, you add a row to the JSON. The same engine runs in the proxy worker, the SDK, the dashboard, and the Python package. One source of truth.

## Storage

| Plane | Where | What lives there |
|-------|-------|------------------|
| Budget state | Cloudflare Durable Objects | Live counters, reservations, plan counters |
| Cost events | Postgres (Supabase) | Per-request cost rows, tag attribution, customer mapping |
| Request bodies (opt-in) | Cloudflare R2 | Encrypted full bodies for Pro/Scale/Enterprise debugging |
| Audit log | Postgres | Org-scoped action history |

The proxy is the only writer to budget state. The dashboard reads cost events directly from Postgres, never the Durable Object — this keeps the hot path fast and the analytical path independent.

## Open source

Everything is Apache-2.0:

- `apps/proxy` — Cloudflare Workers proxy
- `packages/sdk` — TypeScript SDK
- `packages/sdk-python` — Python SDK
- `packages/cost-engine` — pricing arithmetic
- `packages/mcp-server` / `packages/mcp-proxy` — MCP integration
- `packages/claude-agent` — Claude Agent SDK adapter
- `packages/db` — Drizzle ORM schema and migrations

Run the binary we run, on your own infrastructure. Read the exact formula that decides your margin. No black-box billing.

[View the source on GitHub](https://github.com/NullSpend/nullspend)
