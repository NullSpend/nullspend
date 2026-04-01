<p align="center">
  <h1 align="center">NullSpend</h1>
  <p align="center">
    <strong>Every AI request. Authorized. Tracked. Enforced.</strong>
    <br />
    The financial control plane for autonomous AI agents.
  </p>
</p>

<p align="center">
  <a href="https://github.com/NullSpend/nullspend/actions"><img src="https://github.com/NullSpend/nullspend/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/NullSpend/nullspend/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License" /></a>
  <a href="https://nullspend.com/docs"><img src="https://img.shields.io/badge/docs-nullspend.com-brightgreen" alt="Docs" /></a>
</p>

---

Your agents make thousands of LLM calls. Each one costs money. One bad loop can burn your entire monthly budget in minutes — and you won't know until the invoice lands.

NullSpend is the layer that sits between your agents and the AI providers. Every request is authorized against your budget **before** it executes. Not after. Not on a cron job. Not in a batch. Before.

```
OPENAI_BASE_URL=https://proxy.nullspend.com/v1
```

One line. Sub-millisecond enforcement. Zero overspend. Full visibility.

## Why This Exists

The current state of AI cost management is broken. Most tools watch spend passively and send you a notification after you've already blown your budget. A $50 limit becomes a $764 invoice because enforcement runs on a 60-second polling loop.

NullSpend takes a fundamentally different approach:

- **Pre-request authorization** — budgets are checked and reserved before the LLM call, not reconciled after
- **Sub-millisecond enforcement** — real-time synchronous budget checks on every request
- **Network-level control** — the proxy is the single chokepoint every request passes through. Can't be bypassed, can't be misconfigured, can't be forgotten
- **Anomaly detection** — velocity limits detect runaway loops and spending spikes automatically
- **$0 overspend guarantee** — a $50 budget stays a $50 budget

## Get Started in 2 Minutes

### OpenAI

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});

// Every call is now tracked, budgeted, and enforced. Your code doesn't change.
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

### Anthropic

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});
```

### Claude Agent SDK

```typescript
import { withNullSpend } from "@nullspend/claude-agent";

const agent = new Agent({
  client,
  model: "claude-sonnet-4-6",
  ...withNullSpend({
    apiKey: process.env.NULLSPEND_API_KEY,
    tags: { agent: "research-bot", customer: "acme-corp" },
  }),
});
```

### TypeScript SDK

```typescript
import OpenAI from "openai";
import { NullSpend } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://app.nullspend.com",
  apiKey: process.env.NULLSPEND_API_KEY,
  costReporting: {},
});

const openai = new OpenAI({ fetch: ns.createTrackedFetch("openai") });
```

### Python

```python
from nullspend import NullSpend

ns = NullSpend(api_key="ns_live_sk_...")
```

## How It Works

```
┌─────────────┐     ┌──────────────────────────────┐     ┌──────────────┐
│             │     │       NullSpend Proxy         │     │              │
│  Your Agent ├────>│                              ├────>│   OpenAI /   │
│             │<────┤  Authorize ─> Reserve ─>     │<────│   Anthropic  │
│             │     │  Forward ─> Track ─> Settle  │     │              │
└─────────────┘     └──────────────────────────────┘     └──────────────┘
                         <1ms enforcement overhead
                    Cloudflare Workers · Durable Objects · Global Edge
```

Every request follows the same path: **authorize** the spend against your budget, **reserve** the estimated cost, **forward** to the provider, **track** the actual usage, **settle** the final cost. If the budget can't cover it, the request never leaves.

## What You Get

### Pre-Request Budget Enforcement
Set spend limits per user, per API key, or per tag. Budgets are enforced **before** the request reaches the provider — if it would exceed the limit, the proxy returns `429` without ever calling the upstream API. Atomic reservation-based deductions with sub-millisecond latency.

### Velocity Limits
Sliding-window spend rate detection. When an agent starts burning money faster than normal — 200 requests/min when the baseline is 10 — the circuit breaker trips. Automatically recovers when the spike subsides.

### Session Limits
Cap total spend per agent session. Control how much a single conversation, task, or workflow can cost before it stops or escalates to a human.

### Cost Attribution
Tag every request with customer ID, team, agent, feature, environment — any dimension you need. Break down spend by any combination of tags. Know exactly where every dollar goes and who's responsible for it.

### Webhooks
15 event types with HMAC-SHA256 signed delivery: budget thresholds, budget exceeded, request blocked, velocity spikes, HITL actions, and more. Wire them into Slack, PagerDuty, or your own alerting stack.

### Human-in-the-Loop Approval
Propose risky actions — sending emails, calling APIs, writing to databases — wait for human approval, then execute. Full SDK with polling, timeouts, and status tracking. Give your agents autonomy with guardrails.

### Unified LLM + MCP Budgets
One budget covers API calls and tool calls. Gate MCP tool calls through approval workflows with `@nullspend/mcp-proxy`, or expose approval tools directly to MCP clients with `@nullspend/mcp-server`.

### Tracing
W3C traceparent support for correlating requests across distributed systems. Link cost events to traces, sessions, and approval actions for full observability.

### Cost Engine
47 models across OpenAI (23), Anthropic (22), and Google (2). Accurate token-to-cost calculation with support for cached tokens, reasoning tokens, and Anthropic cache write tiers.

## Packages

| Package | Description |
|---|---|
| [`apps/proxy`](apps/proxy/) | Cloudflare Workers proxy — auth, cost tracking, budget enforcement, velocity limits, webhooks, streaming |
| [`@nullspend/sdk`](packages/sdk/) | TypeScript SDK — tracked fetch, cost reporting, HITL approval workflows, budget status |
| [`nullspend`](packages/sdk-python/) | Python SDK |
| [`@nullspend/cost-engine`](packages/cost-engine/) | Pricing catalog and cost calculation for 47 models |
| [`@nullspend/claude-agent`](packages/claude-agent/) | Claude Agent SDK adapter — `withNullSpend()` and `withNullSpendAsync()` for budget-aware agents |
| [`@nullspend/mcp-server`](packages/mcp-server/) | MCP server exposing NullSpend approval tools to any MCP client |
| [`@nullspend/mcp-proxy`](packages/mcp-proxy/) | MCP proxy — gate risky tool calls through approval before forwarding |
| [`@nullspend/docs`](packages/docs-mcp-server/) | MCP server that serves NullSpend docs to AI coding tools |
| [`@nullspend/db`](packages/db/) | Drizzle ORM schema and types |

## Hosted Dashboard

The open-source packages handle tracking and enforcement. The [hosted dashboard at nullspend.com](https://nullspend.com) adds real-time analytics, attribution breakdowns, budget management UI, webhook configuration, team management, and session replay.

## Proxy Endpoints

| Endpoint | Provider |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages` | Anthropic |

Streaming and non-streaming. Your provider API key forwards transparently.

## Development

```bash
git clone https://github.com/NullSpend/nullspend.git && cd nullspend
pnpm install

# Build (dependency order)
pnpm db:build && pnpm cost-engine:build && pnpm sdk:build

# Test
pnpm proxy:test         # Proxy worker tests
pnpm sdk:test           # SDK tests
pnpm cost-engine:test   # Cost engine tests
pnpm claude-agent:test  # Claude agent adapter tests
pnpm mcp:test           # MCP server tests
pnpm mcp-proxy:test     # MCP proxy tests
pnpm db:test            # DB schema tests
pnpm docs-mcp:test      # Docs MCP tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## Documentation

- [Overview](docs/overview.md)
- [Quick Start — OpenAI](docs/quickstart/openai.md)
- [Quick Start — Anthropic](docs/quickstart/anthropic.md)
- [API Reference](docs/api-reference/overview.md)
- [Webhooks](docs/webhooks/overview.md)
- [Full docs at nullspend.com](https://nullspend.com/docs)

## License

Apache-2.0 — see [LICENSE](LICENSE).
