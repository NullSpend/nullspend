<p align="center">
  <h1 align="center">NullSpend</h1>
  <p align="center">
    <strong>Stop your AI agents from burning money.</strong>
    <br />
    Cost tracking, budget enforcement, and spend controls for every LLM call.
  </p>
</p>

<p align="center">
  <a href="https://github.com/NullSpend/nullspend/actions"><img src="https://github.com/NullSpend/nullspend/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/NullSpend/nullspend/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License" /></a>
  <a href="https://nullspend.com/docs"><img src="https://img.shields.io/badge/docs-nullspend.com-brightgreen" alt="Docs" /></a>
</p>

---

NullSpend is the financial control layer for AI agents. Two lines of config give you per-request cost tracking, hard budget limits that block requests *before* they hit the provider, velocity controls, session spend caps, cost attribution, webhook alerts, and a real-time dashboard — across OpenAI, Anthropic, and 47 models.

Your agent keeps working. Your wallet stops bleeding.

## Why NullSpend?

AI agents are expensive and unpredictable. A single runaway loop can burn hundreds of dollars in minutes. Most teams find out *after* the invoice arrives.

NullSpend fixes this:

- **Know what you're spending** — every request tracked with model, tokens, cost, and custom tags
- **Set hard limits** — budgets are enforced *before* the request reaches OpenAI/Anthropic, not after
- **Attribute costs** — break down spend by customer, team, agent, feature, or any tag you define
- **Get alerts** — webhooks fire on thresholds, budget exceeded, velocity spikes, and blocked requests
- **Control agent autonomy** — human-in-the-loop approval for risky actions, with SDK support

## Get Started in 2 Minutes

### OpenAI — change one line

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://proxy.nullspend.com/v1",  // <-- just this
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});

// That's it. Every call is now tracked, budgeted, and visible in your dashboard.
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

### Anthropic — same deal

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});
```

### Claude Agent SDK — one function

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

### Python

```python
from nullspend import NullSpend

ns = NullSpend(api_key="ns_live_sk_...")
```

### TypeScript SDK — client-side tracking

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

## Features

### Cost Tracking
Every LLM request is tracked with provider, model, input/output/cached/reasoning tokens, cost in microdollars, duration, and custom tags. Streaming fully supported. 47 models across OpenAI (23), Anthropic (22), and Google (2).

### Budget Enforcement
Set spend limits per user, per API key, or per tag. The proxy checks budgets **before** forwarding — if a request would exceed the limit, it returns `429` without ever calling the provider. No surprise bills.

### Velocity Limits
Cap spend-per-minute with sliding window enforcement. Automatically blocks requests during spend spikes and recovers when the window clears. Circuit breaker for runaway agents.

### Session Limits
Cap total spend per agent session. Track and limit how much a single conversation or task can cost.

### Cost Attribution
Tag every request with custom metadata — customer ID, team, feature, environment, agent name. Break down costs by any dimension in the dashboard or via API.

### Webhooks
15 event types with HMAC-SHA256 signed delivery: budget warnings, budget exceeded, request blocked, velocity spikes, HITL actions, and more. Integrate with Slack, PagerDuty, or your own systems.

### Human-in-the-Loop Approval
Propose risky actions (emails, API calls, database writes), wait for human approval, then execute. Full SDK with polling, timeouts, and status tracking.

### MCP Integration
Gate MCP tool calls through approval workflows with `@nullspend/mcp-proxy`, or expose approval tools directly to MCP clients with `@nullspend/mcp-server`.

### Tracing
W3C traceparent support for correlating requests across distributed systems. Link cost events to traces, sessions, and HITL actions.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────┐     ┌──────────────┐
│             │     │       NullSpend Proxy         │     │              │
│  Your Agent ├────>│                              ├────>│   OpenAI /   │
│             │<────┤  Auth ─> Budget Check ─>     │<────│   Anthropic  │
│             │     │  Forward ─> Track Cost ─>    │     │              │
└─────────────┘     │  Webhooks ─> Dashboard       │     └──────────────┘
                    └──────────────────────────────┘
                         ~0ms overhead (p50)
                      Cloudflare Workers (global edge)
```

The proxy runs on Cloudflare Workers at the edge, adding ~0ms overhead at p50. Budget enforcement uses Durable Objects for consistent, low-latency state. Cost events are queued and batch-written for reliability.

## Packages

| Package | What it does |
|---|---|
| [`apps/proxy`](apps/proxy/) | Cloudflare Workers proxy — the core. Auth, cost tracking, budget enforcement, webhooks, streaming. |
| [`@nullspend/sdk`](packages/sdk/) | TypeScript SDK — tracked fetch, cost reporting, HITL approval workflows, budget status. |
| [`nullspend`](packages/sdk-python/) | Python SDK. |
| [`@nullspend/cost-engine`](packages/cost-engine/) | Pricing catalog for 47 models. Calculate costs from token counts. |
| [`@nullspend/claude-agent`](packages/claude-agent/) | Claude Agent SDK adapter — `withNullSpend()` and `withNullSpendAsync()` for budget-aware agents. |
| [`@nullspend/mcp-server`](packages/mcp-server/) | MCP server exposing NullSpend approval tools to any MCP client. |
| [`@nullspend/mcp-proxy`](packages/mcp-proxy/) | MCP proxy — gate risky tool calls through approval before forwarding upstream. |
| [`@nullspend/docs`](packages/docs-mcp-server/) | MCP server that serves NullSpend docs to AI coding tools. |
| [`@nullspend/db`](packages/db/) | Drizzle ORM schema and types. |

## Hosted Dashboard

The open-source packages handle tracking and enforcement. The [hosted dashboard](https://nullspend.com) adds:

- Real-time cost analytics and charts
- Per-model, per-key, per-tag attribution breakdowns
- Budget and webhook management UI
- Team and organization management
- Session replay and request inspection

Sign up at [nullspend.com](https://nullspend.com).

## Proxy Endpoints

| Endpoint | Provider |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages` | Anthropic |

All endpoints support streaming and non-streaming. The proxy forwards your provider API key transparently.

## Development

```bash
git clone https://github.com/NullSpend/nullspend.git && cd nullspend
pnpm install

# Build (order matters — dependencies first)
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
