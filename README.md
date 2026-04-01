<p align="center">
  <h1 align="center">NullSpend</h1>
  <p align="center">
    <strong>Financial infrastructure for the autonomous AI economy.</strong>
    <br />
    The first FinOps platform purpose-built for AI agents.
  </p>
</p>

<p align="center">
  <a href="https://github.com/NullSpend/nullspend/actions"><img src="https://github.com/NullSpend/nullspend/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/NullSpend/nullspend/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License" /></a>
  <a href="https://nullspend.com/docs"><img src="https://img.shields.io/badge/docs-nullspend.com-brightgreen" alt="Docs" /></a>
</p>

---

AI agents are becoming autonomous economic actors. They negotiate, transact, and spend — across providers, tools, and workflows — at machine speed. The infrastructure to govern that spend doesn't exist yet.

**NullSpend is building it.**

We're creating the financial infrastructure layer for the autonomous AI economy: real-time budget authorization, spend velocity controls, cost attribution, cross-provider governance, and human-in-the-loop approval — all through a transparent proxy that integrates in one line and enforces in under a millisecond.

```
OPENAI_BASE_URL=https://proxy.nullspend.com/v1
```

This isn't observability. This isn't logging. This is **financial authorization** — every request checked against your budget before it executes, with the spend reserved atomically and the cost reconciled on completion.

## The Problem Is Structural

Today's AI cost tools are built on a fundamentally broken model: they **observe** spend and **notify** you after the fact. A $50 budget limit enforced on a 60-second polling loop becomes a $764 invoice. A runaway agent loop burns $127K in four hours, and the team finds out on their monthly bill.

Agents don't need dashboards. They need authorization infrastructure.

NullSpend provides:

- **Pre-request budget authorization** — spend is checked and reserved before the LLM call, not reconciled after
- **Sub-millisecond enforcement** — real-time synchronous budget checks on every single request
- **Network-level governance** — the proxy is the single control point every request passes through. One env var. Every provider. No escape route.
- **Velocity circuit breakers** — automatically detect and halt runaway spend patterns
- **Unified LLM + tool budgets** — one budget governs API calls and MCP tool calls together

## Get Started in 2 Minutes

### OpenAI

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});

// Every call is now authorized, tracked, and enforced. Your code doesn't change.
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

### Proxy Mode — full enforcement

```mermaid
flowchart LR
    A["Your Agent"] -->|LLM request| B["NullSpend Proxy"]
    B -->|forward| C["OpenAI / Anthropic"]
    C -->|response| B
    B -->|response + cost| A
    B -->|events| D["Dashboard & Webhooks"]

    subgraph B["NullSpend Proxy"]
        direction TB
        B1["1. Authorize"] --> B2["2. Reserve budget"]
        B2 --> B3["3. Forward to provider"]
        B3 --> B4["4. Track usage & cost"]
        B4 --> B5["5. Settle & reconcile"]
    end

    style B fill:#eff6ff,stroke:#2563eb,stroke-width:2px
```

Every request follows the same path: **authorize** the spend against your budget, **reserve** the estimated cost atomically, **forward** to the provider, **track** the actual token usage, **settle** the final cost. If the budget can't cover it, the request never leaves. Sub-millisecond enforcement overhead on the global edge via Cloudflare Workers and Durable Objects.

### SDK Mode — direct calls with client-side enforcement

```mermaid
flowchart LR
    subgraph A["Your Agent"]
        direction TB
        S["@nullspend/sdk"] --> S1["Check budget & mandates"]
        S1 --> S2["Call provider directly"]
        S2 --> S3["Calculate cost"]
        S3 --> S4["Report async"]
    end

    A ==>|"direct API call"| C["OpenAI / Anthropic"]
    C ==>|response| A
    S4 -.->|"cost events"| D["NullSpend API"]
    D -.->|"policy"| S

    style A fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style D fill:#f9fafb,stroke:#d1d5db
```

Don't want to route traffic through a proxy? The SDK wraps your existing fetch call with `createTrackedFetch()`. With `enforcement: true`, it checks budget and model mandates **before** the request — blocking it client-side if the budget would be exceeded. Cost is calculated locally using the built-in pricing engine and reported asynchronously. Your requests go directly to the provider.

## Platform Capabilities

### Budget Authorization
Real-time, pre-request budget enforcement. Set spend limits per user, per API key, or per tag. If a request would exceed the limit, the proxy returns `429` without ever calling the upstream provider. Atomic reservation-based deductions with sub-millisecond latency.

### Velocity Controls
Sliding-window spend velocity detection. When an agent starts burning money faster than normal — 200 requests/min when the baseline is 10 — the circuit breaker trips automatically. Recovers when the anomaly subsides.

### Session Governance
Cap total spend per agent session. Control how much a single conversation, task, or workflow can cost before it stops or escalates to a human.

### Cost Attribution
Tag every request with customer ID, team, agent, feature, environment — any dimension you care about. Break down spend by any combination of tags across your entire fleet. Full visibility into who's spending what, where, and why.

### Webhook Event System
15 event types with HMAC-SHA256 signed delivery: budget thresholds, budget exceeded, request blocked, velocity spikes, approval actions, and more. Wire them into Slack, PagerDuty, or your own alerting and automation systems.

### Human-in-the-Loop Approval
Propose high-stakes actions — sending emails, calling external APIs, writing to production databases — and wait for human approval before execution. Full SDK with polling, timeouts, and lifecycle tracking. Give your agents autonomy with governance.

### Unified LLM + MCP Budgets
One budget governs API calls and tool calls together. Gate MCP tool calls through approval workflows with `@nullspend/mcp-proxy`, or expose approval tools directly to MCP clients with `@nullspend/mcp-server`. A single financial policy across your entire agent stack.

### Distributed Tracing
W3C traceparent support for correlating requests across services. Link cost events to traces, sessions, and approval actions for end-to-end financial observability.

### Cost Engine
47 models across OpenAI (23), Anthropic (22), and Google (2). Accurate token-to-cost calculation with support for cached tokens, reasoning tokens, and Anthropic cache write tiers.

## Packages

| Package | Description |
|---|---|
| [`apps/proxy`](apps/proxy/) | Cloudflare Workers proxy — auth, budget authorization, cost tracking, velocity controls, webhooks, streaming |
| [`@nullspend/sdk`](packages/sdk/) | TypeScript SDK — tracked fetch, cost reporting, HITL approval workflows, budget status |
| [`nullspend`](packages/sdk-python/) | Python SDK |
| [`@nullspend/cost-engine`](packages/cost-engine/) | Pricing catalog and cost calculation for 47 models |
| [`@nullspend/claude-agent`](packages/claude-agent/) | Claude Agent SDK adapter — `withNullSpend()` and `withNullSpendAsync()` for budget-aware agents |
| [`@nullspend/mcp-server`](packages/mcp-server/) | MCP server exposing NullSpend approval tools to any MCP client |
| [`@nullspend/mcp-proxy`](packages/mcp-proxy/) | MCP proxy — gate tool calls through approval before forwarding |
| [`@nullspend/docs`](packages/docs-mcp-server/) | MCP server that serves NullSpend docs to AI coding tools |
| [`@nullspend/db`](packages/db/) | Drizzle ORM schema and types |

## Hosted Platform

The open-source packages handle authorization and enforcement at the network layer. The [hosted platform at nullspend.com](https://nullspend.com) adds real-time analytics, attribution dashboards, budget management, webhook configuration, team governance, and session replay.

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
