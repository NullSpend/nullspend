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

AI agents spend money autonomously — and nobody's watching. One runaway loop burns $127K in four hours. The best "budget enforcement" in the market runs on a 60-second cron job. A $50 budget hits $764 before the next check.

**NullSpend enforces budgets _before_ the request reaches the provider.** Like Visa authorizing a transaction, not like a bank statement you read after the money's gone. A $50 budget stays a $50 budget.

```
OPENAI_BASE_URL=https://proxy.nullspend.com/v1
```

One environment variable. Real-time enforcement. $0 overspend guarantee.

## Everyone Watches. We Enforce.

|  | NullSpend | LiteLLM | Portkey | Helicone |
|---|---|---|---|---|
| **Pre-request enforcement** | 7ms sync | 60s stale | Async | No |
| **Velocity / loop detection** | Yes | No | No | No |
| **Unified LLM + MCP budget** | Yes | No | No | No |
| **HITL approval** | Yes | No | No | No |
| **Open source** | Yes | Yes | Partial | Yes |

Competitors claiming sub-millisecond latency achieve it by not enforcing anything — they're passthrough or async. NullSpend does real-time synchronous enforcement in 7ms, faster than most competitors can log a cost event.

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

## Built Like Visa, Not Like Datadog

### Authorization, Not Notification

Budget reserved before the LLM call — like Visa authorizing a card swipe, not like a bank statement you read after the money's gone. A $50 budget never becomes a $764 invoice.

### Network-Level Chokepoint

The proxy is the one layer every LLM call must pass through. SDK wrappers get bypassed by raw API calls. Framework middleware disappears when you switch tools. Dashboard alerts fire after the money's gone. One env var, every provider, no escape route.

### Financial Circuit Breaker

Visa-style anomaly detection for AI agents. Detects runaway loops and spending velocity — 200 requests/min when the baseline is 10. Budget enforcement catches the limit. Velocity detection catches the anomaly.

## What You Get

### Budget Enforcement
Set spend limits per user, per API key, or per tag. Budgets are enforced **before** the request reaches the provider — if it would exceed the limit, the proxy returns `429` without ever calling OpenAI/Anthropic. Atomic reservation-based deductions with 7ms latency.

### Velocity Limits
Sliding-window spend rate detection. Automatically blocks requests during spend spikes and recovers when the window clears. The financial circuit breaker for runaway agents.

### Session Limits
Cap total spend per agent session. Control how much a single conversation or task can cost before it needs human intervention.

### Cost Attribution
Tag every request with customer ID, team, agent, feature, environment — any dimension. Break down spend by any tag in the dashboard or via API. Know exactly where every dollar goes.

### Webhooks
15 event types with HMAC-SHA256 signed delivery: budget thresholds, budget exceeded, request blocked, velocity spikes, HITL actions, and more. Integrate with Slack, PagerDuty, or your own systems.

### Human-in-the-Loop Approval
Propose risky actions (emails, API calls, database writes), wait for human approval, then execute. Full SDK with polling, timeouts, and status tracking.

### Unified LLM + MCP Budgets
One budget covers API calls and tool calls. Gate MCP tool calls through approval workflows with `@nullspend/mcp-proxy`, or expose approval tools directly to MCP clients with `@nullspend/mcp-server`. Nobody else does this.

### Tracing
W3C traceparent support for correlating requests across distributed systems. Link cost events to traces, sessions, and HITL actions.

### Cost Engine
47 models across OpenAI (23), Anthropic (22), and Google (2). Accurate token-to-cost calculation with support for cached tokens, reasoning tokens, and Anthropic cache write tiers.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────┐     ┌──────────────┐
│             │     │       NullSpend Proxy         │     │              │
│  Your Agent ├────>│                              ├────>│   OpenAI /   │
│             │<────┤  Auth ─> Budget ─> Forward   │<────│   Anthropic  │
│             │     │  ─> Track Cost ─> Reconcile  │     │              │
└─────────────┘     └──────────────────────────────┘     └──────────────┘
                      7ms enforcement · <1ms p50 overhead
                    Cloudflare Workers · Durable Objects · Global Edge
```

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
