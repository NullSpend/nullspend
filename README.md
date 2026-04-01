# NullSpend

FinOps layer for AI agents — cost tracking, budget enforcement, and human-in-the-loop approval.

NullSpend sits between your agents and AI providers, tracking every token, enforcing budgets before requests reach the provider, and giving you a dashboard with real-time analytics and attribution.

## What's in This Repo

| Package | Description |
|---|---|
| [`apps/proxy`](apps/proxy/) | Cloudflare Workers proxy — sits between agents and OpenAI/Anthropic, tracks costs, enforces budgets |
| [`@nullspend/sdk`](packages/sdk/) | TypeScript SDK — propose actions, wait for approval, execute with cost correlation |
| [`@nullspend/cost-engine`](packages/cost-engine/) | Model pricing catalog and cost calculation (47 models across OpenAI, Anthropic, Google) |
| [`@nullspend/claude-agent`](packages/claude-agent/) | Claude Agent SDK adapter — one function to route agent calls through the proxy |
| [`@nullspend/mcp-server`](packages/mcp-server/) | MCP server exposing NullSpend approval tools to MCP clients |
| [`@nullspend/mcp-proxy`](packages/mcp-proxy/) | MCP proxy that gates risky tool calls through approval before forwarding |
| [`@nullspend/docs`](packages/docs-mcp-server/) | MCP server that serves NullSpend docs to AI coding tools |
| [`@nullspend/db`](packages/db/) | Drizzle ORM schema and types |
| [`nullspend` (Python)](packages/sdk-python/) | Python SDK for NullSpend |

The [hosted dashboard](https://nullspend.com) provides analytics, attribution breakdowns, webhook alerts, and team management. The packages in this repo are the open-source building blocks.

## Quick Start

### Option 1: Proxy (recommended)

Route your LLM calls through the NullSpend proxy. Zero code changes beyond config.

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});

// Use OpenAI as normal — costs are tracked automatically
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

Works with Anthropic too:

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});
```

### Option 2: Claude Agent SDK

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { Agent } from "@anthropic-ai/claude-agent-sdk";
import { withNullSpend } from "@nullspend/claude-agent";

const client = new Anthropic();
const agent = new Agent({
  client,
  model: "claude-sonnet-4-6",
  ...withNullSpend({
    apiKey: process.env.NULLSPEND_API_KEY,
    tags: { agent: "my-agent" },
  }),
});
```

### Option 3: SDK (client-side cost tracking)

```typescript
import OpenAI from "openai";
import { NullSpend } from "@nullspend/sdk";

const ns = new NullSpend({
  baseUrl: "https://app.nullspend.com",
  apiKey: process.env.NULLSPEND_API_KEY,
  costReporting: {},
});

// Pass the tracked fetch to the OpenAI SDK — costs are reported automatically
const openai = new OpenAI({ fetch: ns.createTrackedFetch("openai") });
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Agent /    │────>│  NullSpend Proxy │────>│   OpenAI /   │
│   Your App   │<────│  (CF Workers)    │<────│   Anthropic  │
└─────────────┘     └──────┬───────────┘     └──────────────┘
                           │
                    ┌──────┴───────┐
                    │   Auth       │  API key validation
                    │   Budget     │  Check before forwarding
                    │   Cost Track │  Parse response, calculate cost
                    │   Webhooks   │  Threshold/exceeded alerts
                    └──────────────┘
```

The proxy adds ~0ms overhead (measured p50) — it authenticates, forwards to the upstream provider, parses the response for token usage, calculates cost, and enforces budgets. Streaming is fully supported.

## Proxy Endpoints

| Endpoint | Provider |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages` | Anthropic |

## Development

```bash
# Prerequisites: Node.js >= 20.11.0, pnpm >= 10
pnpm install

# Build (order matters)
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Documentation

- [Overview](docs/overview.md)
- [Quick Start — OpenAI](docs/quickstart/openai.md)
- [Quick Start — Anthropic](docs/quickstart/anthropic.md)
- [API Reference](docs/api-reference/overview.md)
- [Webhooks](docs/webhooks/overview.md)
- [Full docs at nullspend.com](https://nullspend.com/docs)

## License

Apache-2.0 — see [LICENSE](LICENSE).
