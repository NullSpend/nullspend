# NullSpend

FinOps layer for AI agents — cost tracking, budget enforcement, and human-in-the-loop approval.

## The Problem

AI agents call LLMs autonomously. Without guardrails, a single runaway loop can burn through thousands of dollars before anyone notices. Existing observability tools can _alert_ you after the damage is done. NullSpend **prevents** it.

## How It Works

```
Your App ──► NullSpend Proxy ──► OpenAI / Anthropic
                │                       │
                │  ◄── streams back ────┘
                │
                ▼
         Dashboard & Webhooks
```

Two config changes. No SDK. No code rewrite.

```bash
# Before                                    # After
OPENAI_BASE_URL=https://api.openai.com/v1   OPENAI_BASE_URL=https://proxy.nullspend.dev/v1
```

Add an `X-NullSpend-Key` header and every request is tracked, budgeted, and visible.

## Features

| Feature | What It Does |
|---|---|
| **[Cost tracking](features/cost-tracking.md)** | Per-request cost calculation for every model — input, output, cached, and reasoning tokens |
| **[Budget enforcement](features/budgets.md)** | Hard spending ceilings. The proxy returns `429` before the request reaches the provider |
| **[Velocity limits](features/budgets.md#velocity-limits)** | Detect runaway loops — block when spend rate exceeds a threshold within a time window |
| **[Session limits](features/budgets.md#session-limits)** | Per-conversation spend caps tied to a session ID |
| **[Tags](features/tags.md)** | Attribute costs to teams, environments, features, or anything else via `X-NullSpend-Tags` |
| **[Tracing](features/tracing.md)** | W3C `traceparent` propagation and custom trace IDs for request correlation |
| **[Webhooks](webhooks/overview.md)** | 18 event types with HMAC-SHA256 signing — cost events, budget exceeded, velocity alerts, threshold crossings, margin alerts |
| **[HITL approvals](features/human-in-the-loop.md)** | Human-in-the-loop approval workflow for high-cost or sensitive operations |
| **[Margins](features/margins.md)** | Connect Stripe to see per-customer profitability — auto-match, health tiers, trajectory projection, Slack alerts, CSV export |
| **Multi-provider** | OpenAI and Anthropic in a single dashboard with provider breakdown |
| **MCP support** | Budget enforcement for Model Context Protocol servers and proxies |

## Trust Model

The proxy never modifies your requests or responses. Your provider API keys stay with you — they pass through to the upstream provider and are never stored. NullSpend sees the token counts in the response to calculate cost; it does not log prompt content.

## Pricing

| | Free | Pro | Team |
|---|---|---|---|
| **Price** | $0/mo | $49/mo | $199/mo |
| **Proxied spend** | $1K/mo | $50K/mo | $250K/mo |
| **Budgets** | 1 | Unlimited | Unlimited |
| **Data retention** | 7 days | 30 days | 90 days |
| **Key features** | Cost tracking, 1 budget | Webhooks, API access | Multi-user, team budgets, advanced analytics |

## Get Started

Set up cost tracking in under 2 minutes:

- [OpenAI Quickstart](quickstart/openai.md)
- [Anthropic Quickstart](quickstart/anthropic.md)
- [Migrating from Helicone](guides/migrating-from-helicone.md)

## API Reference

Build programmatic integrations with the NullSpend API:

- [API Overview](api-reference/overview.md) — authentication, pagination, errors, ID formats
- [Cost Events](api-reference/cost-events-api.md) — ingest and query cost data
- [API Keys](api-reference/api-keys-api.md) — key management and identity introspection
- [Budgets](api-reference/budgets-api.md) — spending limits and budget status
- [Webhooks](api-reference/webhooks-api.md) — endpoint management and delivery history
- [Actions](api-reference/actions-api.md) — human-in-the-loop approval workflows
- [Margins](api-reference/margins-api.md) — Stripe revenue sync, customer profitability, mappings

## SDKs

Client libraries and adapters for integrating NullSpend into your stack:

- [JavaScript SDK](sdks/javascript.md) — `@nullspend/sdk` — TypeScript/JavaScript client for the NullSpend API
- [Claude Agent Adapter](sdks/claude-agent.md) — `@nullspend/claude-agent` — routes Claude Agent SDK calls through the proxy
- [MCP Server](sdks/mcp-server.md) — `@nullspend/mcp-server` — exposes approval tools to any MCP client
- [MCP Proxy](sdks/mcp-proxy.md) — `@nullspend/mcp-proxy` — gates upstream MCP tool calls through approval and budget enforcement
