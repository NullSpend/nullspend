---
title: "NullSpend"
description: "FinOps layer for AI agents — cost tracking, budget enforcement, and human-in-the-loop approval."
---

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
OPENAI_BASE_URL=https://api.openai.com/v1   OPENAI_BASE_URL=https://proxy.nullspend.com/v1
```

Add an `X-NullSpend-Key` header and every request is tracked, budgeted, and visible.

## Features

| Feature | What It Does |
|---|---|
| **[Cost tracking](features/cost-tracking.md)** | Per-request cost calculation for every model — input, output, cached, and reasoning tokens |
| **[Cost attribution](features/cost-attribution.md)** | Break down AI spend by customer, team, or feature. Group by API key or any tag, drill into daily trends, export CSV for finance |
| **[Budget enforcement](features/budgets.md)** | Hard spending ceilings. The proxy returns `429` before the request reaches the provider |
| **[Velocity limits](features/velocity-limits.md)** | Detect runaway loops — block when spend rate exceeds a threshold within a time window |
| **[Session limits](features/budgets.md#session-limits)** | Per-conversation spend caps tied to a session ID |
| **[Tags](features/tags.md)** | Attribute costs to teams, environments, features, or anything else via `X-NullSpend-Tags` |
| **[Tracing](features/tracing.md)** | W3C `traceparent` propagation and custom trace IDs for request correlation |
| **[Webhooks](webhooks/overview.md)** | 15 event types with HMAC-SHA256 signing — cost events, budget exceeded, velocity alerts, threshold crossings |
| **[HITL approvals](features/human-in-the-loop.md)** | Human-in-the-loop approval workflow for high-cost or sensitive operations |
| **[Organizations](features/organizations.md)** | Team collaboration with roles (owner, admin, member, viewer), per-org billing, and invitation management |
| **[Request logging](features/cost-tracking.md#request--response-body-logging)** | Opt-in capture of full request/response bodies (including streaming) for debugging and audit (Pro/Enterprise) |
| **[Session replay](features/cost-tracking.md#session-replay)** | Group LLM calls by session ID, view the full chronological timeline, expand to see request/response bodies |
| **Multi-provider** | OpenAI and Anthropic in a single dashboard with provider breakdown |
| **MCP support** | Budget enforcement for Model Context Protocol servers and proxies |

## Trust Model

The proxy never modifies your requests or responses. Your provider API keys stay with you — they pass through to the upstream provider and are never stored. By default, NullSpend sees only the token counts in the response to calculate cost — prompt content is not logged. Pro and Enterprise plans can opt in to **request/response body logging** for debugging and audit purposes; bodies are stored encrypted in R2 with per-org lifecycle policies.

## Pricing

| | Free | Pro | Enterprise |
|---|---|---|---|
| **Price** | $0/mo | $49/mo | Custom |
| **Proxied spend** | $5K/mo | $50K/mo | Unlimited |
| **Budgets** | 3 | Unlimited | Unlimited |
| **API keys** | 10 | Unlimited | Unlimited |
| **Team members** | 3 (viewers unlimited) | Unlimited | Unlimited |
| **Webhooks** | 2 endpoints | 25 endpoints | Unlimited |
| **Data retention** | 30 days | 90 days | Unlimited |
| **Request logging** | -- | Full request/response bodies | Full request/response bodies |
| **Key features** | Cost tracking, budgets, team orgs | Unlimited keys/budgets/members, request logging | SSO/SAML, custom RBAC, dedicated support |

## Get Started

Set up cost tracking in under 2 minutes:

- [OpenAI Quickstart](quickstart/openai.md)
- [Anthropic Quickstart](quickstart/anthropic.md)
- [Use with AI Coding Assistants](guides/ai-coding-assistant.md) — copy-paste blocks for Cursor, Claude Code, Copilot
- [Migrating from Helicone](guides/migrating-from-helicone.md)

**For AI agents:** fetch [`/llms.txt`](https://nullspend.com/llms.txt) for the complete machine-readable API reference.

## API Reference

Build programmatic integrations with the NullSpend API:

- [API Overview](api-reference/overview.md) — authentication, pagination, errors, ID formats
- [Cost Events](api-reference/cost-events-api.md) — ingest and query cost data
- [API Keys](api-reference/api-keys-api.md) — key management and identity introspection
- [Budgets](api-reference/budgets-api.md) — spending limits and budget status
- [Webhooks](api-reference/webhooks-api.md) — endpoint management and delivery history
- [Actions](api-reference/actions-api.md) — human-in-the-loop approval workflows

## SDKs

Client libraries and adapters for integrating NullSpend into your stack:

- [JavaScript SDK](sdks/javascript.md) — `@nullspend/sdk` — TypeScript/JavaScript client for the NullSpend API
- [Python SDK](sdks/python.md) — `nullspend` — Python client for the NullSpend API
- [Claude Agent Adapter](sdks/claude-agent.md) — `@nullspend/claude-agent` — routes Claude Agent SDK calls through the proxy
- [MCP Server](sdks/mcp-server.md) — `@nullspend/mcp-server` — exposes approval tools to any MCP client
- [MCP Proxy](sdks/mcp-proxy.md) — `@nullspend/mcp-proxy` — gates upstream MCP tool calls through approval and budget enforcement
