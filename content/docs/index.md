---
title: "NullSpend"
description: "The open-source financial operations platform for AI-native SaaS. Cost tracking, attribution, enforcement, and margin intelligence across SDK, proxy, and MCP surfaces."
---

The open-source financial operations platform for AI-native SaaS. Cost tracking, attribution, enforcement, and margin intelligence across SDK, proxy, and MCP surfaces.

## The Problem

AI-native SaaS companies have 10–100x per-customer cost variance they cannot see, cannot control, and cannot bill correctly for. Two power users can quietly burn 40% of your inference bill while one runaway agent loop consumes thousands of dollars before anyone notices. Existing observability tools alert you after the spend has happened. NullSpend instruments cost at the request level, attributes it to customers and tags, enforces budgets *before* the upstream call fires, and syncs margins back to Stripe so finance, engineering, and the dashboard agree on the same numbers.

## How It Works

NullSpend ships as four integration surfaces backed by one accounting engine. Pick the surface that matches how your code already calls LLMs — they share the same dashboard, budgets, and webhooks.

```
  SDK            Proxy           Claude Agent       MCP
  (TS / Python)  (drop-in URL)   (adapter)          (server + proxy)
        │             │               │                │
        └───────────┴───────────────┴───────────────┘
                              ▼
                       NullSpend core
             cost engine · enforcement · attribution
                              │
         ┌─────────────────────┼────────────────────┐
         ▼                      ▼                      ▼
     Dashboard              Webhooks                Stripe sync
```

**SDK** — wrap your existing provider client with `createTrackedFetch` (`@nullspend/sdk` or `nullspend` for Python). Your provider clients stay as-is, no URL change.

**Proxy** — swap one environment variable. Useful when you can't modify the call site.

```bash
# Before                                    # After
OPENAI_BASE_URL=https://api.openai.com/v1   OPENAI_BASE_URL=https://proxy.nullspend.dev/v1
```

**Claude Agent adapter** — `@nullspend/claude-agent` routes Claude Agent SDK calls through the proxy with no other code changes.

**MCP** — `@nullspend/mcp-server` exposes approval tools to any MCP client; `@nullspend/mcp-proxy` gates upstream MCP tool calls through budget enforcement and HITL.

Add an `X-NullSpend-Key` header (proxy) or pass an API key (SDK) and every request is tracked, budgeted, attributed, and visible.

## What You Get

Three capability pillars, each available across every integration surface.

### Cost intelligence

| | |
|---|---|
| **[Cost tracking](features/cost-tracking.md)** | Per-request cost calculation across 56 models — input, output, cached, and reasoning tokens. BIGINT microdollar precision end-to-end. |
| **[Cost attribution](features/cost-attribution.md)** | Break down spend by customer, team, feature, environment, or any tag. Drill into daily trends, export CSV for finance. |
| **[Tags](features/tags.md)** | Attach arbitrary key/value pairs to every request via `X-NullSpend-Tags`. Becomes the join key for attribution, budgets, and webhooks. |
| **[Tracing](features/tracing.md)** | W3C `traceparent` propagation and custom trace IDs for request correlation across services. |
| **[Request logging](features/cost-tracking.md#request--response-body-logging)** | Opt-in capture of full request/response bodies (including streaming) for debugging and audit (Pro / Scale / Enterprise). |
| **[Session replay](features/cost-tracking.md#session-replay)** | Group LLM calls by session ID, view the full chronological timeline, expand to inspect bodies. |

### Enforcement

| | |
|---|---|
| **[Budgets](features/budgets.md)** | Hard spending ceilings. The proxy returns `429` before the request reaches the provider. |
| **[Velocity limits](features/velocity-limits.md)** | Detect runaway loops — block when spend rate exceeds a threshold within a time window. |
| **[Session limits](features/session-limits.md)** | Per-conversation spend caps tied to a session ID. |
| **[Plan limits](features/plan-limits.md)** | Per-tier monthly governed-request cap — Free hard-blocks, Pro / Scale bill overage. |
| **[HITL approvals](features/human-in-the-loop.md)** | Human-in-the-loop approval workflow for high-cost or sensitive operations. |
| **[Organizations](features/organizations.md)** | Team collaboration with roles (owner, admin, member, viewer), per-org billing, invitation management. |

### Margin and billing

| | |
|---|---|
| **Stripe margin sync** | Pull customer revenue from Stripe, join against NullSpend cost data, surface per-customer margin in the dashboard. |
| **Customer health tiers** | Automatic classification of every customer as Healthy / At-Risk / Burning based on margin trend, spend velocity, and revenue ratio. |
| **[Webhooks](webhooks/overview.md)** | 20 event types with HMAC-SHA256 signing — cost events, budget exceeded, velocity alerts, threshold crossings, loop detection, plan-limit denials, margin alerts. |
| **Multi-provider** | OpenAI, Anthropic, and Google Gemini in a single dashboard with provider breakdown. |

## Trust Model

Neither the proxy nor the SDK modifies your requests or responses. Your provider API keys stay with you — they pass through to the upstream provider and are never stored. By default, NullSpend sees only the token counts in the response to calculate cost; prompt content is not logged. Pro, Scale, and Enterprise plans can opt in to **request/response body logging** for debugging and audit purposes; bodies are stored encrypted in R2 with per-org lifecycle policies. The full stack is Apache-2.0 licensed and self-hostable — every line of code that touches your traffic is auditable.

## Pricing

| | Free | Pro | Scale | Enterprise |
|---|---|---|---|---|
| **Price** | $0/mo | $49/mo | $199/mo | Custom |
| **Governed requests / mo** | 100,000 | 500,000 | 2,000,000 | Unlimited |
| **Overage** | Hard block | $0.010 / request | $0.005 / request | Custom |
| **Budgets** | 3 | Unlimited | Unlimited | Unlimited |
| **API keys** | 10 | Unlimited | Unlimited | Unlimited |
| **Team members** | Unlimited | Unlimited | Unlimited | Unlimited |
| **Webhooks** | 2 endpoints | Unlimited | Unlimited | Unlimited |
| **Data retention** | 7 days | 60 days | 180 days | Custom |
| **Request logging** | -- | Full request/response bodies | Full request/response bodies | Full request/response bodies |
| **Enforcement (velocity, session, mandates, tag-budgets, multi-org)** | Included | Included | Included | Included |
| **Margin intelligence + Stripe sync** | -- | Included | Included | Included |
| **SSO / SAML / SCIM, EU residency** | -- | -- | -- | Included |

Full plan comparison and FAQ live on the [pricing page](https://nullspend.dev/pricing).

## Get Started

Pick the integration surface that matches your stack. Setup is under 2 minutes for any of them.

**SDK — recommended for new projects:**
- [JavaScript / TypeScript SDK](sdks/javascript.md)
- [Python SDK](sdks/python.md)

**Proxy — recommended when you can't modify the call site:**
- [OpenAI Quickstart](quickstart/openai.md)
- [Anthropic Quickstart](quickstart/anthropic.md)

**Agent + MCP integrations:**
- [Claude Agent adapter](sdks/claude-agent.md)
- [MCP server](sdks/mcp-server.md) and [MCP proxy](sdks/mcp-proxy.md)

**Migrations and tooling:**
- [Use with AI Coding Assistants](guides/ai-coding-assistant.md) — copy-paste blocks for Cursor, Claude Code, Copilot
- [Migrating from Helicone](guides/migrating-from-helicone.md)

**For AI agents:** fetch [`/llms.txt`](https://nullspend.dev/llms.txt) for the complete machine-readable API reference.

## API Reference

Build programmatic integrations with the NullSpend API:

- [API Overview](api-reference/overview.md) — authentication, pagination, errors, ID formats
- [Cost Events](api-reference/cost-events-api.md) — ingest and query cost data
- [API Keys](api-reference/api-keys-api.md) — key management and identity introspection
- [Budgets](api-reference/budgets-api.md) — spending limits and budget status
- [Webhooks](api-reference/webhooks-api.md) — endpoint management and delivery history
- [Actions](api-reference/actions-api.md) — human-in-the-loop approval workflows

## SDKs and adapters

Client libraries and adapters for integrating NullSpend into your stack:

- [JavaScript SDK](sdks/javascript.md) — `@nullspend/sdk` — TypeScript/JavaScript client for the NullSpend API
- [Python SDK](sdks/python.md) — `nullspend` — Python client for the NullSpend API
- [Claude Agent Adapter](sdks/claude-agent.md) — `@nullspend/claude-agent` — routes Claude Agent SDK calls through the proxy
- [MCP Server](sdks/mcp-server.md) — `@nullspend/mcp-server` — exposes approval tools to any MCP client
- [MCP Proxy](sdks/mcp-proxy.md) — `@nullspend/mcp-proxy` — gates upstream MCP tool calls through approval and budget enforcement
