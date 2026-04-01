---
title: "Use with AI Coding Assistants"
description: "Copy-paste reference blocks for Cursor, Claude Code, GitHub Copilot, and other AI coding tools. Give your assistant full context on the NullSpend API in one paste."
---

Give your AI coding assistant full context on the NullSpend API in one paste. Copy the block below into your system prompt, project rules, or paste it directly into a conversation.

## For Cursor

Add to `.cursor/rules` or paste into a conversation:

````markdown
# NullSpend API Reference

NullSpend is a proxy for OpenAI/Anthropic that tracks costs, enforces budgets, and provides attribution. Integration: change base URL + add one header.

## Setup
```bash
OPENAI_BASE_URL=https://proxy.nullspend.com/v1
NULLSPEND_API_KEY=ns_live_sk_...  # from dashboard
```

## Required Header
Every request must include: `X-NullSpend-Key: <your NULLSPEND_API_KEY>`

## Optional Headers
- `X-NullSpend-Tags: {"customer_id":"acme","team":"eng"}` — JSON, max 10 keys, for cost attribution
- `X-NullSpend-Session: session-123` — session ID for per-session spend limits (max 256 chars)
- `X-NullSpend-Trace-Id: a1b2c3d4e5f67890a1b2c3d4e5f67890` — 32 hex chars, for tracing
- `X-NullSpend-Action-Id: ns_act_...` — link costs to HITL approval actions

## OpenAI Integration
```typescript
import OpenAI from "openai";
const openai = new OpenAI({
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});
// Use openai.chat.completions.create() as normal
```

## Anthropic Integration
```typescript
import Anthropic from "@anthropic-ai/sdk";
const anthropic = new Anthropic({
  baseURL: "https://proxy.nullspend.com/v1",
  defaultHeaders: { "X-NullSpend-Key": process.env.NULLSPEND_API_KEY },
});
// Use anthropic.messages.create() as normal
```

## API Endpoints (API Key auth via X-NullSpend-Key header)
- `POST /api/cost-events` — ingest cost event: `{ provider, model, inputTokens, outputTokens, costMicrodollars, tags? }`
- `POST /api/cost-events/batch` — batch ingest (max 100): `{ events: [...] }`
- `GET /api/budgets/status` — check remaining budget: returns `{ entities: [{ limitMicrodollars, spendMicrodollars, remainingMicrodollars }] }`
- `POST /api/actions` — create HITL approval: `{ agentId, actionType, payload }` → returns `{ data: { id, status } }`
- `GET /api/actions/:id` — poll action status
- `POST /api/actions/:id/result` — mark action result: `{ status: "executing" | "executed" | "failed" }`

## Error Format
`{ error: { code: "machine_code", message: "Human readable.", details: null } }`
Codes: `authentication_required` (401), `budget_exceeded` (429), `rate_limited` (429), `validation_error` (400)

## Cost Units
All costs in microdollars: 1 microdollar = $0.000001. $1 = 1,000,000 microdollars.

## JS SDK
```typescript
import { NullSpend } from "@nullspend/sdk";
const ns = new NullSpend({ baseUrl: "https://nullspend.com", apiKey: process.env.NULLSPEND_API_KEY });
await ns.reportCost({ provider: "openai", model: "gpt-4o", inputTokens: 1000, outputTokens: 500, costMicrodollars: 6750 });
const budget = await ns.checkBudget();
```

## Docs: https://nullspend.com/docs
## llms.txt: https://nullspend.com/llms.txt
````

## For Claude Code

Add to your project's `CLAUDE.md` file:

````markdown
## NullSpend Integration

This project uses NullSpend for AI cost tracking and budget enforcement.

- Proxy URL: `https://proxy.nullspend.com/v1` (set as OPENAI_BASE_URL or ANTHROPIC_BASE_URL)
- API key: stored in NULLSPEND_API_KEY env var, sent as `X-NullSpend-Key` header
- Tags: `X-NullSpend-Tags` header with JSON `{"key":"value"}` for per-customer attribution
- Sessions: `X-NullSpend-Session` header for per-conversation spend limits
- Budget check: `GET /api/budgets/status` with X-NullSpend-Key header
- Cost ingest: `POST /api/cost-events` with `{ provider, model, inputTokens, outputTokens, costMicrodollars }`
- HITL: `POST /api/actions` → poll `GET /api/actions/:id` → `POST /api/actions/:id/result`
- Error format: `{ error: { code, message, details } }`
- Costs in microdollars (1M microdollars = $1)
- JS SDK: `@nullspend/sdk` — NullSpend class with reportCost, checkBudget, createAction, proposeAndWait
- Python SDK: `nullspend` — NullSpendClient class with same methods
- Full API reference: https://nullspend.com/llms.txt
````

## For GitHub Copilot

Create `.github/copilot-instructions.md`:

````markdown
When working with NullSpend:
- The proxy sits at proxy.nullspend.com/v1 (replaces api.openai.com/v1 or anthropic base URL)
- Every request needs the X-NullSpend-Key header with the API key from NULLSPEND_API_KEY env var
- Cost attribution tags go in X-NullSpend-Tags header as JSON: {"customer_id":"acme"}
- Session tracking uses X-NullSpend-Session header
- Costs are in microdollars (1,000,000 = $1)
- Budget status: GET /api/budgets/status with the API key header
- Error shape: { error: { code, message, details } }
- JS SDK: @nullspend/sdk, Python SDK: nullspend
- Docs: https://nullspend.com/docs
- Machine-readable API reference: https://nullspend.com/llms.txt
````

## Machine-Readable API Reference

NullSpend publishes a structured text file at [`/llms.txt`](https://nullspend.com/llms.txt) that AI tools can fetch for complete API context. This follows the [llms.txt standard](https://llmstxt.org/) for making documentation agent-discoverable.

If your AI tool supports URL fetching, point it at `https://nullspend.com/llms.txt` instead of copying blocks manually.
