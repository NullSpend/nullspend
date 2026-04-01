# @nullspend/claude-agent

NullSpend adapter for the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) — route LLM calls through the NullSpend proxy for automatic cost tracking and budget enforcement.

## Install

```bash
npm install @nullspend/claude-agent
```

## Quick Start

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
    tags: { agent: "my-agent", team: "engineering" },
  }),
});
```

### With Budget Awareness (async)

```typescript
import { withNullSpendAsync } from "@nullspend/claude-agent";

const agent = new Agent({
  client,
  model: "claude-sonnet-4-6",
  ...(await withNullSpendAsync({
    apiKey: process.env.NULLSPEND_API_KEY,
    tags: { agent: "my-agent" },
  })),
});
```

`withNullSpendAsync` fetches the API key's policy from the proxy and injects budget constraints into the agent's system prompt via `appendSystemPrompt`.

## API

### `withNullSpend(options): Options`

Synchronous. Returns a partial Claude Agent SDK `Options` object that routes LLM calls through the NullSpend proxy.

### `withNullSpendAsync(options): Promise<Options>`

Async variant. Same as `withNullSpend`, but also fetches the API key's budget policy and sets `appendSystemPrompt` with budget constraints. Policy responses are cached for 60 seconds.

### Options

| Option             | Type                    | Required | Default                          | Description                                              |
| ------------------ | ----------------------- | -------- | -------------------------------- | -------------------------------------------------------- |
| `apiKey`           | `string`                | Yes      |                                  | NullSpend API key (`ns_live_sk_...` or `ns_test_sk_...`) |
| `proxyUrl`         | `string`                | No       | `https://proxy.nullspend.com`    | Override the proxy URL                                   |
| `tags`             | `Record<string,string>` | No       |                                  | Cost attribution tags (max 10 keys, key max 64 chars, value max 256 chars) |
| `budgetSessionId`  | `string`                | No       |                                  | Session ID for budget-level cost grouping                |
| `autoSession`      | `boolean`               | No       | `true`                           | Auto-generate a session ID if `budgetSessionId` is not provided |
| `traceId`          | `string`                | No       |                                  | 32-char lowercase hex trace ID for request correlation   |
| `actionId`         | `string`                | No       |                                  | Link cost events to a HITL action (`ns_act_<UUID>`)      |
| `budgetAwareness`  | `boolean`               | No       | `true`                           | Fetch policy and inject constraints (`withNullSpendAsync` only) |

All other Claude Agent SDK options passed alongside these are preserved and passed through unchanged.

### What It Returns

Both functions return an `Options` object with:

- `env.ANTHROPIC_BASE_URL` — set to the proxy URL
- `env.ANTHROPIC_CUSTOM_HEADERS` — newline-delimited NullSpend headers (`x-nullspend-key`, `x-nullspend-session`, `x-nullspend-tags`, etc.)
- `appendSystemPrompt` — budget constraints (only set by `withNullSpendAsync` when policy fetch succeeds)

### Validation

- `apiKey` — required, must not contain newline characters
- `tags` — max 10 keys; keys must match `[a-zA-Z0-9_-]+` (max 64 chars); values max 256 chars
- `traceId` — must be 32 lowercase hex characters
- `actionId` — must match `ns_act_<UUID>` format
- `budgetSessionId` — must not contain newline characters

## How It Works

`withNullSpend` configures the Agent SDK to send LLM requests through the NullSpend proxy instead of directly to Anthropic. The proxy authenticates the request, forwards it to Anthropic, tracks token usage and cost, enforces any active budgets, and reports the cost event to your dashboard.

## License

Apache-2.0
