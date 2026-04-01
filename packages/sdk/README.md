# @nullspend/sdk

TypeScript SDK for [NullSpend](https://github.com/NullSpend/nullspend) — propose, approve, and execute risky AI agent actions.

## Quick start

```typescript
import { NullSpend } from "@nullspend/sdk";

const seam = new NullSpend({
  baseUrl: "http://localhost:3000",
  apiKey: "ns_live_sk_...", // from your Settings page
});

const result = await seam.proposeAndWait({
  agentId: "my-agent",
  actionType: "send_email",
  payload: { to: "sarah@example.com", subject: "Follow up" },
  execute: async () => {
    return await sendEmail("sarah@example.com", "Follow up", "...");
  },
});
```

The SDK creates a pending action in NullSpend, polls until a human approves or rejects it, then either runs your `execute` callback or throws.

## API

### `new NullSpend(config)`

| Option             | Type     | Required | Description                                      |
| ------------------ | -------- | -------- | ------------------------------------------------ |
| `baseUrl`          | `string` | Yes      | URL of your NullSpend instance                   |
| `apiKey`           | `string` | Yes      | API key from Settings                            |
| `fetch`            | `fetch`  | No       | Custom fetch implementation (defaults to global)  |
| `requestTimeoutMs` | `number` | No       | Per-request timeout in ms (default: 30000). Set to 0 to disable. |

### `seam.proposeAndWait(options)`

High-level: proposes an action, waits for approval, executes, and reports the result.

| Option           | Type                          | Default  | Description                        |
| ---------------- | ----------------------------- | -------- | ---------------------------------- |
| `agentId`        | `string`                      | required | Identifier for the agent           |
| `actionType`     | `string`                      | required | e.g. `send_email`, `http_post`     |
| `payload`        | `Record<string, unknown>`     | required | Action details shown in the inbox  |
| `metadata`       | `Record<string, unknown>`     | optional | Extra context (environment, etc.)  |
| `execute`        | `(context?: ExecuteContext) => T \| Promise<T>` | required | Runs only if approved. Receives `{ actionId }` for cost correlation. |
| `expiresInSeconds` | `number \| null`              | optional | Server-side TTL. Omit for default (1 hour). Set to 0 or null for never-expire. |
| `pollIntervalMs` | `number`                      | 2000     | ms between status polls            |
| `timeoutMs`      | `number`                      | 300000   | Total timeout in ms                |
| `onPoll`         | `(action: ActionRecord) => void` | optional | Called each poll cycle           |

### Lower-level methods

```typescript
// createAction returns { id, status: "pending", expiresAt: string | null }
const action = await seam.createAction({ agentId, actionType, payload, expiresInSeconds: 600 });
const fetched = await seam.getAction(actionId);
const decided = await seam.waitForDecision(actionId, { pollIntervalMs, timeoutMs });
await seam.markResult(actionId, { status: "executed", result: { ... } });
```

### Cost correlation

When using the NullSpend proxy for LLM calls, pass the `actionId` from the execute context as a header to link cost events to the action:

```typescript
const result = await seam.proposeAndWait({
  agentId: "my-agent",
  actionType: "http_post",
  payload: { prompt: "Summarize this document" },
  execute: async (context) => {
    const res = await fetch("https://proxy.nullspend.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-nullspend-action-id": context?.actionId ?? "",
      },
      body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "Hello" }] }),
    });
    return res.json();
  },
});
```

The linked cost data will appear on the action detail page in the dashboard.

## Error handling

```typescript
import { RejectedError, TimeoutError, NullSpendError } from "@nullspend/sdk";

try {
  await seam.proposeAndWait({ ... });
} catch (err) {
  if (err instanceof RejectedError) {
    // Human rejected (or action expired)
  } else if (err instanceof TimeoutError) {
    // No decision within timeoutMs
  } else if (err instanceof NullSpendError) {
    // API error (err.statusCode has the HTTP status)
  }
}
```

## Demos

Three runnable demos in [`examples/`](examples/) show the approval loop for different action types:

| Demo | Action type | What it does |
|---|---|---|
| [`demo-send-email.ts`](examples/demo-send-email.ts) | `send_email` | Simulates sending an email (no real mail sent) |
| [`demo-http-post.ts`](examples/demo-http-post.ts) | `http_post` | POSTs a CRM lead payload to a real public API |
| [`demo-shell-command.ts`](examples/demo-shell-command.ts) | `shell_command` | Executes a safe shell command on the host |

```bash
# Terminal 1: start the app
pnpm dev

# Terminal 2: run any demo
NULLSPEND_API_KEY=ns_live_sk_... pnpm tsx packages/sdk/examples/demo-send-email.ts
NULLSPEND_API_KEY=ns_live_sk_... pnpm tsx packages/sdk/examples/demo-http-post.ts
NULLSPEND_API_KEY=ns_live_sk_... pnpm tsx packages/sdk/examples/demo-shell-command.ts
```

Then open <http://localhost:3000/app/inbox> and approve the action.
