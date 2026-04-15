---
title: "Gemini Quickstart"
description: "Get cost tracking for your Google Gemini calls in under 2 minutes."
---

Get cost tracking for your Google Gemini calls in under 2 minutes.

## Prerequisites

- A NullSpend account ([sign up](https://nullspend.dev/signup))
- An existing app that calls the Google Gemini API
- A Google AI Studio API key ([get one](https://aistudio.google.com/apikey))

## Step 1: Create an API Key

1. Log in to the [NullSpend dashboard](https://nullspend.dev/app/analytics)
2. Go to **Settings** → **Create API Key**
3. Copy the key (starts with `ns_live_sk_`) — you won't see it again

## Step 2: Point Your Requests at the Proxy

NullSpend supports Google's native Gemini REST API. Change the base URL from `generativelanguage.googleapis.com` to `proxy.nullspend.dev` and add your NullSpend key. No SDK wrapper needed, no body format changes.

### cURL

```bash
# Non-streaming
curl "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:generateContent" \
  -H "x-goog-api-key: $GOOGLE_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Hello, what can you do?"}]}],
    "generationConfig": {"maxOutputTokens": 200}
  }'

# Streaming
curl "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:streamGenerateContent" \
  -H "x-goog-api-key: $GOOGLE_API_KEY" \
  -H "X-NullSpend-Key: $NULLSPEND_API_KEY" \
  -H "Content-Type: application/json" \
  --no-buffer \
  -d '{
    "contents": [{"parts": [{"text": "Count from 1 to 10"}]}],
    "generationConfig": {"maxOutputTokens": 200}
  }'
```

### TypeScript (fetch)

```typescript
const response = await fetch(
  "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:generateContent",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": process.env.GOOGLE_API_KEY!,
      "X-NullSpend-Key": process.env.NULLSPEND_API_KEY!,
    },
    body: JSON.stringify({
      contents: [{ parts: [{ text: "Hello" }] }],
    }),
  },
);

const data = await response.json();
console.log(data.candidates[0].content.parts[0].text);
```

### TypeScript (Google GenAI SDK)

```typescript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY!,
  httpOptions: {
    baseUrl: "https://proxy.nullspend.dev/v1beta",
    headers: {
      "X-NullSpend-Key": process.env.NULLSPEND_API_KEY!,
    },
  },
});

const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "Hello, what can you do?",
});
console.log(response.text);
```

### Python (requests)

```python
import requests
import os

response = requests.post(
    "https://proxy.nullspend.dev/v1beta/models/gemini-2.5-flash:generateContent",
    headers={
        "Content-Type": "application/json",
        "x-goog-api-key": os.environ["GOOGLE_API_KEY"],
        "X-NullSpend-Key": os.environ["NULLSPEND_API_KEY"],
    },
    json={
        "contents": [{"parts": [{"text": "Hello"}]}],
    },
)
print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
```

### Python (Google GenAI SDK)

> **Note:** The Python GenAI SDK currently only supports custom `base_url` with `vertexai=True`. For Gemini API (non-Vertex) proxying, use the `requests` example above or set the `HTTPS_PROXY` environment variable.

**Note on `maxOutputTokens`:** Gemini 2.5 models are "thinking" models that use part of the output token budget for internal reasoning. If you set `maxOutputTokens` too low (e.g., 20), the model may spend all tokens on thinking and return no visible output. Use at least 200 for short responses.

## Step 3: Check the Dashboard

Open the [NullSpend dashboard](https://nullspend.dev/app/analytics). Cost events appear within seconds. You'll see:

- **Daily spend chart** — cost over time
- **Model breakdown** — Gemini models with `provider: google`
- **Thinking tokens** — tagged as `_ns_thinking_tokens` for Gemini 2.5 models

### What Gets Tracked

| Field | Source |
|---|---|
| Input tokens | `usageMetadata.promptTokenCount` |
| Output tokens | `usageMetadata.candidatesTokenCount` |
| Cached tokens | `usageMetadata.cachedContentTokenCount` |
| Thinking tokens | `usageMetadata.thoughtsTokenCount` (tag: `_ns_thinking_tokens`) |
| Google response ID | `responseId` (tag: `_ns_google_response_id`) |

## How It Differs from OpenAI/Anthropic

| Aspect | OpenAI/Anthropic | Gemini |
|---|---|---|
| Model location | In request body (`"model": "gpt-4o"`) | In URL path (`/models/gemini-2.5-flash:generateContent`) |
| Streaming | Body field `"stream": true` | Different endpoint (`:streamGenerateContent`) |
| Auth header | `Authorization: Bearer` | `x-goog-api-key` (or `Authorization: Bearer`) |
| SSE format | Delta-based (partial chunks) | Complete response per event |
| Request body | NullSpend convention | Native Gemini format (passthrough, no transformation) |

## What's Next

- **[Set a budget](../features/budgets.md)** — The proxy blocks Gemini requests with `429` when the budget ceiling is hit.
- **[Add tags](../features/tags.md)** — Attribute Gemini costs to teams or features with the [`X-NullSpend-Tags`](../api-reference/custom-headers.md#x-nullspend-tags) header.
- **[Configure webhooks](../webhooks/overview.md)** — Get notified on cost events and budget thresholds.
- **OpenAI too?** — [OpenAI Quickstart](openai.md)
- **Anthropic?** — [Anthropic Quickstart](anthropic.md)

## Troubleshooting

**401 Unauthorized**
Your `X-NullSpend-Key` header is missing or invalid. Your Google API key (`x-goog-api-key`) is separate and forwards to Google unchanged.

**404 Not Found**
Check the URL path. Gemini endpoints must match `/v1beta/models/{model}:generateContent` or `:streamGenerateContent` exactly. Other Gemini methods (e.g., `:countTokens`, `:embedContent`) are not yet supported.

**429 Too Many Requests**
Either a NullSpend budget was exceeded (check `error.code`: `budget_exceeded`, `velocity_exceeded`) or you hit the rate limit. Google-side 429s (quota exceeded) pass through with the original error message.

**Empty response (no visible output)**
Gemini 2.5 models use thinking tokens from the output budget. Increase `maxOutputTokens` in `generationConfig` (try 500+). Check `usageMetadata.thoughtsTokenCount` to see how many tokens went to thinking.

**Streaming returns JSON array instead of SSE**
The proxy automatically appends `?alt=sse` to streaming requests. If you're hitting Google directly (not through the proxy), you need to add `?alt=sse` to the URL yourself.
