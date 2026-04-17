import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

if (process.env.RECORD !== "1") {
  console.error("This script requires RECORD=1. Use `pnpm smoke:record`.");
  process.exit(1);
}

const envPath = ".env.smoke";
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      const val = trimmed.slice(eqIdx + 1).trim();
      if (!process.env[key]) process.env[key] = val;
    }
  }
}

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY missing. Populate .env.smoke first.");
  process.exit(1);
}
if (!ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY missing. Populate .env.smoke first.");
  process.exit(1);
}

const CASSETTES_DIR = new URL(
  "../src/__tests__/fixtures/cassettes/",
  import.meta.url,
);

async function writeCassette(name: string, payload: unknown) {
  const path = new URL(`${name}.json`, CASSETTES_DIR);
  await mkdir(dirname(fileURLToPath(path)), { recursive: true });
  await writeFile(
    fileURLToPath(path),
    JSON.stringify(payload, null, 2) + "\n",
  );
  console.log(`wrote ${name}.json`);
}

async function recordOpenAIChatCompletion() {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${OPENAI_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [
        { role: "user", content: "Say 'cassette recorded' and nothing else." },
      ],
      max_tokens: 10,
    }),
  });
  if (!res.ok) {
    throw new Error(
      `OpenAI non-streaming ${res.status}: ${await res.text()}`,
    );
  }
  const body = (await res.json()) as Record<string, unknown>;
  body.id = "cassette-openai-nonstream";
  body.created = 0;
  body.system_fingerprint = "cassette";
  await writeCassette("openai-chat-completion", body);
}

async function recordOpenAIChatStreaming() {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${OPENAI_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "Say 'cassette' and nothing else." }],
      max_tokens: 5,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });
  if (!res.ok) {
    throw new Error(`OpenAI streaming ${res.status}: ${await res.text()}`);
  }
  const raw = await res.text();
  const normalized = raw
    .replace(/"id":"chatcmpl-[A-Za-z0-9]+"/g, '"id":"cassette-openai-stream"')
    .replace(/"created":\d+/g, '"created":0')
    .replace(/"system_fingerprint":"[^"]*"/g, '"system_fingerprint":"cassette"');
  await writeCassette("openai-chat-streaming", {
    contentType: "text/event-stream",
    body: normalized,
  });
}

async function recordAnthropicMessages() {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-3-haiku-20240307",
      max_tokens: 10,
      messages: [
        { role: "user", content: "Say 'cassette recorded' and nothing else." },
      ],
    }),
  });
  if (!res.ok) {
    throw new Error(`Anthropic non-streaming ${res.status}: ${await res.text()}`);
  }
  const body = (await res.json()) as Record<string, unknown>;
  body.id = "msg_cassette_anthropic_nonstream";
  await writeCassette("anthropic-messages", body);
}

async function main() {
  await recordOpenAIChatCompletion();
  await recordOpenAIChatStreaming();
  await recordAnthropicMessages();
  console.log(
    "\nDone. Review git diff on cassettes — only real shape drift should appear.",
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
