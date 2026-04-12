/**
 * Streaming-specific smoke tests for the Anthropic proxy route.
 * Verifies that Anthropic's named-event SSE format is preserved through
 * the proxy without modification (transparent passthrough).
 *
 * Requires:
 *   - `pnpm proxy:dev` running on localhost:8787
 *   - Real Anthropic API key in ANTHROPIC_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 *
 * Run with: npx vitest run smoke-anthropic-streaming.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { ANTHROPIC_API_KEY, anthropicAuthHeaders, BASE, isServerUp } from "./smoke-test-helpers.js";

function parseSSEEvents(text: string): Array<{ event?: string; data?: string }> {
  const events: Array<{ event?: string; data?: string }> = [];
  let currentEvent: { event?: string; data?: string } = {};

  for (const line of text.split("\n")) {
    if (line.startsWith("event: ")) {
      currentEvent.event = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      currentEvent.data = line.slice(6).trim();
    } else if (line.trim() === "" && (currentEvent.event || currentEvent.data)) {
      events.push(currentEvent);
      currentEvent = {};
    }
  }

  if (currentEvent.event || currentEvent.data) {
    events.push(currentEvent);
  }

  return events;
}

describe("Anthropic streaming format smoke tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error("Proxy dev server is not running.");
    }
    if (!ANTHROPIC_API_KEY) {
      throw new Error("ANTHROPIC_API_KEY env var is required.");
    }
  });

  it("SSE events contain event: field lines (named events not stripped)", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 20,
        messages: [{ role: "user", content: "Say hello" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();

    const eventLines = text
      .split("\n")
      .filter((l) => l.startsWith("event: "));
    expect(eventLines.length).toBeGreaterThan(0);
  }, 30_000);

  it("message_start event present with input token usage", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say hi" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();
    const events = parseSSEEvents(text);

    const messageStart = events.find((e) => e.event === "message_start");
    expect(messageStart).toBeDefined();
    expect(messageStart!.data).toBeTruthy();

    const payload = JSON.parse(messageStart!.data!);
    expect(payload.type).toBe("message_start");
    expect(payload.message).toHaveProperty("usage");
    expect(payload.message.usage.input_tokens).toBeGreaterThan(0);
  }, 30_000);

  it("content_block_delta events contain text content", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 20,
        messages: [{ role: "user", content: "Say hello world" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();
    const events = parseSSEEvents(text);

    const deltas = events.filter((e) => e.event === "content_block_delta");
    expect(deltas.length).toBeGreaterThan(0);

    for (const delta of deltas) {
      const payload = JSON.parse(delta.data!);
      expect(payload.type).toBe("content_block_delta");
      expect(payload.delta).toHaveProperty("type");
      if (payload.delta.type === "text_delta") {
        expect(typeof payload.delta.text).toBe("string");
      }
    }
  }, 30_000);

  it("message_delta event present with output token usage", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say ok" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();
    const events = parseSSEEvents(text);

    const messageDelta = events.find((e) => e.event === "message_delta");
    expect(messageDelta).toBeDefined();
    expect(messageDelta!.data).toBeTruthy();

    const payload = JSON.parse(messageDelta!.data!);
    expect(payload.type).toBe("message_delta");
    expect(payload.usage).toHaveProperty("output_tokens");
    expect(payload.usage.output_tokens).toBeGreaterThan(0);
  }, 30_000);

  it("message_stop event terminates stream", async () => {
    const res = await fetch(`${BASE}/v1/messages`, {
      method: "POST",
      headers: anthropicAuthHeaders(),
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 10,
        messages: [{ role: "user", content: "Say ok" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();
    const events = parseSSEEvents(text);

    const messageStop = events.find((e) => e.event === "message_stop");
    expect(messageStop).toBeDefined();

    const payload = JSON.parse(messageStop!.data!);
    expect(payload.type).toBe("message_stop");

    // message_stop should be the last named event
    const lastEvent = events[events.length - 1];
    expect(lastEvent.event).toBe("message_stop");
  }, 30_000);
});
