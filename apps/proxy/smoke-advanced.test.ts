/**
 * Advanced smoke tests for the OpenAI proxy.
 * Tests real-world scenarios that go beyond basic request/response:
 *
 * - Tool calling / function calling responses
 * - Multiple choices (n > 1)
 * - Long streaming responses (100+ tokens)
 * - Health/ready endpoint
 * - JSON mode responses
 * - Streaming + non-streaming cost consistency
 * - Response format edge cases
 * - Proxy overhead measurement
 *
 * Requires:
 *   - `pnpm proxy:dev` running
 *   - Real OpenAI API key in OPENAI_API_KEY env var
 *   - NULLSPEND_API_KEY for proxy auth
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, OPENAI_API_KEY, authHeaders, isServerUp } from "./smoke-test-helpers.js";

describe("Advanced proxy scenarios", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error("Proxy dev server is not running. Start it with `pnpm proxy:dev`.");
    }
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY env var is required.");
    }
  });

  // ── Tool calling / function calling ──

  describe("Tool calling responses", () => {
    it("non-streaming tool call response has correct structure", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "What's the weather in San Francisco?" }],
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get the current weather in a location",
                parameters: {
                  type: "object",
                  properties: {
                    location: { type: "string", description: "City name" },
                  },
                  required: ["location"],
                },
              },
            },
          ],
          tool_choice: "auto",
          max_tokens: 100,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");
      expect(body.usage.prompt_tokens).toBeGreaterThan(0);
      expect(body).toHaveProperty("choices");

      const choice = body.choices[0];
      if (choice.finish_reason === "tool_calls") {
        expect(choice.message.tool_calls).toBeDefined();
        expect(Array.isArray(choice.message.tool_calls)).toBe(true);
        expect(choice.message.tool_calls[0]).toHaveProperty("function");
        expect(choice.message.tool_calls[0].function).toHaveProperty("name", "get_weather");
        expect(choice.message.tool_calls[0].function).toHaveProperty("arguments");
        const args = JSON.parse(choice.message.tool_calls[0].function.arguments);
        expect(args).toHaveProperty("location");
      }
    }, 30_000);

    it("streaming tool call response completes with [DONE]", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get weather",
                parameters: {
                  type: "object",
                  properties: { location: { type: "string" } },
                  required: ["location"],
                },
              },
            },
          ],
          tool_choice: "auto",
          stream: true,
          max_tokens: 100,
        }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("data:");
      expect(text).toContain("[DONE]");
    }, 30_000);

    it("tool call with tool_choice 'none' returns text content, not tool calls", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "What's the weather?" }],
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get weather",
                parameters: { type: "object", properties: { location: { type: "string" } } },
              },
            },
          ],
          tool_choice: "none",
          max_tokens: 50,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices[0].message.content).toBeTruthy();
      expect(body.choices[0].message.tool_calls).toBeUndefined();
      expect(body.choices[0].finish_reason).toBe("stop");
    }, 30_000);
  });

  // ── Multiple choices (n > 1) ──

  describe("Multiple choices", () => {
    it("n=2 non-streaming returns two choices with usage", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Pick a random number between 1 and 10" }],
          n: 2,
          max_tokens: 5,
          temperature: 1,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices.length).toBe(2);
      expect(body.choices[0].index).toBe(0);
      expect(body.choices[1].index).toBe(1);
      expect(body.choices[0].message.content).toBeTruthy();
      expect(body.choices[1].message.content).toBeTruthy();
      expect(body.usage.completion_tokens).toBeGreaterThan(0);
    }, 30_000);

    it("n=3 streaming returns valid SSE and completes", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say a color" }],
          n: 3,
          max_tokens: 3,
          stream: true,
          temperature: 1,
        }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");

      const dataLines = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim())
        .filter((l) => l !== "[DONE]");

      const indices = new Set<number>();
      for (const line of dataLines) {
        const chunk = JSON.parse(line);
        if (chunk.choices?.[0]?.index !== undefined) {
          indices.add(chunk.choices[0].index);
        }
      }
      expect(indices.size).toBe(3);
    }, 30_000);
  });

  // ── Long streaming responses ──

  describe("Long streaming responses", () => {
    it("200+ token streaming response completes without corruption", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "user",
              content: "List the first 20 prime numbers, one per line, with their position number.",
            },
          ],
          stream: true,
          max_tokens: 300,
        }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");

      const dataLines = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim())
        .filter((l) => l !== "[DONE]");

      expect(dataLines.length).toBeGreaterThan(10);

      // Every data line should be valid JSON
      for (const line of dataLines) {
        expect(() => JSON.parse(line)).not.toThrow();
      }

      // Usage should be present in the final chunk
      const lastChunk = JSON.parse(dataLines[dataLines.length - 1]);
      expect(lastChunk).toHaveProperty("usage");
      expect(lastChunk.usage.completion_tokens).toBeGreaterThan(50);
    }, 60_000);
  });

  // ── Health/ready endpoint ──

  describe("Health and readiness", () => {
    it("/health/ready returns 200", async () => {
      const res = await fetch(`${BASE}/health/ready`);
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");
    });

    it("/health responds faster than 500ms", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/health`);
      const elapsed = performance.now() - start;
      expect(res.status).toBe(200);
      expect(elapsed).toBeLessThan(500);
    });
  });

  // ── JSON mode ──

  describe("JSON mode responses", () => {
    it("response_format json_object returns valid JSON content", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: "You always respond with valid JSON." },
            { role: "user", content: "Give me a JSON object with fields name and age." },
          ],
          response_format: { type: "json_object" },
          max_tokens: 50,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body).toHaveProperty("usage");

      const content = body.choices[0].message.content;
      expect(() => JSON.parse(content)).not.toThrow();
      const parsed = JSON.parse(content);
      expect(parsed).toHaveProperty("name");
    }, 30_000);

    it("streaming with json_object format completes with usage", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: "Respond only with valid JSON." },
            { role: "user", content: "Return {\"status\": \"ok\"}" },
          ],
          response_format: { type: "json_object" },
          stream: true,
          max_tokens: 20,
        }),
      });

      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).toContain("[DONE]");
      expect(text).toContain('"usage"');
    }, 30_000);
  });

  // ── Cost consistency: streaming vs non-streaming ──

  describe("Cost consistency", () => {
    it("streaming and non-streaming produce comparable token counts for identical prompts", async () => {
      const body = {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "What is 1+1? Answer with just the number." }],
        max_tokens: 5,
        temperature: 0,
        seed: 99,
      };

      const nonStreamRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ ...body, stream: false }),
      });
      const nonStreamBody = await nonStreamRes.json();

      const streamRes = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ ...body, stream: true }),
      });
      const streamText = await streamRes.text();

      // Extract usage from streaming
      const streamDataLines = streamText
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim())
        .filter((l) => l !== "[DONE]");

      const lastChunk = JSON.parse(streamDataLines[streamDataLines.length - 1]);

      // Prompt tokens should be identical (same input)
      expect(nonStreamBody.usage.prompt_tokens).toBe(lastChunk.usage.prompt_tokens);

      // Completion tokens should be very close (same seed, temp 0)
      expect(Math.abs(nonStreamBody.usage.completion_tokens - lastChunk.usage.completion_tokens)).toBeLessThanOrEqual(1);
    }, 60_000);
  });

  // ── Proxy overhead measurement ──

  describe("Proxy overhead", () => {
    it("non-streaming request completes in under 10 seconds", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say hi" }],
          max_tokens: 3,
        }),
      });
      const elapsed = performance.now() - start;

      expect(res.status).toBe(200);
      expect(elapsed).toBeLessThan(10_000);
      await res.json();
    }, 15_000);

    it("auth rejection is fast (under 500ms)", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "x-nullspend-key": "wrong-key",
        },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] }),
      });
      const elapsed = performance.now() - start;

      expect(res.status).toBe(401);
      expect(elapsed).toBeLessThan(500);
      await res.text();
    });

    it("body validation rejection is fast (under 500ms)", async () => {
      const start = performance.now();
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: "not json",
      });
      const elapsed = performance.now() - start;

      expect(res.status).toBe(400);
      expect(elapsed).toBeLessThan(500);
      await res.text();
    });
  });

  // ── Response field completeness ──

  describe("Response field completeness", () => {
    it("non-streaming response includes all OpenAI-standard usage sub-fields", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say hello" }],
          max_tokens: 5,
        }),
      });

      const body = await res.json();
      const usage = body.usage;

      expect(usage).toHaveProperty("prompt_tokens");
      expect(usage).toHaveProperty("completion_tokens");
      expect(usage).toHaveProperty("total_tokens");
      expect(usage.total_tokens).toBe(usage.prompt_tokens + usage.completion_tokens);

      // prompt_tokens_details may or may not be present depending on model
      if (usage.prompt_tokens_details) {
        expect(usage.prompt_tokens_details).toHaveProperty("cached_tokens");
      }

      // completion_tokens_details may include reasoning_tokens for reasoning models
      if (usage.completion_tokens_details) {
        expect(typeof usage.completion_tokens_details).toBe("object");
      }
    }, 30_000);

    it("streaming chunks include consistent id and model across all chunks", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Count to 5" }],
          stream: true,
          max_tokens: 20,
        }),
      });

      const text = await res.text();
      const chunks = text
        .split("\n")
        .filter((l) => l.trim().startsWith("data:"))
        .map((l) => l.trim().slice(5).trim())
        .filter((l) => l !== "[DONE]")
        .map((l) => JSON.parse(l));

      const ids = new Set(chunks.map((c) => c.id));
      const models = new Set(chunks.map((c) => c.model));

      // All chunks should share the same completion ID
      expect(ids.size).toBe(1);
      // All chunks should reference the same model
      expect(models.size).toBe(1);
    }, 30_000);
  });

  // ── Edge case: empty assistant responses ──

  describe("Edge case responses", () => {
    it("max_tokens: 1 produces a minimal but valid response", async () => {
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "Say something" }],
          max_tokens: 1,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices[0].finish_reason).toBe("length");
      expect(body.usage.completion_tokens).toBe(1);
    }, 30_000);

    it("very long system prompt is handled correctly", async () => {
      const longSystem = "You are a helpful assistant. ".repeat(200);
      const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: longSystem },
            { role: "user", content: "Say ok" },
          ],
          max_tokens: 5,
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.usage.prompt_tokens).toBeGreaterThan(500);
      expect(body).toHaveProperty("choices");
    }, 30_000);
  });
});
