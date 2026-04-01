/**
 * HTTP smoke tests for the proxy worker.
 * Requires `pnpm proxy:dev` to be running on localhost:8787.
 *
 * Run with: npx vitest run smoke.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { BASE, isServerUp } from "./smoke-test-helpers.js";

describe("proxy smoke tests", () => {
  beforeAll(async () => {
    const up = await isServerUp();
    if (!up) {
      throw new Error(
        "Proxy dev server is not running. Start it with `pnpm proxy:dev` before running smoke tests.",
      );
    }
  });

  describe("GET /health", () => {
    it("returns 200 with correct JSON shape", async () => {
      const res = await fetch(`${BASE}/health`);
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain("application/json");

      const body = await res.json();
      expect(body).toEqual({ status: "ok", service: "nullspend-proxy" });
    });

    it("responds within 100ms (lightweight no-op)", async () => {
      const start = performance.now();
      await fetch(`${BASE}/health`);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(100);
    });

    it("responds to HEAD requests", async () => {
      const res = await fetch(`${BASE}/health`, { method: "HEAD" });
      // CF Workers treat HEAD same as GET for fetch handler
      expect([200, 405]).toContain(res.status);
    });
  });

  describe("GET /health/ready", () => {
    it("returns 200 with valid JSON", async () => {
      const res = await fetch(`${BASE}/health/ready`);
      expect(res.status).toBe(200);

      const body = await res.json();
      expect(body.status).toBe("ok");
      expect(body.service).toBe("nullspend-proxy");
    });
  });

  describe("404 handling", () => {
    it("unknown path returns 404", async () => {
      const res = await fetch(`${BASE}/nonexistent`);
      expect(res.status).toBe(404);
      const body = await res.json();
      expect(body).toEqual({ error: "not_found" });
    });

    it("root path returns 404", async () => {
      const res = await fetch(`${BASE}/`);
      expect(res.status).toBe(404);
    });

    it("deeply nested path returns 404", async () => {
      const res = await fetch(`${BASE}/a/b/c/d/e/f`);
      expect(res.status).toBe(404);
    });

    it("path with query params returns 404", async () => {
      const res = await fetch(`${BASE}/nonexistent?foo=bar&baz=qux`);
      expect(res.status).toBe(404);
    });

    it("path with special characters returns 404", async () => {
      const res = await fetch(`${BASE}/%00%01%02`);
      expect(res.status).toBe(404);
    });

    it("/health with trailing slash returns 404 (strict matching)", async () => {
      const res = await fetch(`${BASE}/health/`);
      expect(res.status).toBe(404);
    });
  });

  describe("HTTP methods", () => {
    it("POST to /health still returns 200 (no method filtering yet)", async () => {
      const res = await fetch(`${BASE}/health`, { method: "POST" });
      // Workers don't filter by method unless coded to
      expect(res.status).toBe(200);
    });

    it("PUT to unknown path returns 404", async () => {
      const res = await fetch(`${BASE}/unknown`, { method: "PUT" });
      expect(res.status).toBe(404);
    });

    it("DELETE to unknown path returns 404", async () => {
      const res = await fetch(`${BASE}/unknown`, { method: "DELETE" });
      expect(res.status).toBe(404);
    });

    it("OPTIONS request doesn't crash", async () => {
      const res = await fetch(`${BASE}/health`, { method: "OPTIONS" });
      expect(res.status).toBeLessThan(500);
    });
  });

  describe("response format consistency", () => {
    it("all responses use application/json content-type", async () => {
      const paths = ["/health", "/health/ready", "/nonexistent"];
      for (const path of paths) {
        const res = await fetch(`${BASE}${path}`);
        expect(
          res.headers.get("content-type"),
          `${path} should return JSON`,
        ).toContain("application/json");
      }
    });

    it("responses are valid parseable JSON", async () => {
      const paths = ["/health", "/health/ready", "/nonexistent", "/", "/a/b/c"];
      for (const path of paths) {
        const res = await fetch(`${BASE}${path}`);
        const text = await res.text();
        expect(() => JSON.parse(text), `${path} should return valid JSON`).not.toThrow();
      }
    });
  });

  describe("concurrent request handling", () => {
    it("handles 20 concurrent /health requests", async () => {
      const requests = Array.from({ length: 20 }, () => fetch(`${BASE}/health`));
      const responses = await Promise.all(requests);
      for (const res of responses) {
        expect(res.status).toBe(200);
      }
    });

    it("handles mixed concurrent requests to different endpoints", async () => {
      const requests = [
        fetch(`${BASE}/health`),
        fetch(`${BASE}/health`),
        fetch(`${BASE}/health/ready`),
        fetch(`${BASE}/nonexistent`),
        fetch(`${BASE}/health`),
        fetch(`${BASE}/another-404`),
      ];
      const responses = await Promise.all(requests);
      expect(responses[0].status).toBe(200);
      expect(responses[1].status).toBe(200);
      expect([200, 503]).toContain(responses[2].status);
      expect(responses[3].status).toBe(404);
      expect(responses[4].status).toBe(200);
      expect(responses[5].status).toBe(404);
    });
  });

  describe("request body handling", () => {
    it("POST with JSON body to unknown path returns 404 (no crash)", async () => {
      const res = await fetch(`${BASE}/api/proxy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "hi" }] }),
      });
      expect(res.status).toBe(404);
    });

    it("POST with malformed JSON to unknown path returns 404 (no crash)", async () => {
      const res = await fetch(`${BASE}/api/proxy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{not valid json!!!",
      });
      expect(res.status).toBe(404);
    });

    it("POST with empty body to unknown path returns 404", async () => {
      const res = await fetch(`${BASE}/api/proxy`, {
        method: "POST",
        body: "",
      });
      expect(res.status).toBe(404);
    });

    it("POST with very large body to unknown path returns 404 (no hang)", async () => {
      const largeBody = "x".repeat(100_000);
      const res = await fetch(`${BASE}/api/proxy`, {
        method: "POST",
        body: largeBody,
      });
      expect(res.status).toBe(404);
    });
  });
});
