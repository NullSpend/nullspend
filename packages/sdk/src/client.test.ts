import { describe, it, expect, vi, afterEach } from "vitest";
import { NullSpend } from "./client.js";
import { NullSpendError, RejectedError, TimeoutError } from "./errors.js";
import type { ActionRecord, CreateActionResponse, NullSpendConfig } from "./types.js";

function mockAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  return {
    id: "act-1",
    agentId: "test-agent",
    actionType: "send_email",
    status: "pending",
    payload: { to: "a@b.com" },
    metadata: null,
    createdAt: "2026-01-01T00:00:00Z",
    approvedAt: null,
    rejectedAt: null,
    executedAt: null,
    expiresAt: null,
    expiredAt: null,
    approvedBy: null,
    rejectedBy: null,
    result: null,
    errorMessage: null,
    environment: null,
    sourceFramework: null,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 ? "OK" : status === 500 ? "Internal Server Error" : `Status ${status}`,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

/** Returns a factory that creates a fresh Response per call (avoids body-reuse issues). */
function jsonResponseFactory(body: unknown, status = 200, headers?: Record<string, string>) {
  return () => jsonResponse(body, status, headers);
}

function createClient(fetchFn: typeof globalThis.fetch): NullSpend {
  return new NullSpend({
    baseUrl: "http://localhost:3000",
    apiKey: "ns_live_sk_test0001",
    fetch: fetchFn,
    maxRetries: 0,
  });
}

function createRetryClient(fetchFn: typeof globalThis.fetch, opts?: Partial<NullSpendConfig>): NullSpend {
  return new NullSpend({
    baseUrl: "http://localhost:3000",
    apiKey: "ns_live_sk_test0001",
    fetch: fetchFn,
    maxRetries: 2,
    ...opts,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe("NullSpend constructor", () => {
  it("throws if baseUrl is missing", () => {
    expect(
      () => new NullSpend({ baseUrl: "", apiKey: "ns_live_sk_x" }),
    ).toThrow("baseUrl is required");
  });

  it("throws if apiKey is missing", () => {
    expect(
      () => new NullSpend({ baseUrl: "http://localhost", apiKey: "" }),
    ).toThrow("apiKey is required");
  });

  it("strips trailing slashes from baseUrl", () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: { id: "1", status: "pending" } }));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000///",
      apiKey: "ns_live_sk_x",
      fetch: fetchFn,
    });
    client.createAction({ agentId: "a", actionType: "send_email", payload: {} });
    expect(fetchFn).toHaveBeenCalledWith(
      "http://localhost:3000/api/actions",
      expect.any(Object),
    );
  });
});

// ---------------------------------------------------------------------------
// createAction
// ---------------------------------------------------------------------------

describe("createAction", () => {
  it("sends correct request and returns id + status", async () => {
    const expected: CreateActionResponse = { id: "act-1", status: "pending" };
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: expected }, 201));
    const client = createClient(fetchFn);

    const result = await client.createAction({
      agentId: "my-agent",
      actionType: "send_email",
      payload: { to: "a@b.com" },
      metadata: { env: "test" },
    });

    expect(result).toEqual(expected);
    expect(fetchFn).toHaveBeenCalledOnce();

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/actions");
    expect(init.method).toBe("POST");
    expect(init.headers["x-nullspend-key"]).toBe("ns_live_sk_test0001");
    expect(init.headers["NullSpend-Version"]).toBe("2026-04-01");
    expect(JSON.parse(init.body)).toEqual({
      agentId: "my-agent",
      actionType: "send_email",
      payload: { to: "a@b.com" },
      metadata: { env: "test" },
    });
  });

  it("sends default NullSpend-Version header", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: { id: "act-1", status: "pending" } }));
    const client = createClient(fetchFn);

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });

    const [, init] = fetchFn.mock.calls[0];
    expect(init.headers["NullSpend-Version"]).toBe("2026-04-01");
  });

  it("sends custom apiVersion when configured", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: { id: "act-1", status: "pending" } }));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      apiVersion: "2099-01-01",
    });

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });

    const [, init] = fetchFn.mock.calls[0];
    expect(init.headers["NullSpend-Version"]).toBe("2099-01-01");
  });

  it("persists NullSpend-Version header across retries", async () => {
    const fetchFn = vi.fn()
      .mockResolvedValueOnce(new Response("", { status: 500, statusText: "Internal Server Error" }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "pending" } }));
    const client = createRetryClient(fetchFn);

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });

    expect(fetchFn).toHaveBeenCalledTimes(2);
    const [, init1] = fetchFn.mock.calls[0];
    const [, init2] = fetchFn.mock.calls[1];
    expect(init1.headers["NullSpend-Version"]).toBe("2026-04-01");
    expect(init2.headers["NullSpend-Version"]).toBe("2026-04-01");
  });

  it("throws NullSpendError on non-OK response", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "bad_request", message: "Bad input", details: null } }, 400),
    );
    const client = createClient(fetchFn);

    await expect(
      client.createAction({ agentId: "a", actionType: "send_email", payload: {} }),
    ).rejects.toThrow(NullSpendError);
  });

  it("populates NullSpendError.code from nested error response", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "validation_error", message: "Invalid input", details: null } }, 400),
    );
    const client = createClient(fetchFn);

    const err = await client.createAction({ agentId: "a", actionType: "send_email", payload: {} }).catch((e: unknown) => e);

    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).code).toBe("validation_error");
    expect((err as NullSpendError).statusCode).toBe(400);
    expect((err as NullSpendError).message).toContain("Invalid input");
  });
});

// ---------------------------------------------------------------------------
// getAction
// ---------------------------------------------------------------------------

describe("getAction", () => {
  it("fetches action by id", async () => {
    const action = mockAction({ id: "act-42", status: "approved" });
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: action }));
    const client = createClient(fetchFn);

    const result = await client.getAction("act-42");
    expect(result.id).toBe("act-42");
    expect(result.status).toBe("approved");

    const [url] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/actions/act-42");
  });
});

// ---------------------------------------------------------------------------
// markResult
// ---------------------------------------------------------------------------

describe("markResult", () => {
  it("posts result for executed action", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "act-1", status: "executed" } }),
    );
    const client = createClient(fetchFn);

    const result = await client.markResult("act-1", {
      status: "executed",
      result: { message: "sent" },
    });

    expect(result.status).toBe("executed");
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/actions/act-1/result");
    expect(init.method).toBe("POST");
  });

  it("posts error for failed action", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "act-1", status: "failed" } }),
    );
    const client = createClient(fetchFn);

    await client.markResult("act-1", {
      status: "failed",
      errorMessage: "SMTP timeout",
    });

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.status).toBe("failed");
    expect(body.errorMessage).toBe("SMTP timeout");
  });
});

// ---------------------------------------------------------------------------
// waitForDecision
// ---------------------------------------------------------------------------

describe("waitForDecision", () => {
  it("returns immediately when action is already approved", async () => {
    const approved = mockAction({ status: "approved" });
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: approved }));
    const client = createClient(fetchFn);

    const result = await client.waitForDecision("act-1", {
      pollIntervalMs: 50,
      timeoutMs: 500,
    });

    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledOnce();
  });

  it("polls until status changes from pending", async () => {
    const pending = mockAction({ status: "pending" });
    const approved = mockAction({ status: "approved" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: pending }))
      .mockResolvedValueOnce(jsonResponse({ data: pending }))
      .mockResolvedValueOnce(jsonResponse({ data: approved }));

    const client = createClient(fetchFn);
    const onPoll = vi.fn();

    const result = await client.waitForDecision("act-1", {
      pollIntervalMs: 10,
      timeoutMs: 5000,
      onPoll,
    });

    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(3);
    expect(onPoll).toHaveBeenCalledTimes(3);
  });

  it("throws TimeoutError when deadline passes", async () => {
    const fetchFn = vi.fn().mockImplementation(() =>
      Promise.resolve(jsonResponse({ data: mockAction({ status: "pending" }) })),
    );
    const client = createClient(fetchFn);

    await expect(
      client.waitForDecision("act-1", {
        pollIntervalMs: 10,
        timeoutMs: 50,
      }),
    ).rejects.toThrow(TimeoutError);
  });

  it("returns on terminal statuses like rejected", async () => {
    const rejected = mockAction({ status: "rejected" });
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: rejected }));
    const client = createClient(fetchFn);

    const result = await client.waitForDecision("act-1", {
      pollIntervalMs: 10,
      timeoutMs: 500,
    });

    expect(result.status).toBe("rejected");
  });
});

// ---------------------------------------------------------------------------
// proposeAndWait
// ---------------------------------------------------------------------------

describe("proposeAndWait", () => {
  it("creates, waits, executes, and reports result on approval", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });
    const markResp = { id: "act-1", status: "executed" };

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision poll 1
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult (executing)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult (executed)
      .mockResolvedValueOnce(jsonResponse({ data: markResp }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue({ sent: true });

    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "send_email",
      payload: { to: "a@b.com" },
      execute,
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toEqual({ sent: true });
    expect(execute).toHaveBeenCalledOnce();
    expect(fetchFn).toHaveBeenCalledTimes(4);
  });

  it("throws RejectedError when action is rejected", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const rejected = mockAction({ id: "act-1", status: "rejected" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: rejected }));

    const client = createClient(fetchFn);
    const execute = vi.fn();

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow(RejectedError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("reports failure when execute throws", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult (executing)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult (failed)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "failed" } }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockRejectedValue(new Error("SMTP timeout"));

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow("SMTP timeout");

    // Should have called markResult with failed status
    const lastCall = fetchFn.mock.calls[3];
    const body = JSON.parse(lastCall[1].body);
    expect(body.status).toBe("failed");
    expect(body.errorMessage).toBe("SMTP timeout");
  });

  it("preserves original error when markResult(failed) also fails", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult(failed) also fails
      .mockResolvedValueOnce(jsonResponse({ error: { code: "server_down", message: "server down", details: null } }, 500));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockRejectedValue(new Error("original execute error"));

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow("original execute error");
  });

  it("passes ExecuteContext with actionId to execute callback", async () => {
    const createResp: CreateActionResponse = { id: "act-ctx-1", status: "pending" };
    const approved = mockAction({ id: "act-ctx-1", status: "approved" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-ctx-1", status: "executing" } }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-ctx-1", status: "executed" } }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue({ done: true });

    await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "http_post",
      payload: { url: "https://example.com" },
      execute,
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(execute).toHaveBeenCalledWith({ actionId: "act-ctx-1" });
  });

  it("works with execute callbacks that ignore context (backwards compat)", async () => {
    const createResp: CreateActionResponse = { id: "act-compat", status: "pending" };
    const approved = mockAction({ id: "act-compat", status: "approved" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-compat", status: "executing" } }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-compat", status: "executed" } }));

    const client = createClient(fetchFn);

    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "send_email",
      payload: { to: "test@example.com" },
      execute: async () => "sent",
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toBe("sent");
  });

  it("handles primitive return values from execute", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executed" } }));

    const client = createClient(fetchFn);
    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "shell_command",
      payload: { cmd: "echo hi" },
      execute: async () => 42,
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toBe(42);

    // Primitive gets wrapped as { value: 42 } for the API
    const resultBody = JSON.parse(fetchFn.mock.calls[3][1].body);
    expect(resultBody.result).toEqual({ value: 42 });
  });

  it("markResult(executed) failure does NOT trigger markResult(failed)", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → ok
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult(executed) → 500
      .mockResolvedValueOnce(jsonResponse({ error: { code: "server_down", message: "server down", details: null } }, 500));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue({ done: true });

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow(NullSpendError);

    // execute ran successfully
    expect(execute).toHaveBeenCalledOnce();
    // Only 4 fetch calls — no 5th markResult(failed) call
    expect(fetchFn).toHaveBeenCalledTimes(4);
  });
});

// ---------------------------------------------------------------------------
// Network & parse error wrapping
// ---------------------------------------------------------------------------

describe("request error wrapping", () => {
  it("wraps network errors in NullSpendError", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    const client = createClient(fetchFn);

    const error = await client
      .getAction("act-1")
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(NullSpendError);
    expect((error as NullSpendError).message).toContain("network error");
    expect((error as NullSpendError).message).toContain("fetch failed");
    expect((error as NullSpendError).statusCode).toBeUndefined();
  });

  it("wraps invalid 2xx JSON in NullSpendError", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      new Response("not json", { status: 200 }),
    );
    const client = createClient(fetchFn);

    const error = await client
      .getAction("act-1")
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(NullSpendError);
    expect((error as NullSpendError).message).toContain("invalid JSON");
    expect((error as NullSpendError).statusCode).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// Retry behavior
// ---------------------------------------------------------------------------

describe("Retry behavior", () => {
  it("retries on 500 → succeeds on second attempt", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ status: "approved" }) }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.getAction("act-1");
    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it.each([502, 503, 504])("retries on %d → succeeds on retry", async (status) => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, status))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ status: "approved" }) }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.getAction("act-1");
    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("retries on 429 with Retry-After → succeeds on retry", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "rate_limited", message: "rate limited", details: null } }, 429, { "Retry-After": "1" }))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ status: "approved" }) }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.getAction("act-1");
    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(2);
    // Should have used Retry-After value (1s = 1000ms)
    expect(sleepSpy).toHaveBeenCalledWith(1000);
  });

  it("retries on network error (TypeError) → succeeds on retry", async () => {
    const fetchFn = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ status: "approved" }) }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.getAction("act-1");
    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("retries on request timeout (DOMException TimeoutError) → succeeds on retry", async () => {
    const fetchFn = vi
      .fn()
      .mockRejectedValueOnce(new DOMException("signal timed out", "TimeoutError"))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ status: "approved" }) }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.getAction("act-1");
    expect(result.status).toBe("approved");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("does NOT retry on 400", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "bad_request", message: "bad request", details: null } }, 400));
    const client = createRetryClient(fetchFn);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it.each([401, 403, 404, 409, 422])("does NOT retry on %d", async (status) => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "nope", message: "nope", details: null } }, status));
    const client = createRetryClient(fetchFn);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("exhausts retries on persistent 500 → throws NullSpendError with status 500 and code", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "broken", message: "broken", details: null } }, 500));
    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const err = await client.getAction("act-1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).statusCode).toBe(500);
    expect((err as NullSpendError).code).toBe("broken");
    expect((err as NullSpendError).message).toContain("broken");
    // 1 initial + 2 retries = 3 — final attempt parses JSON and extracts code
    expect(fetchFn).toHaveBeenCalledTimes(3);
  });

  it("exhausts retries on persistent TypeError → throws NullSpendError (not raw TypeError)", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new TypeError("network down"));
    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const err = await client.getAction("act-1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NullSpendError);
    expect(err).not.toBeInstanceOf(TypeError);
    expect((err as NullSpendError).message).toContain("network error");
    expect(fetchFn).toHaveBeenCalledTimes(3);
  });

  it("maxRetries: 0 disables retry (fetch called 1x on 500)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createClient(fetchFn); // maxRetries: 0

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("respects Retry-After header value", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "wait", message: "wait", details: null } }, 429, { "Retry-After": "2" }))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(sleepSpy).toHaveBeenCalledWith(2000);
  });

  it("Retry-After capped at max delay", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "wait", message: "wait", details: null } }, 429, { "Retry-After": "60" }))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(sleepSpy).toHaveBeenCalledWith(5000); // capped at DEFAULT_MAX_RETRY_DELAY_MS
  });

  it("Retry-After: 0 → immediate retry", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "wait", message: "wait", details: null } }, 429, { "Retry-After": "0" }))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(sleepSpy).toHaveBeenCalledWith(0);
  });

  it("response body consumed on retry (response.text() called)", async () => {
    const textSpy = vi.fn().mockResolvedValue("error body");
    const retryResponse = {
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      headers: new Headers(),
      text: textSpy,
    } as unknown as Response;

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(retryResponse)
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(textSpy).toHaveBeenCalledOnce();
  });

  it("custom retryBaseDelayMs is respected", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn, { retryBaseDelayMs: 200 });
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    // attempt 0, base 200: floor(0.5 * min(200 * 1, 5000)) = floor(100) = 100
    expect(sleepSpy).toHaveBeenCalledWith(100);
  });
});

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

describe("Config validation", () => {
  it("maxRetries: -1 → clamped to 0 (no retries)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createRetryClient(fetchFn, { maxRetries: -1 });

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("maxRetries: NaN → falls back to default (2)", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createRetryClient(fetchFn, { maxRetries: NaN });
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(3); // default: 1 + 2 retries
  });

  it("maxRetries: Infinity → falls back to default (2)", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createRetryClient(fetchFn, { maxRetries: Infinity });
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(3); // 1 + 2 retries (default)
  });

  it("retryBaseDelayMs: -100 → clamped to 0", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn, { retryBaseDelayMs: -100 });
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    // base 0 → ceiling = 0, floor(0.5 * 0) = 0, but min 1
    expect(sleepSpy).toHaveBeenCalledWith(1);
  });

  it("retryBaseDelayMs: 0 → works (minimal delay)", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn, { retryBaseDelayMs: 0 });
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    // min 1
    expect(sleepSpy).toHaveBeenCalledWith(1);
  });

  it("retryBaseDelayMs: NaN → falls back to default (500)", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn, { retryBaseDelayMs: NaN });
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    // attempt 0, base 500: floor(0.5 * 500) = 250
    expect(sleepSpy).toHaveBeenCalledWith(250);
  });

  it("maxRetries: 0.5 → floored to 0 (no retries)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createRetryClient(fetchFn, { maxRetries: 0.5 });

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("requestTimeoutMs: NaN → falls back to default (30000)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: mockAction() }));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      requestTimeoutMs: NaN,
      maxRetries: 0,
    });

    await client.getAction("act-1");

    const init = fetchFn.mock.calls[0][1];
    // Signal should be set (default 30s timeout, not disabled)
    expect(init.signal).toBeDefined();
  });

  it("requestTimeoutMs: Infinity → falls back to default (30000)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: mockAction() }));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      requestTimeoutMs: Infinity,
      maxRetries: 0,
    });

    await client.getAction("act-1");

    const init = fetchFn.mock.calls[0][1];
    // Signal should be set (default 30s timeout, not crashing on Infinity)
    expect(init.signal).toBeDefined();
  });

  it("requestTimeoutMs: 0 → disables timeout (no signal)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: mockAction() }));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      requestTimeoutMs: 0,
      maxRetries: 0,
    });

    await client.getAction("act-1");

    const init = fetchFn.mock.calls[0][1];
    expect(init.signal).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// onRetry callback
// ---------------------------------------------------------------------------

describe("onRetry callback", () => {
  it("calls onRetry before each retry with correct info", async () => {
    const onRetry = vi.fn();
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 502))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 2,
      onRetry,
    });
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");

    expect(onRetry).toHaveBeenCalledTimes(2);

    const first = onRetry.mock.calls[0][0];
    expect(first.attempt).toBe(0);
    expect(first.method).toBe("GET");
    expect(first.path).toBe("/api/actions/act-1");
    expect(first.error).toBeInstanceOf(NullSpendError);
    expect(first.delayMs).toBeGreaterThanOrEqual(1);

    const second = onRetry.mock.calls[1][0];
    expect(second.attempt).toBe(1);
  });

  it("onRetry returning false aborts retry", async () => {
    const onRetry = vi.fn().mockReturnValue(false);
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 2,
      onRetry,
    });

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1); // no retry happened
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("onRetry returning undefined (void) does not abort", async () => {
    const onRetry = vi.fn(); // returns undefined
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 2,
      onRetry,
    });
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("onRetry not called when maxRetries: 0", async () => {
    const onRetry = vi.fn();
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      onRetry,
    });

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("onRetry that throws propagates the thrown error (not NullSpendError)", async () => {
    const onRetry = vi.fn().mockImplementation(() => {
      throw new Error("callback exploded");
    });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 2,
      onRetry,
    });

    const err = await client.getAction("act-1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toBe("callback exploded");
    // Not a NullSpendError — user callback errors propagate as-is
    expect(err).not.toBeInstanceOf(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(1); // no retry after callback threw
  });
});

// ---------------------------------------------------------------------------
// maxRetryTimeMs (total retry wall-time cap)
// ---------------------------------------------------------------------------

describe("maxRetryTimeMs", () => {
  it("aborts retry when total wall-time cap is exceeded", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 5,
      maxRetryTimeMs: 1, // 1ms cap — will expire before retry
    });
    // Don't mock sleep — let the 1ms cap trigger naturally
    vi.spyOn(client as any, "sleep").mockImplementation(async () => {
      // Simulate enough time passing to exceed the 1ms cap
      await new Promise((r) => setTimeout(r, 5));
    });

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    // Should have stopped before exhausting all 5 retries
    expect(fetchFn.mock.calls.length).toBeLessThanOrEqual(3);
  });

  it("aborts retry on network errors when wall-time cap exceeded", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new TypeError("network down"));

    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 5,
      maxRetryTimeMs: 1,
    });
    vi.spyOn(client as any, "sleep").mockImplementation(async () => {
      await new Promise((r) => setTimeout(r, 5));
    });

    const err = await client.getAction("act-1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).message).toContain("network error");
    expect(fetchFn.mock.calls.length).toBeLessThanOrEqual(3);
  });

  it("maxRetryTimeMs: 0 means no cap (default)", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = createRetryClient(fetchFn); // no maxRetryTimeMs
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(3); // all retries exhausted
  });
});

// ---------------------------------------------------------------------------
// Retry delay progression
// ---------------------------------------------------------------------------

describe("Retry delay progression", () => {
  it("delay increases across consecutive retries", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 3,
      retryBaseDelayMs: 100,
    });
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);

    // With random=0.5, base=100: delay = floor(0.5 * min(100 * 2^attempt, 5000))
    // attempt 0: floor(0.5 * 100) = 50
    // attempt 1: floor(0.5 * 200) = 100
    // attempt 2: floor(0.5 * 400) = 200
    expect(sleepSpy).toHaveBeenCalledTimes(3);
    const delays = sleepSpy.mock.calls.map((c: any[]) => c[0] as number);
    expect(delays[0]).toBe(50);
    expect(delays[1]).toBe(100);
    expect(delays[2]).toBe(200);
    // Verify strictly increasing
    expect(delays[1]).toBeGreaterThan(delays[0]);
    expect(delays[2]).toBeGreaterThan(delays[1]);
  });
});

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

describe("Idempotency", () => {
  it("POST (createAction) includes Idempotency-Key matching /^ns_/", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: { id: "act-1", status: "pending" } }, 201));
    const client = createClient(fetchFn);

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toMatch(/^ns_[0-9a-f-]+$/);
  });

  it("GET (getAction) does NOT include Idempotency-Key", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: mockAction() }));
    const client = createClient(fetchFn);

    await client.getAction("act-1");

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toBeUndefined();
  });

  it("same key reused across retries (500 then 200 → both calls have same key)", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "pending" } }, 201));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });

    const key1 = fetchFn.mock.calls[0][1].headers["Idempotency-Key"];
    const key2 = fetchFn.mock.calls[1][1].headers["Idempotency-Key"];
    expect(key1).toBeTruthy();
    expect(key1).toBe(key2);
  });

  it("different createAction calls get different keys", async () => {
    const fetchFn = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "pending" } }, 201))
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-2", status: "pending" } }, 201));
    const client = createClient(fetchFn);

    await client.createAction({ agentId: "a", actionType: "send_email", payload: {} });
    await client.createAction({ agentId: "b", actionType: "send_email", payload: {} });

    const key1 = fetchFn.mock.calls[0][1].headers["Idempotency-Key"];
    const key2 = fetchFn.mock.calls[1][1].headers["Idempotency-Key"];
    expect(key1).not.toBe(key2);
  });

  it("markResult includes Idempotency-Key", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ data: { id: "act-1", status: "executed" } }));
    const client = createClient(fetchFn);

    await client.markResult("act-1", { status: "executed" });

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toMatch(/^ns_[0-9a-f-]+$/);
  });
});

// ---------------------------------------------------------------------------
// Integration (retry + higher-level methods)
// ---------------------------------------------------------------------------

describe("Integration", () => {
  it("proposeAndWait retries transient createAction failure (500 then success)", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction: 500 first, then success
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult (executing)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult (executed)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executed" } }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "send_email",
      payload: {},
      execute: async () => "done",
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toBe("done");
    expect(fetchFn).toHaveBeenCalledTimes(5);
  });

  it("waitForDecision retries transient getAction failure during polling (502 then approved)", async () => {
    const pending = mockAction({ status: "pending" });
    const approved = mockAction({ status: "approved" });

    const fetchFn = vi
      .fn()
      // poll 1: pending
      .mockResolvedValueOnce(jsonResponse({ data: pending }))
      // poll 2: 502 then approved on retry
      .mockResolvedValueOnce(jsonResponse({ error: { code: "bad_gateway", message: "bad gateway", details: null } }, 502))
      .mockResolvedValueOnce(jsonResponse({ data: approved }));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.waitForDecision("act-1", {
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result.status).toBe("approved");
  });

  it("default maxRetries=2 → persistent 500 calls fetch 3 times", async () => {
    const fetchFn = vi.fn().mockImplementation(jsonResponseFactory({ error: { code: "fail", message: "fail", details: null } }, 500));
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
    });
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await expect(client.getAction("act-1")).rejects.toThrow(NullSpendError);
    expect(fetchFn).toHaveBeenCalledTimes(3); // 1 initial + 2 retries
  });
});

// ---------------------------------------------------------------------------
// proposeAndWait 409 resilience
// ---------------------------------------------------------------------------

describe("proposeAndWait 409 resilience", () => {
  it("markResult(executing) gets 409, getAction shows 'executing' → execute callback runs", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → 409
      .mockResolvedValueOnce(jsonResponse({ error: { code: "already_executing", message: "already executing", details: null } }, 409))
      // getAction fallback → executing
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ id: "act-1", status: "executing" }) }))
      // markResult(executed)
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executed" } }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue({ done: true });

    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "send_email",
      payload: {},
      execute,
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toEqual({ done: true });
    expect(execute).toHaveBeenCalledOnce();
  });

  it("markResult(executed) gets 409, getAction shows 'executed' → returns successfully", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → ok
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult(executed) → 409
      .mockResolvedValueOnce(jsonResponse({ error: { code: "already_executed", message: "already executed", details: null } }, 409))
      // getAction fallback → executed
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ id: "act-1", status: "executed" }) }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue("result-value");

    const result = await client.proposeAndWait({
      agentId: "agent-1",
      actionType: "send_email",
      payload: {},
      execute,
      pollIntervalMs: 10,
      timeoutMs: 5000,
    });

    expect(result).toBe("result-value");
  });

  it("markResult(executing) gets 409, getAction shows 'approved' → throws (genuine conflict)", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → 409
      .mockResolvedValueOnce(jsonResponse({ error: { code: "conflict", message: "conflict", details: null } }, 409))
      // getAction fallback → approved (not executing!)
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ id: "act-1", status: "approved" }) }));

    const client = createClient(fetchFn);
    const execute = vi.fn();

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow(NullSpendError);

    expect(execute).not.toHaveBeenCalled();
  });

  it("markResult(executed) genuine 409 does NOT trigger markResult(failed)", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → ok
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult(executed) → 409
      .mockResolvedValueOnce(jsonResponse({ error: { code: "conflict", message: "conflict", details: null } }, 409))
      // getAction fallback → executing (not executed!)
      .mockResolvedValueOnce(jsonResponse({ data: mockAction({ id: "act-1", status: "executing" }) }));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue("result");

    await expect(
      client.proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      }),
    ).rejects.toThrow(NullSpendError);

    // execute ran, but only 5 fetch calls — no 6th markResult(failed) attempt
    expect(execute).toHaveBeenCalledOnce();
    expect(fetchFn).toHaveBeenCalledTimes(5);
  });

  it("getAction failure during 409 fallback propagates without markResult(failed)", async () => {
    const createResp: CreateActionResponse = { id: "act-1", status: "pending" };
    const approved = mockAction({ id: "act-1", status: "approved" });

    const fetchFn = vi
      .fn()
      // createAction
      .mockResolvedValueOnce(jsonResponse({ data: createResp }, 201))
      // waitForDecision
      .mockResolvedValueOnce(jsonResponse({ data: approved }))
      // markResult(executing) → ok
      .mockResolvedValueOnce(jsonResponse({ data: { id: "act-1", status: "executing" } }))
      // markResult(executed) → 409
      .mockResolvedValueOnce(jsonResponse({ error: { code: "conflict", message: "conflict", details: null } }, 409))
      // getAction fallback → network error
      .mockRejectedValueOnce(new TypeError("network down"));

    const client = createClient(fetchFn);
    const execute = vi.fn().mockResolvedValue("result");

    const err = await client
      .proposeAndWait({
        agentId: "agent-1",
        actionType: "send_email",
        payload: {},
        execute,
        pollIntervalMs: 10,
        timeoutMs: 5000,
      })
      .catch((e: unknown) => e);

    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).message).toContain("network error");
    // 5 fetch calls — no markResult(failed) attempt
    expect(fetchFn).toHaveBeenCalledTimes(5);
  });
});

// ---------------------------------------------------------------------------
// Retry-After on 503
// ---------------------------------------------------------------------------

describe("Retry-After on 503", () => {
  it("503 with Retry-After: 2 → sleep called with ~2000ms", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "maintenance", message: "maintenance", details: null } }, 503, { "Retry-After": "2" }))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    expect(sleepSpy).toHaveBeenCalledWith(2000);
  });

  it("503 without Retry-After → exponential backoff", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "maintenance", message: "maintenance", details: null } }, 503))
      .mockResolvedValueOnce(jsonResponse({ data: mockAction() }));

    const client = createRetryClient(fetchFn);
    const sleepSpy = vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    await client.getAction("act-1");
    // attempt 0, base 500: floor(0.5 * min(500 * 1, 5000)) = 250
    expect(sleepSpy).toHaveBeenCalledWith(250);
  });
});

// ---------------------------------------------------------------------------
// Non-JSON error body
// ---------------------------------------------------------------------------

describe("Non-JSON error body", () => {
  it("500 with plain text body on final attempt → error message falls back to statusText", async () => {
    const response = new Response("Internal Server Error", {
      status: 500,
      statusText: "Internal Server Error",
      headers: { "Content-Type": "text/plain" },
    });

    const fetchFn = vi.fn().mockResolvedValue(response);
    const client = createClient(fetchFn); // maxRetries: 0

    const err = await client.getAction("act-1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).message).toContain("Internal Server Error");
    expect((err as NullSpendError).statusCode).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// Cost reporting (Phase 2C)
// ---------------------------------------------------------------------------

describe("reportCost", () => {
  it("sends POST to /api/cost-events with correct body", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "ce-1", createdAt: "2026-03-18T00:00:00Z" } }, 201),
    );
    const client = createClient(fetchFn);

    const result = await client.reportCost({
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      costMicrodollars: 1500,
    });

    expect(result).toEqual({
      id: "ce-1",
      createdAt: "2026-03-18T00:00:00Z",
    });

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/cost-events");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      costMicrodollars: 1500,
    });
  });

  it("includes Idempotency-Key header", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "ce-1", createdAt: "2026-03-18T00:00:00Z" } }, 201),
    );
    const client = createClient(fetchFn);

    await client.reportCost({
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      costMicrodollars: 1500,
    });

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toMatch(/^ns_/);
  });

  it("includes optional fields when provided", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "ce-2", createdAt: "2026-03-18T00:00:00Z" } }, 201),
    );
    const client = createClient(fetchFn);

    await client.reportCost({
      provider: "anthropic",
      model: "claude-sonnet-4-5-20250514",
      inputTokens: 200,
      outputTokens: 100,
      cachedInputTokens: 50,
      reasoningTokens: 30,
      costMicrodollars: 3000,
      durationMs: 1200,
      sessionId: "sess-1",
      eventType: "llm",
    });

    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.cachedInputTokens).toBe(50);
    expect(body.reasoningTokens).toBe(30);
    expect(body.durationMs).toBe(1200);
    expect(body.sessionId).toBe("sess-1");
    expect(body.eventType).toBe("llm");
  });

  it("retries on 429/5xx", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "rate_limit", message: "rate limit", details: null } }, 429, { "Retry-After": "1" }))
      .mockResolvedValueOnce(
        jsonResponse({ data: { id: "ce-3", createdAt: "2026-03-18T00:00:00Z" } }, 201),
      );

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.reportCost({
      provider: "openai",
      model: "gpt-4o",
      inputTokens: 100,
      outputTokens: 50,
      costMicrodollars: 1500,
    });

    expect(result.id).toBe("ce-3");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("throws NullSpendError on 400", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "validation_error", message: "invalid", details: null } }, 400),
    );
    const client = createClient(fetchFn);

    await expect(
      client.reportCost({
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 100,
        outputTokens: 50,
        costMicrodollars: 1500,
      }),
    ).rejects.toThrow(NullSpendError);
  });

  it("throws NullSpendError on 401", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "authentication_required", message: "authentication_required", details: null } }, 401),
    );
    const client = createClient(fetchFn);

    const err = await client
      .reportCost({
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 100,
        outputTokens: 50,
        costMicrodollars: 1500,
      })
      .catch((e: unknown) => e);

    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).statusCode).toBe(401);
  });
});

describe("reportCostBatch", () => {
  it("sends POST to /api/cost-events/batch with events array", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 2, ids: ["ce-1", "ce-2"] }, 201),
    );
    const client = createClient(fetchFn);

    const events = [
      {
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 100,
        outputTokens: 50,
        costMicrodollars: 1500,
      },
      {
        provider: "anthropic",
        model: "claude-sonnet-4-5-20250514",
        inputTokens: 200,
        outputTokens: 100,
        costMicrodollars: 3000,
      },
    ];

    const result = await client.reportCostBatch(events);

    expect(result).toEqual({ inserted: 2, ids: ["ce-1", "ce-2"] });

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/cost-events/batch");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ events });
  });

  it("includes Idempotency-Key header", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 1, ids: ["ce-1"] }, 201),
    );
    const client = createClient(fetchFn);

    await client.reportCostBatch([
      {
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 100,
        outputTokens: 50,
        costMicrodollars: 1500,
      },
    ]);

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toMatch(/^ns_/);
  });

  it("retries on 5xx", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "fail", message: "fail", details: null } }, 500))
      .mockResolvedValueOnce(
        jsonResponse({ inserted: 1, ids: ["ce-1"] }, 201),
      );

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.reportCostBatch([
      {
        provider: "openai",
        model: "gpt-4o",
        inputTokens: 100,
        outputTokens: 50,
        costMicrodollars: 1500,
      },
    ]);

    expect(result.inserted).toBe(1);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });
});

// ---------------------------------------------------------------------------
// checkBudget (Phase 2E)
// ---------------------------------------------------------------------------

describe("checkBudget", () => {
  const budgetResponse = {
    entities: [
      {
        entityType: "user",
        entityId: "user-1",
        limitMicrodollars: 10_000_000,
        spendMicrodollars: 3_000_000,
        remainingMicrodollars: 7_000_000,
        policy: "strict_block",
        resetInterval: "monthly",
        currentPeriodStart: "2026-03-01T00:00:00.000Z",
      },
    ],
  };

  it("sends GET to /api/budgets/status", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(budgetResponse));
    const client = createClient(fetchFn);

    await client.checkBudget();

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/budgets/status");
    expect(init.method).toBe("GET");
  });

  it("parses response correctly (entities with all fields)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(budgetResponse));
    const client = createClient(fetchFn);

    const result = await client.checkBudget();

    expect(result.entities).toHaveLength(1);
    expect(result.entities[0]).toEqual({
      entityType: "user",
      entityId: "user-1",
      limitMicrodollars: 10_000_000,
      spendMicrodollars: 3_000_000,
      remainingMicrodollars: 7_000_000,
      policy: "strict_block",
      resetInterval: "monthly",
      currentPeriodStart: "2026-03-01T00:00:00.000Z",
    });
  });

  it("does not send Idempotency-Key header (GET method)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(budgetResponse));
    const client = createClient(fetchFn);

    await client.checkBudget();

    const headers = fetchFn.mock.calls[0][1].headers;
    expect(headers["Idempotency-Key"]).toBeUndefined();
  });

  it("returns empty entities when no budgets", async () => {
    const emptyResponse = {
      entities: [],
    };
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(emptyResponse));
    const client = createClient(fetchFn);

    const result = await client.checkBudget();

    expect(result.entities).toEqual([]);
  });

  it("retries on 429/5xx", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "rate_limit", message: "rate limit", details: null } }, 429, { "Retry-After": "1" }))
      .mockResolvedValueOnce(jsonResponse(budgetResponse));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.checkBudget();

    expect(result.entities).toHaveLength(1);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("retries on network error", async () => {
    const fetchFn = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(jsonResponse(budgetResponse));

    const client = createRetryClient(fetchFn);
    vi.spyOn(client as any, "sleep").mockResolvedValue(undefined);

    const result = await client.checkBudget();

    expect(result.entities).toHaveLength(1);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("throws NullSpendError on 401", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "authentication_required", message: "authentication_required", details: null } }, 401),
    );
    const client = createClient(fetchFn);

    const err = await client.checkBudget().catch((e: unknown) => e);

    expect(err).toBeInstanceOf(NullSpendError);
    expect((err as NullSpendError).statusCode).toBe(401);
  });
});

// ---------------------------------------------------------------------------
// Client-side batching (Phase 2F)
// ---------------------------------------------------------------------------

function makeCostEvent(id = 1) {
  return {
    provider: "openai",
    model: "gpt-4o",
    inputTokens: 100 * id,
    outputTokens: 50 * id,
    costMicrodollars: 500 * id,
  };
}

describe("client-side batching", () => {
  it("queueCost() queues locally, flush() sends batch to /api/cost-events/batch", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 2, ids: ["id-1", "id-2"] }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: { batchSize: 100, flushIntervalMs: 60000 },
    });

    client.queueCost(makeCostEvent(1));
    client.queueCost(makeCostEvent(2));

    expect(fetchFn).not.toHaveBeenCalled();

    await client.flush();

    expect(fetchFn).toHaveBeenCalledOnce();
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/cost-events/batch");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.events).toHaveLength(2);

    await client.shutdown();
  });

  it("queueCost() throws when batching not configured", () => {
    const fetchFn = vi.fn();
    const client = createClient(fetchFn);

    expect(() => client.queueCost(makeCostEvent())).toThrow(
      "queueCost() requires costReporting to be configured",
    );
  });

  it("queueCost() throws after shutdown()", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 0, ids: [] }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: { batchSize: 100, flushIntervalMs: 60000 },
    });

    await client.shutdown();

    let thrown: unknown;
    try {
      client.queueCost(makeCostEvent());
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(NullSpendError);
    expect((thrown as Error).message).toBe("CostReporter is shut down");
  });

  it("reportCost() still works normally with batching configured", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "ce-1", createdAt: "2026-01-01T00:00:00Z" } }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: { batchSize: 10 },
    });

    const result = await client.reportCost(makeCostEvent());
    expect(result.id).toBe("ce-1");

    const [url] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/cost-events");

    await client.shutdown();
  });

  it("reportCost() still returns ReportCostResponse with batching configured", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ data: { id: "ce-2", createdAt: "2026-03-18T00:00:00Z" } }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: {},
    });

    const result = await client.reportCost(makeCostEvent());
    expect(result).toEqual({ id: "ce-2", createdAt: "2026-03-18T00:00:00Z" });

    await client.shutdown();
  });

  it("flush() / shutdown() are no-ops when batching not configured", async () => {
    const fetchFn = vi.fn();
    const client = createClient(fetchFn);

    // Should not throw or call fetch
    await client.flush();
    await client.shutdown();
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("reportCostBatch() is not affected by batching config (bypasses CostReporter)", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 1, ids: ["id-1"] }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: { batchSize: 100 },
    });

    const result = await client.reportCostBatch([makeCostEvent()]);
    expect(result.inserted).toBe(1);

    const [url] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/cost-events/batch");

    await client.shutdown();
  });

  it("costReporting: {} (empty config) enables batching with all defaults", async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ inserted: 1, ids: ["id-1"] }),
    );
    const client = new NullSpend({
      baseUrl: "http://localhost:3000",
      apiKey: "ns_live_sk_test0001",
      fetch: fetchFn,
      maxRetries: 0,
      costReporting: {},
    });

    // Should not throw — batching is enabled
    client.queueCost(makeCostEvent());
    await client.flush();

    expect(fetchFn).toHaveBeenCalledOnce();

    await client.shutdown();
  });
});
