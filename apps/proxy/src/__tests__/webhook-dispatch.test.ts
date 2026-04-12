import { describe, it, expect, vi, beforeEach } from "vitest";

const mockQueueSend = vi.fn().mockResolvedValue(undefined);

import { createWebhookDispatcher, dispatchToEndpoints } from "../lib/webhook-dispatch.js";
import type { WebhookEndpointWithSecret } from "../lib/webhook-cache.js";
import type { WebhookEvent, ThinWebhookEvent } from "../lib/webhook-events.js";

function makeQueue(): Queue {
  return { send: mockQueueSend, sendBatch: vi.fn() } as unknown as Queue;
}

function makeEndpoint(overrides: Partial<WebhookEndpointWithSecret> = {}): WebhookEndpointWithSecret {
  return {
    id: "ep-1",
    url: "https://hooks.example.com/webhook",
    signingSecret: "whsec_secret",
    previousSigningSecret: null,
    secretRotatedAt: null,
    eventTypes: [],
    apiVersion: "2026-04-01",
    payloadMode: "full" as const,
    ...overrides,
  };
}

function makeEvent(overrides: Partial<WebhookEvent> = {}): WebhookEvent {
  return {
    id: "evt_test123",
    type: "cost_event.created",
    api_version: "2026-04-01",
    created_at: 1710547200,
    data: { object: { request_id: "req_1" } },
    ...overrides,
  };
}

describe("createWebhookDispatcher", () => {
  it("returns null when queue is undefined", () => {
    const dispatcher = createWebhookDispatcher(undefined, "user-1");
    expect(dispatcher).toBeNull();
  });

  it("returns a dispatcher when queue is provided", () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1");
    expect(dispatcher).not.toBeNull();
    expect(dispatcher!.dispatch).toBeInstanceOf(Function);
  });
});

describe("dispatcher.dispatch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("enqueues message with correct shape (ownerId, endpointId, event)", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoint = makeEndpoint();
    const event = makeEvent();

    await dispatcher.dispatch(endpoint, event);

    expect(mockQueueSend).toHaveBeenCalledWith({
      ownerId: "user-1",
      endpointId: "ep-1",
      event,
    });
  });

  it("skips dispatch when endpoint filters don't match event type", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoint = makeEndpoint({ eventTypes: ["budget.exceeded"] });
    const event = makeEvent({ type: "cost_event.created" });

    await dispatcher.dispatch(endpoint, event);
    expect(mockQueueSend).not.toHaveBeenCalled();
  });

  it("dispatches when endpoint has matching event type", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoint = makeEndpoint({ eventTypes: ["cost_event.created", "budget.exceeded"] });
    const event = makeEvent({ type: "cost_event.created" });

    await dispatcher.dispatch(endpoint, event);
    expect(mockQueueSend).toHaveBeenCalled();
  });

  it("dispatches when endpoint has empty event types (all events)", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoint = makeEndpoint({ eventTypes: [] });
    const event = makeEvent({ type: "budget.threshold.warning" });

    await dispatcher.dispatch(endpoint, event);
    expect(mockQueueSend).toHaveBeenCalled();
  });

  it("handles thin events (no data field)", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoint = makeEndpoint();
    const thinEvent: ThinWebhookEvent = {
      id: "evt_thin_123",
      type: "cost_event.created",
      api_version: "2026-04-01",
      created_at: 1710547200,
      related_object: {
        id: "req_abc",
        type: "cost_event",
        url: "/api/cost-events?requestId=req_abc&provider=openai",
      },
    };

    await dispatcher.dispatch(endpoint, thinEvent);

    const sentMessage = mockQueueSend.mock.calls[0][0];
    expect(sentMessage.event).toEqual(thinEvent);
    expect(sentMessage.event).not.toHaveProperty("data");
  });
});

describe("dispatchToEndpoints", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("dispatches to all matching endpoints", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoints = [
      makeEndpoint({ id: "ep-1" }),
      makeEndpoint({ id: "ep-2" }),
    ];
    const event = makeEvent();

    await dispatchToEndpoints(dispatcher, endpoints, event);
    expect(mockQueueSend).toHaveBeenCalledTimes(2);
  });

  it("continues dispatching after individual endpoint failure", async () => {
    mockQueueSend
      .mockRejectedValueOnce(new Error("queue send error"))
      .mockResolvedValueOnce(undefined);

    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    const endpoints = [
      makeEndpoint({ id: "ep-1" }),
      makeEndpoint({ id: "ep-2" }),
    ];

    await dispatchToEndpoints(dispatcher, endpoints, makeEvent());
    expect(mockQueueSend).toHaveBeenCalledTimes(2);
  });

  it("does not throw on dispatch errors (fail-open)", async () => {
    mockQueueSend.mockRejectedValue(new Error("all fail"));

    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;
    await expect(
      dispatchToEndpoints(dispatcher, [makeEndpoint()], makeEvent()),
    ).resolves.not.toThrow();
  });

  // PXY-13: Each endpoint should receive its own apiVersion in the event payload
  it("overrides api_version per endpoint (PXY-13)", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;

    const ep1 = makeEndpoint({ id: "ep-1", apiVersion: "2026-04-01" });
    const ep2 = makeEndpoint({ id: "ep-2", apiVersion: "2026-03-01" });
    const event = makeEvent({ api_version: "2026-01-01" }); // original version

    await dispatchToEndpoints(dispatcher, [ep1, ep2], event);

    expect(mockQueueSend).toHaveBeenCalledTimes(2);

    // Each endpoint should get its own apiVersion, not the event's original
    const call1 = mockQueueSend.mock.calls[0][0];
    const call2 = mockQueueSend.mock.calls[1][0];
    expect(call1.event.api_version).toBe("2026-04-01");
    expect(call2.event.api_version).toBe("2026-03-01");
  });

  it("preserves original api_version when endpoint has no apiVersion", async () => {
    const dispatcher = createWebhookDispatcher(makeQueue(), "user-1")!;

    const ep = makeEndpoint({ apiVersion: undefined as unknown as string });
    const event = makeEvent({ api_version: "2026-01-01" });

    await dispatchToEndpoints(dispatcher, [ep], event);

    const call = mockQueueSend.mock.calls[0][0];
    expect(call.event.api_version).toBe("2026-01-01");
  });
});
