import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { interruptibleSleep, waitWithAbort } from "./polling.js";
import { TimeoutError } from "./errors.js";
import type { NullSpend } from "./client.js";
import type { ActionRecord } from "./types.js";

// ---------------------------------------------------------------------------
// interruptibleSleep
// ---------------------------------------------------------------------------

describe("interruptibleSleep", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves immediately when signal is already aborted", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    const start = performance.now();
    await interruptibleSleep(10_000, ctrl.signal);
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(50);
  });

  it("resolves after the timer when signal never fires", async () => {
    const ctrl = new AbortController();
    const promise = interruptibleSleep(1_000, ctrl.signal);
    let resolved = false;
    promise.then(() => { resolved = true; });

    await vi.advanceTimersByTimeAsync(500);
    expect(resolved).toBe(false);
    await vi.advanceTimersByTimeAsync(600);
    expect(resolved).toBe(true);
  });

  it("resolves early when signal fires mid-sleep", async () => {
    const ctrl = new AbortController();
    const promise = interruptibleSleep(10_000, ctrl.signal);
    let resolved = false;
    promise.then(() => { resolved = true; });

    await vi.advanceTimersByTimeAsync(100);
    expect(resolved).toBe(false);
    ctrl.abort();
    await Promise.resolve();
    expect(resolved).toBe(true);
  });

  it("uses { once: true } so the listener auto-removes after one fire", async () => {
    const ctrl = new AbortController();
    const addSpy = vi.spyOn(ctrl.signal, "addEventListener");
    const promise = interruptibleSleep(1_000, ctrl.signal);
    promise.catch(() => {});
    expect(addSpy).toHaveBeenCalledTimes(1);
    expect(addSpy.mock.calls[0][2]).toEqual({ once: true });
    ctrl.abort();
    await promise;
  });

  it("removes the timer when signal aborts", async () => {
    const ctrl = new AbortController();
    const clearSpy = vi.spyOn(globalThis, "clearTimeout");
    const promise = interruptibleSleep(1_000, ctrl.signal);
    ctrl.abort();
    await promise;
    expect(clearSpy).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// waitWithAbort
// ---------------------------------------------------------------------------

function makeSdk(actions: ActionRecord[]): NullSpend {
  let i = 0;
  return {
    getAction: vi.fn().mockImplementation(async () => actions[Math.min(i++, actions.length - 1)]),
  } as unknown as NullSpend;
}

function pending(id = "act_1"): ActionRecord {
  return { id, status: "pending" } as ActionRecord;
}

function approved(id = "act_1"): ActionRecord {
  return { id, status: "approved" } as ActionRecord;
}

describe("waitWithAbort", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns the action immediately when first poll is non-pending", async () => {
    const sdk = makeSdk([approved()]);
    const ctrl = new AbortController();
    const promise = waitWithAbort(sdk, "act_1", 10_000, ctrl.signal, 100);
    await vi.runAllTimersAsync();
    const action = await promise;
    expect(action.status).toBe("approved");
  });

  it("throws TimeoutError when deadline exceeded with no transition", async () => {
    const sdk = makeSdk([pending(), pending(), pending()]);
    const ctrl = new AbortController();
    const promise = waitWithAbort(sdk, "act_1", 250, ctrl.signal, 100);
    promise.catch(() => {}); // suppress unhandled rejection during fake-timer advance
    await vi.advanceTimersByTimeAsync(500);
    await expect(promise).rejects.toBeInstanceOf(TimeoutError);
  });

  it("throws Aborted when signal fires before transition", async () => {
    const sdk = makeSdk([pending(), pending()]);
    const ctrl = new AbortController();
    const promise = waitWithAbort(sdk, "act_1", 10_000, ctrl.signal, 100);
    promise.catch(() => {});
    await vi.advanceTimersByTimeAsync(50);
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(200);
    await expect(promise).rejects.toThrow(/Aborted/);
  });

  it("does not sleep past the deadline (zero-ms remaining short-circuit)", async () => {
    const sdk = makeSdk([pending()]);
    const ctrl = new AbortController();
    // Deadline shorter than the poll interval — first iteration runs, then the
    // remaining-ms check breaks out without scheduling another sleep.
    const promise = waitWithAbort(sdk, "act_1", 10, ctrl.signal, 5_000);
    promise.catch(() => {});
    await vi.advanceTimersByTimeAsync(20);
    await expect(promise).rejects.toBeInstanceOf(TimeoutError);
  });
});
