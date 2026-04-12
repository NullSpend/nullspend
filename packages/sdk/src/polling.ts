import type { NullSpend } from "./client.js";
import type { ActionRecord } from "./types.js";
import { TimeoutError } from "./errors.js";

const DEFAULT_POLL_INTERVAL_MS = 3_000;

/**
 * Poll an action until it leaves "pending" status, respecting both
 * a deadline and an external abort signal.
 *
 * Shared by `@nullspend/mcp-server` and `@nullspend/mcp-proxy`.
 */
export async function waitWithAbort(
  sdk: NullSpend,
  actionId: string,
  timeoutMs: number,
  signal: AbortSignal,
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
): Promise<ActionRecord> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (signal.aborted) {
      throw new Error("Aborted");
    }

    const action = await sdk.getAction(actionId);
    if (action.status !== "pending") {
      return action;
    }

    const remaining = deadline - Date.now();
    if (remaining <= 0) break;

    await interruptibleSleep(Math.min(pollIntervalMs, remaining), signal);
  }

  throw new TimeoutError(actionId, timeoutMs);
}

/**
 * Sleep that resolves early when the given abort signal fires.
 */
export function interruptibleSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }

    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);

    function onAbort() {
      clearTimeout(timer);
      resolve();
    }

    signal.addEventListener("abort", onAbort, { once: true });
  });
}
