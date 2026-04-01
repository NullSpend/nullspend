import { CostReporter } from "./cost-reporter.js";
import { NullSpendError, RejectedError, TimeoutError } from "./errors.js";
import { buildTrackedFetch } from "./tracked-fetch.js";
import { createPolicyCache } from "./policy-cache.js";
import type { PolicyResponse } from "./policy-cache.js";
import {
  isRetryableStatusCode,
  isRetryableError,
  parseRetryAfterMs,
  calculateRetryDelayMs,
  DEFAULT_MAX_RETRIES,
  DEFAULT_RETRY_BASE_DELAY_MS,
  DEFAULT_MAX_RETRY_DELAY_MS,
} from "./retry.js";
import type {
  ActionRecord,
  BudgetStatus,
  CostSummaryPeriod,
  CostSummaryResponse,
  ListBudgetsResponse,
  ListCostEventsOptions,
  ListCostEventsResponse,
  NullSpendConfig,
  CostEventInput,
  CreateActionInput,
  CreateActionResponse,
  MarkResultInput,
  MutateActionResponse,
  ProposeAndWaitOptions,
  ReportCostResponse,
  ReportCostBatchResponse,
  RetryInfo,
  TrackedFetchOptions,
  TrackedProvider,
  WaitForDecisionOptions,
} from "./types.js";

const DEFAULT_POLL_INTERVAL_MS = 2_000;
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const API_KEY_HEADER = "x-nullspend-key";
const MAX_RETRIES_CEILING = 10;
const SDK_API_VERSION = "2026-04-01";

function toFiniteInt(value: number | undefined, fallback: number): number {
  const v = value ?? fallback;
  if (!Number.isFinite(v)) return fallback;
  return Math.floor(v);
}

export class NullSpend {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly _fetch: typeof globalThis.fetch;
  private readonly requestTimeoutMs: number;
  private readonly maxRetries: number;
  private readonly retryBaseDelayMs: number;
  private readonly maxRetryTimeMs: number;
  private readonly apiVersion: string;
  private readonly onRetry: ((info: RetryInfo) => void | boolean) | undefined;
  private readonly costReporter: CostReporter | null;

  constructor(config: NullSpendConfig) {
    if (!config.baseUrl) throw new NullSpendError("baseUrl is required");
    if (!config.apiKey) throw new NullSpendError("apiKey is required");

    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.apiVersion = config.apiVersion ?? SDK_API_VERSION;
    this._fetch = config.fetch ?? globalThis.fetch.bind(globalThis);
    this.requestTimeoutMs = Math.max(
      0,
      toFiniteInt(config.requestTimeoutMs, DEFAULT_REQUEST_TIMEOUT_MS),
    );
    this.maxRetries = Math.min(
      MAX_RETRIES_CEILING,
      Math.max(0, toFiniteInt(config.maxRetries, DEFAULT_MAX_RETRIES)),
    );
    this.retryBaseDelayMs = Math.max(
      0,
      toFiniteInt(config.retryBaseDelayMs, DEFAULT_RETRY_BASE_DELAY_MS),
    );
    this.maxRetryTimeMs = Math.max(
      0,
      toFiniteInt(config.maxRetryTimeMs, 0),
    );
    this.onRetry = config.onRetry;
    this.costReporter = config.costReporting
      ? new CostReporter(config.costReporting, async (events) => {
          await this.reportCostBatch(events);
        })
      : null;
  }

  // -------------------------------------------------------------------------
  // Low-level API methods
  // -------------------------------------------------------------------------

  async createAction(input: CreateActionInput): Promise<CreateActionResponse> {
    const res = await this.request<{ data: CreateActionResponse }>("POST", "/api/actions", input);
    return res.data;
  }

  async getAction(id: string): Promise<ActionRecord> {
    const res = await this.request<{ data: ActionRecord }>("GET", `/api/actions/${id}`);
    return res.data;
  }

  async markResult(
    id: string,
    input: MarkResultInput,
  ): Promise<MutateActionResponse> {
    const res = await this.request<{ data: MutateActionResponse }>(
      "POST",
      `/api/actions/${id}/result`,
      input,
    );
    return res.data;
  }

  // -------------------------------------------------------------------------
  // Cost reporting
  // -------------------------------------------------------------------------

  async reportCost(event: CostEventInput): Promise<ReportCostResponse> {
    const res = await this.request<{ data: ReportCostResponse }>(
      "POST",
      "/api/cost-events",
      event,
    );
    return res.data;
  }

  async reportCostBatch(
    events: CostEventInput[],
  ): Promise<ReportCostBatchResponse> {
    return this.request<ReportCostBatchResponse>(
      "POST",
      "/api/cost-events/batch",
      { events },
    );
  }

  // -------------------------------------------------------------------------
  // Client-side batching
  // -------------------------------------------------------------------------

  queueCost(event: CostEventInput): void {
    if (!this.costReporter) {
      throw new NullSpendError(
        "queueCost() requires costReporting to be configured",
      );
    }
    this.costReporter.enqueue(event);
  }

  async flush(): Promise<void> {
    await this.costReporter?.flush();
  }

  async shutdown(): Promise<void> {
    await this.costReporter?.shutdown();
  }

  // -------------------------------------------------------------------------
  // Tracked fetch (provider wrappers)
  // -------------------------------------------------------------------------

  /**
   * Create a tracked fetch function for a specific LLM provider.
   * Automatically tracks cost events for OpenAI and Anthropic API calls.
   *
   * ```ts
   * const openai = new OpenAI({ fetch: ns.createTrackedFetch("openai") });
   * ```
   */
  createTrackedFetch(
    provider: TrackedProvider,
    options?: TrackedFetchOptions,
  ): typeof globalThis.fetch {
    if (!this.costReporter) {
      throw new NullSpendError(
        "createTrackedFetch() requires costReporting to be configured",
      );
    }

    let policyCache = null;
    if (options?.enforcement) {
      policyCache = createPolicyCache(async (): Promise<PolicyResponse> => {
        return this.request<PolicyResponse>("GET", "/api/policy");
      });
    }

    return buildTrackedFetch(
      provider,
      options,
      (event) => this.queueCost(event),
      policyCache,
    );
  }

  // -------------------------------------------------------------------------
  // Budget status
  // -------------------------------------------------------------------------

  async checkBudget(): Promise<BudgetStatus> {
    return this.request<BudgetStatus>("GET", "/api/budgets/status");
  }

  // -------------------------------------------------------------------------
  // Read APIs — budgets, cost events, spend summary
  // -------------------------------------------------------------------------

  /** List all budgets for the authenticated org. */
  async listBudgets(): Promise<ListBudgetsResponse> {
    return this.request<ListBudgetsResponse>("GET", "/api/budgets");
  }

  /** Get a spend summary for the given period (7d, 30d, or 90d). */
  async getCostSummary(period: CostSummaryPeriod = "30d"): Promise<CostSummaryResponse> {
    const res = await this.request<{ data: CostSummaryResponse }>("GET", `/api/cost-events/summary?period=${period}`);
    return res.data;
  }

  /** List recent cost events with optional pagination. */
  async listCostEvents(options?: ListCostEventsOptions): Promise<ListCostEventsResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.cursor) params.set("cursor", options.cursor);
    const qs = params.toString();
    return this.request<ListCostEventsResponse>("GET", `/api/cost-events${qs ? `?${qs}` : ""}`);
  }

  // -------------------------------------------------------------------------
  // Polling
  // -------------------------------------------------------------------------

  async waitForDecision(
    actionId: string,
    options?: WaitForDecisionOptions,
  ): Promise<ActionRecord> {
    const pollInterval = options?.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
    const timeout = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline = Date.now() + timeout;

    while (Date.now() < deadline) {
      const action = await this.getAction(actionId);
      options?.onPoll?.(action);

      if (action.status !== "pending") {
        return action;
      }

      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.sleep(Math.min(pollInterval, remaining));
    }

    throw new TimeoutError(actionId, timeout);
  }

  // -------------------------------------------------------------------------
  // High-level orchestrator
  // -------------------------------------------------------------------------

  async proposeAndWait<T>(options: ProposeAndWaitOptions<T>): Promise<T> {
    const { id } = await this.createAction({
      agentId: options.agentId,
      actionType: options.actionType,
      payload: options.payload,
      metadata: options.metadata,
      expiresInSeconds: options.expiresInSeconds,
    });

    const decision = await this.waitForDecision(id, {
      pollIntervalMs: options.pollIntervalMs,
      timeoutMs: options.timeoutMs,
      onPoll: options.onPoll,
    });

    if (decision.status !== "approved") {
      throw new RejectedError(id, decision.status);
    }

    // markResult(executing) with 409 resilience
    try {
      await this.markResult(id, { status: "executing" });
    } catch (err) {
      if (err instanceof NullSpendError && err.statusCode === 409) {
        const current = await this.getAction(id);
        if (current.status !== "executing") throw err;
        // Already executing — proceed (lost response on a successful write)
      } else {
        throw err;
      }
    }

    // Execute the callback — errors here trigger the failure path
    let result: T;
    try {
      result = await options.execute({ actionId: id });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      try {
        await this.markResult(id, { status: "failed", errorMessage: message });
      } catch {
        // Best-effort: don't shadow the original execute error
      }
      throw err;
    }

    // Report success — errors here do NOT trigger markResult(failed)
    // because the execute callback already succeeded
    const serializable =
      result !== null && typeof result === "object"
        ? (result as Record<string, unknown>)
        : { value: result };

    try {
      await this.markResult(id, { status: "executed", result: serializable });
    } catch (err) {
      if (err instanceof NullSpendError && err.statusCode === 409) {
        const current = await this.getAction(id);
        if (current.status !== "executed") throw err;
        // Already executed — proceed (lost response on a successful write)
      } else {
        throw err;
      }
    }

    return result;
  }

  // -------------------------------------------------------------------------
  // HTTP helper with retry + idempotency
  // -------------------------------------------------------------------------

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    const headers: Record<string, string> = {
      [API_KEY_HEADER]: this.apiKey,
      "NullSpend-Version": this.apiVersion,
      Accept: "application/json",
    };

    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }

    // Idempotency key: generated once for mutating methods, reused across retries
    if (method !== "GET") {
      headers["Idempotency-Key"] = `ns_${crypto.randomUUID()}`;
    }

    // Body serialized once before loop
    const serializedBody = body !== undefined ? JSON.stringify(body) : undefined;

    let lastError: NullSpendError | undefined;
    let retryAfterMs: number | null = null;
    const retryDeadline =
      this.maxRetryTimeMs > 0 ? Date.now() + this.maxRetryTimeMs : 0;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      if (attempt > 0) {
        // Check total retry wall-time cap
        if (retryDeadline > 0 && Date.now() >= retryDeadline) {
          throw lastError;
        }

        const delay =
          retryAfterMs ?? calculateRetryDelayMs(attempt - 1, this.retryBaseDelayMs, DEFAULT_MAX_RETRY_DELAY_MS);
        retryAfterMs = null; // use Retry-After only once

        // onRetry callback — return false to abort
        if (this.onRetry) {
          const cont = this.onRetry({
            attempt: attempt - 1,
            delayMs: delay,
            error: lastError!,
            method,
            path,
          });
          if (cont === false) throw lastError;
        }

        await this.sleep(delay);
      }

      let response: Response;
      try {
        const fetchOptions: RequestInit = {
          method,
          headers,
          body: serializedBody,
        };

        if (this.requestTimeoutMs > 0) {
          fetchOptions.signal = AbortSignal.timeout(this.requestTimeoutMs);
        }

        response = await this._fetch(url, fetchOptions);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        lastError = new NullSpendError(`${method} ${path} network error: ${msg}`);

        if (isRetryableError(err) && attempt < this.maxRetries) {
          continue;
        }
        throw lastError;
      }

      if (response.ok) {
        try {
          return (await response.json()) as T;
        } catch {
          throw new NullSpendError(
            `${method} ${path} returned invalid JSON`,
            response.status,
          );
        }
      }

      // Non-OK response — check if retryable
      if (isRetryableStatusCode(response.status) && attempt < this.maxRetries) {
        retryAfterMs = parseRetryAfterMs(
          response.headers.get("Retry-After"),
          DEFAULT_MAX_RETRY_DELAY_MS,
        );
        // Drain response body to prevent connection leak
        try { await response.text(); } catch { /* ignore */ }

        let detail: string;
        try {
          detail = response.statusText || `HTTP ${response.status}`;
        } catch {
          detail = `HTTP ${response.status}`;
        }
        lastError = new NullSpendError(
          `${method} ${path} failed: ${detail}`,
          response.status,
        );
        continue;
      }

      // Non-retryable error or final attempt
      let detail: string;
      let code: string | undefined;
      try {
        const json = (await response.json()) as Record<string, unknown>;
        const errObj = json.error;
        if (errObj && typeof errObj === "object") {
          const err = errObj as Record<string, unknown>;
          code = typeof err.code === "string" ? err.code : undefined;
          detail = String(err.message ?? err.code ?? response.statusText);
        } else {
          detail = response.statusText;
        }
      } catch {
        detail = response.statusText;
      }
      throw new NullSpendError(
        `${method} ${path} failed: ${detail}`,
        response.status,
        code,
      );
    }

    // Unreachable: every loop iteration either returns or throws.
    // TypeScript needs this for control-flow analysis.
    throw lastError;
  }

  // -------------------------------------------------------------------------
  // Utilities
  // -------------------------------------------------------------------------

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
