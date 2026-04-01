import type { ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";

const LOG_PREFIX = "[nullspend-proxy]";

function log(message: string): void {
  process.stderr.write(`${LOG_PREFIX} ${message}\n`);
}

// ---------------------------------------------------------------------------
// Cost Estimation
// ---------------------------------------------------------------------------

const TIER_FREE = 0;
const TIER_READ = 10_000; // $0.01
const TIER_WRITE = 100_000; // $0.10

/**
 * Suggest a cost based on MCP annotations — for UI display only, never auto-applied.
 * Used by discovery to populate suggested_cost in the dashboard.
 */
export function suggestToolCost(
  annotations: ToolAnnotations | undefined,
): number {
  if (!annotations) return TIER_READ;

  if (annotations.readOnlyHint && annotations.openWorldHint === false) {
    return TIER_FREE;
  }

  if (annotations.destructiveHint && annotations.openWorldHint) {
    return TIER_WRITE;
  }

  return TIER_READ;
}

// ---------------------------------------------------------------------------
// MCP Cost Event
// ---------------------------------------------------------------------------

export interface McpCostEvent {
  toolName: string;
  serverName: string;
  durationMs: number;
  costMicrodollars: number;
  status: string;
  reservationId?: string;
  actionId?: string;
}

// ---------------------------------------------------------------------------
// EventBatcher
// ---------------------------------------------------------------------------

const DEFAULT_BATCH_SIZE = 20;
const DEFAULT_FLUSH_INTERVAL_MS = 5_000;
const MAX_QUEUE_SIZE = 4096;

export class EventBatcher {
  private queue: McpCostEvent[] = [];
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  private inflight: Promise<void> | null = null;
  private isShuttingDown = false;
  private readonly batchSize: number;
  private readonly backendUrl: string;
  private readonly headers: Record<string, string>;

  constructor(opts: {
    apiKey: string;
    backendUrl: string;
    batchSize?: number;
    flushIntervalMs?: number;
  }) {
    this.backendUrl = opts.backendUrl;
    this.batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;

    this.headers = {
      "Content-Type": "application/json",
      "x-nullspend-key": opts.apiKey,
    };

    const interval = opts.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;
    this.flushTimer = setInterval(() => this.flush(), interval);
    if (this.flushTimer.unref) {
      this.flushTimer.unref();
    }
  }

  push(event: McpCostEvent): void {
    if (this.isShuttingDown) return; // Don't accept events during shutdown
    if (this.queue.length >= MAX_QUEUE_SIZE) {
      this.queue.shift(); // drop oldest
    }
    this.queue.push(event);

    if (this.queue.length >= this.batchSize) {
      this.flush();
    }
  }

  flush(): void {
    if (this.queue.length === 0) return;

    const batch = this.queue.splice(0, this.batchSize);
    this.inflight = this.sendBatch(batch);
  }

  private async sendBatch(batch: McpCostEvent[], isRetry = false): Promise<void> {
    try {
      const url = `${this.backendUrl}/v1/mcp/events`;
      const resp = await fetch(url, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ events: batch }),
        signal: AbortSignal.timeout(5_000),
      });
      if (!resp.ok) {
        log(`Event batch rejected: ${resp.status}`);
        if (!isRetry) this.requeue(batch);
      }
    } catch (err) {
      log(`Event batch failed: ${err instanceof Error ? err.message : String(err)}`);
      if (!isRetry) this.requeue(batch);
    }
  }

  private requeue(batch: McpCostEvent[]): void {
    // Re-queue at the front for retry on next flush cycle.
    // Respect MAX_QUEUE_SIZE — drop oldest new events if needed.
    const space = MAX_QUEUE_SIZE - this.queue.length;
    if (space > 0) {
      this.queue.unshift(...batch.slice(0, space));
    }
  }

  async shutdown(): Promise<void> {
    this.isShuttingDown = true;
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }

    // Flush remaining events (isRetry=true: no re-queue during shutdown)
    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.batchSize);
      await this.sendBatch(batch, true);
    }

    // Wait for any in-flight request
    if (this.inflight) {
      await this.inflight;
    }
  }
}

// ---------------------------------------------------------------------------
// BudgetClient
// ---------------------------------------------------------------------------

interface BudgetCheckResponse {
  allowed: boolean;
  reservationId?: string;
  denied?: boolean;
  remaining?: number;
}

const CIRCUIT_FAILURE_THRESHOLD = 5;
const CIRCUIT_COOLDOWN_MS = 30_000;
const CACHE_TTL_MS = 60_000;
const REQUEST_TIMEOUT_MS = 2_000;

export class BudgetClient {
  private readonly backendUrl: string;
  private readonly headers: Record<string, string>;
  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;
  private lastGoodResponse: BudgetCheckResponse | null = null;
  private lastGoodAt = 0;

  constructor(opts: {
    apiKey: string;
    backendUrl: string;
  }) {
    this.backendUrl = opts.backendUrl;

    this.headers = {
      "Content-Type": "application/json",
      "x-nullspend-key": opts.apiKey,
    };
  }

  async check(
    toolName: string,
    serverName: string,
    estimateMicrodollars: number,
  ): Promise<BudgetCheckResponse> {
    const now = Date.now();

    // Circuit breaker: open
    if (now < this.circuitOpenUntil) {
      return this.fallback();
    }

    try {
      const url = `${this.backendUrl}/v1/mcp/budget/check`;
      const resp = await fetch(url, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ toolName, serverName, estimateMicrodollars }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });

      if (!resp.ok) {
        throw new Error(`Budget check returned ${resp.status}`);
      }

      const data = (await resp.json()) as BudgetCheckResponse;
      this.consecutiveFailures = 0;
      // Only cache allowed responses — caching a denial would turn
      // fail-open fallback into fail-closed if the circuit opens later
      if (data.allowed) {
        this.lastGoodResponse = data;
        this.lastGoodAt = Date.now();
      }
      return data;
    } catch (err) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= CIRCUIT_FAILURE_THRESHOLD) {
        this.circuitOpenUntil = Date.now() + CIRCUIT_COOLDOWN_MS;
        log(
          `Budget circuit breaker OPEN after ${this.consecutiveFailures} failures (cooldown ${CIRCUIT_COOLDOWN_MS}ms)`,
        );
      }
      log(
        `Budget check failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      return this.fallback();
    }
  }

  private fallback(): BudgetCheckResponse {
    // Use cached response if fresh enough
    if (
      this.lastGoodResponse &&
      Date.now() - this.lastGoodAt < CACHE_TTL_MS
    ) {
      return this.lastGoodResponse;
    }

    // Fail-open: allow the call
    return { allowed: true };
  }
}

// ---------------------------------------------------------------------------
// ToolCostRegistry
// ---------------------------------------------------------------------------

const STARTUP_TIMEOUT_MS = 5_000;

export interface DiscoverToolPayload {
  name: string;
  description?: string | null;
  annotations?: Record<string, unknown> | null;
  tierCost: number;
  suggestedCost: number;
}

export class ToolCostRegistry {
  private costs = new Map<string, number>();
  private readonly nullspendUrl: string;
  private readonly apiKey: string;
  private readonly serverName: string;

  constructor(opts: {
    nullspendUrl: string;
    apiKey: string;
    serverName: string;
  }) {
    this.nullspendUrl = opts.nullspendUrl;
    this.apiKey = opts.apiKey;
    this.serverName = opts.serverName;
  }

  async fetchCosts(): Promise<void> {
    try {
      const url = `${this.nullspendUrl}/api/tool-costs`;
      const resp = await fetch(url, {
        headers: { "x-nullspend-key": this.apiKey },
        signal: AbortSignal.timeout(STARTUP_TIMEOUT_MS),
      });

      if (resp.status === 401) {
        log(
          `API key not recognized (401). Create a managed key at ${this.nullspendUrl}/app/settings and use it as NULLSPEND_API_KEY.`,
        );
        return;
      }

      if (!resp.ok) {
        log(`Failed to fetch tool costs: ${resp.status}`);
        return;
      }

      const json = await resp.json();

      if (!json || !Array.isArray(json.data)) {
        log("Tool cost response missing 'data' array — using annotation tiers");
        return;
      }

      for (const row of json.data) {
        if (
          typeof row?.serverName === "string" &&
          typeof row?.toolName === "string" &&
          typeof row?.costMicrodollars === "number"
        ) {
          this.costs.set(`${row.serverName}/${row.toolName}`, row.costMicrodollars);
        }
      }

      log(`Loaded ${this.costs.size} tool cost(s) from dashboard`);
    } catch (err) {
      log(
        `Tool cost fetch failed (using annotation tiers): ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  async discoverTools(tools: DiscoverToolPayload[]): Promise<void> {
    if (tools.length === 0) return;

    // Chunk into batches of 500 (server-side limit)
    const BATCH_SIZE = 500;
    const chunks: DiscoverToolPayload[][] = [];
    for (let i = 0; i < tools.length; i += BATCH_SIZE) {
      chunks.push(tools.slice(i, i + BATCH_SIZE));
    }

    let totalRegistered = 0;

    for (const chunk of chunks) {
      try {
        const url = `${this.nullspendUrl}/api/tool-costs/discover`;
        const resp = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-nullspend-key": this.apiKey,
          },
          body: JSON.stringify({
            serverName: this.serverName,
            tools: chunk,
          }),
          signal: AbortSignal.timeout(STARTUP_TIMEOUT_MS),
        });

        if (resp.status === 401) {
          log(
            `API key not recognized (401). Create a managed key at ${this.nullspendUrl}/app/settings and use it as NULLSPEND_API_KEY.`,
          );
          return;
        }

        if (!resp.ok) {
          log(`Tool discovery registration failed: ${resp.status}`);
          return;
        }

        totalRegistered += chunk.length;
      } catch (err) {
        log(
          `Tool discovery failed (continuing with annotation tiers): ${err instanceof Error ? err.message : String(err)}`,
        );
        return;
      }
    }

    log(`Registered ${totalRegistered} tool(s) with dashboard`);
  }

  getCost(toolName: string): number | undefined {
    return this.costs.get(`${this.serverName}/${toolName}`);
  }
}

// ---------------------------------------------------------------------------
// CostTracker (facade)
// ---------------------------------------------------------------------------

export interface CostTrackerConfig {
  backendUrl: string;
  serverName: string;
  budgetEnforcementEnabled: boolean;
  toolCostOverrides: Record<string, number>;
  apiKey: string;
}

export class CostTracker {
  readonly batcher: EventBatcher;
  readonly budgetClient: BudgetClient;
  readonly config: CostTrackerConfig;
  private registry: ToolCostRegistry | null = null;

  constructor(config: CostTrackerConfig) {
    this.config = config;

    this.batcher = new EventBatcher({
      backendUrl: config.backendUrl,
      apiKey: config.apiKey,
    });
    this.budgetClient = new BudgetClient({
      backendUrl: config.backendUrl,
      apiKey: config.apiKey,
    });
  }

  setRegistry(registry: ToolCostRegistry): void {
    this.registry = registry;
  }

  resolveToolCost(toolName: string): number {
    if (toolName in this.config.toolCostOverrides) {
      return this.config.toolCostOverrides[toolName];
    }

    if (this.registry) {
      const registryCost = this.registry.getCost(toolName);
      if (registryCost !== undefined) {
        return registryCost;
      }
    }

    // Unpriced: track the call at $0 until user sets a real price
    return 0;
  }

  async checkBudget(
    toolName: string,
    estimateMicrodollars: number,
  ): Promise<BudgetCheckResponse> {
    if (!this.config.budgetEnforcementEnabled) {
      return { allowed: true };
    }
    return this.budgetClient.check(
      toolName,
      this.config.serverName,
      estimateMicrodollars,
    );
  }

  reportEvent(event: McpCostEvent): void {
    this.batcher.push(event);
  }

  async shutdown(): Promise<void> {
    await this.batcher.shutdown();
  }
}
