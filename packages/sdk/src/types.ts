export const ACTION_TYPES = [
  "send_email",
  "http_post",
  "http_delete",
  "shell_command",
  "db_write",
  "file_write",
  "file_delete",
  "budget_increase",
] as const;

export type ActionType = (typeof ACTION_TYPES)[number];

export const ACTION_STATUSES = [
  "pending",
  "approved",
  "rejected",
  "expired",
  "executing",
  "executed",
  "failed",
] as const;

export type ActionStatus = (typeof ACTION_STATUSES)[number];

export const TERMINAL_STATUSES: ReadonlySet<ActionStatus> = new Set([
  "rejected",
  "expired",
  "executed",
  "failed",
]);

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

export interface CostReportingConfig {
  /** Flush when queue reaches this size. Default: 10. Clamped [1, 100]. */
  batchSize?: number;
  /** Flush on a timer interval in ms. Default: 5000. Min 100. */
  flushIntervalMs?: number;
  /** Max queued events before dropping oldest. Default: 1000. Min 1. */
  maxQueueSize?: number;
  /** Called when events are dropped due to queue overflow. */
  onDropped?: (count: number) => void;
  /** Called when a batch flush fails after retries. */
  onFlushError?: (error: Error, events: CostEventInput[]) => void;
}

export interface NullSpendConfig {
  baseUrl: string;
  apiKey: string;
  /**
   * Proxy base URL used to detect when a tracked fetch request is going through
   * the NullSpend proxy. Detection is by URL ORIGIN match (scheme + host + port).
   * When the request URL's origin matches, the SDK skips client-side cost tracking
   * and lets the proxy handle accounting.
   *
   * **IMPORTANT — strict port comparison**: if your proxy listens on a non-default
   * port (e.g. `https://proxy.example.com:8443`), you MUST include that port in
   * `proxyUrl`. Mismatched ports cause the SDK to take the direct path and
   * double-count cost events. This strictness is intentional — port-normalization
   * would silently mask real misconfigurations.
   *
   * Optional — header-based detection (`x-nullspend-key` set on the request) still
   * works as a fallback when `proxyUrl` is unset.
   *
   * Example: `"https://nullspend.cjones6489.workers.dev"`
   */
  proxyUrl?: string;
  /** Override the API version sent via NullSpend-Version header. Defaults to the SDK's built-in version. */
  apiVersion?: string;
  /** Custom fetch implementation (defaults to globalThis.fetch). */
  fetch?: typeof globalThis.fetch;
  /** Per-request timeout in milliseconds. Default: 30000 (30s). Set to 0 to disable. */
  requestTimeoutMs?: number;
  /** Max retries on transient failures (429, 5xx, network). Default: 2. Set 0 to disable. */
  maxRetries?: number;
  /** Base delay between retries in ms. Default: 500. */
  retryBaseDelayMs?: number;
  /** Total wall-time cap for all retry attempts in ms. Default: 0 (no cap). */
  maxRetryTimeMs?: number;
  /** Called before each retry. Return false to abort retrying. */
  onRetry?: (info: RetryInfo) => void | boolean;
  /** Enable client-side cost event batching. Presence enables batching. */
  costReporting?: CostReportingConfig;
}

export interface RetryInfo {
  /** Zero-based retry attempt (0 = first retry, i.e. second overall request). */
  attempt: number;
  /** Delay in ms before this retry fires. */
  delayMs: number;
  /** The error that triggered the retry. */
  error: Error;
  /** HTTP method of the request. */
  method: string;
  /** URL path of the request. */
  path: string;
}

// ---------------------------------------------------------------------------
// API request / response shapes
// ---------------------------------------------------------------------------

export interface CreateActionInput {
  agentId: string;
  actionType: ActionType | (string & {});
  payload: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  /** Server-side TTL in seconds. Omit for server default (1 hour). Set to 0 or null to never expire. */
  expiresInSeconds?: number | null;
}

export interface CreateActionResponse {
  id: string;
  status: "pending";
  expiresAt: string | null;
}

export interface ActionRecord {
  id: string;
  agentId: string;
  actionType: string;
  status: ActionStatus;
  payload: Record<string, unknown>;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  approvedAt: string | null;
  rejectedAt: string | null;
  executedAt: string | null;
  expiresAt: string | null;
  expiredAt: string | null;
  approvedBy: string | null;
  rejectedBy: string | null;
  result: Record<string, unknown> | null;
  errorMessage: string | null;
  environment: string | null;
  sourceFramework: string | null;
}

export interface MarkResultInput {
  status: "executing" | "executed" | "failed";
  result?: Record<string, unknown>;
  errorMessage?: string;
}

export interface MutateActionResponse {
  id: string;
  status: ActionStatus;
  approvedAt?: string | null;
  rejectedAt?: string | null;
  executedAt?: string | null;
  budgetIncrease?: {
    previousLimit: number;
    newLimit: number;
    amount: number;
    requestedAmount: number;
  };
}

// ---------------------------------------------------------------------------
// proposeAndWait options
// ---------------------------------------------------------------------------

/** Context passed to the execute callback with the action ID for cost correlation. */
export interface ExecuteContext {
  actionId: string;
}

export interface ProposeAndWaitOptions<T> {
  agentId: string;
  actionType: ActionType | (string & {});
  payload: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  /** Server-side TTL in seconds. Omit for server default (1 hour). Set to 0 or null to never expire. */
  expiresInSeconds?: number | null;
  /**
   * Called after the action is approved. Receives context with `actionId` which
   * can be sent as the `x-nullspend-action-id` header when calling the proxy
   * to correlate cost events with this action.
   */
  execute: (context?: ExecuteContext) => T | Promise<T>;
  /** Milliseconds between polls. Default: 2000 */
  pollIntervalMs?: number;
  /** Total timeout in milliseconds. Default: 300000 (5 min) */
  timeoutMs?: number;
  /** Called each time the SDK polls. Useful for logging. */
  onPoll?: (action: ActionRecord) => void;
}

export interface WaitForDecisionOptions {
  pollIntervalMs?: number;
  timeoutMs?: number;
  onPoll?: (action: ActionRecord) => void;
}

// ---------------------------------------------------------------------------
// Cost reporting (Phase 2C)
// ---------------------------------------------------------------------------

export interface CostEventInput {
  provider: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
  costMicrodollars: number;
  costBreakdown?: {
    input: number;
    output: number;
    cached: number;
    reasoning?: number;
  };
  durationMs?: number;
  sessionId?: string;
  traceId?: string;
  eventType?: "llm" | "tool" | "custom";
  toolName?: string;
  toolServer?: string;
  tags?: Record<string, string>;
  /** Customer identifier for per-customer profitability tracking. */
  customer?: string;
}

// ---------------------------------------------------------------------------
// Tracked fetch (provider wrappers)
// ---------------------------------------------------------------------------

export type TrackedProvider = "openai" | "anthropic";

export interface TrackedFetchOptions {
  /** Customer identifier for per-customer profitability tracking. */
  customer?: string;
  sessionId?: string;
  tags?: Record<string, string>;
  traceId?: string;
  actionId?: string;
  /** Enable cooperative budget + mandate enforcement. */
  enforcement?: boolean;
  /** Per-session spend limit in microdollars. Takes precedence over policy-fetched limit. */
  sessionLimitMicrodollars?: number;
  /** Called when cost tracking encounters a non-fatal error. */
  onCostError?: (error: Error) => void;
  /** Called before throwing BudgetExceededError, MandateViolationError, or SessionLimitExceededError. */
  onDenied?: (reason: DenialReason) => void;
}

export type DenialReason =
  | { type: "budget"; remaining: number; entityType?: string; entityId?: string; limit?: number; spend?: number }
  | { type: "mandate"; mandate: string; requested: string; allowed: string[] }
  | { type: "session_limit"; sessionSpend: number; sessionLimit: number }
  | { type: "velocity"; retryAfterSeconds?: number; limit?: number; window?: number; current?: number }
  | { type: "tag_budget"; tagKey?: string; tagValue?: string; remaining?: number; limit?: number; spend?: number };

// ---------------------------------------------------------------------------
// Customer session
// ---------------------------------------------------------------------------

export interface CustomerSessionOptions {
  /** Customer's plan tier (attached as a tag for filtering/enforcement). */
  plan?: string;
  /** Session ID for session-level spend limits. */
  sessionId?: string;
  /** Per-session spend limit in microdollars. */
  sessionLimitMicrodollars?: number;
  /** Additional tags merged with customer context. */
  tags?: Record<string, string>;
  /** Enable cooperative budget + mandate enforcement. */
  enforcement?: boolean;
  /** Called when cost tracking encounters a non-fatal error. */
  onCostError?: (error: Error) => void;
  /** Called before throwing enforcement errors. */
  onDenied?: (reason: DenialReason) => void;
}

export interface CustomerSession {
  /** Pre-configured fetch for OpenAI SDK. Usage: `new OpenAI({ fetch: session.openai })` */
  openai: typeof globalThis.fetch;
  /** Pre-configured fetch for Anthropic SDK. Usage: `new Anthropic({ fetch: session.anthropic })` */
  anthropic: typeof globalThis.fetch;
  /** Get a tracked fetch for a specific provider. */
  fetch: (provider: TrackedProvider) => typeof globalThis.fetch;
  /** The customer ID this session is scoped to. */
  readonly customerId: string;
}

export interface ReportCostResponse {
  id: string;
  createdAt: string;
}

export interface ReportCostBatchResponse {
  inserted: number;
  ids: string[];
}

// ---------------------------------------------------------------------------
// Budget status (Phase 2E)
// ---------------------------------------------------------------------------

export interface BudgetEntity {
  entityType: string;
  entityId: string;
  limitMicrodollars: number;
  spendMicrodollars: number;
  remainingMicrodollars: number;
  policy: string;
  resetInterval: string | null;
  currentPeriodStart: string | null;
}

export interface BudgetStatus {
  entities: BudgetEntity[];
}

// ---------------------------------------------------------------------------
// Budget list (full detail)
// ---------------------------------------------------------------------------

export interface BudgetRecord {
  id: string;
  entityType: string;
  entityId: string;
  maxBudgetMicrodollars: number;
  spendMicrodollars: number;
  policy: string;
  resetInterval: string | null;
  currentPeriodStart: string | null;
  thresholdPercentages: number[];
  velocityLimitMicrodollars: number | null;
  velocityWindowSeconds: number | null;
  velocityCooldownSeconds: number | null;
  sessionLimitMicrodollars: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface ListBudgetsResponse {
  data: BudgetRecord[];
}

// ---------------------------------------------------------------------------
// Cost events (read)
// ---------------------------------------------------------------------------

export interface CostEventRecord {
  id: string;
  provider: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  costMicrodollars: number;
  durationMs: number | null;
  sessionId: string | null;
  traceId: string | null;
  eventType: string;
  toolName: string | null;
  tags: Record<string, string> | null;
  customerId: string | null;
  costBreakdown?: {
    input?: number;
    output?: number;
    cached?: number;
    reasoning?: number;
    toolDefinition?: number;
  } | null;
  createdAt: string;
}

export interface ListCostEventsResponse {
  data: CostEventRecord[];
  cursor: { createdAt: string; id: string } | null;
}

export interface ListCostEventsOptions {
  limit?: number;
  /**
   * Pagination cursor. Accepts either a raw JSON-stringified cursor or the
   * `{ createdAt, id }` object returned by `ListCostEventsResponse.cursor`.
   * Pass the response cursor directly to fetch the next page — the SDK
   * stringifies it for the server.
   */
  cursor?: string | { createdAt: string; id: string };
}

// ---------------------------------------------------------------------------
// Cost summary
// ---------------------------------------------------------------------------

export interface ModelBreakdown {
  provider: string;
  model: string;
  totalCostMicrodollars: number;
  requestCount: number;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
}

export interface ProviderBreakdown {
  provider: string;
  totalCostMicrodollars: number;
  requestCount: number;
}

export interface KeyBreakdown {
  apiKeyId: string;
  keyName: string;
  totalCostMicrodollars: number;
  requestCount: number;
}

export interface ToolBreakdown {
  model: string;
  totalCostMicrodollars: number;
  requestCount: number;
  avgDurationMs: number;
}

export interface SourceBreakdown {
  source: string;
  totalCostMicrodollars: number;
  requestCount: number;
}

export interface TraceBreakdown {
  traceId: string;
  totalCostMicrodollars: number;
  requestCount: number;
}

export interface CostBreakdownTotals {
  inputCost: number;
  outputCost: number;
  cachedCost: number;
  reasoningCost: number;
}

export interface CostSummaryResponse {
  daily: Array<{ date: string; totalCostMicrodollars: number }>;
  models: ModelBreakdown[];
  providers: ProviderBreakdown[];
  keys: KeyBreakdown[];
  tools: ToolBreakdown[];
  sources: SourceBreakdown[];
  traces: TraceBreakdown[];
  totals: {
    totalCostMicrodollars: number;
    totalRequests: number;
    period: CostSummaryPeriod;
  };
  costBreakdown: CostBreakdownTotals;
}

export type CostSummaryPeriod = "7d" | "30d" | "90d";
