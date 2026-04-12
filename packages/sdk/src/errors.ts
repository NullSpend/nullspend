/** Coerce to a finite non-negative number, defaulting to 0. */
function safeFiniteNonNeg(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

export class NullSpendError extends Error {
  public readonly statusCode: number | undefined;
  public readonly code: string | undefined;

  constructor(message: string, statusCode?: number, code?: string) {
    super(message);
    this.name = "NullSpendError";
    this.statusCode = statusCode;
    this.code = code;
  }
}

export class TimeoutError extends NullSpendError {
  public readonly actionId: string;
  public readonly timeoutMs: number;

  constructor(actionId: string, timeoutMs: number) {
    const safeActionId = typeof actionId === "string" ? actionId : String(actionId ?? "unknown");
    const safeTimeoutMs = Number.isFinite(timeoutMs) && timeoutMs >= 0 ? timeoutMs : 0;
    super(
      `Timed out waiting for decision on action ${safeActionId} after ${safeTimeoutMs}ms`,
    );
    this.name = "TimeoutError";
    this.actionId = safeActionId;
    this.timeoutMs = safeTimeoutMs;
  }
}

export class RejectedError extends NullSpendError {
  public readonly actionId: string;
  public readonly actionStatus: string;

  constructor(actionId: string, status: string) {
    const safeActionId = typeof actionId === "string" ? actionId : String(actionId ?? "unknown");
    const safeStatus = typeof status === "string" ? status : String(status ?? "unknown");
    super(`Action ${safeActionId} was ${safeStatus}`);
    this.name = "RejectedError";
    this.actionId = safeActionId;
    this.actionStatus = safeStatus;
  }
}

export class BudgetExceededError extends NullSpendError {
  public readonly remainingMicrodollars: number;
  public readonly entityType: string | undefined;
  public readonly entityId: string | undefined;
  public readonly limitMicrodollars: number | undefined;
  public readonly spendMicrodollars: number | undefined;
  /**
   * Plan-upgrade URL surfaced by the proxy when the denying org has
   * configured one (org-level via dashboard Settings > General, or
   * per-customer via customer_mappings.upgrade_url). Supports the
   * `{customer_id}` placeholder which the proxy substitutes at denial
   * time. Undefined when no upgrade_url is configured.
   */
  public readonly upgradeUrl: string | undefined;

  constructor(details: number | {
    remaining: number;
    entityType?: string;
    entityId?: string;
    limit?: number;
    spend?: number;
    upgradeUrl?: string;
  }) {
    const d = typeof details === "number" ? { remaining: details } : details;
    const safeRemaining = safeFiniteNonNeg(d.remaining);
    super(
      `Budget exceeded: ${safeRemaining} microdollars remaining`,
    );
    this.name = "BudgetExceededError";
    this.remainingMicrodollars = safeRemaining;
    this.entityType = d.entityType;
    this.entityId = d.entityId;
    this.limitMicrodollars = d.limit !== undefined ? safeFiniteNonNeg(d.limit) : undefined;
    this.spendMicrodollars = d.spend !== undefined ? safeFiniteNonNeg(d.spend) : undefined;
    this.upgradeUrl = d.upgradeUrl;
  }
}

export class MandateViolationError extends NullSpendError {
  public readonly mandate: string;
  public readonly requested: string;
  public readonly allowed: string[];

  constructor(mandate: string, requested: string, allowed: string[]) {
    super(
      `Mandate violation: ${mandate} does not allow "${requested}". Allowed: ${allowed.join(", ")}`,
    );
    this.name = "MandateViolationError";
    this.mandate = mandate;
    this.requested = requested;
    this.allowed = allowed;
  }
}

export class SessionLimitExceededError extends NullSpendError {
  public readonly sessionSpendMicrodollars: number;
  public readonly sessionLimitMicrodollars: number;

  constructor(sessionSpend: number, sessionLimit: number) {
    const safeSpend = safeFiniteNonNeg(sessionSpend);
    const safeLimit = safeFiniteNonNeg(sessionLimit);
    super(
      `Session limit exceeded: ${safeSpend} of ${safeLimit} microdollars spent`,
    );
    this.name = "SessionLimitExceededError";
    this.sessionSpendMicrodollars = safeSpend;
    this.sessionLimitMicrodollars = safeLimit;
  }
}

export class VelocityExceededError extends NullSpendError {
  public readonly retryAfterSeconds: number | undefined;
  public readonly limitMicrodollars: number | undefined;
  public readonly windowSeconds: number | undefined;
  public readonly currentMicrodollars: number | undefined;

  constructor(details?: {
    retryAfterSeconds?: number;
    limit?: number;
    window?: number;
    current?: number;
  }) {
    const retryAfter = details?.retryAfterSeconds !== undefined
      ? safeFiniteNonNeg(details.retryAfterSeconds) : undefined;
    super(
      `Velocity limit exceeded${retryAfter ? ` — retry after ${retryAfter}s` : ""}`,
    );
    this.name = "VelocityExceededError";
    this.retryAfterSeconds = retryAfter;
    this.limitMicrodollars = details?.limit !== undefined ? safeFiniteNonNeg(details.limit) : undefined;
    this.windowSeconds = details?.window !== undefined ? safeFiniteNonNeg(details.window) : undefined;
    this.currentMicrodollars = details?.current !== undefined ? safeFiniteNonNeg(details.current) : undefined;
  }
}

export class TagBudgetExceededError extends NullSpendError {
  public readonly tagKey: string | undefined;
  public readonly tagValue: string | undefined;
  public readonly remainingMicrodollars: number | undefined;
  public readonly limitMicrodollars: number | undefined;
  public readonly spendMicrodollars: number | undefined;

  constructor(details?: {
    tagKey?: string;
    tagValue?: string;
    remaining?: number;
    limit?: number;
    spend?: number;
  }) {
    const tag = details?.tagKey ? `${details.tagKey}=${details.tagValue}` : "unknown";
    super(`Tag budget exceeded for ${tag}`);
    this.name = "TagBudgetExceededError";
    this.tagKey = details?.tagKey;
    this.tagValue = details?.tagValue;
    this.remainingMicrodollars = details?.remaining !== undefined ? safeFiniteNonNeg(details.remaining) : undefined;
    this.limitMicrodollars = details?.limit !== undefined ? safeFiniteNonNeg(details.limit) : undefined;
    this.spendMicrodollars = details?.spend !== undefined ? safeFiniteNonNeg(details.spend) : undefined;
  }
}
