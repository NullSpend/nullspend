/** Coerce to a finite non-negative number, defaulting to 0. */
function safeFiniteNonNeg(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

/**
 * Per-error-class default docs URLs (D-1). Used when the proxy doesn't
 * supply a `recovery.docs` URL — gives every error a stable place to point
 * developers for resolution guidance. Keyed by `error.name`.
 */
const DEFAULT_ERROR_DOCS_URL: Record<string, string> = {
  NullSpendError: "https://nullspend.dev/docs/errors",
  TimeoutError: "https://nullspend.dev/docs/errors/timeout",
  RejectedError: "https://nullspend.dev/docs/errors/rejected",
  BudgetExceededError: "https://nullspend.dev/docs/errors/budget-exceeded",
  MandateViolationError: "https://nullspend.dev/docs/errors/mandate-violation",
  SessionLimitExceededError: "https://nullspend.dev/docs/errors/session-limit",
  VelocityExceededError: "https://nullspend.dev/docs/errors/velocity",
  LoopDetectedError: "https://nullspend.dev/docs/errors/loop-detected",
  TagBudgetExceededError: "https://nullspend.dev/docs/errors/tag-budget",
  PlanLimitExceededError: "https://nullspend.dev/docs/errors/plan-limit",
};

export class NullSpendError extends Error {
  public readonly statusCode: number | undefined;
  public readonly code: string | undefined;

  constructor(message: string, statusCode?: number, code?: string) {
    super(message);
    this.name = "NullSpendError";
    this.statusCode = statusCode;
    this.code = code;
  }

  /**
   * Documentation URL for this error type. Prefers a proxy-supplied
   * `recovery.docs` URL when available (subclasses with `recovery` override
   * this getter to thread that value through), otherwise falls back to a
   * stable per-class default. (D-1)
   */
  get docsUrl(): string {
    return DEFAULT_ERROR_DOCS_URL[this.name] ?? DEFAULT_ERROR_DOCS_URL.NullSpendError;
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

/** Machine-readable recovery hints from proxy 429 denials. */
export type Recovery = {
  retryable: boolean;
  ownerActionRequired: boolean;
  retryAfterSeconds: number | null;
  docs: string | null;
};

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
  /** Finalization reserve in microdollars held back from available budget. */
  public readonly finalizationReserveMicrodollars: number | undefined;
  /** Remaining budget after subtracting finalization reserve. */
  public readonly finalizationRemainingMicrodollars: number | undefined;
  /** Structured recovery hints from proxy. Undefined for old proxy versions. */
  public readonly recovery: Recovery | undefined;

  constructor(details: number | {
    remaining: number;
    entityType?: string;
    entityId?: string;
    limit?: number;
    spend?: number;
    upgradeUrl?: string;
    finalizationReserve?: number;
    finalizationRemaining?: number;
    recovery?: Recovery;
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
    this.finalizationReserveMicrodollars = d.finalizationReserve !== undefined ? safeFiniteNonNeg(d.finalizationReserve) : undefined;
    this.finalizationRemainingMicrodollars = d.finalizationRemaining !== undefined ? safeFiniteNonNeg(d.finalizationRemaining) : undefined;
    this.recovery = d.recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
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
  public readonly recovery: Recovery | undefined;

  constructor(sessionSpend: number, sessionLimit: number, recovery?: Recovery) {
    const safeSpend = safeFiniteNonNeg(sessionSpend);
    const safeLimit = safeFiniteNonNeg(sessionLimit);
    super(
      `Session limit exceeded: ${safeSpend} of ${safeLimit} microdollars spent`,
    );
    this.name = "SessionLimitExceededError";
    this.sessionSpendMicrodollars = safeSpend;
    this.sessionLimitMicrodollars = safeLimit;
    this.recovery = recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
  }
}

export class VelocityExceededError extends NullSpendError {
  public readonly retryAfterSeconds: number | undefined;
  public readonly limitMicrodollars: number | undefined;
  public readonly windowSeconds: number | undefined;
  public readonly currentMicrodollars: number | undefined;
  public readonly recovery: Recovery | undefined;

  constructor(details?: {
    retryAfterSeconds?: number;
    limit?: number;
    window?: number;
    current?: number;
    recovery?: Recovery;
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
    this.recovery = details?.recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
  }
}

export class LoopDetectedError extends NullSpendError {
  public readonly model: string;
  public readonly callCount: number;
  public readonly windowSeconds: number;
  public readonly maxCalls: number;
  public readonly detectionType: string;
  public readonly recovery: Recovery | undefined;

  constructor(details: {
    model: string;
    callCount: number;
    windowSeconds: number;
    maxCalls: number;
    detectionType?: string;
    recovery?: Recovery;
  }) {
    const model = details.model || "unknown";
    const callCount = safeFiniteNonNeg(details.callCount);
    const windowSeconds = safeFiniteNonNeg(details.windowSeconds);
    const maxCalls = safeFiniteNonNeg(details.maxCalls);
    super(
      `Loop detected: ${model} called ${callCount} times with identical ` +
      `content in ${windowSeconds}s (limit: ${maxCalls}). ` +
      `Check for retry loops or stuck agent logic. ` +
      `Adjust at https://nullspend.dev/app/budgets or set loop_max_calls=0 to disable.`,
      429,
      "loop_detected",
    );
    this.name = "LoopDetectedError";
    this.model = model;
    this.callCount = callCount;
    this.windowSeconds = windowSeconds;
    this.maxCalls = maxCalls;
    this.detectionType = details.detectionType ?? "per_key";
    this.recovery = details.recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
  }
}

export class TagBudgetExceededError extends NullSpendError {
  public readonly tagKey: string | undefined;
  public readonly tagValue: string | undefined;
  public readonly remainingMicrodollars: number | undefined;
  public readonly limitMicrodollars: number | undefined;
  public readonly spendMicrodollars: number | undefined;
  public readonly recovery: Recovery | undefined;

  constructor(details?: {
    tagKey?: string;
    tagValue?: string;
    remaining?: number;
    limit?: number;
    spend?: number;
    recovery?: Recovery;
  }) {
    const tag = details?.tagKey ? `${details.tagKey}=${details.tagValue}` : "unknown";
    super(`Tag budget exceeded for ${tag}`);
    this.name = "TagBudgetExceededError";
    this.tagKey = details?.tagKey;
    this.tagValue = details?.tagValue;
    this.remainingMicrodollars = details?.remaining !== undefined ? safeFiniteNonNeg(details.remaining) : undefined;
    this.limitMicrodollars = details?.limit !== undefined ? safeFiniteNonNeg(details.limit) : undefined;
    this.spendMicrodollars = details?.spend !== undefined ? safeFiniteNonNeg(details.spend) : undefined;
    this.recovery = details?.recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
  }
}

/**
 * PR-2c: NullSpend-tier plan-limit denial. Thrown when an org's governed-request
 * count exceeds its plan cap (Free = 100K per period). Unlike BudgetExceededError
 * (org-configured budget), this comes from NullSpend's pricing tiers — the
 * `upgradeUrl` points to NullSpend's canonical pricing page, and `selfHostUrl`
 * gives the alternative remediation path. Both are top-level `error.*` fields
 * on the 429 body, parsed by `dispatchDenialCode` in `tracked-fetch.ts`.
 *
 * `count`, `blockAt`, and `tier` carry the decision frozen at the ORIGINAL
 * request time (PR-2c codex-round-3 C1 — DO idempotency replay persists the
 * original outcome; retries of the same request produce identical errors
 * regardless of current counter state).
 */
export class PlanLimitExceededError extends NullSpendError {
  public readonly count: number;
  public readonly blockAt: number;
  public readonly tier: string;
  public readonly upgradeUrl: string | undefined;
  public readonly selfHostUrl: string | undefined;
  public readonly recovery: Recovery | undefined;

  constructor(details: {
    count: number;
    blockAt: number;
    tier: string;
    upgradeUrl?: string;
    selfHostUrl?: string;
    recovery?: Recovery;
  }) {
    const safeCount = safeFiniteNonNeg(details.count);
    const safeBlockAt = safeFiniteNonNeg(details.blockAt);
    const safeTier = typeof details.tier === "string" && details.tier.length > 0 ? details.tier : "unknown";
    super(
      `Plan limit reached: ${safeCount} of ${safeBlockAt} governed requests on ${safeTier} plan. Upgrade or wait for period reset.`,
      429,
      "plan_limit_exceeded",
    );
    this.name = "PlanLimitExceededError";
    this.count = safeCount;
    this.blockAt = safeBlockAt;
    this.tier = safeTier;
    this.upgradeUrl = details.upgradeUrl;
    this.selfHostUrl = details.selfHostUrl;
    this.recovery = details.recovery;
  }

  override get docsUrl(): string {
    return this.recovery?.docs ?? super.docsUrl;
  }
}
