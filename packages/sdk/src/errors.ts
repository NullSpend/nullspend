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
  constructor(actionId: string, timeoutMs: number) {
    super(
      `Timed out waiting for decision on action ${actionId} after ${timeoutMs}ms`,
    );
    this.name = "TimeoutError";
  }
}

export class RejectedError extends NullSpendError {
  public readonly actionId: string;
  public readonly actionStatus: string;

  constructor(actionId: string, status: string) {
    super(`Action ${actionId} was ${status}`);
    this.name = "RejectedError";
    this.actionId = actionId;
    this.actionStatus = status;
  }
}

export class BudgetExceededError extends NullSpendError {
  public readonly remainingMicrodollars: number;

  constructor(remainingMicrodollars: number) {
    super(
      `Budget exceeded: ${remainingMicrodollars} microdollars remaining`,
    );
    this.name = "BudgetExceededError";
    this.remainingMicrodollars = remainingMicrodollars;
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
