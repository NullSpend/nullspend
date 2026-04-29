const CALVER_REGEX = /^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$/;

export function isCalVer(value: unknown): value is string {
  if (typeof value !== "string" || !CALVER_REGEX.test(value)) return false;
  // Reject semantically-invalid dates the regex can't catch (Feb 30, Apr 31).
  // `new Date(...).toISOString()` round-trips ONLY for real calendar dates.
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return false;
  return parsed.toISOString().slice(0, 10) === value;
}

export class ApiVersion {
  readonly value: string;
  readonly index: number;

  constructor(value: string, index: number) {
    this.value = value;
    this.index = index;
  }

  toString(): string {
    return this.value;
  }

  toJSON(): string {
    return this.value;
  }

  private cmpIndex(other: ApiVersion): number {
    return this.index - other.index;
  }

  eq(other: ApiVersion): boolean {
    return this.cmpIndex(other) === 0;
  }

  lt(other: ApiVersion): boolean {
    return this.cmpIndex(other) < 0;
  }

  lte(other: ApiVersion): boolean {
    return this.cmpIndex(other) <= 0;
  }

  gt(other: ApiVersion): boolean {
    return this.cmpIndex(other) > 0;
  }

  gte(other: ApiVersion): boolean {
    return this.cmpIndex(other) >= 0;
  }
}
