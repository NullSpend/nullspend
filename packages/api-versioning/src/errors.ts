export class InvalidVersionError extends Error {
  readonly code = "invalid_version";
  readonly supported: readonly string[];
  constructor(value: string, supported: readonly string[]) {
    super(`Invalid API version: ${JSON.stringify(value)}. Expected CalVer YYYY-MM-DD.`);
    this.name = "InvalidVersionError";
    this.supported = supported;
  }
}

export class UnknownVersionError extends Error {
  readonly code = "unsupported_version";
  readonly supported: readonly string[];
  constructor(value: string, supported: readonly string[]) {
    super(`Unsupported API version: ${JSON.stringify(value)}. Supported: ${supported.join(", ")}.`);
    this.name = "UnknownVersionError";
    this.supported = supported;
  }
}

export class RegistryAssertionError extends Error {
  constructor(message: string) {
    super(`API versioning registry assertion: ${message}`);
    this.name = "RegistryAssertionError";
  }
}
