export interface DocsServerConfig {
  serverName: string;
  version: string;
}

export function loadConfig(): DocsServerConfig {
  return {
    serverName: process.env.NULLSPEND_DOCS_SERVER_NAME ?? "nullspend-docs",
    version: "0.1.0",
  };
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}
