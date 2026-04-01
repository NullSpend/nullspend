# Security Policy

## Reporting a Vulnerability

If you discover a security issue in NullSpend, please report it responsibly.

**Email:** chris@nullspend.dev

Please include:

- Description of the issue
- Steps to reproduce
- Affected versions/packages
- Any potential impact assessment

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 5 business days
- **Fix timeline:** Depends on severity, typically within 30 days for critical issues

## Scope

The following are in scope:

- `@nullspend/sdk` — TypeScript SDK
- `@nullspend/cost-engine` — Cost calculation engine
- `@nullspend/claude-agent` — Claude Agent SDK adapter
- `@nullspend/mcp-server` — MCP server
- `@nullspend/mcp-proxy` — MCP proxy
- `@nullspend/proxy` — Cloudflare Workers proxy
- `@nullspend/db` — Database schema

## Out of Scope

- The hosted NullSpend dashboard (report via email)
- Third-party dependencies (report upstream, but let us know)
- Social engineering

## Disclosure

We follow coordinated disclosure. We ask that you give us reasonable time to address the issue before public disclosure. We will credit reporters in release notes unless anonymity is requested.
