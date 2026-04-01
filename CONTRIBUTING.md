# Contributing to NullSpend

We welcome contributions to NullSpend. This guide covers the development setup, testing, and PR process.

## Development Setup

### Prerequisites

- Node.js >= 20.11.0
- pnpm >= 10

### Getting Started

```bash
git clone https://github.com/NullSpend/nullspend.git
cd nullspend
pnpm install
```

### Build Order

Some packages depend on others. Build in this order:

```bash
pnpm db:build           # @nullspend/db (required first)
pnpm cost-engine:build  # @nullspend/cost-engine
pnpm sdk:build          # @nullspend/sdk (depends on cost-engine)
pnpm claude-agent:build # @nullspend/claude-agent
pnpm docs-mcp:build     # @nullspend/docs
```

### Running Tests

```bash
pnpm proxy:test         # Proxy worker tests
pnpm sdk:test           # SDK tests
pnpm cost-engine:test   # Cost engine tests
pnpm claude-agent:test  # Claude agent adapter tests
pnpm mcp:test           # MCP server tests
pnpm mcp-proxy:test     # MCP proxy tests
pnpm docs-mcp:test      # Docs MCP server tests
pnpm db:test            # DB schema tests
```

### Type Checking & Linting

```bash
pnpm proxy:typecheck    # Proxy TypeScript check
pnpm lint               # ESLint
```

## Project Structure

```
nullspend/
├── apps/proxy/              # Cloudflare Workers proxy
├── packages/
│   ├── sdk/                 # @nullspend/sdk
│   ├── sdk-python/          # Python SDK
│   ├── cost-engine/         # @nullspend/cost-engine
│   ├── claude-agent/        # @nullspend/claude-agent
│   ├── mcp-server/          # @nullspend/mcp-server
│   ├── mcp-proxy/           # @nullspend/mcp-proxy
│   ├── docs-mcp-server/     # @nullspend/docs
│   └── db/                  # @nullspend/db
└── docs/                    # Documentation
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Make your changes
3. Add or update tests for any changed functionality
4. Ensure all tests pass (`pnpm proxy:test && pnpm sdk:test && pnpm cost-engine:test`)
5. Ensure type checking passes (`pnpm proxy:typecheck`)
6. Ensure linting passes (`pnpm lint`)
7. Open a pull request with a clear title and description

### PR Guidelines

- Keep PRs focused on a single change
- Include test coverage for new functionality
- Update documentation if you're changing public APIs
- Use clear commit messages

### Commit Messages

Follow conventional commit style:

```
feat: add Gemini model support to cost-engine
fix: correct cached token calculation for Anthropic
docs: update SDK quick start example
test: add edge cases for budget enforcement
```

## Code Style

- TypeScript for all packages
- ESM module format (`.js` extensions in relative imports for proxy worker)
- Vitest for testing
- No default exports (except where required by frameworks)

## Questions?

Open a [GitHub issue](https://github.com/NullSpend/nullspend/issues) for questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
