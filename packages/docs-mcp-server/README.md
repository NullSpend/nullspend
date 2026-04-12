# @nullspend/docs

MCP server that serves NullSpend documentation to AI coding tools. Zero auth required — no API key, no account needed.

## Quick Start

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nullspend-docs": {
      "command": "npx",
      "args": ["-y", "@nullspend/docs"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "nullspend-docs": {
      "command": "npx",
      "args": ["-y", "@nullspend/docs"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add nullspend-docs -- npx -y @nullspend/docs
```

## Tools

### `nullspend_search_docs`

Search NullSpend documentation by keyword. Returns ranked results with title, description, and path.

**Parameters:**
- `query` (string, required) — search query
- `limit` (number, optional) — max results (default: 10, max: 20)

### `nullspend_fetch_doc`

Fetch the full content of a documentation page.

**Parameters:**
- `path` (string, required) — doc path from search results (e.g. `"quickstart/openai"`, `"features/budgets"`, `"llms.txt"`)

## How It Works

The server bundles 39 markdown documentation pages plus `llms.txt` as static content. Search uses keyword matching with synonym expansion — no network calls, no embeddings, fully offline.

## Development

```bash
pnpm install
pnpm build        # Generates docs content + compiles
pnpm test         # Run tests
pnpm start        # Start server (stdio)
```

Requires Node.js 22+.
