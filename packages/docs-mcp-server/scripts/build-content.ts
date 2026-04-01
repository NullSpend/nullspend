/**
 * Build script: reads content/docs/*.md + public/llms.txt from the monorepo root
 * and generates src/generated/docs.json with all doc content.
 *
 * Run via: tsx scripts/build-content.ts
 */

import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import path from "node:path";

interface DocPage {
  path: string;
  title: string;
  description: string;
  content: string;
}

const REPO_ROOT = path.resolve(import.meta.dirname, "..", "..", "..");
const DOCS_DIR = path.join(REPO_ROOT, "content", "docs");
const LLMS_TXT = path.join(REPO_ROOT, "public", "llms.txt");
const OUTPUT = path.join(import.meta.dirname, "..", "src", "generated", "docs.json");

// Anchored, non-greedy frontmatter regex — only matches at start of file
const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n/;
const TITLE_RE = /^title:\s*"([^"]*)"/m;
const DESC_RE = /^description:\s*"([^"]*)"/m;
const HEADING_RE = /^#\s+(.+)$/m;

function parseFrontmatter(raw: string): { title: string; description: string; body: string } {
  const fmMatch = raw.match(FRONTMATTER_RE);

  if (fmMatch) {
    const fm = fmMatch[1];
    const title = fm.match(TITLE_RE)?.[1] ?? "";
    const description = fm.match(DESC_RE)?.[1] ?? "";
    const body = raw.slice(fmMatch[0].length).trim();
    return { title, description, body };
  }

  // Fallback: extract title from first # heading, description from first paragraph
  const headingMatch = raw.match(HEADING_RE);
  const title = headingMatch?.[1]?.trim() ?? "Untitled";

  // First non-empty, non-heading line as description
  const lines = raw.split(/\r?\n/);
  let description = "";
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      description = trimmed.slice(0, 200);
      break;
    }
  }

  // Strip the heading line from the body to avoid title duplication
  const body = headingMatch
    ? raw.slice(0, headingMatch.index!) + raw.slice(headingMatch.index! + headingMatch[0].length)
    : raw;
  return { title, description, body: body.trim() };
}

function collectMarkdownFiles(dir: string, base: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      files.push(...collectMarkdownFiles(full, base));
    } else if (entry.endsWith(".md")) {
      files.push(full);
    }
  }
  return files;
}

function main() {
  // Validate inputs exist
  let stat;
  try {
    stat = statSync(DOCS_DIR);
  } catch {
    throw new Error(`Docs directory not found: ${DOCS_DIR}`);
  }
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: ${DOCS_DIR}`);
  }

  let llmsTxtContent: string;
  try {
    llmsTxtContent = readFileSync(LLMS_TXT, "utf-8").replace(/\r\n/g, "\n");
  } catch {
    throw new Error(`llms.txt not found: ${LLMS_TXT}`);
  }

  // Collect all markdown files
  const mdFiles = collectMarkdownFiles(DOCS_DIR, DOCS_DIR);

  if (mdFiles.length === 0) {
    throw new Error(`No markdown files found in ${DOCS_DIR}`);
  }

  const docs: DocPage[] = [];

  for (const file of mdFiles) {
    const raw = readFileSync(file, "utf-8").replace(/\r\n/g, "\n");
    const { title, description, body } = parseFrontmatter(raw);

    // Compute path relative to docs dir, strip .md, normalize to forward slashes
    const relPath = path
      .relative(DOCS_DIR, file)
      .replace(/\\/g, "/")
      .replace(/\.md$/, "");

    docs.push({ path: relPath, title, description, content: body });
  }

  // Add llms.txt as a synthetic entry
  docs.push({
    path: "llms.txt",
    title: "LLM-Readable API Reference",
    description: "Machine-readable NullSpend overview for AI agents",
    content: llmsTxtContent.trim(),
  });

  // Sort for deterministic output
  docs.sort((a, b) => a.path.localeCompare(b.path));

  // Ensure output directory exists
  mkdirSync(path.dirname(OUTPUT), { recursive: true });

  writeFileSync(OUTPUT, JSON.stringify(docs, null, 2), "utf-8");

  process.stderr.write(
    `[build-content] Generated ${docs.length} docs (${mdFiles.length} markdown + llms.txt) → ${path.relative(REPO_ROOT, OUTPUT)}\n`,
  );
}

main();
