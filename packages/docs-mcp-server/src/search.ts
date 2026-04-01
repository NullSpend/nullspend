import type { DocPage } from "./content.js";

export interface SearchResult {
  path: string;
  title: string;
  description: string;
  score: number;
}

const SYNONYMS: Record<string, string[]> = {
  hitl: ["human-in-the-loop", "approval"],
  cost: ["spend", "price", "pricing", "budget"],
  spend: ["cost", "budget", "price"],
  budget: ["cost", "spend", "limit"],
  key: ["api-key", "authentication"],
  "api-key": ["key", "authentication"],
  "429": ["error", "rate-limit", "budget-exceeded"],
  limit: ["budget", "velocity", "session-limit"],
  webhook: ["event", "notification"],
  track: ["tracking", "monitor"],
  tracking: ["track", "monitor"],
};

const TITLE_WEIGHT = 10;
const DESCRIPTION_WEIGHT = 5;
const CONTENT_WEIGHT = 1;
const MIN_SCORE_THRESHOLD = 2;
const MIN_TOKEN_LENGTH_FOR_SUBSTRING = 3;

/** Tokenize text: split on non-alphanumeric, lowercase, strip hyphen noise, filter empties */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9-]+/)
    .map((t) => t.replace(/^-+|-+$/g, "")) // strip leading/trailing hyphens
    .filter((t) => t.length > 0 && !/^-+$/.test(t)); // remove empty and hyphen-only tokens
}

/** Expand query tokens with synonyms */
function expandWithSynonyms(tokens: string[]): string[] {
  const expanded = new Set(tokens);
  for (const token of tokens) {
    const syns = SYNONYMS[token];
    if (syns) {
      for (const syn of syns) {
        expanded.add(syn);
      }
    }
  }
  return [...expanded];
}

/** Check if tokens match — exact for short tokens, substring for longer ones (both sides) */
function tokenMatch(queryToken: string, contentToken: string): boolean {
  if (queryToken.length < MIN_TOKEN_LENGTH_FOR_SUBSTRING || contentToken.length < MIN_TOKEN_LENGTH_FOR_SUBSTRING) {
    return queryToken === contentToken;
  }
  return queryToken.includes(contentToken) || contentToken.includes(queryToken);
}

/** Score how well query tokens match a set of content tokens */
function scoreTokens(queryTokens: string[], contentTokens: string[]): number {
  let score = 0;
  for (const qt of queryTokens) {
    for (const ct of contentTokens) {
      if (tokenMatch(qt, ct)) {
        score++;
        break; // count each query token at most once per field
      }
    }
  }
  return score;
}

export interface DocIndex {
  doc: DocPage;
  titleTokens: string[];
  descriptionTokens: string[];
  contentTokens: string[];
}

/** Build a search index from docs (call once, reuse) */
export function buildIndex(docs: DocPage[]): DocIndex[] {
  return docs.map((doc) => ({
    doc,
    titleTokens: tokenize(doc.title),
    descriptionTokens: tokenize(doc.description),
    contentTokens: tokenize(doc.content),
  }));
}

/** Search docs by query, return ranked results */
export function searchDocs(
  query: string,
  index: DocIndex[],
  limit: number = 10,
): SearchResult[] {
  const rawTokens = tokenize(query);
  if (rawTokens.length === 0) {
    // Empty query: return all docs sorted alphabetically, limited
    return index
      .map((entry) => ({
        path: entry.doc.path,
        title: entry.doc.title,
        description: entry.doc.description,
        score: 0,
      }))
      .sort((a, b) => a.path.localeCompare(b.path))
      .slice(0, limit);
  }

  const queryTokens = expandWithSynonyms(rawTokens);

  const scored: SearchResult[] = [];

  for (const entry of index) {
    const titleScore = scoreTokens(queryTokens, entry.titleTokens) * TITLE_WEIGHT;
    const descScore = scoreTokens(queryTokens, entry.descriptionTokens) * DESCRIPTION_WEIGHT;
    const contentScore = scoreTokens(queryTokens, entry.contentTokens) * CONTENT_WEIGHT;
    const total = titleScore + descScore + contentScore;

    if (total > 0) {
      scored.push({
        path: entry.doc.path,
        title: entry.doc.title,
        description: entry.doc.description,
        score: total,
      });
    }
  }

  // Filter results below minimum threshold (per-result, not all-or-nothing)
  const filtered = scored.filter((s) => s.score >= MIN_SCORE_THRESHOLD);

  if (filtered.length === 0) {
    return [];
  }

  // Sort by score descending, then by path for stability
  filtered.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));

  return filtered.slice(0, limit);
}
