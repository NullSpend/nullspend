// Entries must be scheme + host (+ optional non-versioned path segments).
// Do NOT include API version path segments like /v1, /v1beta, /v2 — adapters
// append those when constructing the upstream URL.
const DEFAULT_ALLOWED = new Set([
  "https://api.openai.com",
  "https://api.groq.com/openai",
  "https://api.together.xyz",
  "https://api.fireworks.ai/inference",
  "https://api.mistral.ai",
  "https://openrouter.ai/api",
  "https://generativelanguage.googleapis.com",
]);
// NOTE: Perplexity excluded — uses /chat/completions (no /v1/ prefix),
// not compatible with our ${base}/v1/chat/completions URL construction.

export function isAllowedUpstream(url: string): boolean {
  const normalized = url.replace(/\/+$/, "").toLowerCase();
  return DEFAULT_ALLOWED.has(normalized);
}
