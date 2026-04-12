const DEFAULT_ALLOWED = new Set([
  "https://api.openai.com",
  "https://api.groq.com/openai",
  "https://api.together.xyz",
  "https://api.fireworks.ai/inference",
  "https://api.mistral.ai",
  "https://openrouter.ai/api",
]);
// NOTE: Perplexity excluded — uses /chat/completions (no /v1/ prefix),
// not compatible with our ${base}/v1/chat/completions URL construction.

export function isAllowedUpstream(url: string): boolean {
  const normalized = url.replace(/\/+$/, "").toLowerCase();
  return DEFAULT_ALLOWED.has(normalized);
}
