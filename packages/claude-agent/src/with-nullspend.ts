import type { Options } from "@anthropic-ai/claude-agent-sdk";
import type { NullSpendAgentOptions } from "./types.js";

const DEFAULT_PROXY_URL = "https://proxy.nullspend.dev";
const MAX_TAG_KEYS = 10;
const MAX_TAG_KEY_LENGTH = 64;
const MAX_TAG_VALUE_LENGTH = 256;
const TAG_KEY_PATTERN = /^[a-zA-Z0-9_-]+$/;
const TRACE_ID_PATTERN = /^[0-9a-f]{32}$/;
const ACTION_ID_PATTERN =
  /^ns_act_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const NEWLINE_RE = /[\r\n]/;

function sanitizeHeaderValue(value: string, field: string): string {
  if (NEWLINE_RE.test(value)) {
    throw new Error(
      `withNullSpend: ${field} must not contain newline characters`,
    );
  }
  return value;
}

/**
 * Generate a short random session ID.
 * Format: "ses_{timestamp36}_{random}" (e.g., "ses_mncqabm4_a7f3")
 * Short enough to read in the UI, unique enough to avoid collisions.
 */
function generateSessionId(): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 6);
  return `ses_${ts}_${rand}`;
}

export function withNullSpend(
  options: NullSpendAgentOptions & Options,
): Options {
  const {
    apiKey,
    budgetSessionId,
    autoSession = true,
    tags,
    traceId,
    actionId,
    proxyUrl,
    ...sdkOptions
  } = options;

  if (!apiKey) throw new Error("withNullSpend: apiKey is required");

  // Auto-generate a session ID if not provided and autoSession is enabled
  const resolvedSessionId = budgetSessionId ?? (autoSession ? generateSessionId() : undefined);

  // Validate traceId format (must match proxy's ^[0-9a-f]{32}$)
  if (traceId && !TRACE_ID_PATTERN.test(traceId)) {
    throw new Error(
      'withNullSpend: traceId must be a 32-char lowercase hex string (e.g. "abcdef0123456789abcdef0123456789")',
    );
  }

  // Validate actionId format (must match proxy's stripNsPrefix("ns_act_", ...))
  if (actionId && !ACTION_ID_PATTERN.test(actionId)) {
    throw new Error(
      'withNullSpend: actionId must be in ns_act_<UUID> format (e.g. "ns_act_550e8400-e29b-41d4-a716-446655440000")',
    );
  }

  if (tags) {
    const keys = Object.keys(tags);
    if (keys.length > MAX_TAG_KEYS) {
      throw new Error(
        `withNullSpend: tags must have at most ${MAX_TAG_KEYS} keys`,
      );
    }
    for (const k of keys) {
      if (!TAG_KEY_PATTERN.test(k)) {
        throw new Error(
          `withNullSpend: tag key "${k}" must match [a-zA-Z0-9_-]+`,
        );
      }
      if (k.length > MAX_TAG_KEY_LENGTH) {
        throw new Error(
          `withNullSpend: tag key "${k}" exceeds ${MAX_TAG_KEY_LENGTH} chars`,
        );
      }
      if (tags[k].length > MAX_TAG_VALUE_LENGTH) {
        throw new Error(
          `withNullSpend: tag value for "${k}" exceeds ${MAX_TAG_VALUE_LENGTH} chars`,
        );
      }
    }
  }

  // Sanitize all header values against newline injection
  const safeApiKey = sanitizeHeaderValue(apiKey, "apiKey");
  const customHeaders: string[] = [`x-nullspend-key: ${safeApiKey}`];

  if (resolvedSessionId) {
    const safe = sanitizeHeaderValue(resolvedSessionId, "budgetSessionId");
    customHeaders.push(`x-nullspend-session: ${safe}`);
  }
  if (tags && Object.keys(tags).length > 0)
    customHeaders.push(`x-nullspend-tags: ${JSON.stringify(tags)}`);
  if (traceId) customHeaders.push(`x-nullspend-trace-id: ${traceId}`);
  if (actionId) customHeaders.push(`x-nullspend-action-id: ${actionId}`);

  // Merge with any existing custom headers from the caller
  const existingHeaders = sdkOptions.env?.ANTHROPIC_CUSTOM_HEADERS;
  if (existingHeaders) customHeaders.unshift(existingHeaders);

  // Always include process.env as the base so the child process retains
  // PATH, HOME, ANTHROPIC_API_KEY, etc. User-provided env vars override.
  const existingEnv = {
    ...(typeof process !== "undefined" ? process.env : {}),
    ...(sdkOptions.env ?? {}),
  };

  return {
    ...sdkOptions,
    env: {
      ...existingEnv,
      ANTHROPIC_BASE_URL: proxyUrl ?? DEFAULT_PROXY_URL,
      ANTHROPIC_CUSTOM_HEADERS: customHeaders.join("\n"),
    },
  };
}

// ── Policy cache ───────────────────────────────────────────────────
const policyCache = new Map<string, { data: PolicyResponse; expiresAt: number }>();
const POLICY_CACHE_TTL_MS = 60_000;

interface PolicyResponse {
  budget: {
    remaining_microdollars: number;
    max_microdollars: number;
    spend_microdollars: number;
    period_end: string | null;
    entity_type: string;
    entity_id: string;
  } | null;
  allowed_models: string[] | null;
  allowed_providers: string[] | null;
  cheapest_per_provider: Record<string, { model: string; input_per_mtok: number; output_per_mtok: number }> | null;
  cheapest_overall: { model: string; provider: string; input_per_mtok: number; output_per_mtok: number } | null;
  restrictions_active: boolean;
}

async function fetchPolicy(baseUrl: string, apiKey: string): Promise<PolicyResponse | null> {
  const cacheKey = `${baseUrl}:${apiKey}`;
  const cached = policyCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) return cached.data;

  try {
    const res = await fetch(`${baseUrl}/v1/policy`, {
      headers: { "x-nullspend-key": apiKey },
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return null;
    const data = await res.json() as PolicyResponse;
    policyCache.set(cacheKey, { data, expiresAt: Date.now() + POLICY_CACHE_TTL_MS });
    return data;
  } catch (err) {
    console.warn("[withNullSpendAsync] Policy fetch failed:", err instanceof Error ? err.message : err);
    return null;
  }
}

function buildBudgetContext(policy: PolicyResponse): string {
  const lines: string[] = ["[NullSpend Budget Context]"];

  if (policy.budget) {
    const remaining = (policy.budget.remaining_microdollars / 1_000_000).toFixed(2);
    const periodEnd = policy.budget.period_end
      ? ` (resets ${policy.budget.period_end.split("T")[0]})`
      : "";
    lines.push(`You have $${remaining} remaining in your budget${periodEnd}.`);
  }

  if (policy.allowed_models && policy.allowed_models.length > 0) {
    lines.push(`Allowed models: ${policy.allowed_models.join(", ")}.`);
  }

  if (policy.cheapest_overall) {
    const c = policy.cheapest_overall;
    lines.push(`Preferred model (cheapest): ${c.model} ($${c.input_per_mtok}/MTok input, $${c.output_per_mtok}/MTok output).`);
  }

  lines.push("When possible, prefer cheaper models to conserve budget.");
  return lines.join("\n");
}

/**
 * Async version of withNullSpend that fetches the key's policy
 * and injects budget constraints into the agent's system prompt.
 *
 * The policy fetch is best-effort: if it fails, the agent proceeds
 * without budget context (the proxy still enforces hard limits).
 *
 * Set `budgetAwareness: false` to skip the policy fetch.
 */
export async function withNullSpendAsync(
  options: NullSpendAgentOptions & Options,
): Promise<Options> {
  const base = withNullSpend(options);

  if (options.budgetAwareness === false) return base;

  const resolvedUrl = options.proxyUrl ?? DEFAULT_PROXY_URL;
  const policy = await fetchPolicy(resolvedUrl, options.apiKey);

  if (!policy) return base;

  const budgetContext = buildBudgetContext(policy);

  // appendSystemPrompt is on SDKControlInitializeRequest, not Options,
  // but the Agent SDK passes it through at runtime. Cast to avoid DTS error.
  const existing = (base as Record<string, unknown>).appendSystemPrompt as string | undefined;
  const merged = existing ? `${existing}\n\n${budgetContext}` : budgetContext;
  return {
    ...base,
    appendSystemPrompt: merged,
  } as Options;
}

/** Reset policy cache — exposed for testing only. */
export function _resetPolicyCache(): void {
  policyCache.clear();
}
