import type { CostEventInput, TrackedFetchOptions, TrackedProvider } from "./types.js";
import type { PolicyCache } from "./policy-cache.js";
import { validateCustomerId } from "./customer-id.js";
import {
  BudgetExceededError,
  MandateViolationError,
  SessionLimitExceededError,
  VelocityExceededError,
  TagBudgetExceededError,
} from "./errors.js";
import {
  isTrackedRoute,
  extractModelFromBody,
  isStreamingRequest,
  isStreamingResponse,
  extractOpenAIUsageFromJSON,
  extractAnthropicUsageFromJSON,
} from "./provider-parsers.js";
import {
  createOpenAISSEParser,
  createAnthropicSSEParser,
} from "./sse-parser.js";
import {
  calculateOpenAICostEvent,
  calculateAnthropicCostEvent,
} from "./cost-calculator.js";
import { getModelPricing } from "@nullspend/cost-engine";

/**
 * Build a tracked fetch function that intercepts LLM API calls and
 * automatically reports cost events.
 *
 * @param proxyUrl Optional proxy base URL. Detection is by URL ORIGIN match
 *   (scheme + host + port — strict). When the request URL's origin matches, the
 *   SDK skips client-side cost tracking and lets the proxy handle accounting.
 *   If your proxy listens on a non-default port, include the port in this URL
 *   or the SDK will take the direct path and double-count. Header-based detection
 *   (`x-nullspend-key`) is a secondary fallback ONLY when `proxyUrl` is configured
 *   but the origin check fails (e.g., port mismatch). Without a configured
 *   `proxyUrl`, the header alone cannot trigger the proxy path.
 */
export function buildTrackedFetch(
  provider: TrackedProvider,
  options: TrackedFetchOptions | undefined,
  queueCost: (event: CostEventInput) => void,
  policyCache: PolicyCache | null,
  proxyUrl?: string,
  fetchFn?: typeof globalThis.fetch,
): typeof globalThis.fetch {
  // Validate + normalize customer here so direct callers of createTrackedFetch
  // get the same fail-fast behavior as NullSpend.customer(). Throws on invalid
  // input — that is intentional, we want the error at fetch-builder construction
  // time, not at first request.
  const customer = options?.customer !== undefined ? validateCustomerId(options.customer) : undefined;
  const sessionId = options?.sessionId;
  const tags = options?.tags;
  const traceId = options?.traceId;
  const enforcement = options?.enforcement ?? false;
  const manualSessionLimit = options?.sessionLimitMicrodollars ?? null;
  const onCostError = options?.onCostError ?? defaultCostErrorHandler;
  const onDenied = options?.onDenied;

  const trackSessionSpend = !!(enforcement && sessionId);
  let sessionSpendMicrodollars = 0;
  const costSink = trackSessionSpend
    ? (event: CostEventInput) => {
        sessionSpendMicrodollars += event.costMicrodollars;
        queueCost(event);
      }
    : queueCost;

  const metadata = { sessionId, traceId, tags, customer };
  const doFetch = fetchFn ?? globalThis.fetch;

  return async function trackedFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    // Resolve URL from input
    const url = resolveUrl(input);

    // Inject X-NullSpend-Customer header if customer is set. This MUST happen
    // before the isProxied bailout — otherwise the header never reaches the
    // proxy and customer attribution is silently lost.
    //
    // WHATWG subtlety: when input is a Request and init.headers is set,
    // init.headers REPLACES the Request's headers entirely (dropping any
    // Authorization the caller baked into the Request). So if the caller
    // passed a Request without init.headers, we must inject the header into
    // a cloned Request instead of synthesizing an init.
    if (customer) {
      if (input instanceof Request && !init?.headers) {
        const newHeaders = new Headers(input.headers);
        newHeaders.set("X-NullSpend-Customer", customer);
        input = new Request(input, { headers: newHeaders });
      } else {
        init = addHeader(init, "X-NullSpend-Customer", customer);
      }
    }

    // Proxy path: skip client-side cost tracking + enforcement + stream injection,
    // but still intercept NullSpend denial codes so callers get typed errors.
    // The proxy handles cost tracking and enforcement server-side; running them
    // client-side would double-count. The 429 interception was previously
    // bypassed by an early return — that was the bug fixed in §15c-1.
    if (isProxied(url, init, proxyUrl, input)) {
      const response = await doFetch(input, init);
      if (response.status === 429 && enforcement) {
        const parsed = await parseDenialPayload(response);
        if (parsed) dispatchDenialCode(parsed, customer, onDenied, onCostError);
      }
      return response; // raw response, no cost tracking — preserves no-double-count
    }

    // Non-tracked routes pass through
    const method = init?.method ?? (input instanceof Request ? input.method : "GET");
    if (!isTrackedRoute(provider, url, method)) {
      return doFetch(input, init);
    }

    // Parse body for model + streaming detection
    const bodyStr = await extractBody(input, init);
    const model = (bodyStr && extractModelFromBody(bodyStr)) ?? "unknown";
    const streaming = bodyStr ? isStreamingRequest(bodyStr) : false;

    // Phase 2: Cooperative enforcement
    if (enforcement && policyCache) {
      try {
        await policyCache.getPolicy();
        const mandateResult = policyCache.checkMandate(provider, model);
        if (!mandateResult.allowed) {
          safeDenied(onDenied, { type: "mandate", mandate: mandateResult.mandate!, requested: mandateResult.requested!, allowed: mandateResult.allowed_list! }, onCostError);
          throw new MandateViolationError(
            mandateResult.mandate!,
            mandateResult.requested!,
            mandateResult.allowed_list!,
          );
        }

        // Rough estimate for budget check
        const estimate = estimateCostMicrodollars(provider, model, bodyStr);
        const budgetResult = policyCache.checkBudget(estimate);
        if (!budgetResult.allowed) {
          safeDenied(onDenied, {
            type: "budget",
            remaining: budgetResult.remaining ?? 0,
            entityType: budgetResult.entityType,
            entityId: budgetResult.entityId,
            limit: budgetResult.limit,
            spend: budgetResult.spend,
          }, onCostError);
          throw new BudgetExceededError({
            remaining: budgetResult.remaining ?? 0,
            entityType: budgetResult.entityType,
            entityId: budgetResult.entityId,
            limit: budgetResult.limit,
            spend: budgetResult.spend,
          });
        }

        // Session limit check (only when sessionId is set)
        if (sessionId) {
          const sessionLimit = manualSessionLimit ?? policyCache.getSessionLimit() ?? null;
          if (sessionLimit !== null && sessionSpendMicrodollars + estimate > sessionLimit) {
            warnSessionDenied(sessionSpendMicrodollars, estimate, sessionLimit, sessionId);
            safeDenied(onDenied, { type: "session_limit", sessionSpend: sessionSpendMicrodollars, sessionLimit }, onCostError);
            throw new SessionLimitExceededError(sessionSpendMicrodollars, sessionLimit);
          }
        }
      } catch (err) {
        if (
          err instanceof BudgetExceededError ||
          err instanceof MandateViolationError ||
          err instanceof SessionLimitExceededError
        ) {
          throw err;
        }
        // Policy fetch failure — fall open, but still enforce manual session limit
        if (sessionId && manualSessionLimit !== null) {
          const estimate = estimateCostMicrodollars(provider, model, bodyStr);
          if (sessionSpendMicrodollars + estimate > manualSessionLimit) {
            warnSessionDenied(sessionSpendMicrodollars, estimate, manualSessionLimit, sessionId, true);
            safeDenied(onDenied, { type: "session_limit", sessionSpend: sessionSpendMicrodollars, sessionLimit: manualSessionLimit }, onCostError);
            throw new SessionLimitExceededError(sessionSpendMicrodollars, manualSessionLimit);
          }
        }
        onCostError?.(err instanceof Error ? err : new Error(String(err)));
      }
    }

    // For OpenAI streaming: inject stream_options.include_usage
    let modifiedInit = init;
    if (streaming && provider === "openai" && bodyStr) {
      modifiedInit = injectStreamUsage(bodyStr, init);
    } else if (!bodyStr && provider === "openai") {
      // Body couldn't be extracted (e.g., Request object with ReadableStream body).
      // OpenAI streaming won't include usage without stream_options.include_usage.
      // Non-streaming still works (usage in response JSON).
      onCostError(new Error(
        "Could not extract request body — OpenAI streaming usage will not be tracked. " +
        "Pass fetch(url, init) instead of fetch(request) for full tracking support.",
      ));
    }

    const startTime = performance.now();
    const response = await doFetch(input, modifiedInit ?? init);

    // No 429 interception in direct mode. In direct mode there is no NullSpend
    // proxy to send denial codes; the SDK handles enforcement locally (mandates,
    // budgets, session limits above). Trusting X-NullSpend-Denied from an
    // arbitrary LLM provider origin would be a trust-boundary violation.
    // Proxy-mode 429 interception lives in the isProxied() branch above.

    // Don't track errors
    if (!response.ok) return response;

    // Streaming response
    if (isStreamingResponse(response) && response.body) {
      return handleStreamingResponse(
        provider, model, response, startTime, metadata, costSink, onCostError,
      );
    }

    // Non-streaming response
    return handleNonStreamingResponse(
      provider, model, response, startTime, metadata, costSink, onCostError,
    );
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defaultCostErrorHandler(error: Error): void {
  console.warn("[nullspend] Cost tracking error:", error.message);
}

function safeDenied(
  onDenied: ((reason: import("./types.js").DenialReason) => void) | undefined,
  reason: import("./types.js").DenialReason,
  onCostError: (error: Error) => void,
): void {
  let result: unknown;
  try {
    result = onDenied?.(reason);
  } catch (cbErr) {
    // User's onDenied threw synchronously. Surface to onCostError, but
    // ALSO swallow if onCostError throws — a buggy onCostError must not
    // escape and prevent the typed error from being thrown by the caller
    // (audit Risk 4).
    try {
      onCostError(cbErr instanceof Error ? cbErr : new Error(String(cbErr)));
    } catch {
      // Both callbacks are buggy; nothing more we can do.
    }
  }
  // The DenialReason callback type is `(reason) => void`, but TypeScript
  // accepts `async (reason) => { ... }` as a valid `void`-returning function.
  // If the user wrote an async onDenied, attach a no-op catch so an async
  // rejection doesn't surface as an UnhandledPromiseRejectionWarning (or
  // crash on `--unhandled-rejections=strict`). Audit Risk 5.
  if (result && typeof (result as { catch?: unknown }).catch === "function") {
    (result as Promise<unknown>).catch((asyncErr) => {
      try {
        onCostError(asyncErr instanceof Error ? asyncErr : new Error(String(asyncErr)));
      } catch {
        // Same defensive swallow.
      }
    });
  }
}

type DenialPayload = {
  code: string;
  details: Record<string, unknown> | undefined;
  retryAfterSeconds: number | undefined;
  /**
   * Plan-upgrade URL surfaced by the proxy on `budget_exceeded` and
   * `customer_budget_exceeded` denials. Comes from `error.upgrade_url`
   * (top-level under `error`, not nested in `details`). Undefined when
   * the proxy didn't include one.
   */
  upgradeUrl: string | undefined;
};

/**
 * Parse a 429 response body for a NullSpend denial. Returns null if the
 * response isn't a NullSpend denial — caller treats null as "fall through,
 * return raw response".
 *
 * Identification is gated on the `X-NullSpend-Denied: 1` response header. The
 * proxy stamps this header on every denial Response (5 paths in shared.ts,
 * 4 in mcp.ts). Upstream provider 429s (OpenAI rate_limit_exceeded, Anthropic
 * rate limit) never carry this header so they fall through SILENTLY without
 * any body parsing. This is a stronger contract than the previous code-probing
 * approach: collisions are structurally impossible, not just empirically
 * unlikely.
 */
async function parseDenialPayload(
  response: Response,
): Promise<DenialPayload | null> {
  // Header gate — only NullSpend denials carry this. Upstream provider 429s
  // never have it, so they fall through with zero body parsing.
  if (response.headers.get("X-NullSpend-Denied") !== "1") return null;

  let json: Record<string, unknown>;
  try {
    const cloned = response.clone();
    json = await cloned.json() as Record<string, unknown>;
  } catch {
    return null; // Header was present but body is malformed — fall through
  }
  // Runtime-narrow errObj before treating it as a record. JSON parsing can
  // produce error: null, error: 0, error: false, error: "string", error: [], etc.
  const rawErrObj = json.error;
  if (!rawErrObj || typeof rawErrObj !== "object" || Array.isArray(rawErrObj)) return null;
  const errObj = rawErrObj as Record<string, unknown>;
  const code = errObj.code;
  if (typeof code !== "string") return null; // Non-string code — defensive
  // Same runtime narrowing for details: must be a plain object, not null/array/primitive.
  const rawDetails = errObj.details;
  const details = rawDetails && typeof rawDetails === "object" && !Array.isArray(rawDetails)
    ? rawDetails as Record<string, unknown>
    : undefined;
  // upgrade_url lives at error.upgrade_url (top-level under error, peer of
  // code/message/details). Per Phase 0 finish architecture decision 6, this
  // is the chosen location and the SDK reads from there only.
  const rawUpgradeUrl = errObj.upgrade_url;
  const upgradeUrl = typeof rawUpgradeUrl === "string" && rawUpgradeUrl.length > 0
    ? rawUpgradeUrl
    : undefined;
  const retryAfterRaw = parseInt(response.headers.get("Retry-After") ?? "", 10);
  return {
    code,
    details,
    upgradeUrl,
    // RFC 7231 Retry-After is a non-negative integer; reject negatives defensively.
    retryAfterSeconds: Number.isFinite(retryAfterRaw) && retryAfterRaw >= 0 ? retryAfterRaw : undefined,
  };
}

/**
 * Dispatch a parsed NullSpend denial: fire onDenied via safeDenied, then throw
 * the matching typed error. Handles 5 known codes; unknown codes are surfaced
 * to onCostError as a drift signal (caller returns raw response, no throw).
 *
 * Drift signaling is now safe because parseDenialPayload gates on the
 * X-NullSpend-Denied response header. Upstream provider 429s never carry that
 * header so they fall through before reaching this function. Anything that
 * reaches here is, by contract, a NullSpend denial — so an unknown code can
 * only mean the proxy shipped a code the SDK hasn't been updated to handle.
 */
function dispatchDenialCode(
  parsed: DenialPayload,
  customer: string | undefined,
  onDenied: ((reason: import("./types.js").DenialReason) => void) | undefined,
  onCostError: (error: Error) => void,
): void {
  const { code, details, retryAfterSeconds } = parsed;

  if (code === "budget_exceeded") {
    const entityType = details?.entity_type as string | undefined;
    const entityId = details?.entity_id as string | undefined;
    const limit = toFiniteNumber(details?.budget_limit_microdollars);
    const spend = toFiniteNumber(details?.budget_spend_microdollars);
    const remaining = Math.max(0, (limit ?? 0) - (spend ?? 0));
    safeDenied(onDenied, { type: "budget", remaining, entityType, entityId, limit, spend }, onCostError);
    throw new BudgetExceededError({
      remaining,
      entityType,
      entityId,
      limit,
      spend,
      upgradeUrl: parsed.upgradeUrl,
    });
  }

  if (code === "customer_budget_exceeded") {
    // Proxy emits a distinct code for customer-entity denials.
    // Details shape: { customer_id, budget_limit_microdollars, budget_spend_microdollars }
    // customer_id can be null per shared.ts:212; coerce to undefined.
    // Fall back to the SDK-side customer when the proxy didn't echo it.
    const rawCustomerId = details?.customer_id;
    const entityId = (typeof rawCustomerId === "string" ? rawCustomerId : undefined) ?? customer ?? undefined;
    const limit = toFiniteNumber(details?.budget_limit_microdollars);
    const spend = toFiniteNumber(details?.budget_spend_microdollars);
    const remaining = Math.max(0, (limit ?? 0) - (spend ?? 0));
    safeDenied(onDenied, { type: "budget", remaining, entityType: "customer", entityId, limit, spend }, onCostError);
    throw new BudgetExceededError({
      remaining,
      entityType: "customer",
      entityId,
      limit,
      spend,
      upgradeUrl: parsed.upgradeUrl,
    });
  }

  if (code === "velocity_exceeded") {
    // Proxy sends details: { limitMicrodollars, windowSeconds, currentMicrodollars }
    // Retry-after is in the Retry-After HTTP header, not in the JSON body
    const velLimit = toFiniteNumber(details?.limitMicrodollars);
    const velWindow = toFiniteNumber(details?.windowSeconds);
    const velCurrent = toFiniteNumber(details?.currentMicrodollars);
    safeDenied(onDenied, {
      type: "velocity",
      retryAfterSeconds,
      limit: velLimit,
      window: velWindow,
      current: velCurrent,
    }, onCostError);
    throw new VelocityExceededError({
      retryAfterSeconds,
      limit: velLimit,
      window: velWindow,
      current: velCurrent,
    });
  }

  if (code === "session_limit_exceeded") {
    const sessionSpend = toFiniteNumber(details?.session_spend_microdollars);
    const sessionLimit = toFiniteNumber(details?.session_limit_microdollars);
    safeDenied(onDenied, { type: "session_limit", sessionSpend: sessionSpend ?? 0, sessionLimit: sessionLimit ?? 0 }, onCostError);
    throw new SessionLimitExceededError(sessionSpend ?? 0, sessionLimit ?? 0);
  }

  if (code === "tag_budget_exceeded") {
    // Proxy sends budget_spend_microdollars, not remaining — compute remaining
    const tagKey = details?.tag_key as string | undefined;
    const tagValue = details?.tag_value as string | undefined;
    const limit = toFiniteNumber(details?.budget_limit_microdollars);
    const spend = toFiniteNumber(details?.budget_spend_microdollars);
    const remaining = Math.max(0, (limit ?? 0) - (spend ?? 0));
    safeDenied(onDenied, { type: "tag_budget", tagKey, tagValue, remaining, limit, spend }, onCostError);
    throw new TagBudgetExceededError({ tagKey, tagValue, remaining, limit, spend });
  }

  // Unknown code — by header-gate contract this is a drift signal (proxy
  // shipped a code the SDK doesn't support yet). Surface to onCostError so
  // operators see it; do NOT fire onDenied (we have nothing meaningful to
  // dispatch) and do NOT throw (caller's existing 429 handling should run).
  onCostError(new Error(
    `[nullspend] Proxy returned unknown denial code: ${code}. ` +
    `This is a drift signal — the proxy may be ahead of the SDK. ` +
    `Update @nullspend/sdk to the latest version.`,
  ));
}

function warnSessionDenied(
  spend: number,
  estimate: number,
  limit: number,
  sessionId: string,
  fallback = false,
): void {
  const tag = fallback ? " (fallback)" : "";
  console.warn(
    `[nullspend] Session limit denied${tag}: spend=${spend} estimate=${estimate} limit=${limit} sessionId=${sessionId}`,
  );
}

function resolveUrl(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function isProxied(
  url: string,
  init: RequestInit | undefined,
  proxyUrl: string | undefined,
  input: RequestInfo | URL,
): boolean {
  // Configurable proxyUrl match via URL origin (not substring — substring
  // matching allows "https://host.evil.com" to spoof "https://host").
  // The client constructor validates proxyUrl with `new URL()`, so we only
  // need to handle parse failures on the request URL side defensively.
  //
  // IMPORTANT: origin comparison is strict on PORT. If your proxy listens
  // on a non-default port (e.g. https://proxy.example.com:8443) you MUST
  // include that port in the proxyUrl you pass to the SDK constructor.
  // Mismatched ports cause the SDK to take the direct path and double-count
  // cost events. This is intentional — port-normalization would mask real
  // misconfigurations.
  if (proxyUrl) {
    try {
      const requestOrigin = new URL(url).origin;
      const proxyOrigin = new URL(proxyUrl).origin;
      if (requestOrigin === proxyOrigin) return true;
    } catch {
      // Invalid URL on request side — fall through to header check
    }
  }
  // Header-based detection is ONLY used when proxyUrl is configured AND the
  // URL origin check above failed (e.g., the proxy URL was set without a port
  // but the request includes one). Without a configured proxyUrl, the presence
  // of x-nullspend-key alone is NOT sufficient — a caller could add the header
  // to a direct OpenAI request and bypass all SDK tracking. (Codex finding #1)
  if (!proxyUrl) return false;

  // With proxyUrl configured but origin mismatch (port difference, etc.),
  // fall back to header check as a secondary signal.
  if (init?.headers) {
    const headers = init.headers;
    if (headers instanceof Headers) {
      if (headers.has("x-nullspend-key")) return true;
    } else if (Array.isArray(headers)) {
      if (headers.some(([k]) => k.toLowerCase() === "x-nullspend-key")) return true;
    } else if (typeof headers === "object") {
      if ("x-nullspend-key" in headers) return true;
    }
  }
  // Edge case: caller passed a Request with x-nullspend-key in its headers
  // and no init (or init without headers). Per WHATWG fetch, init.headers
  // REPLACES the Request's headers entirely when set, so we only fall back
  // to inspecting the Request's own headers when the caller didn't override.
  if (input instanceof Request && !init?.headers && input.headers.has("x-nullspend-key")) {
    return true;
  }
  return false;
}

/**
 * Coerce an unknown value to a finite number, or undefined if not coercible.
 * Guards against proxy details fields being strings or non-finite values.
 */
function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

/**
 * Add a header to a RequestInit, returning a new RequestInit. Handles all three
 * header formats (Headers object, array of tuples, plain object) without
 * mutating the caller's input.
 *
 * Case-insensitively dedupes existing entries with the same name so we never
 * emit duplicate headers (which have ambiguous semantics across fetch runtimes).
 */
function addHeader(init: RequestInit | undefined, name: string, value: string): RequestInit {
  const existing = init?.headers;
  if (existing instanceof Headers) {
    const next = new Headers(existing);
    next.set(name, value);  // Headers.set() is case-insensitive per spec
    return { ...init, headers: next };
  }
  const lowerName = name.toLowerCase();
  if (Array.isArray(existing)) {
    // Strip any existing entries with the same header name (case-insensitive),
    // then append the new one.
    const filtered = existing.filter(([k]) => k.toLowerCase() !== lowerName);
    return { ...init, headers: [...filtered, [name, value]] };
  }
  if (existing && typeof existing === "object") {
    // Strip any existing keys with the same name (case-insensitive),
    // then set the new one. Preserves the canonical casing the caller uses.
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(existing as Record<string, string>)) {
      if (k.toLowerCase() !== lowerName) next[k] = v;
    }
    next[name] = value;
    return { ...init, headers: next };
  }
  return { ...init, headers: { [name]: value } };
}

async function extractBody(input: RequestInfo | URL, init?: RequestInit): Promise<string | null> {
  // init.body takes precedence (per WHATWG fetch spec: init replaces Request fields)
  if (init?.body !== undefined) {
    return typeof init.body === "string" ? init.body : null;
  }
  // Request body fallback: clone so the original body stream stays consumable
  // for the actual fetch call. Provider SDKs typically pass (url, init), but
  // some callers pass a pre-built Request object.
  if (input instanceof Request) {
    try {
      return await input.clone().text();
    } catch {
      return null;
    }
  }
  return null;
}

function injectStreamUsage(bodyStr: string, init?: RequestInit): RequestInit | undefined {
  try {
    const parsed = JSON.parse(bodyStr);
    // Merge with existing stream_options, don't overwrite
    const streamOptions = parsed.stream_options ?? {};
    streamOptions.include_usage = true;
    parsed.stream_options = streamOptions;
    const newBody = JSON.stringify(parsed);

    // Clone init, replace body, strip Content-Length (will be recalculated)
    const newInit = { ...init, body: newBody };
    if (newInit.headers) {
      const headers = new Headers(newInit.headers as HeadersInit);
      headers.delete("content-length");
      newInit.headers = headers;
    }
    return newInit;
  } catch {
    return init;
  }
}

function estimateCostMicrodollars(
  provider: string,
  model: string,
  bodyStr: string | null,
): number {
  const pricing = getModelPricing(provider, model);
  if (!pricing) return 0;

  // Rough estimate: use max_tokens from body for output, 1000 for input
  let maxTokens = 4096;
  if (bodyStr) {
    try {
      const parsed = JSON.parse(bodyStr);
      if (typeof parsed.max_tokens === "number") maxTokens = parsed.max_tokens;
      if (typeof parsed.max_completion_tokens === "number") maxTokens = parsed.max_completion_tokens;
    } catch {
      // ignore
    }
  }

  const inputEstimate = 1000 * pricing.inputPerMTok;
  const outputEstimate = maxTokens * pricing.outputPerMTok;
  return Math.round(inputEstimate + outputEstimate);
}

async function handleStreamingResponse(
  provider: TrackedProvider,
  model: string,
  response: Response,
  startTime: number,
  metadata: { sessionId?: string; traceId?: string; tags?: Record<string, string>; customer?: string },
  queueCost: (event: CostEventInput) => void,
  onCostError?: (error: Error) => void,
): Promise<Response> {
  const body = response.body!;

  if (provider === "openai") {
    const { readable, resultPromise } = createOpenAISSEParser(body);

    // Fire-and-forget cost tracking
    resultPromise
      .then((result) => {
        if (!result.usage) return; // cancelled or no usage
        const durationMs = Math.round(performance.now() - startTime);
        const resolvedModel = result.model ?? model;
        const costEvent = calculateOpenAICostEvent(resolvedModel, result.usage, durationMs, metadata);
        queueCost(costEvent);
      })
      .catch((err) => {
        onCostError?.(err instanceof Error ? err : new Error(String(err)));
      });

    return new Response(readable, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  // Anthropic
  const { readable, resultPromise } = createAnthropicSSEParser(body);

  resultPromise
    .then((result) => {
      if (!result.usage) return;
      const durationMs = Math.round(performance.now() - startTime);
      const resolvedModel = result.model ?? model;
      const costEvent = calculateAnthropicCostEvent(
        resolvedModel, result.usage, result.cacheCreationDetail, durationMs, metadata,
      );
      queueCost(costEvent);
    })
    .catch((err) => {
      onCostError?.(err instanceof Error ? err : new Error(String(err)));
    });

  return new Response(readable, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

async function handleNonStreamingResponse(
  provider: TrackedProvider,
  model: string,
  response: Response,
  startTime: number,
  metadata: { sessionId?: string; traceId?: string; tags?: Record<string, string>; customer?: string },
  queueCost: (event: CostEventInput) => void,
  onCostError?: (error: Error) => void,
): Promise<Response> {
  try {
    const cloned = response.clone();
    const json = await cloned.json();
    const durationMs = Math.round(performance.now() - startTime);

    // Extract model from response if available
    const responseModel =
      json && typeof json === "object" && typeof (json as Record<string, unknown>).model === "string"
        ? (json as Record<string, unknown>).model as string
        : null;
    const resolvedModel = responseModel ?? model;

    if (provider === "openai") {
      const usage = extractOpenAIUsageFromJSON(json);
      if (usage) {
        const costEvent = calculateOpenAICostEvent(resolvedModel, usage, durationMs, metadata);
        queueCost(costEvent);
      }
    } else {
      const result = extractAnthropicUsageFromJSON(json);
      if (result) {
        const costEvent = calculateAnthropicCostEvent(
          resolvedModel, result.usage, result.cacheDetail, durationMs, metadata,
        );
        queueCost(costEvent);
      }
    }
  } catch (err) {
    onCostError?.(err instanceof Error ? err : new Error(String(err)));
  }

  return response;
}
