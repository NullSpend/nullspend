// ---------------------------------------------------------------------------
// Policy types
// ---------------------------------------------------------------------------

export interface PolicyBudget {
  remaining_microdollars: number;
  max_microdollars: number;
  spend_microdollars: number;
  period_end: string | null;
  entity_type: string;
  entity_id: string;
  finalization_reserve_microdollars?: number;
  policy?: string;
}

interface CheapestModel {
  model: string;
  provider?: string;
  input_per_mtok: number;
  output_per_mtok: number;
}

export interface PolicyResponse {
  budget: PolicyBudget | null;
  allowed_models: string[] | null;
  allowed_providers: string[] | null;
  cheapest_per_provider: Record<string, CheapestModel> | null;
  cheapest_overall: CheapestModel | null;
  restrictions_active: boolean;
  session_limit_microdollars?: number | null;
}

// ---------------------------------------------------------------------------
// Policy cache
// ---------------------------------------------------------------------------

export interface PolicyCache {
  /** Get cached policy, fetching if stale. */
  getPolicy(): Promise<PolicyResponse | null>;
  /** Check if model/provider is allowed by mandates. */
  checkMandate(provider: string, model: string): {
    allowed: boolean;
    mandate?: string;
    requested?: string;
    allowed_list?: string[];
  };
  /** Check if estimated cost fits within budget. */
  checkBudget(estimateMicrodollars: number, options?: { finalize?: boolean }): {
    allowed: boolean;
    remaining?: number;
    entityType?: string;
    entityId?: string;
    limit?: number;
    spend?: number;
  };
  /** Get the session limit from cached policy, or null if not set. */
  getSessionLimit(): number | null;
  /** Get the finalization reserve from cached policy, or null if not set. */
  getFinalizationReserve(): number | null;
  /** Invalidate cached policy. */
  invalidate(): void;
}

const DEFAULT_TTL_MS = 60_000;
const DEFAULT_FETCH_TIMEOUT_MS = 5_000;

/**
 * Create a single-entry policy cache that fetches from the dashboard
 * and caches for the configured TTL.
 *
 * `fetchTimeoutMs` bounds how long any single policy fetch can pend before the
 * cache falls open. The SDK's `requestTimeoutMs` covers individual API calls
 * but not the cached promise waiters, so an unresponsive policy endpoint would
 * otherwise block enforcement-path callers indefinitely. (PERF-2)
 */
export function createPolicyCache(
  fetchPolicy: () => Promise<PolicyResponse>,
  ttlMs: number = DEFAULT_TTL_MS,
  onError?: (error: Error) => void,
  fetchTimeoutMs: number = DEFAULT_FETCH_TIMEOUT_MS,
): PolicyCache {
  let cached: PolicyResponse | null = null;
  let cachedAt = 0;
  let inflightPromise: Promise<PolicyResponse | null> | null = null;

  async function getPolicy(): Promise<PolicyResponse | null> {
    const now = Date.now();
    if (cached && now - cachedAt < ttlMs) return cached;

    // Dedup in-flight fetches
    if (inflightPromise) return inflightPromise;

    const fetchWithTimeout: Promise<PolicyResponse> = fetchTimeoutMs > 0
      ? new Promise<PolicyResponse>((resolve, reject) => {
          const timer = setTimeout(
            () => reject(new Error(`policy fetch timed out after ${fetchTimeoutMs}ms`)),
            fetchTimeoutMs,
          );
          fetchPolicy().then(
            (value) => {
              clearTimeout(timer);
              resolve(value);
            },
            (err) => {
              clearTimeout(timer);
              reject(err);
            },
          );
        })
      : fetchPolicy();

    inflightPromise = fetchWithTimeout
      .then((policy) => {
        cached = policy;
        cachedAt = Date.now();
        return policy;
      })
      .catch((err) => {
        // Fetch failure (including timeout) falls open — return stale cache or null.
        // Surface the error so operators know the policy endpoint is unreachable.
        onError?.(err instanceof Error ? err : new Error(String(err)));
        return cached;
      })
      .finally(() => {
        inflightPromise = null;
      });

    return inflightPromise;
  }

  function checkMandate(
    provider: string,
    model: string,
  ): {
    allowed: boolean;
    mandate?: string;
    requested?: string;
    allowed_list?: string[];
  } {
    if (!cached) return { allowed: true };

    // Check allowed providers
    if (cached.allowed_providers !== null) {
      if (!cached.allowed_providers.includes(provider)) {
        return {
          allowed: false,
          mandate: "allowed_providers",
          requested: provider,
          allowed_list: cached.allowed_providers,
        };
      }
    }

    // Check allowed models
    if (cached.allowed_models !== null) {
      if (!cached.allowed_models.includes(model)) {
        return {
          allowed: false,
          mandate: "allowed_models",
          requested: model,
          allowed_list: cached.allowed_models,
        };
      }
    }

    return { allowed: true };
  }

  function checkBudget(estimateMicrodollars: number, options?: { finalize?: boolean }): {
    allowed: boolean;
    remaining?: number;
    entityType?: string;
    entityId?: string;
    limit?: number;
    spend?: number;
  } {
    if (!cached || !cached.budget) return { allowed: true };

    const b = cached.budget;
    const reserve = b.finalization_reserve_microdollars ?? 0;
    const policy = b.policy ?? "strict_block";
    const skipReserve = options?.finalize || policy !== "strict_block";
    const effectiveRemaining = skipReserve
      ? b.remaining_microdollars
      : Math.max(0, b.remaining_microdollars - reserve);
    if (estimateMicrodollars > effectiveRemaining) {
      return {
        allowed: false,
        remaining: effectiveRemaining,
        entityType: b.entity_type,
        entityId: b.entity_id,
        limit: b.max_microdollars,
        spend: b.spend_microdollars,
      };
    }
    return { allowed: true, remaining: effectiveRemaining };
  }

  function getSessionLimit(): number | null {
    if (!cached) return null;
    return cached.session_limit_microdollars ?? null;
  }

  function getFinalizationReserve(): number | null {
    if (!cached || !cached.budget) return null;
    return cached.budget.finalization_reserve_microdollars ?? null;
  }

  function invalidate(): void {
    cached = null;
    cachedAt = 0;
  }

  return { getPolicy, checkMandate, checkBudget, getSessionLimit, getFinalizationReserve, invalidate };
}
