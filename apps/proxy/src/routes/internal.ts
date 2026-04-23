import { invalidateAuthCacheForOwner, authenticateApiKey } from "../lib/api-key-auth.js";
import { doBudgetRemove, doBudgetResetSpend, doBudgetUpsertEntities, doBudgetGetVelocityState, doIncrementPlanCounter } from "../lib/budget-do-client.js";
import { lookupBudgetsForDO } from "../lib/budget-do-lookup.js";
import { errorResponse } from "../lib/errors.js";
import { optionalBinding } from "../lib/env.js";
import { emitMetric } from "../lib/metrics.js";
import { timingSafeStringEqual } from "../lib/timing-safe-equal.js";
import { retrieveBodies } from "../lib/body-storage.js";
import { resolvePeriodBounds } from "../lib/period-math.js";

/**
 * PR-2d / Decision #40: dual-secret authentication for internal endpoints.
 *
 * Accept Bearer tokens matching `INTERNAL_SECRET` OR `INTERNAL_SECRET_NEXT`.
 * During rotation, ops sets `INTERNAL_SECRET_NEXT` to the current value (on both
 * proxy and dashboard sides) and rolls the dashboard's `PROXY_INTERNAL_SECRET`
 * to the new value. When cutover completes, ops clears `INTERNAL_SECRET_NEXT`
 * and the proxy rotates `INTERNAL_SECRET` to match.
 *
 * Short-circuits on first match — the constant-time guarantee is per-secret
 * byte compare, which each `timingSafeStringEqual` call provides. Revealing
 * which-of-two-valid-secrets-matched does NOT help an attacker guess either.
 *
 * Returns a `Response` on failure (500/401), or `null` on success.
 *
 * Typed via cast because `INTERNAL_SECRET_NEXT` is not yet in the auto-generated
 * `worker-configuration.d.ts` (opt-in per-deploy secret, added to `.dev.vars.example`
 * for docs). Runtime presence is enforced by `typeof === "string" && length > 0`.
 */
export function authenticateInternal(request: Request, env: Env): Response | null {
  const e = env as unknown as { INTERNAL_SECRET?: string; INTERNAL_SECRET_NEXT?: string };
  const secrets = [e.INTERNAL_SECRET, e.INTERNAL_SECRET_NEXT].filter(
    (s): s is string => typeof s === "string" && s.length > 0,
  );
  if (secrets.length === 0) {
    console.error("[internal] INTERNAL_SECRET not configured");
    return errorResponse("internal_error", "Server misconfigured", 500);
  }

  const authHeader = request.headers.get("authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return errorResponse("unauthorized", "Missing or malformed Authorization header", 401);
  }

  const token = authHeader.slice(7);
  for (const secret of secrets) {
    if (timingSafeStringEqual(token, secret)) return null;
  }
  return errorResponse("unauthorized", "Invalid token", 401);
}

interface InvalidationBody {
  action: "remove" | "reset_spend" | "sync" | "auth_only";
  ownerId: string;
  entityType: string;
  entityId: string;
  sentAt?: number;
}

const MAX_FIELD_LENGTH = 256;

function isNonEmptyString(val: unknown): val is string {
  return typeof val === "string" && val.trim().length > 0 && val.length <= MAX_FIELD_LENGTH;
}

function parseBody(raw: unknown): InvalidationBody | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const obj = raw as Record<string, unknown>;

  if (
    obj.action !== "remove" &&
    obj.action !== "reset_spend" &&
    obj.action !== "sync" &&
    obj.action !== "auth_only"
  ) return null;
  if (!isNonEmptyString(obj.ownerId)) return null;

  // auth_only: skip the DO sync entirely. Used by metadata-only updates
  // (like upgradeUrl) where there's no budget entity to sync. entityType
  // and entityId may be omitted; if present, they're ignored.
  if (obj.action === "auth_only") {
    return {
      action: "auth_only",
      ownerId: obj.ownerId.trim(),
      entityType: "",
      entityId: "",
      ...(typeof obj.sentAt === "number" && { sentAt: obj.sentAt }),
    };
  }

  if (!isNonEmptyString(obj.entityType)) return null;
  if (!isNonEmptyString(obj.entityId)) return null;

  return {
    action: obj.action,
    ownerId: obj.ownerId.trim(),
    entityType: obj.entityType.trim(),
    entityId: obj.entityId.trim(),
    ...(typeof obj.sentAt === "number" && { sentAt: obj.sentAt }),
  };
}

export async function handleBudgetInvalidation(
  request: Request,
  env: Env,
): Promise<Response> {
  const authFail = authenticateInternal(request, env);
  if (authFail) return authFail;

  // Parse and validate body
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return errorResponse("bad_request", "Invalid JSON body", 400);
  }

  const body = parseBody(raw);
  if (!body) {
    return errorResponse(
      "bad_request",
      "Missing or invalid fields: action (remove|reset_spend|sync|auth_only), ownerId. entityType+entityId required for non-auth_only actions.",
      400,
    );
  }

  // Execute
  try {
    if (body.action === "auth_only") {
      // Metadata-only invalidation: no DO sync, just clear the auth cache.
      // Used by upgradeUrl edits and other org-metadata changes that
      // affect the cached identity but no budget entity.
      // Falls through to the invalidateAuthCacheForOwner call below.
    } else if (body.action === "remove") {
      await doBudgetRemove(env, body.ownerId, body.entityType, body.entityId);
    } else if (body.action === "reset_spend") {
      await doBudgetResetSpend(env, body.ownerId, body.entityType, body.entityId);
    } else {
      // action === "sync": look up specific entity from Postgres and upsert into DO
      // Uses populateIfEmpty (single-entity upsert) — does NOT purge sibling budgets
      const connectionString = env.HYPERDRIVE.connectionString;
      let identity: { keyId: string | null; orgId: string | null; userId: string | null; tags: Record<string, string> };
      if (body.entityType === "tag") {
        const eqIdx = body.entityId.indexOf("=");
        const tagObj = eqIdx > 0
          ? { [body.entityId.slice(0, eqIdx)]: body.entityId.slice(eqIdx + 1) }
          : { [body.entityId]: "" };
        identity = { keyId: null, orgId: body.ownerId, userId: null, tags: tagObj };
      } else if (body.entityType === "customer") {
        // customer entities: pass the customer ID via tags["customer"] so
        // lookupBudgetsForDO's customer branch finds it.
        identity = { keyId: null, orgId: body.ownerId, userId: null, tags: { customer: body.entityId } };
      } else if (body.entityType === "user") {
        // entityId is the userId for "user" entity budgets
        identity = { keyId: null, orgId: body.ownerId, userId: body.entityId, tags: {} };
      } else {
        // api_key entities: entityId is the key ID
        identity = { keyId: body.entityId, orgId: body.ownerId, userId: null, tags: {} };
      }
      const entities = await lookupBudgetsForDO(connectionString, identity);
      // Empty on sync is unexpected (removes use action: "remove") — may indicate
      // Postgres commit timing or stale reads.
      if (entities.length === 0) {
        console.warn(
          `[internal] sync returned 0 entities from Postgres`,
          { ownerId: body.ownerId, entityType: body.entityType, entityId: body.entityId },
        );
        emitMetric("budget_sync_empty", {
          ownerId: body.ownerId,
          entityType: body.entityType,
          entityId: body.entityId,
        });
      }
      await doBudgetUpsertEntities(env, body.ownerId, entities);
    }

    invalidateAuthCacheForOwner(body.ownerId);

    if (body.sentAt) {
      emitMetric("budget_sync_latency_ms", { ms: Math.max(0, Date.now() - body.sentAt), action: body.action });
    }

    emitMetric("budget_invalidation", {
      action: body.action,
      ownerId: body.ownerId,
      entityType: body.entityType,
      entityId: body.entityId,
      status: "ok",
    });

    return Response.json({ ok: true });
  } catch (err) {
    console.error("[internal] Budget invalidation failed:", err);

    emitMetric("budget_invalidation", {
      action: body.action,
      ownerId: body.ownerId,
      entityType: body.entityType,
      entityId: body.entityId,
      status: "error",
    });

    return errorResponse("internal_error", "Invalidation failed", 500);
  }
}

export async function handleVelocityState(
  request: Request,
  env: Env,
): Promise<Response> {
  const authFail = authenticateInternal(request, env);
  if (authFail) return authFail;

  // Read ownerId from query param
  const url = new URL(request.url);
  const ownerId = url.searchParams.get("ownerId");
  if (!ownerId || ownerId.length === 0 || ownerId.length > MAX_FIELD_LENGTH) {
    return errorResponse("bad_request", "Missing or invalid ownerId query parameter", 400);
  }

  try {
    const velocityState = await doBudgetGetVelocityState(env, ownerId);

    emitMetric("velocity_state_lookup", {
      ownerId,
      count: velocityState.length,
      status: "ok",
    });

    return Response.json({ velocityState });
  } catch (err) {
    console.error("[internal] Velocity state lookup failed:", err);

    emitMetric("velocity_state_lookup", {
      ownerId,
      status: "error",
    });

    return errorResponse("internal_error", "Velocity state lookup failed", 500);
  }
}

/**
 * GET /internal/request-bodies/:requestId?ownerId=...
 *
 * Retrieves stored request/response bodies from R2.
 * Auth: INTERNAL_SECRET (same as other internal endpoints).
 * The ownerId query param scopes the R2 key prefix — the dashboard
 * must supply the ownerId that owns the cost event.
 */
export async function handleRequestBodies(
  request: Request,
  env: Env,
): Promise<Response> {
  const authFail = authenticateInternal(request, env);
  if (authFail) return authFail;

  const url = new URL(request.url);
  const ownerId = url.searchParams.get("ownerId");
  if (!ownerId || ownerId.length === 0 || ownerId.length > MAX_FIELD_LENGTH) {
    return errorResponse("bad_request", "Missing or invalid ownerId query parameter", 400);
  }

  // Extract requestId from path: /internal/request-bodies/{requestId}
  const pathParts = url.pathname.split("/");
  const requestId = pathParts[pathParts.length - 1];
  if (!requestId || requestId.length === 0 || requestId.length > MAX_FIELD_LENGTH) {
    return errorResponse("bad_request", "Missing or invalid requestId in path", 400);
  }

  // Defense-in-depth: reject path traversal characters in R2 key components
  const SAFE_ID_RE = /^[a-zA-Z0-9_\-.:]+$/;
  if (!SAFE_ID_RE.test(ownerId) || !SAFE_ID_RE.test(requestId)) {
    return errorResponse("bad_request", "ownerId and requestId must be alphanumeric", 400);
  }

  const bodyBucket = optionalBinding(env, "BODY_STORAGE");
  if (!bodyBucket) {
    return errorResponse("internal_error", "Body storage not configured", 500);
  }

  try {
    const bodies = await retrieveBodies(bodyBucket, ownerId, requestId);

    emitMetric("body_storage_read", {
      hasRequest: bodies.requestBody !== null,
      hasResponse: bodies.responseBody !== null,
      responseFormat: bodies.responseFormat,
    });

    // Parse each body independently — one corrupt body should not prevent retrieval of the other
    let requestBody: unknown = null;
    let responseBody: unknown = null;
    if (bodies.requestBody) {
      try { requestBody = JSON.parse(bodies.requestBody); }
      catch { requestBody = null; }
    }
    if (bodies.responseBody) {
      if (bodies.responseFormat === "sse") {
        // Wrap raw SSE text so the dashboard can detect and render it appropriately
        responseBody = { _format: "sse", text: bodies.responseBody };
      } else {
        try { responseBody = JSON.parse(bodies.responseBody); }
        catch { responseBody = null; }
      }
    }

    return Response.json({ requestBody, responseBody });
  } catch (err) {
    console.error("[internal] Request body retrieval failed:", err);
    return errorResponse("internal_error", "Body retrieval failed", 500);
  }
}

// ---------------------------------------------------------------------------
// POST /internal/plan-counter/increment  (PR-2d / Decision #26)
// ---------------------------------------------------------------------------

/**
 * Body schema: `{ apiKey: string, idempotencyKeys: string[] (1–100) }`.
 *
 * Fail-open reconciliation from the dashboard's `/api/cost-events` path — the
 * dashboard holds the caller's raw API key (SDK direct-mode), computes per-event
 * idempotency keys as `sha256Hex(JSON.stringify([requestId, provider]))` (64 hex
 * chars, well under the DO's 256-char limit + collision-safe against delimiters),
 * and POSTs the batch here. The proxy resolves the apiKey to an identity
 * server-side — caller-supplied orgId would be an attack surface (C36).
 */
const PLAN_COUNTER_APIKEY_MAX_LENGTH = 256;
const PLAN_COUNTER_IDEMPOTENCY_KEY_MAX_LENGTH = 256;  // matches DO's MAX_IDEMPOTENCY_KEY_LENGTH
const PLAN_COUNTER_MAX_KEYS = 100;                    // matches costEventBatchInputSchema cap

interface PlanCounterIncrementBody {
  apiKey: string;
  idempotencyKeys: string[];
}

function parsePlanCounterBody(raw: unknown): PlanCounterIncrementBody | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const obj = raw as Record<string, unknown>;

  if (typeof obj.apiKey !== "string") return null;
  if (obj.apiKey.length === 0 || obj.apiKey.length > PLAN_COUNTER_APIKEY_MAX_LENGTH) return null;

  if (!Array.isArray(obj.idempotencyKeys)) return null;
  const keys = obj.idempotencyKeys;
  if (keys.length === 0 || keys.length > PLAN_COUNTER_MAX_KEYS) return null;
  for (const k of keys) {
    if (typeof k !== "string") return null;
    if (k.length === 0 || k.length > PLAN_COUNTER_IDEMPOTENCY_KEY_MAX_LENGTH) return null;
  }

  return { apiKey: obj.apiKey, idempotencyKeys: keys as string[] };
}

/**
 * POST /internal/plan-counter/increment
 *
 * Increment the DO plan-counter for each supplied idempotency key. Replay-safe:
 * DO-side dedup ensures a previously-seen key is a no-op. Returns the final
 * call's result so the dashboard helper can observe current count/status.
 *
 * Auth: dual-secret Bearer (INTERNAL_SECRET or INTERNAL_SECRET_NEXT).
 * Scope: derived from `apiKey` — caller cannot target arbitrary orgs.
 */
export async function handlePlanCounterIncrement(
  request: Request,
  env: Env,
): Promise<Response> {
  const authFail = authenticateInternal(request, env);
  if (authFail) return authFail;

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return errorResponse("bad_request", "Invalid JSON body", 400);
  }

  const body = parsePlanCounterBody(raw);
  if (!body) {
    return errorResponse(
      "bad_request",
      "Body must be { apiKey: string, idempotencyKeys: string[] (1–100, each 1–256 chars) }",
      400,
    );
  }

  const connectionString = env.HYPERDRIVE.connectionString;

  // Resolve identity from the caller-supplied apiKey. The hash → DB lookup
  // + positive/negative caching already live inside `authenticateApiKey`.
  let identity;
  try {
    identity = await authenticateApiKey(body.apiKey, connectionString, env);
  } catch (err) {
    console.error("[internal] plan-counter apiKey lookup failed:", err);
    emitMetric("plan_counter_endpoint", { status: "db_error" });
    return errorResponse("internal_error", "Identity lookup failed", 503);
  }
  if (!identity) {
    emitMetric("plan_counter_endpoint", { status: "unknown_key" });
    return errorResponse("unauthorized", "Unknown API key", 401);
  }

  const { periodStart, periodEnd } = resolvePeriodBounds(identity, Date.now());

  // Per-key loop. Each iteration is its own DO transactionSync — a mid-batch
  // throw leaves prior increments durably committed (replay-safe via dedup).
  // The LAST result is the return value; `count` reflects the current DO total.
  const startMs = Date.now();
  let lastResult: Awaited<ReturnType<typeof doIncrementPlanCounter>> | null = null;
  try {
    for (const key of body.idempotencyKeys) {
      lastResult = await doIncrementPlanCounter(env, {
        orgId: identity.orgId,
        planLimitBlockAt: identity.planLimitBlockAt,
        planLimitMode: identity.planLimitMode,
        tier: identity.tierLabel,
        periodStart,
        periodEnd,
        idempotencyKey: key,
      });
    }
  } catch (err) {
    console.error("[internal] plan-counter DO increment failed:", err);
    emitMetric("plan_counter_endpoint", {
      status: "do_error",
      tier: identity.tierLabel,
      keyCount: body.idempotencyKeys.length,
    });
    return errorResponse("internal_error", "Plan counter increment failed", 500);
  }

  emitMetric("plan_counter_endpoint", {
    status: "ok",
    finalStatus: lastResult?.status ?? "unknown",
    tier: identity.tierLabel,
    keyCount: body.idempotencyKeys.length,
    durationMs: Date.now() - startMs,
  });

  // Return last call's result verbatim (discriminated union carries count/blockAt/reason).
  return Response.json(lastResult ?? { status: "skipped", reason: "no_keys" });
}
