import { waitUntil } from "cloudflare:workers";
import type { RequestContext } from "../lib/context.js";
import { errorResponse } from "../lib/errors.js";
import { lookupBudgetsForDO, lookupCustomerUpgradeUrl, type BudgetEntity } from "../lib/budget-do-lookup.js";
import { logCostEventsBatchQueued, getCostEventQueue } from "../lib/cost-event-queue.js";
import { checkBudget, reconcileBudgetQueued, getReconcileQueue } from "../lib/budget-orchestrator.js";
import { getWebhookEndpoints, getWebhookEndpointsWithSecrets } from "../lib/webhook-cache.js";
import { buildCostEventPayload, buildThinCostEventPayload, buildVelocityExceededPayload, buildSessionLimitExceededPayload, buildTagBudgetExceededPayload, buildCustomerBudgetExceededPayload, buildBudgetExceededPayload } from "../lib/webhook-events.js";
import { detectThresholdCrossings } from "../lib/webhook-thresholds.js";
import { dispatchToEndpoints } from "../lib/webhook-dispatch.js";
import { expireRotatedSecrets } from "../lib/webhook-expiry.js";
import { UUID_RE } from "../lib/validation.js";
import { emitMetric } from "../lib/metrics.js";
import { dispatchDenialWebhook, dispatchVelocityRecoveryWebhooks, buildBudgetHeaders } from "./shared.js";
import { resolveUpgradeUrl } from "../lib/upgrade-url.js";

// ---------------------------------------------------------------------------
// POST /v1/mcp/budget/check
// ---------------------------------------------------------------------------

interface BudgetCheckBody {
  toolName: string;
  serverName: string;
  estimateMicrodollars: number;
}

function validateBudgetCheckBody(
  body: Record<string, unknown>,
): BudgetCheckBody | null {
  if (
    typeof body.toolName !== "string" ||
    body.toolName.length === 0 ||
    typeof body.serverName !== "string" ||
    body.serverName.length === 0 ||
    typeof body.estimateMicrodollars !== "number" ||
    !Number.isFinite(body.estimateMicrodollars) ||
    body.estimateMicrodollars < 0
  ) {
    return null;
  }
  return body as unknown as BudgetCheckBody;
}

export async function handleMcpBudgetCheck(
  request: Request,
  env: Env,
  ctx: RequestContext,
): Promise<Response> {
  const parsed = validateBudgetCheckBody(ctx.body);
  if (!parsed) {
    const resp = errorResponse(
      "bad_request",
      "Body must include toolName (string), serverName (string), estimateMicrodollars (non-negative number)",
      400,
    );
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    if (ctx.sessionId) resp.headers.set("X-NullSpend-Session", ctx.sessionId);
    return resp;
  }

  try {
    const budgetStartMs = performance.now();
    const outcome = await checkBudget(env, ctx, parsed.estimateMicrodollars);
    if (ctx.stepTiming) ctx.stepTiming.budgetCheckMs = Math.round(performance.now() - budgetStartMs);

    // On denial we don't reserve against the DO, so budget headers reflect
    // the state right before this (rejected) request.
    const budgetEntities = outcome.budgetEntities;
    const budgetHeaders = buildBudgetHeaders(budgetEntities, 0);

    // Metric emission deferred to the per-branch returns below so the
    // upgradeUrlEmitted tag reflects the actual response. See E5.
    const reason = outcome.status === "denied"
      ? (outcome.velocityDenied ? "velocity_exceeded"
         : outcome.sessionLimitDenied ? "session_limit_exceeded"
         : outcome.tagBudgetDenied ? "tag_budget_exceeded"
         : outcome.deniedEntityType === "customer" ? "customer_budget_exceeded"
         : "budget_exceeded")
      : null;
    function emitDenialMetric(upgradeUrlEmitted: boolean, upgradeUrlSource: "per_customer" | "org" | "none"): void {
      emitMetric("budget_denied", {
        reason: reason ?? "unknown",
        provider: "mcp",
        entityType: outcome.deniedEntityType ?? "unknown",
        upgradeUrlEmitted,
        upgradeUrlSource,
      });
    }

    if (outcome.status === "denied" && outcome.velocityDenied) {
      emitDenialMetric(false, "none");
      dispatchDenialWebhook(ctx, env, "[mcp-route]", () =>
        buildVelocityExceededPayload({
          budgetEntityType: outcome.deniedEntityType ?? "unknown",
          budgetEntityId: outcome.deniedEntityId ?? "unknown",
          velocityLimitMicrodollars: outcome.velocityDetails?.limitMicrodollars ?? 0,
          velocityWindowSeconds: outcome.velocityDetails?.windowSeconds ?? 60,
          velocityCurrentMicrodollars: outcome.velocityDetails?.currentMicrodollars ?? 0,
          cooldownSeconds: outcome.retryAfterSeconds ?? 60,
          model: `${parsed.serverName}/${parsed.toolName}`,
          provider: "mcp",
        }, ctx.auth.apiVersion),
      );
      return new Response(
        JSON.stringify({
          error: {
            code: "velocity_exceeded",
            message: "Request blocked: spending rate exceeds velocity limit. Retry after cooldown.",
            details: outcome.velocityDetails ?? null,
          },
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "NullSpend-Version": ctx.resolvedApiVersion,
            "Retry-After": String(outcome.retryAfterSeconds ?? 60),
            "X-NullSpend-Trace-Id": ctx.traceId,
            "X-NullSpend-Denied": "1",
            ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
            ...budgetHeaders,
          },
        },
      );
    }

    // Session limit denial
    if (outcome.status === "denied" && outcome.sessionLimitDenied) {
      emitDenialMetric(false, "none");
      dispatchDenialWebhook(ctx, env, "[mcp-route]", () =>
        buildSessionLimitExceededPayload({
          budgetEntityType: outcome.deniedEntityType ?? "unknown",
          budgetEntityId: outcome.deniedEntityId ?? "unknown",
          sessionId: outcome.sessionId ?? "unknown",
          sessionSpendMicrodollars: outcome.sessionSpend ?? 0,
          sessionLimitMicrodollars: outcome.sessionLimit ?? 0,
          model: `${parsed.serverName}/${parsed.toolName}`,
          provider: "mcp",
        }, ctx.auth.apiVersion),
      );
      return new Response(
        JSON.stringify({
          error: {
            code: "session_limit_exceeded",
            message: "Request blocked: session spend exceeds session limit. Start a new session.",
            details: {
              session_id: outcome.sessionId ?? null,
              session_spend_microdollars: outcome.sessionSpend ?? 0,
              session_limit_microdollars: outcome.sessionLimit ?? 0,
            },
          },
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "NullSpend-Version": ctx.resolvedApiVersion,
            "X-NullSpend-Trace-Id": ctx.traceId,
            "X-NullSpend-Denied": "1",
            ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
            ...budgetHeaders,
          },
        },
      );
    }

    // Tag budget denial
    if (outcome.status === "denied" && outcome.tagBudgetDenied) {
      emitDenialMetric(false, "none");
      dispatchDenialWebhook(ctx, env, "[mcp-route]", () =>
        buildTagBudgetExceededPayload({
          tagKey: outcome.tagKey ?? "unknown",
          tagValue: outcome.tagValue ?? "unknown",
          budgetEntityId: outcome.deniedEntityId ?? "unknown",
          budgetLimitMicrodollars: outcome.maxBudget ?? 0,
          budgetSpendMicrodollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
          estimatedRequestCostMicrodollars: parsed.estimateMicrodollars,
          model: `${parsed.serverName}/${parsed.toolName}`,
          provider: "mcp",
        }, ctx.auth.apiVersion),
      );
      return new Response(
        JSON.stringify({
          error: {
            code: "tag_budget_exceeded",
            message: "Request blocked: estimated cost exceeds tag budget limit.",
            details: {
              tag_key: outcome.tagKey ?? null,
              tag_value: outcome.tagValue ?? null,
              budget_limit_microdollars: outcome.maxBudget ?? 0,
              budget_spend_microdollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
            },
          },
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "NullSpend-Version": ctx.resolvedApiVersion,
            "X-NullSpend-Trace-Id": ctx.traceId,
            "X-NullSpend-Denied": "1",
            ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
            ...budgetHeaders,
          },
        },
      );
    }

    // Customer budget denial — new in MCP after the envelope migration.
    // Detected via deniedEntityType === "customer". Per decision 5 + 10 of
    // the plan, this is one of two denial codes that carries upgrade_url.
    if (outcome.status === "denied" && outcome.deniedEntityType === "customer") {
      dispatchDenialWebhook(ctx, env, "[mcp-route]", () =>
        buildCustomerBudgetExceededPayload({
          customerId: outcome.deniedEntityId ?? "unknown",
          budgetLimitMicrodollars: outcome.maxBudget ?? 0,
          budgetSpendMicrodollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
          estimatedRequestCostMicrodollars: parsed.estimateMicrodollars,
          model: `${parsed.serverName}/${parsed.toolName}`,
          provider: "mcp",
        }, ctx.auth.apiVersion),
      );

      // Resolve upgrade_url: per-customer override takes priority over
      // org-level. Cold-path DB query, fails open to null.
      const customerId = outcome.deniedEntityId ?? ctx.customerId ?? null;
      // T7: defensive — log + metric if somehow we enter this branch with
      // no customer_id. Never expected today but observable if it ever fires.
      if (customerId === null) {
        console.warn(
          `[mcp-route] customer_budget_exceeded with null customer_id (denied_entity=${outcome.deniedEntityId} ctx_customer=${ctx.customerId})`,
        );
        emitMetric("customer_denial_missing_id", {
          provider: "mcp",
          orgId: ctx.auth.orgId ?? "unknown",
        });
      }
      const customerUrl = customerId && ctx.auth.orgId
        ? await lookupCustomerUpgradeUrl(ctx.connectionString, ctx.auth.orgId, customerId)
        : null;
      const upgradeUrl = resolveUpgradeUrl(ctx.auth.orgUpgradeUrl, customerUrl, customerId);
      const source: "per_customer" | "org" | "none" =
        customerUrl != null ? "per_customer"
        : (upgradeUrl != null ? "org" : "none");
      emitDenialMetric(upgradeUrl != null, source);

      return new Response(
        JSON.stringify({
          error: {
            code: "customer_budget_exceeded",
            message: "Request blocked: estimated cost exceeds customer budget limit.",
            ...(upgradeUrl ? { upgrade_url: upgradeUrl } : {}),
            details: {
              customer_id: outcome.deniedEntityId ?? null,
              budget_limit_microdollars: outcome.maxBudget ?? 0,
              budget_spend_microdollars: (outcome.spend ?? 0) + (outcome.reserved ?? 0),
            },
          },
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "NullSpend-Version": ctx.resolvedApiVersion,
            "X-NullSpend-Trace-Id": ctx.traceId,
            "X-NullSpend-Denied": "1",
            ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
            ...budgetHeaders,
          },
        },
      );
    }

    // Generic budget denial
    if (outcome.status === "denied") {
      const entityType = outcome.deniedEntityType ?? budgetEntities?.[0]?.entityType ?? "unknown";
      const entityId = outcome.deniedEntityId ?? budgetEntities?.[0]?.entityId ?? "unknown";
      const budgetLimit = outcome.maxBudget ?? 0;
      const budgetSpend = (outcome.spend ?? 0) + (outcome.reserved ?? 0);
      dispatchDenialWebhook(ctx, env, "[mcp-route]", () =>
        buildBudgetExceededPayload({
          budgetEntityType: entityType,
          budgetEntityId: entityId,
          budgetLimitMicrodollars: budgetLimit,
          budgetSpendMicrodollars: budgetSpend,
          estimatedRequestCostMicrodollars: parsed.estimateMicrodollars,
          model: `${parsed.serverName}/${parsed.toolName}`,
          provider: "mcp",
        }),
      );

      // Generic budget_exceeded uses org-level upgrade URL only (no
      // per-customer lookup — the denying entity isn't a customer).
      const genericUpgradeUrl = resolveUpgradeUrl(
        ctx.auth.orgUpgradeUrl,
        null,
        ctx.customerId ?? null,
      );
      emitDenialMetric(
        genericUpgradeUrl != null,
        genericUpgradeUrl != null ? "org" : "none",
      );

      return new Response(
        JSON.stringify({
          error: {
            code: "budget_exceeded",
            message: "Request blocked: estimated cost exceeds remaining budget.",
            ...(genericUpgradeUrl ? { upgrade_url: genericUpgradeUrl } : {}),
            details: {
              entity_type: entityType,
              entity_id: entityId,
              budget_limit_microdollars: budgetLimit,
              budget_spend_microdollars: budgetSpend,
              estimated_cost_microdollars: parsed.estimateMicrodollars,
            },
          },
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "NullSpend-Version": ctx.resolvedApiVersion,
            "X-NullSpend-Trace-Id": ctx.traceId,
            "X-NullSpend-Denied": "1",
            ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
            ...budgetHeaders,
          },
        },
      );
    }

    dispatchVelocityRecoveryWebhooks(outcome, ctx, env, "mcp");

    // Approved: MCP's success contract stays flat for backward compat
    // with the existing cost-tracker consumer in packages/mcp-proxy.
    // Budget headers are stamped alongside — they're header-only, don't
    // affect body parsing.
    const approvedBudgetHeaders = buildBudgetHeaders(budgetEntities, parsed.estimateMicrodollars);
    return Response.json({
      allowed: true,
      reservationId: outcome.reservationId,
      traceId: ctx.traceId,
    }, {
      headers: {
        "NullSpend-Version": ctx.resolvedApiVersion,
        "X-NullSpend-Trace-Id": ctx.traceId,
        ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
        ...approvedBudgetHeaders,
      },
    });
  } catch (err) {
    // Budget check or denial handling failed. Log for diagnostics so
    // production issues don't disappear into a silent 503.
    console.error("[mcp-route] Budget check or denial handling failed:", err);
    const budgetUnavailResp = errorResponse("budget_unavailable", "Budget service unavailable", 503);
    budgetUnavailResp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    if (ctx.sessionId) budgetUnavailResp.headers.set("X-NullSpend-Session", ctx.sessionId);
    return budgetUnavailResp;
  }
}

// ---------------------------------------------------------------------------
// POST /v1/mcp/events
// ---------------------------------------------------------------------------

interface McpCostEvent {
  toolName: string;
  serverName: string;
  durationMs: number;
  costMicrodollars: number;
  status: string;
  reservationId?: string;
  actionId?: string;
}

function validateEvents(body: Record<string, unknown>): McpCostEvent[] | null {
  if (!Array.isArray(body.events)) return null;
  if (body.events.length === 0 || body.events.length > 50) return null;

  for (const event of body.events) {
    if (
      typeof event !== "object" ||
      event === null ||
      typeof event.toolName !== "string" ||
      event.toolName.length === 0 ||
      typeof event.serverName !== "string" ||
      event.serverName.length === 0 ||
      typeof event.durationMs !== "number" ||
      !Number.isFinite(event.durationMs) ||
      event.durationMs < 0 ||
      typeof event.costMicrodollars !== "number" ||
      !Number.isFinite(event.costMicrodollars) ||
      event.costMicrodollars < 0 ||
      typeof event.status !== "string"
    ) {
      return null;
    }
  }

  return body.events as McpCostEvent[];
}

export async function handleMcpEvents(
  request: Request,
  env: Env,
  ctx: RequestContext,
): Promise<Response> {
  const events = validateEvents(ctx.body);
  if (!events) {
    const resp = errorResponse(
      "bad_request",
      "Body must include events array (1-50 items) with toolName, serverName, durationMs, costMicrodollars, status",
      400,
    );
    resp.headers.set("X-NullSpend-Trace-Id", ctx.traceId);
    if (ctx.sessionId) resp.headers.set("X-NullSpend-Session", ctx.sessionId);
    return resp;
  }

  const userId = ctx.auth.userId;
  const apiKeyId = ctx.auth.keyId;

  const accepted = events.length;

  waitUntil(
    (async () => {
      // Phase 1: Build all cost event rows
      const costEventRows = events.map((event) => ({
        requestId: crypto.randomUUID(),
        provider: "mcp" as const,
        model: `${event.serverName}/${event.toolName}`,
        inputTokens: 0,
        outputTokens: 0,
        cachedInputTokens: 0,
        reasoningTokens: 0,
        costMicrodollars: event.costMicrodollars,
        durationMs: event.durationMs,
        userId,
        orgId: ctx.auth.orgId,
        apiKeyId,
        actionId: event.actionId && UUID_RE.test(event.actionId) ? event.actionId : null,
        eventType: "tool" as const,
        toolName: event.toolName,
        toolServer: event.serverName,
        sessionId: ctx.sessionId,
        traceId: ctx.traceId,
        tags: ctx.tags,
        customerId: ctx.customerId,
        source: "mcp" as const,
      }));

      // Phase 2: Single batch INSERT (never throws — errors logged internally)
      try {
        await logCostEventsBatchQueued(getCostEventQueue(env), ctx.connectionString, costEventRows);
      } catch (err) {
        console.error("[mcp-events] Failed to batch-insert cost events:", err);
      }

      // Phase 3: Budget lookup + reconcile reservations
      // Hoisted so budgetEntities are available for both reconciliation and threshold detection.
      let budgetEntities: BudgetEntity[] = [];
      const eventsWithReservations = events.filter((e) => e.reservationId);
      const needsBudgetLookup = eventsWithReservations.length > 0 || (ctx.webhookDispatcher && ctx.auth.hasWebhooks);
      if (needsBudgetLookup) {
        try {
          const doEntities = await lookupBudgetsForDO(ctx.connectionString, { keyId: apiKeyId, userId, tags: ctx.tags });
          budgetEntities = doEntities.map((e) => ({
            entityKey: `{budget}:${e.entityType}:${e.entityId}`,
            entityType: e.entityType,
            entityId: e.entityId,
            maxBudget: e.maxBudget,
            spend: e.spend,
            reserved: 0,
            policy: e.policy,
            thresholdPercentages: e.thresholdPercentages ?? [50, 80, 90, 95],
            sessionLimit: e.sessionLimit ?? null,
          }));
        } catch {
          // best-effort
        }
      }

      let reconcileFailures = 0;
      if (budgetEntities.length > 0 && eventsWithReservations.length > 0) {
        for (const event of eventsWithReservations) {
          try {
            await reconcileBudgetQueued(getReconcileQueue(env), env, ctx.ownerId, ctx.auth.orgId, event.reservationId!, event.costMicrodollars, budgetEntities, ctx.connectionString);
          } catch (err) {
            reconcileFailures++;
            console.error("[mcp-events] Failed to reconcile reservation:", { reservationId: event.reservationId, error: err });
          }
        }
        if (reconcileFailures > 0) {
          console.error(`[mcp-events] ${reconcileFailures}/${eventsWithReservations.length} reconciliations failed — reservations will expire via alarm`);
        }
      }

      // Phase 4: Webhook dispatch (one per cost event per endpoint, single secrets fetch)
      if (ctx.webhookDispatcher && ctx.auth.hasWebhooks) {
        try {
          const cached = await getWebhookEndpoints(ctx.connectionString, ctx.ownerId, env.CACHE_KV);
          if (cached.length > 0) {
            const endpoints = await getWebhookEndpointsWithSecrets(ctx.connectionString, ctx.ownerId);
            for (const row of costEventRows) {
              for (const ep of endpoints) {
                if ((ep.payloadMode ?? "full") === "thin") {
                  await ctx.webhookDispatcher!.dispatch(ep, buildThinCostEventPayload(row.requestId, row.provider, ep.apiVersion));
                } else {
                  await ctx.webhookDispatcher!.dispatch(ep, buildCostEventPayload({ ...row, createdAt: new Date().toISOString() }, ep.apiVersion));
                }
              }
            }

            // Threshold detection (aligned with OpenAI/Anthropic routes)
            if (budgetEntities.length > 0) {
              const epVersion = endpoints[0]?.apiVersion ?? ctx.auth.apiVersion;
              const totalCost = costEventRows.reduce((sum, r) => sum + r.costMicrodollars, 0);
              if (totalCost > 0) {
                const thresholdEvents = detectThresholdCrossings(budgetEntities, totalCost, costEventRows[0].requestId, epVersion);
                for (const te of thresholdEvents) {
                  await dispatchToEndpoints(ctx.webhookDispatcher!, endpoints, te);
                }
              }
            }

            expireRotatedSecrets(ctx.connectionString, endpoints).catch(() => {});
          }
        } catch (err) {
          console.error("[mcp-events] Webhook dispatch failed:", err);
        }
      }
    })(),
  );

  return Response.json({ accepted }, {
    headers: {
      "NullSpend-Version": ctx.resolvedApiVersion,
      "X-NullSpend-Trace-Id": ctx.traceId,
      ...(ctx.sessionId ? { "X-NullSpend-Session": ctx.sessionId } : {}),
    },
  });
}
