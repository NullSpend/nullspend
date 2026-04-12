// SYNC: Dashboard WebhookEvent interface in lib/webhooks/dispatch.ts must match this shape

export const CURRENT_API_VERSION = "2026-04-01";

export type WebhookEventType =
  | "cost_event.created"
  | "budget.threshold.warning"
  | "budget.threshold.critical"
  | "budget.exceeded"
  | "budget.increased"
  | "budget.reset"
  | "request.blocked"
  | "action.created"
  | "action.approved"
  | "action.rejected"
  | "action.expired"
  | "velocity.exceeded"
  | "velocity.recovered"
  | "session.limit_exceeded"
  | "tag_budget.exceeded"
  | "customer_budget.exceeded"
  | "test.ping";

export interface WebhookEvent {
  id: string;
  type: WebhookEventType;
  api_version: string;
  created_at: number;
  data: { object: Record<string, unknown> };
}

export interface ThinWebhookEvent {
  id: string;
  type: WebhookEventType;
  api_version: string;
  created_at: number;
  related_object: { id: string; type: string; url: string };
}

export type AnyWebhookEvent = WebhookEvent | ThinWebhookEvent;

/**
 * Webhook payload input. Aligned with Omit<NewCostEventRow, "id" | "createdAt">
 * (the cost calculator return type). Fields with Drizzle defaults or nullable
 * columns are optional to match the insert type callers spread.
 */
interface CostEventData {
  // Required (notNull, no default)
  requestId: string;
  provider: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  costMicrodollars: number;
  // Nullable (notNull: false, no default) — optional in insert type
  durationMs?: number | null;
  apiKeyId?: string | null;
  userId?: string | null;
  actionId?: string | null;
  toolName?: string | null;
  toolServer?: string | null;
  toolCallsRequested?: { name: string; id: string }[] | null;
  upstreamDurationMs?: number | null;
  sessionId?: string | null;
  traceId?: string | null;
  costBreakdown?: { input?: number; output?: number; cached?: number; reasoning?: number; toolDefinition?: number } | null;
  // Has default (optional in insert type)
  cachedInputTokens?: number;
  reasoningTokens?: number;
  eventType?: string;
  toolDefinitionTokens?: number;
  source?: string;
  tags?: Record<string, string>;
  // Extra field added by callers (not in DB schema)
  createdAt?: string;
}

export function buildCostEventPayload(
  costEvent: CostEventData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "cost_event.created",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        request_id: costEvent.requestId,
        event_type: costEvent.eventType ?? "llm",
        provider: costEvent.provider,
        model: costEvent.model,
        input_tokens: costEvent.inputTokens,
        output_tokens: costEvent.outputTokens,
        cached_input_tokens: costEvent.cachedInputTokens ?? 0,
        cost_microdollars: costEvent.costMicrodollars,
        duration_ms: costEvent.durationMs,
        upstream_duration_ms: costEvent.upstreamDurationMs ?? null,
        session_id: costEvent.sessionId ?? null,
        trace_id: costEvent.traceId ?? null,
        tool_name: costEvent.toolName ?? null,
        tool_server: costEvent.toolServer ?? null,
        tool_calls_requested: costEvent.toolCallsRequested ?? null,
        tool_definition_tokens: costEvent.toolDefinitionTokens ?? 0,
        api_key_id: costEvent.apiKeyId,
        source: costEvent.source ?? null,
        tags: costEvent.tags ?? {},
        created_at: costEvent.createdAt ?? new Date().toISOString(),
      },
    },
  };
}

interface BudgetExceededData {
  budgetEntityType: string;
  budgetEntityId: string;
  budgetLimitMicrodollars: number;
  budgetSpendMicrodollars: number;
  estimatedRequestCostMicrodollars: number;
  model: string;
  provider: string;
}

export function buildBudgetExceededPayload(
  data: BudgetExceededData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "budget.exceeded",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        budget_limit_microdollars: data.budgetLimitMicrodollars,
        budget_spend_microdollars: data.budgetSpendMicrodollars,
        estimated_request_cost_microdollars: data.estimatedRequestCostMicrodollars,
        model: data.model,
        provider: data.provider,
        blocked_at: new Date().toISOString(),
      },
    },
  };
}

interface ThresholdData {
  budgetEntityType: string;
  budgetEntityId: string;
  budgetLimitMicrodollars: number;
  budgetSpendMicrodollars: number;
  thresholdPercent: number;
  triggeredByRequestId: string;
  isCritical?: boolean;
}

export function buildThresholdPayload(
  data: ThresholdData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  // isCritical can be explicitly set by the caller (per-entity thresholds).
  // Fallback: >= 90 is critical (preserves backward compat for default thresholds).
  const type: WebhookEventType = (data.isCritical ?? data.thresholdPercent >= 90)
    ? "budget.threshold.critical"
    : "budget.threshold.warning";

  return {
    id: `evt_${crypto.randomUUID()}`,
    type,
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        budget_limit_microdollars: data.budgetLimitMicrodollars,
        budget_spend_microdollars: data.budgetSpendMicrodollars,
        threshold_percent: data.thresholdPercent,
        budget_remaining_microdollars:
          data.budgetLimitMicrodollars - data.budgetSpendMicrodollars,
        triggered_by_request_id: data.triggeredByRequestId,
      },
    },
  };
}

interface BudgetResetData {
  budgetEntityType: string;
  budgetEntityId: string;
  budgetLimitMicrodollars: number;
  previousSpendMicrodollars: number;
  newPeriodStart: string;
  resetInterval: string;
}

export function buildBudgetResetPayload(
  data: BudgetResetData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "budget.reset",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        budget_limit_microdollars: data.budgetLimitMicrodollars,
        previous_spend_microdollars: data.previousSpendMicrodollars,
        new_period_start: data.newPeriodStart,
        reset_interval: data.resetInterval,
      },
    },
  };
}

interface RequestBlockedData {
  reason: "budget" | "rate_limit" | "policy";
  model: string;
  provider: string;
  apiKeyId: string | null;
  details: string | null;
}

export function buildRequestBlockedPayload(
  data: RequestBlockedData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "request.blocked",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        reason: data.reason,
        model: data.model,
        provider: data.provider,
        api_key_id: data.apiKeyId,
        details: data.details,
      },
    },
  };
}

interface VelocityExceededData {
  budgetEntityType: string;
  budgetEntityId: string;
  velocityLimitMicrodollars: number;
  velocityWindowSeconds: number;
  velocityCurrentMicrodollars: number;
  cooldownSeconds: number;
  model: string;
  provider: string;
}

export function buildVelocityExceededPayload(
  data: VelocityExceededData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "velocity.exceeded",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        velocity_limit_microdollars: data.velocityLimitMicrodollars,
        velocity_window_seconds: data.velocityWindowSeconds,
        velocity_current_microdollars: data.velocityCurrentMicrodollars,
        cooldown_seconds: data.cooldownSeconds,
        model: data.model,
        provider: data.provider,
        blocked_at: new Date().toISOString(),
      },
    },
  };
}

interface VelocityRecoveredData {
  budgetEntityType: string;
  budgetEntityId: string;
  velocityLimitMicrodollars: number;
  velocityWindowSeconds: number;
  velocityCooldownSeconds: number;
}

export function buildVelocityRecoveredPayload(
  data: VelocityRecoveredData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "velocity.recovered",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        velocity_limit_microdollars: data.velocityLimitMicrodollars,
        velocity_window_seconds: data.velocityWindowSeconds,
        velocity_cooldown_seconds: data.velocityCooldownSeconds,
        recovered_at: new Date().toISOString(),
      },
    },
  };
}

interface SessionLimitExceededData {
  budgetEntityType: string;
  budgetEntityId: string;
  sessionId: string;
  sessionSpendMicrodollars: number;
  sessionLimitMicrodollars: number;
  model: string;
  provider: string;
}

export function buildSessionLimitExceededPayload(
  data: SessionLimitExceededData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "session.limit_exceeded",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: data.budgetEntityType,
        budget_entity_id: data.budgetEntityId,
        session_id: data.sessionId,
        session_spend_microdollars: data.sessionSpendMicrodollars,
        session_limit_microdollars: data.sessionLimitMicrodollars,
        model: data.model,
        provider: data.provider,
        blocked_at: new Date().toISOString(),
      },
    },
  };
}

interface TagBudgetExceededData {
  tagKey: string;
  tagValue: string;
  budgetEntityId: string;
  budgetLimitMicrodollars: number;
  budgetSpendMicrodollars: number;
  estimatedRequestCostMicrodollars: number;
  model: string;
  provider: string;
}

export function buildTagBudgetExceededPayload(
  data: TagBudgetExceededData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "tag_budget.exceeded",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: "tag",
        budget_entity_id: data.budgetEntityId,
        tag_key: data.tagKey,
        tag_value: data.tagValue,
        budget_limit_microdollars: data.budgetLimitMicrodollars,
        budget_spend_microdollars: data.budgetSpendMicrodollars,
        estimated_request_cost_microdollars: data.estimatedRequestCostMicrodollars,
        model: data.model,
        provider: data.provider,
        blocked_at: new Date().toISOString(),
      },
    },
  };
}

interface CustomerBudgetExceededData {
  customerId: string;
  budgetLimitMicrodollars: number;
  budgetSpendMicrodollars: number;
  estimatedRequestCostMicrodollars: number;
  model: string;
  provider: string;
}

export function buildCustomerBudgetExceededPayload(
  data: CustomerBudgetExceededData,
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "customer_budget.exceeded",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        budget_entity_type: "customer",
        budget_entity_id: data.customerId,
        customer_id: data.customerId,
        budget_limit_microdollars: data.budgetLimitMicrodollars,
        budget_spend_microdollars: data.budgetSpendMicrodollars,
        estimated_request_cost_microdollars: data.estimatedRequestCostMicrodollars,
        model: data.model,
        provider: data.provider,
        blocked_at: new Date().toISOString(),
      },
    },
  };
}

export function buildTestPingPayload(
  apiVersion: string = CURRENT_API_VERSION,
): WebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "test.ping",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    data: {
      object: {
        message: "Test webhook event",
      },
    },
  };
}

export function buildThinCostEventPayload(
  requestId: string,
  provider: string,
  apiVersion: string = CURRENT_API_VERSION,
): ThinWebhookEvent {
  return {
    id: `evt_${crypto.randomUUID()}`,
    type: "cost_event.created",
    api_version: apiVersion,
    created_at: Math.floor(Date.now() / 1000),
    related_object: {
      id: requestId,
      type: "cost_event",
      url: `/api/cost-events?requestId=${encodeURIComponent(requestId)}&provider=${encodeURIComponent(provider)}`,
    },
  };
}
