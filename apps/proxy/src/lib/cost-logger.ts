import type { NewCostEventRow } from "@nullspend/db";
import { emitMetric } from "./metrics.js";
import { getSql } from "./db.js";

/**
 * Persist a cost event to Postgres via Hyperdrive.
 * Uses raw postgres.js tagged templates for minimal bundle size.
 * Never throws — this runs inside waitUntil().
 *
 * When skipDbWrites is true (local dev without Hyperdrive), falls back
 * to console logging to avoid workerd crashes from unreachable Postgres.
 */
export async function logCostEvent(
  connectionString: string,
  event: Omit<NewCostEventRow, "id" | "createdAt">,
  options?: { throwOnError?: boolean; skipDbWrites?: boolean },
): Promise<void> {
  if (options?.skipDbWrites) {
    console.log("[cost-logger] Local dev — cost event (not persisted):", {
      requestId: event.requestId,
      provider: event.provider,
      model: event.model,
      inputTokens: event.inputTokens,
      outputTokens: event.outputTokens,
      costMicrodollars: event.costMicrodollars,
      durationMs: event.durationMs,
      eventType: event.eventType,
      toolName: event.toolName,
      sessionId: event.sessionId,
      traceId: event.traceId,
    });
    return;
  }

  try {
    const sql = getSql(connectionString);
    await sql`
      INSERT INTO cost_events (
        request_id, api_key_id, user_id, org_id, provider, model,
        input_tokens, output_tokens, cached_input_tokens, reasoning_tokens,
        cost_microdollars, duration_ms, action_id, event_type,
        tool_name, tool_server, tool_calls_requested, tool_definition_tokens,
        upstream_duration_ms, session_id, trace_id, source, cost_breakdown, tags,
        budget_status, stop_reason, estimated_cost_microdollars, customer_id
      ) VALUES (
        ${event.requestId}, ${event.apiKeyId ?? null}, ${event.userId ?? null}, ${event.orgId ?? null},
        ${event.provider}, ${event.model},
        ${event.inputTokens}, ${event.outputTokens},
        ${event.cachedInputTokens ?? 0}, ${event.reasoningTokens ?? 0},
        ${event.costMicrodollars}, ${event.durationMs ?? null},
        ${event.actionId ?? null}, ${event.eventType ?? "llm"},
        ${event.toolName ?? null}, ${event.toolServer ?? null},
        ${event.toolCallsRequested ? sql.json(event.toolCallsRequested) : null},
        ${event.toolDefinitionTokens ?? 0},
        ${event.upstreamDurationMs ?? null}, ${event.sessionId ?? null},
        ${event.traceId ?? null}, ${event.source ?? "proxy"},
        ${event.costBreakdown ? sql.json(event.costBreakdown) : null},
        ${event.tags ? sql.json(event.tags) : sql.json({})},
        ${event.budgetStatus ?? null}, ${event.stopReason ?? null},
        ${event.estimatedCostMicrodollars ?? null}, ${event.customerId ?? null}
      )
      ON CONFLICT (request_id, provider) DO NOTHING
    `;
  } catch (err) {
    emitMetric("cost_event_drop", { reason: "pg_error" });
    console.error(
      "[cost-logger] Failed to write cost event:",
      err instanceof Error ? err.message : "Unknown error",
    );
    if (options?.throwOnError) throw err;
  }
}

/**
 * Persist multiple cost events in a single multi-row INSERT.
 * Same guarantees as logCostEvent: never throws, falls back to console in local dev.
 */
export async function logCostEventsBatch(
  connectionString: string,
  events: Omit<NewCostEventRow, "id" | "createdAt">[],
  options?: { throwOnError?: boolean; skipDbWrites?: boolean },
): Promise<void> {
  if (events.length === 0) return;

  if (options?.skipDbWrites) {
    for (const event of events) {
      console.log("[cost-logger] Local dev — cost event (not persisted):", {
        requestId: event.requestId,
        provider: event.provider,
        model: event.model,
        inputTokens: event.inputTokens,
        outputTokens: event.outputTokens,
        costMicrodollars: event.costMicrodollars,
        durationMs: event.durationMs,
        eventType: event.eventType,
        toolName: event.toolName,
        sessionId: event.sessionId,
        traceId: event.traceId,
      });
    }
    return;
  }

  try {
    const sql = getSql(connectionString);
    // Use postgres.js array helper for multi-row INSERT
    await sql`
      INSERT INTO cost_events ${sql(
        events.map((e) => ({
          request_id: e.requestId,
          api_key_id: e.apiKeyId ?? null,
          user_id: e.userId ?? null,
          org_id: e.orgId ?? null,
          provider: e.provider,
          model: e.model,
          input_tokens: e.inputTokens,
          output_tokens: e.outputTokens,
          cached_input_tokens: e.cachedInputTokens ?? 0,
          reasoning_tokens: e.reasoningTokens ?? 0,
          cost_microdollars: e.costMicrodollars,
          duration_ms: e.durationMs ?? null,
          action_id: e.actionId ?? null,
          event_type: e.eventType ?? "llm",
          tool_name: e.toolName ?? null,
          tool_server: e.toolServer ?? null,
          tool_calls_requested: e.toolCallsRequested ? sql.json(e.toolCallsRequested) : null,
          tool_definition_tokens: e.toolDefinitionTokens ?? 0,
          upstream_duration_ms: e.upstreamDurationMs ?? null,
          session_id: e.sessionId ?? null,
          trace_id: e.traceId ?? null,
          source: e.source ?? "proxy",
          cost_breakdown: e.costBreakdown ? sql.json(e.costBreakdown) : null,
          tags: e.tags ? sql.json(e.tags) : sql.json({}),
          budget_status: e.budgetStatus ?? null,
          stop_reason: e.stopReason ?? null,
          estimated_cost_microdollars: e.estimatedCostMicrodollars ?? null,
          customer_id: e.customerId ?? null,
        }))
      )}
      ON CONFLICT (request_id, provider) DO NOTHING
    `;
  } catch (err) {
    emitMetric("cost_event_drop", { reason: "batch_pg_error", count: events.length });
    console.error(
      "[cost-logger] Failed to write cost event batch:",
      err instanceof Error ? err.message : "Unknown error",
      `(${events.length} events)`,
    );
    if (options?.throwOnError) throw err;
  }
}
