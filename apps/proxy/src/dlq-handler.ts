import type { ReconciliationMessage } from "./lib/reconciliation-queue.js";
import { reconcileBudget } from "./lib/budget-orchestrator.js";
import type { BudgetEntity } from "./lib/budget-do-lookup.js";
import { emitMetric } from "./lib/metrics.js";

export const DLQ_QUEUE_NAME = "nullspend-reconcile-dlq";

/**
 * Cloudflare Queue consumer for dead-lettered reconciliation messages.
 *
 * For each message: emit a metric, log a structured error, then attempt one
 * final best-effort reconciliation (no throwOnError — never throws). Always
 * acks in a finally block so messages are never retried at the queue level.
 *
 * Note: `reconcileBudget` without `throwOnError` logs errors internally
 * ("[budget-orchestrator] Reconciliation failed:"). This produces a second
 * log line on failure — intentional, as the DLQ log adds context (message
 * ID, age) that the orchestrator doesn't have.
 */
export async function handleDlqQueue(
  batch: MessageBatch<ReconciliationMessage>,
  env: Env,
): Promise<void> {
  const connectionString = env.HYPERDRIVE.connectionString;

  for (const message of batch.messages) {
    try {
      const msg = message.body;
      const ageMs =
        typeof msg.enqueuedAt === "number" && msg.enqueuedAt > 0
          ? Date.now() - msg.enqueuedAt
          : -1;

      emitMetric("reconciliation_dlq", {
        reservationId: msg.reservationId,
        costMicrodollars: msg.actualCostMicrodollars,
        ownerId: msg.ownerId ?? "unknown",
        ageMs,
        entityCount: msg.budgetEntities.length,
      });

      console.error(
        "[dlq] Dead-lettered reconciliation message:",
        safeStringify(msg),
      );

      const budgetEntities: BudgetEntity[] = msg.budgetEntities.map((e) => ({
        entityKey: e.entityKey,
        entityType: e.entityType,
        entityId: e.entityId,
        maxBudget: 0,
        spend: 0,
        reserved: 0,
        policy: "strict_block",
      }));

      await reconcileBudget(
        env,
        msg.ownerId,
        msg.reservationId,
        msg.actualCostMicrodollars,
        budgetEntities,
        connectionString,
      );
    } catch (err) {
      console.error("[dlq] Unexpected error processing message:", err);
    } finally {
      message.ack();
    }
  }
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
