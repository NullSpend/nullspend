import type { ReconciliationMessage } from "./lib/reconciliation-queue.js";
import { reconcileBudget } from "./lib/budget-orchestrator.js";
import type { BudgetEntity } from "./lib/budget-do-lookup.js";

/**
 * Cloudflare Queue consumer for budget reconciliation messages.
 * Each message is processed individually: ack on success, retry on failure.
 * After max_retries (configured in wrangler.jsonc), messages go to DLQ.
 */
export async function handleReconciliationQueue(
  batch: MessageBatch<ReconciliationMessage>,
  env: Env,
): Promise<void> {
  const connectionString = env.HYPERDRIVE.connectionString;

  for (const message of batch.messages) {
    try {
      const msg = message.body;
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
        msg.orgId ?? null,
        msg.reservationId,
        msg.actualCostMicrodollars,
        budgetEntities,
        connectionString,
        { throwOnError: true },
      );
      message.ack();
    } catch (err) {
      console.error("[queue] Reconciliation failed, retrying:", err);
      message.retry();
    }
  }
}
