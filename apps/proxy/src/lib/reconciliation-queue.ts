export interface ReconciliationMessage {
  type: "reconcile";
  reservationId: string;
  actualCostMicrodollars: number;
  budgetEntities: Array<{ entityKey: string; entityType: string; entityId: string }>;
  ownerId: string | null;
  enqueuedAt: number;
}

/**
 * Enqueue a reconciliation message to Cloudflare Queues.
 * Resolves in <1ms (message written to disk), decoupling reconciliation
 * from the request lifecycle and the 30s waitUntil limit.
 */
export async function enqueueReconciliation(
  queue: Queue,
  msg: ReconciliationMessage,
): Promise<void> {
  await queue.send(msg);
}
