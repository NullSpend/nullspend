/**
 * Lightweight structured-log emitter for reconciliation and cost metrics.
 * Outputs JSON to console.log in a format parseable by a tail worker.
 * No external dependencies.
 */
export function emitMetric(name: string, tags: Record<string, string | number | boolean>): void {
  console.log(JSON.stringify({ _metric: name, ...tags, _ts: Date.now() }));
}
