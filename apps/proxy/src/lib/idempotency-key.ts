/**
 * Plan-counter idempotency-key hashing (PR-2d / Decision #26 / codex R3#2 / R4#2 / build-audit F1).
 *
 * **MUST stay byte-identical to `lib/cost-events/idempotency-key.ts` on the
 * dashboard side.** The two callers — the dashboard's live-path sync
 * (`/api/cost-events`) and the proxy's reconciliation cron — produce keys for
 * the SAME logical event; if the hashes diverge, DO dedup can't catch the
 * partial-success replay (build-audit F1). One source of truth would be ideal,
 * but the dashboard is Node and the proxy is workerd — no shared package today.
 * Mechanically duplicating a 5-line crypto.subtle call is the lowest-risk fix.
 *
 * Rationale: the DO's `plan_counter_idempotency` table enforces a 256-char key
 * cap. Raw `[requestId, provider]` concatenation blows that cap on long
 * requestIds (200–300 chars are real in prod) and naive `${requestId}::${provider}`
 * collides on any delimiter present in either field.
 *
 * SHA-256 hex digest of `JSON.stringify([requestId, provider])` solves both:
 *   - Fixed 64 hex chars, well under the DO's 256-char limit.
 *   - JSON-tuple encoding escapes delimiters, so `("a::b", "c")` and
 *     `("a", "b::c")` produce distinct hashes (codex R3#2 / C58c).
 *   - 2^256 hash space — effective collision probability is zero.
 */
export async function sha256IdempotencyKey(
  requestId: string,
  provider: string,
): Promise<string> {
  const input = JSON.stringify([requestId, provider]);
  const buffer = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");
}
