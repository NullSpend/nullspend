/**
 * Plan-limit check — pure function.
 *
 * Inputs come from the SQL-derived columns on `ApiKeyIdentity` (see
 * `api-key-auth.ts` — `planLimitBlockAt`, `planLimitMode`). The proxy is
 * tier-blind: SQL knows tier→limits, JS only reads the primitives.
 *
 * - `blockAt = null` → unlimited (self-hosted, Enterprise).
 * - `mode = "soft"` → count-only; never denies (Pro/Scale soft cap; overage
 *   billed via downstream invoicing).
 * - `mode = "hard"` → deny when `currentCount > blockAt` (Free tier).
 *
 * Per PR-2a Decision #15 (proxy stays tier-blind via SQL-derived columns) +
 * Decision #4 (Free tier includes full atomic enforcement).
 */
export type PlanLimitMode = "hard" | "soft";

export type PlanLimitResult =
  | { status: "approved" }
  | { status: "denied"; blockAt: number };

export function checkPlanLimit(
  blockAt: number | null,
  mode: PlanLimitMode,
  currentCount: number,
): PlanLimitResult {
  if (blockAt === null) return { status: "approved" };
  if (mode === "soft") return { status: "approved" };
  if (currentCount > blockAt) return { status: "denied", blockAt };
  return { status: "approved" };
}
