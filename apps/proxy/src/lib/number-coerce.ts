/**
 * Coerce a DB-returned value to a finite JS number, or null.
 *
 * Needed because `postgres.js` with `fetch_types: false` (see apps/proxy/src/lib/db.ts)
 * returns BIGINT columns as `string` or `BigInt`, NOT JS `number`. Without this
 * coercion, the `ApiKeyIdentity.subscriptionPeriodStart` type contract
 * (`number | null`) is fiction and downstream `Date.UTC(...)` math returns NaN.
 *
 * Accepts string, number, BigInt, null, undefined. Rejects everything else
 * (boolean, object, array, NaN, Infinity) as null — fail-soft on surprise types
 * rather than pass through ambiguous values.
 *
 * **Subtle edge (per edge-audit E3):** empty string `""` coerces via `Number()` to
 * `0`, which is a finite number → this helper returns `0`, not `null`. Same for
 * `"0"`. For BIGINT columns from a trusted Postgres source, this matches JS
 * idiom: zero is a valid number, and postgres.js doesn't emit `""` for BIGINT
 * in practice. If downstream code needs to distinguish "empty" from "zero",
 * check at the call site rather than in this helper. The period-invariant
 * check in `resolvePeriodBounds` catches epoch-0 as invalid bounds downstream.
 *
 * Mirrors the existing defensive coercion pattern at
 * `apps/proxy/src/lib/budget-do-lookup.ts:59-69` and `budget-spend.ts:96-98`.
 *
 * Per PR-2a Decision #36 + plan-audit A6 + codex PR-2a-R3-1 + edge-audit E3.
 */
export function toFiniteNumber(v: unknown): number | null {
  if (v === null || v === undefined) return null;

  // Reject booleans explicitly — `Number(true)===1` would silently pass through.
  if (typeof v === "boolean") return null;

  // Reject symbols BEFORE `Number(v)` — `Number(Symbol())` THROWS TypeError,
  // which would violate the fail-soft contract documented in the JSDoc.
  // Caught by the cubic post-merge reviewer.
  if (typeof v === "symbol") return null;

  // Reject non-primitive types (arrays, objects) — `Number([])===0` ambiguity.
  if (typeof v === "object") return null;

  // `Number(...)` handles string, number, BigInt correctly:
  // - "1744848000000" → 1744848000000
  // - 1744848000000n → 1744848000000
  // - 1744848000000 → 1744848000000 (identity)
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
