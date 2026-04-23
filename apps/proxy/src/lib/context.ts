import type { AuthResult } from "./auth.js";
import type { WebhookDispatcher } from "./webhook-dispatch.js";
import type { StepTiming } from "./headers.js";

/**
 * Pre-auth request metadata. Populated at the TOP of `index.fetch` before
 * auth, body-parse, or any other early-return path. Used by
 * `stampNullspendHeaders` so pre-auth 401/429/400 responses can still echo
 * trace + request + session headers.
 *
 * PR-2c (codex-round-3 H3 + codex-round-4 M4): `RequestContext` extends this
 * interface — post-auth callers pass `ctx` directly to the stamping helper.
 */
export interface IngressMetadata {
  traceId: string;              // from traceparent / x-nullspend-trace-id / auto-generated
  nullspendRequestId: string;   // from x-nullspend-request-id (validated) or crypto.randomUUID()
  sessionId: string | null;     // from x-nullspend-session
}

export interface RequestContext extends IngressMetadata {
  body: Record<string, unknown>;
  bodyText: string;                // original request body text (avoids re-serialize for upstream fetch)
  bodyByteLength: number;          // original request body size (avoids re-stringify for estimation)
  auth: AuthResult;
  ownerId: string;                 // orgId ?? userId — DO keying and webhook/budget scoping
  connectionString: string;
  skipDbWrites: boolean;     // true in local dev without Hyperdrive (env: SKIP_DB_PERSIST)
  tags: Record<string, string>; // from x-nullspend-tags
  customerId: string | null;   // from x-nullspend-customer or tags["customer"]
  customerWarning: string | null; // set when customer header is present but invalid
  webhookDispatcher: WebhookDispatcher | null;
  resolvedApiVersion: string;
  requestStartMs: number;       // performance.now() at request entry
  stepTiming?: StepTiming;      // per-step latency for Server-Timing header
  requestLoggingEnabled: boolean; // pro/enterprise tier — enables R2 body capture
  finalize: boolean;             // from x-nullspend-finalize: "1" — unlocks finalization reserve
}

/**
 * Parse + validate `x-nullspend-request-id` header.
 *
 * PR-2c codex-round-4 M2: prevents malicious/malformed client values from
 * being reflected in response headers. Charset is alphanumeric + `-` + `_`
 * (covers UUID, ULID, default NanoID). Length cap 256 matches the DO's
 * `MAX_IDEMPOTENCY_KEY_LENGTH` constant.
 *
 * Returns the client-supplied value when valid, else `crypto.randomUUID()`.
 * Emits `nullspend_request_id_invalid{reason}` metric on rejection so
 * operators see unexpected client behavior. Non-present header is NOT an
 * error — silent fallback to generated UUID.
 */
export const NULLSPEND_REQUEST_ID_MAX_LENGTH = 256;
export const NULLSPEND_REQUEST_ID_CHARSET = /^[A-Za-z0-9_-]+$/;

export function resolveNullspendRequestId(
  request: Request,
  emit: (reason: "empty" | "too_long" | "bad_charset") => void,
): string {
  const raw = request.headers.get("x-nullspend-request-id");
  if (raw === null) return crypto.randomUUID();
  if (raw.length === 0) {
    emit("empty");
    return crypto.randomUUID();
  }
  if (raw.length > NULLSPEND_REQUEST_ID_MAX_LENGTH) {
    emit("too_long");
    return crypto.randomUUID();
  }
  if (!NULLSPEND_REQUEST_ID_CHARSET.test(raw)) {
    emit("bad_charset");
    return crypto.randomUUID();
  }
  return raw;
}

export type RouteHandler = (
  request: Request,
  env: Env,
  ctx: RequestContext,
) => Promise<Response>;
