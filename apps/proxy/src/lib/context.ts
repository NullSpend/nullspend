import type { AuthResult } from "./auth.js";
import type { WebhookDispatcher } from "./webhook-dispatch.js";
import type { StepTiming } from "./headers.js";

export interface RequestContext {
  body: Record<string, unknown>;
  bodyText: string;                // original request body text (avoids re-serialize for upstream fetch)
  bodyByteLength: number;          // original request body size (avoids re-stringify for estimation)
  auth: AuthResult;
  ownerId: string;                 // orgId ?? userId — DO keying and webhook/budget scoping
  connectionString: string;
  skipDbWrites: boolean;     // true in local dev without Hyperdrive (env: SKIP_DB_PERSIST)
  sessionId: string | null;     // from x-nullspend-session
  traceId: string;              // from traceparent / x-nullspend-trace-id / auto-generated
  tags: Record<string, string>; // from x-nullspend-tags
  customerId: string | null;   // from x-nullspend-customer or tags["customer"]
  customerWarning: string | null; // set when customer header is present but invalid
  webhookDispatcher: WebhookDispatcher | null;
  resolvedApiVersion: string;
  requestStartMs: number;       // performance.now() at request entry
  stepTiming?: StepTiming;      // per-step latency for Server-Timing header
  requestLoggingEnabled: boolean; // pro/enterprise tier — enables R2 body capture
}

export type RouteHandler = (
  request: Request,
  env: Env,
  ctx: RequestContext,
) => Promise<Response>;
