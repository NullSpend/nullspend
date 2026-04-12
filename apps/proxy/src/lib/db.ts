import postgres from "postgres";

/**
 * Per-request postgres.js client for Cloudflare Workers.
 *
 * IMPORTANT: Cloudflare Workers enforce I/O context isolation — connections
 * created in one request's handler cannot be reused by another request.
 * Module-level connection pools violate this and cause:
 *   "Cannot perform I/O on behalf of a different request"
 *
 * Instead, we create a fresh postgres.js instance per call. This is efficient
 * because Hyperdrive handles connection pooling at the infrastructure level —
 * the postgres.js "connection" is actually a fast local socket to the
 * Hyperdrive proxy, not a real TCP+TLS connection to Postgres.
 */

/**
 * DB write skipping is controlled by `ctx.skipDbWrites` (boolean) on
 * RequestContext, computed once per request in index.ts from env vars
 * (FORCE_DB_PERSIST / SKIP_DB_PERSIST). Callers check it directly —
 * no shared utility function needed.
 */

/**
 * Create a postgres.js client for the current request context.
 * Each call returns a fresh instance — do not cache across requests.
 * Hyperdrive handles connection pooling at the infrastructure level.
 */
export function getSql(connectionString: string): ReturnType<typeof postgres> {
  return postgres(connectionString, {
    max: 1,              // single connection per instance; Hyperdrive pools at infra level
    idle_timeout: 20,    // seconds before idle connection is closed
    connect_timeout: 5,  // seconds to wait for connection
    prepare: false,      // required for Hyperdrive compatibility (PgBouncer-style pooling)
    fetch_types: false,  // skip pg_type catalog round-trip — built-in types (text[], int4[]) still parsed
    // Statement-level timeout: kill queries that hang beyond 10s.
    // Prevents cost-logger and budget-lookup from blocking waitUntil indefinitely.
    transform: { undefined: null },
    connection: { statement_timeout: 10_000 },
  });
}

