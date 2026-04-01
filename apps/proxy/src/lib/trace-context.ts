const TRACEPARENT_RE = /^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/;
const ALL_ZEROS_TRACE_ID = "00000000000000000000000000000000";
const ALL_ZEROS_SPAN_ID = "0000000000000000";
const TRACE_ID_RE = /^[0-9a-f]{32}$/;

/**
 * Resolve a 32-char lowercase hex trace ID from request headers.
 *
 * Priority:
 * 1. W3C `traceparent` header → extract trace-id field
 * 2. `X-NullSpend-Trace-Id` custom header
 * 3. Auto-generate via crypto.randomUUID()
 *
 * Never throws. Logs console.warn on invalid headers and falls through.
 */
export function resolveTraceId(request: Request): string {
  // 1. W3C traceparent header
  const traceparent = request.headers.get("traceparent");
  if (traceparent) {
    const match = TRACEPARENT_RE.exec(traceparent);
    if (match) {
      const version = match[1];
      const traceId = match[2];
      const spanId = match[3];
      // Reject version ff (reserved), all-zeros trace-id, and all-zeros span-id (W3C spec)
      if (version !== "ff" && traceId !== ALL_ZEROS_TRACE_ID && spanId !== ALL_ZEROS_SPAN_ID) {
        return traceId;
      }
      console.warn(`[trace-context] Invalid traceparent: version=${version}, traceId=${traceId}, spanId=${spanId}`);
    } else {
      console.warn(`[trace-context] Malformed traceparent header: ${traceparent.slice(0, 200)}`);
    }
  }

  // 2. X-NullSpend-Trace-Id custom header
  const customTraceId = request.headers.get("x-nullspend-trace-id");
  if (customTraceId) {
    if (TRACE_ID_RE.test(customTraceId)) {
      return customTraceId;
    }
    console.warn(`[trace-context] Invalid X-NullSpend-Trace-Id: ${customTraceId.slice(0, 200)}`);
  }

  // 3. Auto-generate
  return crypto.randomUUID().replace(/-/g, "");
}
