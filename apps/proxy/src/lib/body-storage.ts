import { emitMetric } from "./metrics.js";

/**
 * R2 key layout: `{ownerId}/{requestId}/request.json` and `.../response.json`
 * Streaming responses use `.../response.sse` to distinguish raw SSE text.
 *
 * Scoping under ownerId means a future lifecycle policy (e.g. 30-day expiry)
 * can be applied per-prefix, and the internal retrieval endpoint only
 * needs to verify ownerId ownership — no extra DB lookup.
 */
function requestKey(ownerId: string, requestId: string): string {
  return `${ownerId}/${requestId}/request.json`;
}

function responseKey(ownerId: string, requestId: string): string {
  return `${ownerId}/${requestId}/response.json`;
}

function responseSseKey(ownerId: string, requestId: string): string {
  return `${ownerId}/${requestId}/response.sse`;
}

// NOTE: All size checks use string.length (UTF-16 code units), not actual UTF-8 byte
// count. For ASCII-dominated SSE/JSON this is accurate, but multi-byte characters
// (emoji, CJK) could result in slightly >1MB UTF-8 stored in R2. Harmless in practice
// (R2 limit is 5GB). A future cleanup should switch to chunk.byteLength tracking if
// byte-accurate limits are needed — must be done consistently across all store functions.
const MAX_BODY_BYTES = 1_048_576; // 1MB cap per object — matches proxy body limit

/**
 * Store request body in R2. Fire-and-forget — never throws.
 */
export async function storeRequestBody(
  bucket: R2Bucket,
  ownerId: string,
  requestId: string,
  body: string,
): Promise<void> {
  if (body.length > MAX_BODY_BYTES) {
    emitMetric("body_storage_skipped", { type: "request", reason: "too_large" });
    return;
  }
  try {
    await bucket.put(requestKey(ownerId, requestId), body, {
      httpMetadata: { contentType: "application/json" },
    });
    emitMetric("body_storage_write", { type: "request" });
  } catch (err) {
    console.error("[body-storage] Failed to store request body:", err);
    emitMetric("body_storage_error", { type: "request" });
  }
}

/**
 * Store response body in R2. Fire-and-forget — never throws.
 */
export async function storeResponseBody(
  bucket: R2Bucket,
  ownerId: string,
  requestId: string,
  body: string,
): Promise<void> {
  if (body.length > MAX_BODY_BYTES) {
    emitMetric("body_storage_skipped", { type: "response", reason: "too_large" });
    return;
  }
  try {
    await bucket.put(responseKey(ownerId, requestId), body, {
      httpMetadata: { contentType: "application/json" },
    });
    emitMetric("body_storage_write", { type: "response" });
  } catch (err) {
    console.error("[body-storage] Failed to store response body:", err);
    emitMetric("body_storage_error", { type: "response" });
  }
}

/**
 * Store streaming response body (raw SSE text) in R2. Fire-and-forget — never throws.
 */
export async function storeStreamingResponseBody(
  bucket: R2Bucket,
  ownerId: string,
  requestId: string,
  body: string,
): Promise<void> {
  if (body.length > MAX_BODY_BYTES) {
    emitMetric("body_storage_skipped", { type: "response_sse", reason: "too_large" });
    return;
  }
  try {
    await bucket.put(responseSseKey(ownerId, requestId), body, {
      httpMetadata: { contentType: "text/event-stream" },
    });
    emitMetric("body_storage_write", { type: "response_sse" });
  } catch (err) {
    console.error("[body-storage] Failed to store streaming response body:", err);
    emitMetric("body_storage_error", { type: "response_sse" });
  }
}

// --- Stream Body Accumulator ---

export interface StreamBodyAccumulator {
  transform: TransformStream<Uint8Array, Uint8Array>;
  getBody(): string;
  readonly overflow: boolean;
}

/**
 * Create a passthrough TransformStream that accumulates decoded text.
 * Every chunk is enqueued immediately (zero latency impact).
 * After the stream completes, call `getBody()` to retrieve the accumulated text.
 * Text beyond MAX_BODY_BYTES is silently dropped (overflow flag set).
 */
export function createStreamBodyAccumulator(): StreamBodyAccumulator {
  let buffer = "";
  let overflow = false;
  const decoder = new TextDecoder("utf-8", { fatal: false });

  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      controller.enqueue(chunk);

      if (overflow) return;

      const text = decoder.decode(chunk, { stream: true });
      if (buffer.length + text.length > MAX_BODY_BYTES) {
        buffer = buffer + text.slice(0, MAX_BODY_BYTES - buffer.length);
        overflow = true;
        emitMetric("body_storage_overflow", { type: "response_sse" });
        return;
      }
      buffer += text;
    },

    flush() {
      // Flush remaining bytes from the streaming decoder
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      if (!overflow && remaining) {
        if (buffer.length + remaining.length > MAX_BODY_BYTES) {
          buffer = buffer + remaining.slice(0, MAX_BODY_BYTES - buffer.length);
          overflow = true;
          emitMetric("body_storage_overflow", { type: "response_sse" });
        } else {
          buffer += remaining;
        }
      }
    },

    cancel() {
      // Stream cancelled by client — buffer contains partial data, which is fine
      const remaining = decoder.decode(new Uint8Array(), { stream: false });
      if (!overflow && remaining) {
        if (buffer.length + remaining.length > MAX_BODY_BYTES) {
          buffer = buffer + remaining.slice(0, MAX_BODY_BYTES - buffer.length);
          overflow = true;
        } else {
          buffer += remaining;
        }
      }
    },
  });

  return {
    transform,
    getBody() { return buffer; },
    get overflow() { return overflow; },
  };
}

export interface StoredBodies {
  requestBody: string | null;
  responseBody: string | null;
  responseFormat: "json" | "sse" | null;
}

/**
 * Retrieve stored bodies for a request. Returns null for missing objects.
 * Checks for both JSON and SSE response bodies — prefers JSON when both exist.
 */
export async function retrieveBodies(
  bucket: R2Bucket,
  ownerId: string,
  requestId: string,
): Promise<StoredBodies> {
  const [reqObj, resJsonObj, resSseObj] = await Promise.all([
    bucket.get(requestKey(ownerId, requestId)),
    bucket.get(responseKey(ownerId, requestId)),
    bucket.get(responseSseKey(ownerId, requestId)),
  ]);

  let responseBody: string | null = null;
  let responseFormat: "json" | "sse" | null = null;

  if (resJsonObj) {
    responseBody = await resJsonObj.text();
    responseFormat = "json";
  } else if (resSseObj) {
    responseBody = await resSseObj.text();
    responseFormat = "sse";
  }

  return {
    requestBody: reqObj ? await reqObj.text() : null,
    responseBody,
    responseFormat,
  };
}
