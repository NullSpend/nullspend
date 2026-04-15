/**
 * Gemini API type definitions.
 *
 * Based on the Gemini REST API (v1beta):
 * https://ai.google.dev/gemini-api/docs/api-overview
 */

/** Token usage metadata from a Gemini API response. */
export interface GeminiUsageMetadata {
  /** Total input tokens (includes cached). */
  promptTokenCount?: number;
  /** Output tokens. */
  candidatesTokenCount?: number;
  /** Cached input tokens (subset of promptTokenCount). */
  cachedContentTokenCount?: number;
  /** Thinking/reasoning tokens (subset of output, informational). */
  thoughtsTokenCount?: number;
  /** Sum of all token counts (informational, not used in cost calc). */
  totalTokenCount?: number;
}

/** A single candidate from a Gemini response. */
export interface GeminiCandidate {
  content?: {
    role?: string;
    parts?: GeminiPart[];
  };
  finishReason?: string;
  index?: number;
}

/** A part within a Gemini content block. */
export interface GeminiPart {
  text?: string;
  functionCall?: {
    name: string;
    args: Record<string, unknown>;
    id?: string;
  };
}

/** Top-level Gemini generateContent response. */
export interface GeminiResponse {
  candidates?: GeminiCandidate[];
  usageMetadata?: GeminiUsageMetadata;
  modelVersion?: string;
  responseId?: string;
}
