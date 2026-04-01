/**
 * Raw usage object from an Anthropic Messages API response.
 * Field names match Anthropic's API exactly.
 */
export interface AnthropicRawUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

/**
 * Optional TTL-specific cache breakdown from the response body.
 * Nested inside `usage.cache_creation` in API responses — the route
 * handler (Phase 4C) extracts it and passes it as a separate parameter.
 */
export interface AnthropicCacheCreationDetail {
  ephemeral_5m_input_tokens?: number;
  ephemeral_1h_input_tokens?: number;
}
