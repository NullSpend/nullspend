export interface NullSpendAgentOptions {
  /** NullSpend API key (ns_live_sk_... or ns_test_sk_...) */
  apiKey: string;
  /** Session ID for budget-level cost grouping (NOT the SDK's conversation sessionId) */
  budgetSessionId?: string;
  /** Auto-generate a session ID if budgetSessionId is not provided (default: true) */
  autoSession?: boolean;
  /** Key-value tags for cost attribution (max 10 keys, alphanumeric/underscore/hyphen keys, 64-char keys, 256-char values) */
  tags?: Record<string, string>;
  /** 32-char lowercase hex trace ID (e.g. "abcdef0123456789abcdef0123456789") */
  traceId?: string;
  /** NullSpend action ID in ns_act_<UUID> format (e.g. "ns_act_550e8400-e29b-41d4-a716-446655440000") */
  actionId?: string;
  /** Override the proxy URL (default: https://proxy.nullspend.dev) */
  proxyUrl?: string;
  /** Fetch budget policy at init and inject constraints into system prompt (default: true) */
  budgetAwareness?: boolean;
}
