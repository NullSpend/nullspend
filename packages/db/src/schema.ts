import { sql } from "drizzle-orm";
import {
  bigint,
  bigserial,
  boolean,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  real,
  text,
  timestamp,
  uniqueIndex,
  uuid,
} from "drizzle-orm/pg-core";

export const ACTION_TYPES = [
  "send_email",
  "http_post",
  "http_delete",
  "shell_command",
  "db_write",
  "file_write",
  "file_delete",
  "budget_increase",
] as const;

export type ActionType = (typeof ACTION_TYPES)[number];

export const ACTION_STATUSES = [
  "pending",
  "approved",
  "rejected",
  "expired",
  "executing",
  "executed",
  "failed",
] as const;

export type ActionStatus = (typeof ACTION_STATUSES)[number];

export const apiKeys = pgTable("api_keys", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  name: text("name").notNull(),
  keyHash: text("key_hash").notNull(),
  keyPrefix: text("key_prefix").notNull(),
  apiVersion: text("api_version").notNull().default("2026-04-01"),
  environment: text("environment").notNull().default("live"),
  defaultTags: jsonb("default_tags").$type<Record<string, string>>().notNull().default(sql`'{}'`),
  allowedModels: text("allowed_models").array(),
  allowedProviders: text("allowed_providers").array(),
  allowedCustomers: text("allowed_customers").array(),
  requireCustomerId: boolean("require_customer_id").notNull().default(false),
  lastUsedAt: timestamp("last_used_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  revokedAt: timestamp("revoked_at", { withTimezone: true }),
}, (table) => [
  uniqueIndex("api_keys_key_hash_idx").on(table.keyHash),
  index("api_keys_user_id_idx").on(table.userId),
  index("api_keys_org_id_idx").on(table.orgId).where(sql`revoked_at IS NULL`),
]);

export type ApiKeyRow = typeof apiKeys.$inferSelect;
export type NewApiKeyRow = typeof apiKeys.$inferInsert;

export const actions = pgTable("actions", {
  id: uuid("id").defaultRandom().primaryKey(),
  ownerUserId: text("owner_user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  agentId: text("agent_id").notNull(),
  actionType: text("action_type").$type<ActionType>().notNull(),
  status: text("status").$type<ActionStatus>().notNull().default("pending"),
  payloadJson: jsonb("payload_json").$type<Record<string, unknown>>().notNull(),
  metadataJson: jsonb("metadata_json").$type<Record<string, unknown> | null>(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  approvedAt: timestamp("approved_at", { withTimezone: true }),
  rejectedAt: timestamp("rejected_at", { withTimezone: true }),
  executedAt: timestamp("executed_at", { withTimezone: true }),
  expiresAt: timestamp("expires_at", { withTimezone: true }),
  expiredAt: timestamp("expired_at", { withTimezone: true }),
  approvedBy: text("approved_by"),
  rejectedBy: text("rejected_by"),
  resultJson: jsonb("result_json").$type<Record<string, unknown> | null>(),
  errorMessage: text("error_message"),
  environment: text("environment"),
  sourceFramework: text("source_framework"),
}, (table) => [
  index("actions_owner_status_created_idx").on(table.ownerUserId, table.status, table.createdAt),
  index("actions_owner_created_idx").on(table.ownerUserId, table.createdAt),
  index("actions_org_id_idx").on(table.orgId),
]);

export type ActionRow = typeof actions.$inferSelect;
export type NewActionRow = typeof actions.$inferInsert;

export const budgets = pgTable("budgets", {
  id: uuid("id").defaultRandom().primaryKey(),
  // DB CHECK allows: user, agent, api_key, team, tag, customer.
  // "agent" and "team" are reserved for future use — gated by Zod validation
  // in lib/validations/budgets.ts which only accepts: api_key, user, tag, customer.
  entityType: text("entity_type").$type<"user" | "api_key" | "tag" | "customer">().notNull(),
  entityId: text("entity_id").notNull(),
  maxBudgetMicrodollars: bigint("max_budget_microdollars", { mode: "number" }).notNull(),
  spendMicrodollars: bigint("spend_microdollars", { mode: "number" }).notNull().default(0),
  policy: text("policy").$type<"strict_block" | "soft_block" | "warn">().notNull().default("strict_block"),
  resetInterval: text("reset_interval").$type<"daily" | "weekly" | "monthly" | "yearly" | null>(),
  thresholdPercentages: integer("threshold_percentages").array().notNull().default([50, 80, 90, 95]),
  velocityLimitMicrodollars: bigint("velocity_limit_microdollars", { mode: "number" }),
  velocityWindowSeconds: integer("velocity_window_seconds").default(60),
  velocityCooldownSeconds: integer("velocity_cooldown_seconds").default(60),
  loopMaxCalls: integer("loop_max_calls"),
  loopWindowSeconds: integer("loop_window_seconds"),
  loopAggregateMaxKeys: integer("loop_aggregate_max_keys"),
  sessionLimitMicrodollars: bigint("session_limit_microdollars", { mode: "number" }),
  finalizationReserveMicrodollars: bigint("finalization_reserve_microdollars", { mode: "number" }).notNull().default(0),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  currentPeriodStart: timestamp("current_period_start", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("budgets_org_entity_idx").on(table.orgId, table.entityType, table.entityId),
  index("budgets_user_id_idx").on(table.userId),
  index("budgets_org_id_idx").on(table.orgId),
]);

export type BudgetRow = typeof budgets.$inferSelect;
export type NewBudgetRow = typeof budgets.$inferInsert;

export const costEvents = pgTable("cost_events", {
  id: uuid("id").defaultRandom().primaryKey(),
  requestId: text("request_id").notNull(),
  apiKeyId: uuid("api_key_id").references(() => apiKeys.id, { onDelete: "set null" }),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  parentRequestId: text("parent_request_id"),
  provider: text("provider").notNull(),
  model: text("model").notNull(),
  inputTokens: integer("input_tokens").notNull(),
  outputTokens: integer("output_tokens").notNull(),
  cachedInputTokens: integer("cached_input_tokens").notNull().default(0),
  reasoningTokens: integer("reasoning_tokens").notNull().default(0),
  costMicrodollars: bigint("cost_microdollars", { mode: "number" }).notNull(),
  durationMs: integer("duration_ms"),
  actionId: uuid("action_id").references(() => actions.id, { onDelete: "set null" }),
  eventType: text("event_type").$type<"llm" | "tool" | "custom">().notNull().default("llm"),
  toolName: text("tool_name"),
  toolServer: text("tool_server"),
  toolCallsRequested: jsonb("tool_calls_requested").$type<{ name: string; id: string }[] | null>(),
  toolDefinitionTokens: integer("tool_definition_tokens").default(0),
  upstreamDurationMs: integer("upstream_duration_ms"),
  sessionId: text("session_id"),
  traceId: text("trace_id"),
  source: text("source").$type<CostEventSource>().notNull().default("proxy"),
  costBreakdown: jsonb("cost_breakdown").$type<{ input?: number; output?: number; cached?: number; reasoning?: number; toolDefinition?: number } | null>(),
  tags: jsonb("tags").$type<Record<string, string>>().notNull().default(sql`'{}'`),
  customerId: text("customer_id"),
  budgetStatus: text("budget_status").$type<"skipped" | "approved" | "denied">(),
  stopReason: text("stop_reason"),
  estimatedCostMicrodollars: bigint("estimated_cost_microdollars", { mode: "number" }),
  // PR-2d: billing period bounds stamped at ingest via resolvePeriodBounds(identity, createdAt).
  // Paid orgs: Stripe subscription period. Free/unpaid: calendar-month fallback.
  // NEVER null for new rows; legacy rows pre-PR-2d may be NULL.
  // Used by the reconciliation cron to increment the EVENT's original period, not the current
  // period at reconciliation time.
  periodStart: timestamp("period_start", { withTimezone: true }),
  periodEnd: timestamp("period_end", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("cost_events_request_id_provider_idx").on(table.requestId, table.provider),
  index("cost_events_user_id_created_at_idx").on(table.userId, table.createdAt),
  index("cost_events_api_key_id_created_at_idx").on(table.apiKeyId, table.createdAt),
  index("cost_events_provider_model_created_at_idx").on(table.provider, table.model, table.createdAt),
  index("cost_events_action_id_idx").on(table.actionId),
  index("cost_events_event_type_idx").on(table.eventType),
  index("cost_events_org_session_created_idx").on(table.orgId, table.sessionId, table.createdAt).where(sql`session_id IS NOT NULL`),
  index("cost_events_trace_id_idx").on(table.traceId).where(sql`trace_id IS NOT NULL`),
  index("cost_events_tags_idx").using("gin", table.tags),
  index("cost_events_org_id_created_at_idx").on(table.orgId, table.createdAt),
  index("cost_events_customer_id_idx").on(table.customerId).where(sql`customer_id IS NOT NULL`),
  // PR-2d: index for period-scoped queries (reconciliation cron, future billing analytics).
  // Partial index skips legacy NULL rows from pre-PR-2d writes.
  index("idx_cost_events_period_start").on(table.orgId, table.periodStart).where(sql`period_start IS NOT NULL`),
]);

export type CostEventRow = typeof costEvents.$inferSelect;
export type NewCostEventRow = typeof costEvents.$inferInsert;

export const subscriptions = pgTable("subscriptions", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull().unique(),
  stripeCustomerId: text("stripe_customer_id").notNull(),
  stripeSubscriptionId: text("stripe_subscription_id").notNull(),
  tier: text("tier").notNull(),
  status: text("status").notNull(),
  currentPeriodStart: timestamp("current_period_start", { withTimezone: true }),
  currentPeriodEnd: timestamp("current_period_end", { withTimezone: true }),
  cancelAtPeriodEnd: boolean("cancel_at_period_end").notNull().default(false),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("subscriptions_stripe_customer_id_idx").on(table.stripeCustomerId),
]);

export type SubscriptionRow = typeof subscriptions.$inferSelect;
export type NewSubscriptionRow = typeof subscriptions.$inferInsert;

export const COST_EVENT_SOURCES = ["proxy", "api", "mcp", "sdk"] as const;
export type CostEventSource = (typeof COST_EVENT_SOURCES)[number];

export const TOOL_COST_SOURCES = ["discovered", "manual"] as const;

export type ToolCostSource = (typeof TOOL_COST_SOURCES)[number];

export const toolCosts = pgTable("tool_costs", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  serverName: text("server_name").notNull(),
  toolName: text("tool_name").notNull(),
  costMicrodollars: bigint("cost_microdollars", { mode: "number" }).notNull().default(0),
  suggestedCost: bigint("suggested_cost", { mode: "number" }).notNull().default(0),
  source: text("source").$type<ToolCostSource>().notNull().default("discovered"),
  description: text("description"),
  annotations: jsonb("annotations").$type<Record<string, unknown> | null>(),
  lastSeenAt: timestamp("last_seen_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("tool_costs_user_server_tool_idx").on(table.userId, table.serverName, table.toolName),
  index("tool_costs_user_id_idx").on(table.userId),
  index("tool_costs_org_id_idx").on(table.orgId),
]);

export type ToolCostRow = typeof toolCosts.$inferSelect;
export type NewToolCostRow = typeof toolCosts.$inferInsert;

export const webhookEndpoints = pgTable("webhook_endpoints", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: text("user_id").notNull(),
  orgId: uuid("org_id").notNull(),
  url: text("url").notNull(),
  description: text("description"),
  signingSecret: text("signing_secret").notNull(),
  previousSigningSecret: text("previous_signing_secret"),
  secretRotatedAt: timestamp("secret_rotated_at", { withTimezone: true }),
  eventTypes: text("event_types").array().notNull().default([]),
  enabled: boolean("enabled").notNull().default(true),
  apiVersion: text("api_version").notNull().default("2026-04-01"),
  payloadMode: text("payload_mode").$type<"full" | "thin">().notNull().default("full"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index("webhook_endpoints_user_id_idx").on(table.userId),
  index("webhook_endpoints_org_id_idx").on(table.orgId),
]);

export type WebhookEndpointRow = typeof webhookEndpoints.$inferSelect;
export type NewWebhookEndpointRow = typeof webhookEndpoints.$inferInsert;

export const webhookDeliveries = pgTable("webhook_deliveries", {
  id: uuid("id").defaultRandom().primaryKey(),
  endpointId: uuid("endpoint_id").notNull()
    .references(() => webhookEndpoints.id, { onDelete: "cascade" }),
  eventType: text("event_type").notNull(),
  eventId: text("event_id").notNull(),
  status: text("status").notNull().default("pending"),
  attempts: integer("attempts").notNull().default(0),
  lastAttemptAt: timestamp("last_attempt_at", { withTimezone: true }),
  responseStatus: integer("response_status"),
  responseBodyPreview: text("response_body_preview"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index("webhook_deliveries_endpoint_id_idx").on(table.endpointId, table.createdAt),
  index("webhook_deliveries_event_id_idx").on(table.eventId),
]);

export type WebhookDeliveryRow = typeof webhookDeliveries.$inferSelect;
export type NewWebhookDeliveryRow = typeof webhookDeliveries.$inferInsert;

// ---------------------------------------------------------------------------
// Organizations
// ---------------------------------------------------------------------------

export const organizations = pgTable("organizations", {
  id: uuid("id").defaultRandom().primaryKey(),
  name: text("name").notNull(),
  slug: text("slug").notNull().unique(),
  isPersonal: boolean("is_personal").notNull().default(false),
  logoUrl: text("logo_url"),
  metadata: jsonb("metadata").$type<Record<string, unknown>>().notNull().default(sql`'{}'`),
  createdBy: text("created_by").notNull(),
  // If non-null, overrides TIERS[tier].retentionDays for this org. Used to grandfather
  // existing orgs when tier retention changes (plan §1.1) and to implement the 14-day
  // grace window on downgrade (refined spec §9.5). NULL = use tier default.
  retentionOverrideDays: integer("retention_override_days"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("organizations_personal_user_idx").on(table.createdBy).where(sql`is_personal = true`),
]);

export type OrganizationRow = typeof organizations.$inferSelect;
export type NewOrganizationRow = typeof organizations.$inferInsert;

export const orgMemberships = pgTable("org_memberships", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  userId: text("user_id").notNull(),
  role: text("role").$type<"owner" | "admin" | "member" | "viewer">().notNull().default("member"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("org_memberships_org_user_idx").on(table.orgId, table.userId),
  index("org_memberships_user_id_idx").on(table.userId),
]);

export type OrgMembershipRow = typeof orgMemberships.$inferSelect;
export type NewOrgMembershipRow = typeof orgMemberships.$inferInsert;

export const orgInvitations = pgTable("org_invitations", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  email: text("email").notNull(),
  role: text("role").$type<"owner" | "admin" | "member" | "viewer">().notNull().default("member"),
  invitedBy: text("invited_by").notNull(),
  tokenHash: text("token_hash").notNull().unique(),
  tokenPrefix: text("token_prefix").notNull(),
  status: text("status").$type<"pending" | "accepted" | "declined" | "revoked" | "expired">().notNull().default("pending"),
  acceptedBy: text("accepted_by"),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  acceptedAt: timestamp("accepted_at", { withTimezone: true }),
  revokedAt: timestamp("revoked_at", { withTimezone: true }),
}, (table) => [
  index("org_invitations_org_id_idx").on(table.orgId),
  index("org_invitations_email_idx").on(table.email),
  uniqueIndex("org_invitations_pending_idx").on(table.orgId, table.email).where(sql`status = 'pending'`),
]);

export type OrgInvitationRow = typeof orgInvitations.$inferSelect;
export type NewOrgInvitationRow = typeof orgInvitations.$inferInsert;

// ---------------------------------------------------------------------------
// Audit Events
// ---------------------------------------------------------------------------

export const auditEvents = pgTable("audit_events", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull(),
  actorId: text("actor_id").notNull(),
  action: text("action").notNull(),
  resourceType: text("resource_type").notNull(),
  resourceId: text("resource_id"),
  metadata: jsonb("metadata").$type<Record<string, unknown>>(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index("audit_events_org_id_idx").on(table.orgId),
  index("audit_events_created_at_idx").on(table.createdAt),
]);

export type AuditEventRow = typeof auditEvents.$inferSelect;
export type NewAuditEventRow = typeof auditEvents.$inferInsert;

// ── Sessions (materialized from cost_events) ────────────────────────

export const sessions = pgTable("sessions", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull(),
  sessionId: text("session_id").notNull(),
  eventCount: integer("event_count").notNull().default(0),
  totalCostMicrodollars: bigint("total_cost_microdollars", { mode: "number" }).notNull().default(0),
  totalInputTokens: integer("total_input_tokens").notNull().default(0),
  totalOutputTokens: integer("total_output_tokens").notNull().default(0),
  totalDurationMs: integer("total_duration_ms").notNull().default(0),
  firstEventAt: timestamp("first_event_at", { withTimezone: true }).notNull().defaultNow(),
  lastEventAt: timestamp("last_event_at", { withTimezone: true }).notNull().defaultNow(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("sessions_org_session_idx").on(table.orgId, table.sessionId),
  index("sessions_org_last_event_idx").on(table.orgId, table.lastEventAt),
]);

export type SessionRow = typeof sessions.$inferSelect;

// ── Margins (Stripe connections, customer revenue, mappings) ─────────

export const STRIPE_CONNECTION_STATUSES = ["active", "error", "revoked"] as const;
export type StripeConnectionStatus = (typeof STRIPE_CONNECTION_STATUSES)[number];

export const stripeConnections = pgTable("stripe_connections", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  encryptedKey: text("encrypted_key").notNull(),
  keyPrefix: text("key_prefix").notNull(),
  status: text("status").$type<StripeConnectionStatus>().notNull().default("active"),
  lastSyncAt: timestamp("last_sync_at", { withTimezone: true }),
  lastError: text("last_error"),
  lastSyncMeta: jsonb("last_sync_meta").$type<{ skippedCurrencies?: Record<string, number> }>(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("stripe_connections_org_id_idx").on(table.orgId),
]);

export type StripeConnectionRow = typeof stripeConnections.$inferSelect;
export type NewStripeConnectionRow = typeof stripeConnections.$inferInsert;

export const customerRevenue = pgTable("customer_revenue", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  stripeCustomerId: text("stripe_customer_id").notNull(),
  customerName: text("customer_name"),
  customerEmail: text("customer_email"),
  avatarUrl: text("avatar_url"),
  periodStart: timestamp("period_start", { withTimezone: true, mode: "date" }).notNull(),
  amountMicrodollars: bigint("amount_microdollars", { mode: "number" }).notNull(),
  invoiceCount: integer("invoice_count").notNull().default(1),
  currency: text("currency").notNull().default("usd"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("customer_revenue_org_customer_period_idx").on(table.orgId, table.stripeCustomerId, table.periodStart),
  index("customer_revenue_org_period_idx").on(table.orgId, table.periodStart),
]);

export type CustomerRevenueRow = typeof customerRevenue.$inferSelect;
export type NewCustomerRevenueRow = typeof customerRevenue.$inferInsert;

export const MATCH_TYPES = ["auto", "manual"] as const;
export type MatchType = (typeof MATCH_TYPES)[number];

export const customerMappings = pgTable("customer_mappings", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  stripeCustomerId: text("stripe_customer_id").notNull(),
  tagKey: text("tag_key").notNull().default("customer"),
  tagValue: text("tag_value").notNull(),
  matchType: text("match_type").$type<MatchType>().notNull(),
  confidence: real("confidence"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("customer_mappings_org_stripe_tag_idx").on(table.orgId, table.stripeCustomerId, table.tagKey),
  uniqueIndex("customer_mappings_org_tag_value_idx").on(table.orgId, table.tagKey, table.tagValue),
]);

export type CustomerMappingRow = typeof customerMappings.$inferSelect;
export type NewCustomerMappingRow = typeof customerMappings.$inferInsert;

/**
 * Per-customer settings that are independent of Stripe revenue sync.
 * Decoupled from customer_mappings (which is Stripe-specific) so orgs
 * using per-customer budgets WITHOUT Stripe integration can still
 * configure customer-level overrides.
 *
 * Currently holds the optional per-customer upgrade URL surfaced in
 * customer_budget_exceeded denial responses. Fall-back chain:
 *   customer_settings.upgrade_url (this table)
 *     → organizations.metadata.upgradeUrl (org-level default)
 *     → omitted from response
 *
 * Keyed on (org_id, customer_id) where customer_id is the tag_value
 * a client sends via the X-NullSpend-Customer header.
 */
export const customerSettings = pgTable("customer_settings", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  customerId: text("customer_id").notNull(),
  upgradeUrl: text("upgrade_url"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  // Name matches the auto-generated constraint index from the
  // UNIQUE (org_id, customer_id) table constraint declared in
  // drizzle/0056_customer_settings_table.sql. Audit E3 removed the
  // duplicate standalone index; this declaration is structurally a
  // no-op against the live DB because the constraint already enforces
  // uniqueness + creates an equivalent index.
  uniqueIndex("customer_settings_org_id_customer_id_key").on(table.orgId, table.customerId),
]);

export type CustomerSettingsRow = typeof customerSettings.$inferSelect;
export type NewCustomerSettingsRow = typeof customerSettings.$inferInsert;

/**
 * PXY-2: Idempotent budget spend reconciliation.
 *
 * Dedup table that makes `updateBudgetSpend` safe to retry. Each row
 * represents a successfully reconciled (request_id, entity) pair.
 * ON CONFLICT DO NOTHING prevents double-counting on queue retries,
 * outbox alarm retries, or manual recovery.
 *
 * Pruned after 7 days by the DO alarm handler.
 */
export const reconciledRequests = pgTable("reconciled_requests", {
  requestId: text("request_id").notNull(),
  entityType: text("entity_type").notNull(),
  entityId: text("entity_id").notNull(),
  orgId: uuid("org_id").notNull(),
  costMicrodollars: bigint("cost_microdollars", { mode: "number" }).notNull(),
  reconciledAt: timestamp("reconciled_at", { withTimezone: true }).defaultNow().notNull(),
}, (table) => [
  primaryKey({ columns: [table.requestId, table.entityType, table.entityId] }),
  index("reconciled_requests_reconciled_at_idx").on(table.reconciledAt),
]);

// ---------------------------------------------------------------------------
// Margin alert dedup (MRG-5)
// ---------------------------------------------------------------------------

/**
 * Tracks which margin threshold crossings have already been dispatched
 * so the same alert is not re-fired on every sync.
 *
 * Unique on (org_id, tag_value, period, to_tier) — one alert per customer
 * per period per worsening tier transition.
 */
export const marginAlertsSent = pgTable("margin_alerts_sent", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  tagValue: text("tag_value").notNull(),
  period: text("period").notNull(),
  fromTier: text("from_tier").notNull(),
  toTier: text("to_tier").notNull(),
  sentAt: timestamp("sent_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("margin_alerts_sent_dedup_idx").on(table.orgId, table.tagValue, table.period, table.toTier),
  index("margin_alerts_sent_org_period_idx").on(table.orgId, table.period),
]);

// ---------------------------------------------------------------------------
// Governed request usage history (PR-2b)
// ---------------------------------------------------------------------------

/**
 * Per-org billing-period governed-request counts. Populated by the DO's
 * `incrementPlanCounter` path via the `pg_sync_outbox_plan_counter` outbox
 * + alarm handler (`upsertPlanCounterPeriod`). Additive upserts — safe under
 * out-of-order delivery.
 *
 * Primary key `(org_id, period_start)` — one row per org per billing period.
 * Period bounds follow the Stripe subscription if present, calendar-month
 * otherwise (per plan Decision #16). The DO stamps period bounds at
 * increment time so late-arriving deltas land on the ORIGINAL period row.
 */
export const orgPeriodUsage = pgTable("org_period_usage", {
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  periodStart: timestamp("period_start", { withTimezone: true }).notNull(),
  periodEnd: timestamp("period_end", { withTimezone: true }).notNull(),
  governedRequestsCount: bigint("governed_requests_count", { mode: "number" }).notNull().default(0),
  lastUpdatedAt: timestamp("last_updated_at", { withTimezone: true }).notNull().defaultNow(),
  // PR-6a overage foundation (drizzle/0066). All nullable; zero backfill.
  //   overageInvoiceItemId / overageInvoicedAt — PR-6b cron writes on Stripe success. NEVER written in 6a.
  //   tierAtPeriodEnd / statusAtPeriodEnd       — DO's stampPeriodClose writes at period boundary (R2 A2 + R3 P0).
  //   disposition                                — 5-value state machine (PR-6b CX-R2-4).
  //                                                billable_pending  | evaluated_skipped : DO writer stamps at period close.
  //                                                billing_in_flight                     : PR-6b cron atomic-claim in-progress.
  //                                                billed                                : PR-6b cron flipped after finalize.
  //                                                needs_review                          : dead-letter (retry exhausted / orphaned / age-out).
  //   claimedByRunId / claimedAt                — PR-6b atomic-claim columns (drizzle/0071). Populated on billing_in_flight;
  //                                                cleared on revert/complete/sweep. Stale-claim sweeper reverts rows whose
  //                                                claimedAt < now()-15m.
  overageInvoiceItemId: text("overage_invoice_item_id"),
  overageInvoicedAt: timestamp("overage_invoiced_at", { withTimezone: true }),
  tierAtPeriodEnd: text("tier_at_period_end"),
  statusAtPeriodEnd: text("status_at_period_end"),
  disposition: text("disposition").$type<
    | "billable_pending"
    | "evaluated_skipped"
    | "billing_in_flight"
    | "billed"
    | "needs_review"
  >(),
  claimedByRunId: uuid("claimed_by_run_id"),
  claimedAt: timestamp("claimed_at", { withTimezone: true }),
}, (table) => [
  primaryKey({ columns: [table.orgId, table.periodStart] }),
  index("idx_org_period_usage_period").on(table.periodStart),
  // PR-6a R3 P3: partial index covers genuinely-unbilled-but-billable rows only.
  // The PR-6b cron scans this for invoice candidates; keying on disposition
  // prevents the `overage_pending_stalled` alert from false-positiving on
  // evaluated-and-skipped rows.
  index("idx_opu_billable_pending").on(table.periodEnd).where(sql`disposition = 'billable_pending'`),
]);

export type OrgPeriodUsageRow = typeof orgPeriodUsage.$inferSelect;
export type NewOrgPeriodUsageRow = typeof orgPeriodUsage.$inferInsert;

/**
 * PR-6b: per-reconcile audit row for the overage-billing cron.
 *
 * Spec: docs/plans/pricing-pr6b-overage-billing-cron.md §4 (schema) + C2 +
 *       OV-8 + CX-R2-4 (row-per-reconciliation) + CX-R3-1 (invoice_state
 *       state machine) + CX-R3-4 (currency persistence).
 *
 * One row per invocation of `reconcileOverageForPeriod`. INSERTed BEFORE the
 * atomic claim so a mid-run crash leaves an audit row the stale-claim sweeper
 * can reconcile.
 *
 * `invoice_state` drives Stripe-flow replay after partial crash:
 *   none → draft_created → item_attached → finalized
 *                       ↘ voided (on retry exhaustion at any step)
 *
 * `final_status` is NULL while in-flight; terminal transitions populate it.
 * Watcher reads `WHERE final_status='failed' AND completed_at > now()-1h`
 * for its failureCount1h metric.
 */
export const overageCronRuns = pgTable("overage_cron_runs", {
  runId: uuid("run_id").primaryKey(),
  orgId: uuid("org_id").notNull(),
  periodStart: timestamp("period_start", { withTimezone: true }).notNull(),
  startedAt: timestamp("started_at", { withTimezone: true }).notNull().defaultNow(),
  completedAt: timestamp("completed_at", { withTimezone: true }),
  finalStatus: text("final_status").$type<
    | "ok"
    | "failed"
    | "needs_review"
    | "skipped"
    | "abandoned"
  >(),
  attemptCount: integer("attempt_count").notNull().default(0),
  lastErrorSummary: jsonb("last_error_summary").$type<Record<string, unknown>>(),
  invoiceItemId: text("invoice_item_id"),
  stripeDraftInvoiceId: text("stripe_draft_invoice_id"),
  invoiceState: text("invoice_state")
    .$type<"none" | "draft_created" | "item_attached" | "finalized" | "voided">()
    .notNull()
    .default("none"),
  currency: text("currency"),
}, (table) => [
  // completedAt DESC matches the live migration DDL + how queries read this
  // (MAX/latest-first). Drizzle-kit regeneration would otherwise emit an ASC
  // index and silently drift from prod.
  index("idx_overage_cron_runs_final_status_completed_at").on(
    table.finalStatus,
    table.completedAt.desc(),
  ),
  index("idx_overage_cron_runs_org_period").on(table.orgId, table.periodStart),
]);

export type OverageCronRunRow = typeof overageCronRuns.$inferSelect;
export type NewOverageCronRunRow = typeof overageCronRuns.$inferInsert;

/**
 * PR-2b build-audit Finding #1 + edge-case EC6 + codex-final F2: Dedup guard
 * for plan-counter outbox retries, scoped to `(org_id, request_id, period_start)`.
 *
 * The DO alarm writes to `org_period_usage` additively. If the Postgres write
 * succeeds but the DO-local ack of the outbox row fails (DO crash, isolate
 * eviction, network hiccup between the two SQL calls), the next alarm retries
 * the same request and double-adds the delta. This table gives the upsert a
 * Postgres-side idempotency guard: `INSERT ... ON CONFLICT DO NOTHING` on the
 * `(org_id, request_id, period_start)` tuple, then conditional upsert against
 * `org_period_usage` only when the dedup insert actually landed.
 *
 * **Composite key rationale:**
 * - `period_start` (EC6): the DO-side dedup is period-scoped — an SDK caller
 *   reusing a deterministic idempotency key across a billing-period boundary
 *   gets their stale row deleted and counted fresh in the new period. Without
 *   `period_start` in the PK, the new period's upsert would conflict against
 *   the prior period's row and be silently rejected.
 * - `org_id` (codex-final F2): two DIFFERENT orgs using the same deterministic
 *   idempotency key in the SAME billing period would otherwise collide — the
 *   second org's upsert would be silently rejected even though it represents
 *   independent governed-request work. Scoping by `org_id` keeps each org's
 *   dedup namespace isolated.
 *
 * Mirrors the `reconciled_requests` pattern used by `updateBudgetSpend` for
 * the budget-spend path — same M-3 "split-brain on retry" defense.
 *
 * `written_at` retained for TTL pruning via a future housekeeping task
 * (rows older than ~7 days can be deleted; outbox max-retry window is minutes).
 */
export const planCounterSyncRequests = pgTable("plan_counter_sync_requests", {
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  requestId: text("request_id").notNull(),
  periodStart: timestamp("period_start", { withTimezone: true }).notNull(),
  writtenAt: timestamp("written_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.orgId, table.requestId, table.periodStart] }),
  index("idx_plan_counter_sync_requests_written_at").on(table.writtenAt),
]);

// ---------------------------------------------------------------------------
// PR-2e Sub-PR 11: launch-watcher telemetry tables.
//
// All three tables back the new `GET /api/internal/metrics-summary?window=15m`
// dashboard endpoint, which the watcher worker polls every 5 min to detect
// launch-day misconfigurations (M-30 pattern: pricing page advertises an
// enforcement contract the proxy isn't actually enforcing).
//
// Per PR-2e codex R4 #1 cleanup contract:
//   - planCounterCronHistory is GLOBAL (no org_id) -- skipped by the drift
//     test at tests/e2e/lib/orphan-cleanup-tables.test.ts.
//   - planCounterSyncFailures + planCounterDivergences are org-scoped with
//     ON DELETE CASCADE -- excluded from ORG_ID_CLEANUP_TABLE_NAMES per the
//     drift test rule "CASCADE tables MUST NOT be in inventory."
//
// Per PR-2e codex R4 #5 + R5 #3 schema lock: every column is NOT NULL.
// Nullable telemetry columns silently drop or undercount rows.
//
// Per PR-2e codex R3 #1: planCounterSyncFailures.count is the BATCH SIZE
// (1-100), NOT the failure ordinal. The metrics-summary endpoint MUST query
// SUM(count) not COUNT(*).
// ---------------------------------------------------------------------------

export const planCounterCronHistory = pgTable("plan_counter_cron_history", {
  id: bigserial("id", { mode: "number" }).primaryKey(),
  tickAt: timestamp("tick_at", { withTimezone: true }).notNull().defaultNow(),
  batchFull: boolean("batch_full").notNull(),
  candidateCount: integer("candidate_count").notNull(),
  reconciled: integer("reconciled").notNull(),
  failed: integer("failed").notNull(),
}, (table) => [
  index("idx_plan_counter_cron_history_tick_at").on(table.tickAt),
]);

export const planCounterSyncFailures = pgTable("plan_counter_sync_failures", {
  id: bigserial("id", { mode: "number" }).primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  failedAt: timestamp("failed_at", { withTimezone: true }).notNull().defaultNow(),
  reason: text("reason").notNull(),
  count: integer("count").notNull(),
}, (table) => [
  index("idx_plan_counter_sync_failures_failed_at").on(table.failedAt),
  index("idx_plan_counter_sync_failures_org_id").on(table.orgId),
]);

export const planCounterDivergences = pgTable("plan_counter_divergences", {
  id: bigserial("id", { mode: "number" }).primaryKey(),
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  divergenceAt: timestamp("divergence_at", { withTimezone: true }).notNull().defaultNow(),
  tier: text("tier").notNull(),
  storedPeriodStart: timestamp("stored_period_start", { withTimezone: true }).notNull(),
  incomingPeriodStart: timestamp("incoming_period_start", { withTimezone: true }).notNull(),
}, (table) => [
  index("idx_plan_counter_divergences_divergence_at").on(table.divergenceAt),
  index("idx_plan_counter_divergences_org_id").on(table.orgId),
]);

/**
 * PR-2e F2-partial: dedup guard for the divergence outbox (codex R7 Issue 1A).
 *
 * Mirrors `planCounterSyncRequests` — PG-side retry-idempotency for the
 * DO alarm's divergence outbox. DO generates `eventId = crypto.randomUUID()`
 * at divergence-detection time; `upsertPlanCounterDivergence` does
 * `INSERT ON CONFLICT DO NOTHING` on this dedup table, then INSERTs into
 * `plan_counter_divergences` only when the dedup row actually landed.
 *
 * Closes the retry race where PG commit succeeds but DO-local ack fails —
 * without dedup, the next alarm retry would write a second divergence
 * row, inflating the launch-watcher's COUNT(*) over 15m.
 *
 * Per cleanup-contract drift test: CASCADE on organizations → NOT in
 * ORG_ID_CLEANUP_TABLE_NAMES.
 *
 * TODO(TODOS.md): retry-state pruning. `writtenAt` column + index exist for
 * future 7-day pruning. Same gap as `planCounterSyncRequests` (shipped in
 * PR-2b). At current volumes (<100 orgs, ~0-5 divergence events/org/month),
 * the table stays under ~100 rows for the foreseeable future — not a launch
 * blocker. Add both tables to the daily reconcile cron when retry-state
 * volumes grow past observable dashboard-query cost.
 */
export const planCounterDivergenceDedup = pgTable("plan_counter_divergence_dedup", {
  orgId: uuid("org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  eventId: text("event_id").notNull(),
  writtenAt: timestamp("written_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.orgId, table.eventId] }),
  index("idx_plan_counter_divergence_dedup_written_at").on(table.writtenAt),
]);

// ---------------------------------------------------------------------------
// ISSUE-014: Stripe inbound webhook idempotency (drizzle/0072).
//
// Cross-instance dedup table. Replaces the in-memory Map in
// app/api/stripe/webhook/route.ts which couldn't survive multi-instance
// Vercel Fluid Compute or cold starts.
//
// State machine: not-seen → processing → processed.
//   - processing entries older than 5 min are eligible for stale reclaim.
//   - processed entries are terminal; subsequent same-id deliveries dedupe.
//   - On transient failures, the row is DELETEd so Stripe retries are
//     accepted and the next attempt re-claims fresh.
// ---------------------------------------------------------------------------
export const stripeWebhookEventsProcessed = pgTable("stripe_webhook_events_processed", {
  eventId: text("event_id").primaryKey(),
  state: text("state").$type<"processing" | "processed">().notNull(),
  processedAt: timestamp("processed_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index("idx_stripe_webhook_events_processed_processed_at")
    .on(table.processedAt)
    .where(sql`state = 'processing'`),
]);

export type StripeWebhookEventProcessedRow = typeof stripeWebhookEventsProcessed.$inferSelect;
export type NewStripeWebhookEventProcessedRow = typeof stripeWebhookEventsProcessed.$inferInsert;
