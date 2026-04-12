---
title: "Organizations & Teams"
description: "Organizations are the unit of collaboration in NullSpend. All resources — API keys, budgets, cost events, webhooks — belong to an organization, not an individual user."
---

Organizations are the unit of collaboration in NullSpend. All resources — API keys, budgets, cost events, webhooks — belong to an organization, not an individual user.

## How Organizations Work

When you sign up, NullSpend automatically creates a **personal organization** for you. This is your default workspace — it works identically to a single-user account. You don't need to think about organizations until you want to collaborate.

When you're ready to add teammates, create a **team organization** from the org switcher in the sidebar. Invite members by email, assign roles, and everyone shares the same API keys, budgets, and dashboard.

## Personal vs Team Organizations

| | Personal | Team |
|---|---|---|
| Created | Automatically on signup | Manually via org switcher |
| Members | You only | Invite by email |
| Deletable | No | Yes (owner only) |
| Renamable | No | Yes (admin+) |
| Billing | Own subscription | Own subscription |

Both types work identically for cost tracking and budget enforcement. The only difference is who has access.

## Roles

Every member has one of four roles. Roles are hierarchical — each includes all permissions of the roles below it.

| Role | What they can do |
|---|---|
| **Viewer** | View dashboards, analytics, activity, cost events, budgets, and members. Read-only access. |
| **Member** | Everything Viewer can do, plus create budgets and API keys. |
| **Admin** | Everything Member can do, plus revoke keys, delete budgets, manage webhooks, configure Slack, and invite/manage members. |
| **Owner** | Everything Admin can do, plus manage billing (upgrade/downgrade), delete the organization, and transfer ownership. |

**Viewer seats are free.** Only owner, admin, and member roles count toward the team member limit. Invite as many viewers as you need — finance stakeholders, managers, or anyone who needs visibility without write access.

## Creating a Team Organization

1. Click the **org switcher** in the sidebar (shows your current org name)
2. Click **Create organization**
3. Enter a name and slug (URL-safe identifier, auto-generated from the name)
4. Click **Create** — you're automatically switched to the new org

The new org starts empty — no API keys, budgets, or cost data. You become the owner.

## Inviting Members

1. Go to **Settings** → **Members**
2. Enter the member's email address
3. Select a role (admin, member, or viewer)
4. Click **Invite**

The invited user receives a link. When they click it:
- If they have a NullSpend account, they're added to the org immediately
- If they don't, they sign up first and then accept the invitation

Pending invitations expire after **7 days**. Admins and owners can revoke pending invitations at any time.

### Invitation Limits

| Tier | Team Members (seat-counted) | Viewers |
|---|---|---|
| Free | 3 | Unlimited |
| Pro | Unlimited | Unlimited |
| Enterprise | Unlimited | Unlimited |

The seat count includes both active members and pending invitations (excluding viewer invitations).

## Managing Members

Admins and owners can manage existing members in **Settings** → **Members**:

- **Change role** — Click the role badge next to a member to change their role. Admins cannot change other admins or the owner.
- **Remove member** — Click the remove button. Admins cannot remove other admins or the owner. Resources created by a removed member stay with the organization.

### Permission Boundaries

- Admins **cannot** modify other admins (prevents horizontal privilege escalation)
- The owner role **cannot** be changed via role change — use explicit ownership transfer
- You **cannot** modify or remove yourself

## Switching Organizations

If you're a member of multiple organizations, use the **org switcher** in the sidebar to switch between them. Each org has its own:

- API keys and budgets
- Cost events and analytics
- Webhook endpoints
- Slack configuration
- Subscription and billing

Switching is instant — the dashboard reloads with the selected org's data.

## Billing

Each organization has its own subscription. Personal orgs and team orgs are billed independently.

- **Free tier** — includes 3 budgets, 10 API keys, 2 webhooks, 3 team members, $5K/mo spend cap, 30-day retention
- **Pro ($49/mo)** — unlimited budgets, keys, members, 25 webhooks, $50K/mo spend cap, 90-day retention
- **Enterprise** — unlimited everything, SSO/SAML, custom RBAC, dedicated support

Only the **owner** can manage billing (upgrade, downgrade, access the Stripe portal).

### Downgrade Behavior

If you downgrade from Pro to Free and your resource counts exceed Free tier limits:
- Existing resources are **preserved** — nothing is deleted
- New creation is **blocked** until you're within limits
- A persistent upgrade banner appears on affected pages

## Deleting an Organization

Only the **owner** can delete a team organization. Personal organizations cannot be deleted.

Deleting an org:
- Removes all members and pending invitations
- Revokes all API keys
- Cancels the Stripe subscription (if any)
- Deletes all associated resources (budgets, webhooks, cost events, Slack config)

This action is irreversible.

## Related

- [Budget Configuration](../guides/budget-configuration.md) — set up spending limits for your org
- [API Keys](../api-reference/api-keys-api.md) — manage API keys
- [Webhooks](../webhooks/overview.md) — configure notifications
- [Tags](tags.md) — attribute costs to teams or projects within an org
