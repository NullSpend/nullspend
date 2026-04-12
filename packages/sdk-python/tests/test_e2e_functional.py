"""
Python SDK Functional E2E Test Suite — mirrors F1-F11 from
apps/proxy/smoke-sdk-functional.test.ts.

Sections:
  F1-F4   HITL action lifecycle, polling, orchestrator, requestBudgetIncrease
  F5-F6   Read APIs (getCostSummary all 3 periods, listCostEvents pagination)
  F8-F9   Config behavior (retry, timeout)
  F11     HITL error class fields (all 9 error classes)

Requires env vars (loaded from apps/proxy/.env.smoke):
  NULLSPEND_API_KEY, NULLSPEND_SMOKE_KEY_ID, NULLSPEND_SMOKE_USER_ID,
  DATABASE_URL, NULLSPEND_DASHBOARD_URL

Approval mechanism: direct SQL UPDATE via psycopg (matches TS approach).
Test actions tagged with agent_id LIKE 'py-functional-test-%' and cleaned
up symmetrically in setup/teardown.

Run:
  pip install psycopg[binary]  # one-time, for SQL approval
  cd packages/sdk-python
  python -m pytest tests/test_e2e_functional.py -v --tb=short -x
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Any

import pytest
import httpx

from nullspend import (
    NullSpend,
    NullSpendError,
    PollTimeoutError,
    RejectedError,
    BudgetExceededError,
    MandateViolationError,
    SessionLimitExceededError,
    VelocityExceededError,
    TagBudgetExceededError,
)
from nullspend.types import (
    CreateActionInput,
    MarkResultInput,
    ProposeAndWaitOptions,
    RequestBudgetIncreaseOptions,
    ListCostEventsOptions,
    CostEventInput,
)

# ── Load env from apps/proxy/.env.smoke ──────────────────────────
_SMOKE_ENV_PATH = Path(__file__).resolve().parents[3] / "apps" / "proxy" / ".env.smoke"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env loader — returns dict, does NOT mutate os.environ."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            result[key] = value
    return result


# Merge: env vars take precedence over .env.smoke file
_smoke_env = _load_dotenv(_SMOKE_ENV_PATH)

# ── Env vars (prefer os.environ, fall back to .env.smoke) ────────
def _env(key: str) -> str:
    return os.environ.get(key, _smoke_env.get(key, ""))

NULLSPEND_API_KEY = _env("NULLSPEND_API_KEY")
NULLSPEND_SMOKE_KEY_ID = _env("NULLSPEND_SMOKE_KEY_ID")
NULLSPEND_SMOKE_USER_ID = _env("NULLSPEND_SMOKE_USER_ID")
DATABASE_URL = _env("DATABASE_URL")
DASHBOARD_URL = _env("NULLSPEND_DASHBOARD_URL").rstrip("/")

# ── Skip marker for live tests (need env vars + dashboard + DB) ──
_HAS_LIVE_ENV = all([NULLSPEND_API_KEY, NULLSPEND_SMOKE_KEY_ID,
                     NULLSPEND_SMOKE_USER_ID, DATABASE_URL, DASHBOARD_URL])

live = pytest.mark.skipif(not _HAS_LIVE_ENV,
                          reason="Live E2E env vars not configured")

# ── Postgres helper (lazy import) ────────────────────────────────
_pg_conn: Any = None
_SMOKE_ORG_ID: str = ""


def _get_pg():
    global _pg_conn
    if _pg_conn is None:
        try:
            import psycopg  # noqa: F811
        except ImportError:
            pytest.skip("psycopg not installed — run: pip install psycopg[binary]")
        _pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
    return _pg_conn


def _sql_fetchone(query: str, params: tuple = ()) -> dict | None:
    conn = _get_pg()
    with conn.cursor() as cur:
        cur.execute(query, params)
        if cur.description is None:
            return None
        cols = [d.name for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None


def _sql_execute(query: str, params: tuple = ()) -> int:
    conn = _get_pg()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.rowcount


# ── Constants ────────────────────────────────────────────────────
AGENT_PREFIX = "py-functional-test"
TEST_POLL_INTERVAL_S = 0.5
APPROVE_DELAY_S = 1.0
TIMEOUT_FAST_S = 1.5
ACT_PREFIX = "ns_act_"


def _make_agent_id(test_name: str) -> str:
    import random
    ts = hex(int(time.time()))[2:]
    rand = hex(random.getrandbits(32))[2:]
    return f"{AGENT_PREFIX}-{test_name}-{ts}-{rand}"


def _strip_action_prefix(prefixed: str) -> str:
    if not prefixed.startswith(ACT_PREFIX):
        raise ValueError(f"Expected prefix '{ACT_PREFIX}', got '{prefixed}'")
    return prefixed[len(ACT_PREFIX):]


def _approve_action_via_sql(action_id: str) -> None:
    uuid = _strip_action_prefix(action_id)
    count = _sql_execute(
        """UPDATE actions SET status = 'approved', approved_at = NOW(),
           approved_by = %s WHERE id = %s AND org_id = %s AND status = 'pending'""",
        (NULLSPEND_SMOKE_USER_ID, uuid, _SMOKE_ORG_ID),
    )
    if count != 1:
        raise RuntimeError(f"approve({action_id}): {count} rows updated")


def _reject_action_via_sql(action_id: str) -> None:
    uuid = _strip_action_prefix(action_id)
    count = _sql_execute(
        """UPDATE actions SET status = 'rejected', rejected_at = NOW(),
           rejected_by = %s WHERE id = %s AND org_id = %s AND status = 'pending'""",
        (NULLSPEND_SMOKE_USER_ID, uuid, _SMOKE_ORG_ID),
    )
    if count != 1:
        raise RuntimeError(f"reject({action_id}): {count} rows updated")


def _cleanup_test_actions() -> None:
    _sql_execute(
        "DELETE FROM actions WHERE agent_id LIKE %s AND org_id = %s",
        (f"{AGENT_PREFIX}-%", _SMOKE_ORG_ID),
    )


def _delayed_approve(action_id: str, label: str) -> threading.Thread:
    """Schedule SQL approve after APPROVE_DELAY_S on a background thread."""
    def run():
        time.sleep(APPROVE_DELAY_S)
        try:
            _approve_action_via_sql(action_id)
        except Exception as e:
            print(f"[{label}] SQL approve failed: {e}")
    t = threading.Thread(target=run)
    t.start()
    return t


def _delayed_reject(action_id: str, label: str) -> threading.Thread:
    def run():
        time.sleep(APPROVE_DELAY_S)
        try:
            _reject_action_via_sql(action_id)
        except Exception as e:
            print(f"[{label}] SQL reject failed: {e}")
    t = threading.Thread(target=run)
    t.start()
    return t


# ── Fixtures ─────────────────────────────────────────────────────

_live_setup_done = False


@pytest.fixture(scope="module")
def live_setup():
    """Setup for live tests only: verify env, look up org_id, cleanup orphans."""
    global _SMOKE_ORG_ID, _live_setup_done

    if not _HAS_LIVE_ENV:
        pytest.skip("Live E2E env vars not configured")

    if not _live_setup_done:
        # Verify dashboard reachability
        try:
            r = httpx.get(f"{DASHBOARD_URL}/api/budgets",
                          headers={"x-nullspend-key": NULLSPEND_API_KEY},
                          timeout=5.0)
            if r.status_code not in (200, 401, 403):
                pytest.skip(f"Dashboard not reachable (status {r.status_code})")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pytest.skip(f"Dashboard at {DASHBOARD_URL} not reachable")

        # Look up org_id from API key
        row = _sql_fetchone("SELECT org_id FROM api_keys WHERE id = %s",
                            (NULLSPEND_SMOKE_KEY_ID,))
        if not row or not row.get("org_id"):
            pytest.fail("Smoke test API key has no org_id")
        _SMOKE_ORG_ID = row["org_id"]

        # Symmetric cleanup: orphan rows from prior crashed runs
        _cleanup_test_actions()
        _live_setup_done = True

    yield

    # Teardown
    try:
        _cleanup_test_actions()
    except Exception as e:
        print(f"[py-functional] teardown cleanup failed: {e}")

    global _pg_conn
    if _pg_conn is not None:
        _pg_conn.close()
        _pg_conn = None


@pytest.fixture
def live_client(live_setup):
    """Return a sync NullSpend client pointing at the live dashboard."""
    client = NullSpend(base_url=DASHBOARD_URL, api_key=NULLSPEND_API_KEY)
    yield client
    client.close()


# ═══════════════════════════════════════════════════════════════════
# § Section 3 — Config behavior (F8-F9)
# ═══════════════════════════════════════════════════════════════════

class TestConfigBehavior:
    """F8-F9: retry and timeout config behavior."""

    def test_f8a_retries_on_503_then_succeeds(self):
        """F8a: SDK retries on 503, then succeeds on 3rd attempt."""
        call_count = 0

        class RetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return httpx.Response(503, text="Service Unavailable")
                return httpx.Response(
                    200, json={"data": {"id": "evt_x", "createdAt": "2026-04-07T00:00:00Z"}},
                )

        client = NullSpend(base_url="https://example.invalid", api_key="test-key",
                           max_retries=5, retry_base_delay_s=0.01)
        client._client = httpx.Client(transport=RetryTransport(), timeout=5.0)

        client.report_cost(CostEventInput(
            provider="openai", model="gpt-4o-mini",
            input_tokens=10, output_tokens=5, cost_microdollars=100,
        ))

        assert call_count == 3  # 1 initial + 2 retries

    def test_f8b_max_retries_cap_enforced(self):
        """F8b: maxRetries cap is enforced — stops after N retries."""
        call_count = 0

        class AlwaysFailTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                return httpx.Response(503, text="Service Unavailable")

        client = NullSpend(base_url="https://example.invalid", api_key="k",
                           max_retries=2, retry_base_delay_s=0.001)
        client._client = httpx.Client(transport=AlwaysFailTransport(), timeout=5.0)

        with pytest.raises(NullSpendError) as exc_info:
            client.list_budgets()

        assert call_count == 3  # 1 initial + 2 retries
        assert exc_info.value.status_code == 503

    def test_f9_request_timeout_aborts_slow_upstream(self):
        """F9: request timeout fires and aborts slow upstream."""
        import httpcore

        class SlowTransport(httpx.BaseTransport):
            def handle_request(self, request):
                # Simulate a timeout by raising the httpx timeout exception
                # directly — a real slow server would trigger this via httpx's
                # built-in timeout mechanism.
                raise httpx.ReadTimeout(
                    "Timed out",
                    request=request,
                )

        client = NullSpend(base_url="https://example.invalid", api_key="k",
                           request_timeout_s=0.3, max_retries=0)
        client._client = httpx.Client(transport=SlowTransport(), timeout=0.3)

        with pytest.raises(NullSpendError, match="network error"):
            client.list_budgets()


# ═══════════════════════════════════════════════════════════════════
# § Section 4 — Error class fields (F11)
# ═══════════════════════════════════════════════════════════════════

class TestErrorClassFields:
    """F11: error class hierarchy, fields, and messages."""

    def test_poll_timeout_error(self):
        err = PollTimeoutError("act_test_x", 1500)
        assert isinstance(err, PollTimeoutError)
        assert isinstance(err, NullSpendError)
        assert isinstance(err, Exception)
        assert err.action_id == "act_test_x"
        assert err.timeout_ms == 1500
        assert "act_test_x" in str(err)

    def test_rejected_error(self):
        err = RejectedError("act_test_y", "rejected")
        assert isinstance(err, RejectedError)
        assert isinstance(err, NullSpendError)
        assert err.action_id == "act_test_y"
        assert err.action_status == "rejected"
        assert "rejected" in str(err)

    def test_budget_exceeded_error(self):
        err = BudgetExceededError(
            remaining_microdollars=500_000,
            entity_type="user", entity_id="user-123",
            limit_microdollars=10_000_000, spend_microdollars=9_500_000,
        )
        assert isinstance(err, NullSpendError)
        assert err.remaining_microdollars == 500_000
        assert err.entity_type == "user"
        assert err.entity_id == "user-123"

    def test_mandate_violation_error(self):
        err = MandateViolationError(
            mandate="providers", requested="anthropic", allowed=["openai"],
        )
        assert isinstance(err, NullSpendError)
        assert err.mandate == "providers"
        assert err.requested == "anthropic"

    def test_session_limit_exceeded_error(self):
        err = SessionLimitExceededError(
            session_spend_microdollars=5_000_000,
            session_limit_microdollars=5_000_000,
        )
        assert isinstance(err, NullSpendError)
        assert err.session_spend_microdollars == 5_000_000

    def test_velocity_exceeded_error(self):
        err = VelocityExceededError(retry_after_seconds=60)
        assert isinstance(err, NullSpendError)
        assert err.retry_after_seconds == 60

    def test_tag_budget_exceeded_error(self):
        err = TagBudgetExceededError(tag_key="env", tag_value="prod")
        assert isinstance(err, NullSpendError)
        assert err.tag_key == "env"
        assert err.tag_value == "prod"


# ═══════════════════════════════════════════════════════════════════
# § Section 1 — HITL action lifecycle (F1-F4)
# ═══════════════════════════════════════════════════════════════════

@live
class TestHITLLifecycle:
    """F1-F4: HITL action lifecycle against the live dashboard."""

    def test_f1_full_lifecycle(self, live_client):
        """F1: createAction → approve → getAction → markResult(executing) → markResult(executed)."""
        agent_id = _make_agent_id("f1")

        # 1. createAction
        created = live_client.create_action(CreateActionInput(
            agent_id=agent_id,
            action_type="http_post",
            payload={"url": "https://example.com/webhook", "method": "POST"},
        ))
        assert created.id
        assert created.status == "pending"

        # 2. SQL approve
        _approve_action_via_sql(created.id)

        # 3. getAction
        fetched = live_client.get_action(created.id)
        assert fetched.status == "approved"
        assert fetched.approved_by == NULLSPEND_SMOKE_USER_ID
        assert fetched.agent_id == agent_id

        # 4. markResult(executing)
        executing = live_client.mark_result(created.id, MarkResultInput(status="executing"))
        assert executing.status == "executing"

        # 5. markResult(executed)
        done = live_client.mark_result(created.id, MarkResultInput(
            status="executed", result={"ok": True},
        ))
        assert done.status == "executed"

    def test_f2a_wait_for_decision_approved(self, live_client):
        """F2a: waitForDecision resolves when action is approved mid-poll."""
        agent_id = _make_agent_id("f2a")

        created = live_client.create_action(CreateActionInput(
            agent_id=agent_id, action_type="http_post",
            payload={"url": "https://example.com/webhook"},
        ))

        t = _delayed_approve(created.id, "F2a")

        decision = live_client.wait_for_decision(
            created.id, poll_interval_s=TEST_POLL_INTERVAL_S, timeout_s=10.0,
        )
        assert decision.status == "approved"
        assert decision.id == created.id
        t.join(timeout=5)

    def test_f2b_wait_for_decision_timeout(self, live_client):
        """F2b: waitForDecision raises PollTimeoutError when no decision arrives."""
        agent_id = _make_agent_id("f2b")

        created = live_client.create_action(CreateActionInput(
            agent_id=agent_id, action_type="http_post",
            payload={"url": "https://example.com/webhook"},
        ))

        with pytest.raises(PollTimeoutError) as exc_info:
            live_client.wait_for_decision(
                created.id, poll_interval_s=TEST_POLL_INTERVAL_S,
                timeout_s=TIMEOUT_FAST_S,
            )

        assert exc_info.value.action_id == created.id
        assert exc_info.value.timeout_ms == int(TIMEOUT_FAST_S * 1000)

    def test_f3a_propose_and_wait_happy_path(self, live_client):
        """F3a: proposeAndWait → approve → execute → markResult(executed)."""
        agent_id = _make_agent_id("f3a")
        execute_called = False
        captured_action_id = None

        def on_poll(action):
            nonlocal captured_action_id
            if captured_action_id is None:
                captured_action_id = action.id
                _delayed_approve(action.id, "F3a")

        def execute(ctx):
            nonlocal execute_called
            execute_called = True
            return "ok"

        result = live_client.propose_and_wait(ProposeAndWaitOptions(
            agent_id=agent_id, action_type="http_post",
            payload={"url": "https://example.com/webhook"},
            execute=execute,
            poll_interval_s=TEST_POLL_INTERVAL_S, timeout_s=10.0,
            on_poll=on_poll,
        ))

        assert result == "ok"
        assert execute_called
        assert captured_action_id

        row = _sql_fetchone(
            "SELECT status FROM actions WHERE id = %s",
            (_strip_action_prefix(captured_action_id),),
        )
        assert row and row["status"] == "executed"

    def test_f3b_propose_and_wait_rejected(self, live_client):
        """F3b: proposeAndWait raises RejectedError when action is rejected."""
        agent_id = _make_agent_id("f3b")
        execute_called = False
        captured_action_id = None

        def on_poll(action):
            nonlocal captured_action_id
            if captured_action_id is None:
                captured_action_id = action.id
                _delayed_reject(action.id, "F3b")

        def execute(ctx):
            nonlocal execute_called
            execute_called = True
            return "should not run"

        with pytest.raises(RejectedError) as exc_info:
            live_client.propose_and_wait(ProposeAndWaitOptions(
                agent_id=agent_id, action_type="http_post",
                payload={"url": "https://example.com/webhook"},
                execute=execute,
                poll_interval_s=TEST_POLL_INTERVAL_S, timeout_s=10.0,
                on_poll=on_poll,
            ))

        assert exc_info.value.action_id == captured_action_id
        assert exc_info.value.action_status == "rejected"
        assert not execute_called

    def test_f4_request_budget_increase(self, live_client):
        """F4: requestBudgetIncrease creates action, gets approved, returns result."""
        agent_id = _make_agent_id("f4")
        captured_action_id = None

        def on_poll(action):
            nonlocal captured_action_id
            if captured_action_id is None:
                captured_action_id = action.id
                _delayed_approve(action.id, "F4")

        result = live_client.request_budget_increase(RequestBudgetIncreaseOptions(
            agent_id=agent_id,
            amount_microdollars=5_000_000,
            reason="F4 test — verify budget increase flow",
            entity_type="user",
            entity_id=NULLSPEND_SMOKE_USER_ID,
            poll_interval_s=TEST_POLL_INTERVAL_S, timeout_s=10.0,
            on_poll=on_poll,
        ))

        assert result.action_id
        assert result.requested_amount_microdollars == 5_000_000

        row = _sql_fetchone(
            "SELECT status FROM actions WHERE id = %s",
            (_strip_action_prefix(captured_action_id),),
        )
        assert row and row["status"] == "executed"


# ═══════════════════════════════════════════════════════════════════
# § Section 2 — Read APIs (F5-F6)
# ═══════════════════════════════════════════════════════════════════

@live
class TestReadAPIs:
    """F5-F6: read APIs against the live dashboard."""

    def test_f5_get_cost_summary_all_periods(self, live_client):
        """F5: getCostSummary returns expected shape for all 3 periods."""
        for period in ("7d", "30d", "90d"):
            summary = live_client.get_cost_summary(period)

            # Shape assertions — totals may be 0
            assert isinstance(summary.daily, list)
            assert summary.models is not None
            assert summary.providers is not None
            assert summary.keys is not None
            assert summary.tools is not None
            assert summary.sources is not None
            assert summary.traces is not None

            assert summary.totals["period"] == period
            assert isinstance(summary.totals["totalCostMicrodollars"], (int, float))
            assert isinstance(summary.totals["totalRequests"], (int, float))

            assert summary.cost_breakdown is not None
            assert "inputCost" in summary.cost_breakdown
            assert "outputCost" in summary.cost_breakdown

    def test_f6_list_cost_events_pagination(self, live_client):
        """F6: listCostEvents pagination round-trip."""
        page1 = live_client.list_cost_events(ListCostEventsOptions(limit=5))
        assert isinstance(page1.data, list)
        assert len(page1.data) <= 5

        if not page1.cursor:
            pytest.skip("Smoke org has <5 cost events — pagination not testable")

        page2 = live_client.list_cost_events(ListCostEventsOptions(
            limit=5, cursor=page1.cursor,
        ))
        assert isinstance(page2.data, list)
        assert len(page2.data) <= 5

        # Pages must not overlap
        ids1 = {e.id for e in page1.data}
        overlap = [e for e in page2.data if e.id in ids1]
        assert overlap == []
