"""Global spend budget: mandatory (no silent default), period windows, pacing,
and hard cap. Fully offline."""

from datetime import datetime, timezone

import pytest

from agentconnect.common.budget import BudgetManager, _period_window
from agentconnect.common.config import load_profiles, load_providers, load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.providers import ProviderRegistry
from agentconnect.common.quota import QuotaLedger
from agentconnect.common.schemas import (
    PrivacyClass,
    TaskConstraints,
    TaskState,
    TaskSubmission,
)
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.routing import RoutingContext, RoutingEngine
from agentconnect.router.service import RouterService

# A fixed reference instant: 2026-07-15 12:00:00 UTC (mid-month, a Wednesday).
NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def _paid_cfg(mem):
    reg = ProviderRegistry.from_config(load_providers())
    return reg


# --------------------------------------------------------------- unit: manager
def test_not_configured_by_default():
    bm = BudgetManager(SharedMemory(), load_routing())
    assert bm.is_configured() is False
    st = bm.status(NOW)
    assert st["configured"] is False
    assert st["action_required"] == "set_budget"


def test_set_persists_and_configures():
    mem = SharedMemory()
    BudgetManager(mem, load_routing()).set(30.0, "monthly")
    # A fresh manager on the same memory sees it (survives "restart").
    bm2 = BudgetManager(mem, load_routing())
    assert bm2.is_configured()
    assert bm2.config().amount_usd == 30.0 and bm2.config().period == "monthly"


def test_period_windows():
    ds, de = _period_window(NOW, "daily")
    assert datetime.fromtimestamp(ds, timezone.utc).hour == 0
    assert de - ds == 86400
    ws, we = _period_window(NOW, "weekly")
    assert datetime.fromtimestamp(ws, timezone.utc).weekday() == 0  # Monday
    assert round(we - ws) == 7 * 86400
    ms, me = _period_window(NOW, "monthly")
    d0 = datetime.fromtimestamp(ms, timezone.utc)
    assert d0.day == 1 and d0.month == 7
    assert datetime.fromtimestamp(me, timezone.utc).month == 8  # rolls to August


def test_spent_sums_across_providers_and_remaining():
    mem = SharedMemory()
    bm = BudgetManager(mem, load_routing())
    bm.set(10.0, "monthly")
    # Two different providers each spend within this month.
    for prov, cost in (("openai_paid", 2.0), ("rented_h100_pool", 3.0)):
        mem.record_quota_usage({"provider": prov, "task_id": "t", "est_input": 0, "est_output": 0,
                                "act_input": 0, "act_output": 0, "requests": 1,
                                "est_cost_usd": cost, "act_cost_usd": cost, "status": "completed",
                                "failure_reason": None})
    assert bm.spent(NOW) == pytest.approx(5.0)
    assert bm.remaining(NOW) == pytest.approx(5.0)


def test_pacing_and_pressure():
    mem = SharedMemory()
    bm = BudgetManager(mem, load_routing())
    bm.set(31.0, "monthly")  # ~$1/day
    # Mid-month (~day 15) the paced allowance is ~half.
    assert bm.paced_allowance(NOW) == pytest.approx(31.0 * bm.elapsed_fraction(NOW))
    # Spend way ahead of pace -> high pressure; can't afford beyond remaining.
    mem.record_quota_usage({"provider": "openai_paid", "task_id": "t", "est_input": 0, "est_output": 0,
                            "act_input": 0, "act_output": 0, "requests": 1, "est_cost_usd": 28.0,
                            "act_cost_usd": 28.0, "status": "completed", "failure_reason": None})
    assert bm.over_pace_usd(NOW) > 0
    assert bm.pressure(NOW) > 0.5
    ok, reason = bm.can_afford(5.0, NOW)  # only $3 left
    assert not ok and reason == "period_budget_exhausted"


def test_can_afford_requires_configuration():
    bm = BudgetManager(SharedMemory(), load_routing())
    ok, reason = bm.can_afford(0.001, NOW)
    assert not ok and reason == "budget_not_configured"


# ----------------------------------------------------------- engine: gating
def _engine():
    mem = SharedMemory()
    reg = ProviderRegistry.from_config(load_providers())
    return RoutingEngine(reg, load_profiles(), load_routing(), QuotaLedger(memory=mem))


def _paid_ctx(**kw):
    return RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, needed_capabilities=("reasoning",),
        profile="resident_ok", est_input_tokens=1000, est_output_tokens=500,
        allow_external=True, allow_paid=True, **kw,
    )


def test_paid_blocked_when_budget_not_configured():
    eng = _engine()  # default: not configured
    d = eng.route(_paid_ctx(), status=None)
    reasons = {r.provider: r.reason for r in d.rejected_options}
    assert reasons.get("openai_paid") == "budget_not_configured"
    assert reasons.get("rented_h100_pool") == "budget_not_configured"


def test_paid_eligible_after_budget_configured():
    eng = _engine()
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    d = eng.route(_paid_ctx(), status=None)
    provs = {s.provider for s in d.scores}
    assert "openai_paid" in provs  # no longer blocked


def test_exhausted_budget_hard_blocks_paid():
    eng = _engine()
    eng.set_budget_state(configured=True, remaining_usd=0.0001, pressure=1.0, require_explicit=True)
    d = eng.route(_paid_ctx(), status=None)
    reasons = {r.provider: r.reason for r in d.rejected_options}
    assert reasons.get("openai_paid") == "period_budget_exhausted"


def test_pressure_penalizes_paid_score():
    eng = _engine()
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    base = {s.provider: s.total for s in eng.route(_paid_ctx(), status=None).scores}
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=1.0, require_explicit=True)
    hot = {s.provider: s.total for s in eng.route(_paid_ctx(), status=None).scores}
    assert hot["openai_paid"] < base["openai_paid"]


# --------------------------------------------------------------- service e2e
def test_service_set_get_and_required_signal():
    svc = RouterService.create(
        memory=SharedMemory(), local_client=InProcessLocalClient(ResidencyManager())
    )
    # Not configured at first.
    assert svc.get_budget_status()["configured"] is False
    assert svc.get_router_status()["budget"]["action_required"] == "set_budget"
    # Setting it returns a live status.
    st = svc.set_budget(20.0, "weekly")
    assert st["configured"] and st["amount_usd"] == 20.0 and st["period"] == "weekly"
    assert svc.get_router_status()["budget"]["action_required"] is None


def test_rented_only_task_surfaces_budget_required():
    # repo_sensitive + allow_rented, no local node, no budget -> only candidate is the
    # rented node, which is fail-closed -> REJECTED with a prompt-the-user next action.
    svc = RouterService.create(memory=SharedMemory(), local_client=None)
    summary = svc.submit_task(
        TaskSubmission(task="huge private reasoning job", agent_type="repo_scout",
                       constraints=TaskConstraints(privacy_class="repo_sensitive",
                                                   allow_external=False, allow_rented=True))
    )
    assert summary.status == TaskState.REJECTED
    assert "budget" in (summary.recommended_next_action or "").lower()
