"""Rented-GPU node tier (handoff Goal 4): privacy gating, budget, provisioning,
and end-to-end dispatch — all offline via StubProvisioner + an injected in-process
client."""

import pytest

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
from agentconnect.router.provisioning import StubProvisioner, spec_from_provider
from agentconnect.router.routing import RoutingContext, RoutingEngine
from agentconnect.router.service import RouterService


@pytest.fixture
def engine():
    mem = SharedMemory()
    reg = ProviderRegistry.from_config(load_providers())
    eng = RoutingEngine(reg, load_profiles(), load_routing(), QuotaLedger(memory=mem))
    # A budget must be configured for paid/rented to be eligible (mandatory-prompt
    # policy). Give the engine ample headroom for the rented-routing tests.
    eng.set_budget_state(configured=True, remaining_usd=1000.0, pressure=0.0, require_explicit=True)
    return eng


def _ctx(privacy, **kw):
    return RoutingContext(
        task_id="t", privacy_class=privacy, needed_capabilities=("reasoning",),
        profile="default_worker", est_input_tokens=1000, est_output_tokens=500, **kw,
    )


def test_public_task_may_use_rented(engine):
    d = engine.route(_ctx(PrivacyClass.public), status=None)
    provs = {s.provider for s in d.scores}
    assert "rented_h100_pool" in provs  # eligible for public


def test_repo_sensitive_refused_rented_without_optin(engine):
    d = engine.route(_ctx(PrivacyClass.repo_sensitive, allow_external=False), status=None)
    reasons = {r.provider: r.reason for r in d.rejected_options}
    assert reasons.get("rented_h100_pool") == "privacy_policy_blocks_provider_tier"


def test_repo_sensitive_allowed_rented_with_optin_and_trust(engine):
    d = engine.route(_ctx(PrivacyClass.repo_sensitive, allow_external=False, allow_rented=True), status=None)
    provs = {s.provider for s in d.scores}
    # local box has no live status here, so the rented node is the eligible local option
    assert "rented_h100_pool" in provs
    assert d.selected_provider == "rented_h100_pool"
    assert d.decision == "route_to_rented_node"


def test_secret_sensitive_never_reaches_rented(engine):
    d = engine.route(_ctx(PrivacyClass.secret_sensitive, allow_rented=True), status=None)
    assert d.selected_provider is None
    assert d.decision == "blocked_secret_sensitive"


def test_rental_budget_blocks_when_exhausted(engine):
    reg = engine.registry
    cfg = reg.get("rented_h100_pool")
    # Burn the daily rental budget (max_daily_usd = 20.0).
    engine.quota.record_rental_window(cfg, "t0", seconds=6 * 3600)  # 6h * 3.5 = 21 usd
    ok, reason = engine.quota.can_reserve_rental(cfg)
    assert not ok and reason == "daily_rental_budget_exhausted"


def test_trust_policy_enforced_for_repo_sensitive(engine, monkeypatch):
    cfg = engine.registry.get("rented_h100_pool")
    # Strip the trust properties -> repo_sensitive must be refused even with opt-in.
    stripped = cfg.rental.__class__(**{**cfg.rental.__dict__, "trust": {"ephemeral": True}})
    object.__setattr__(cfg, "rental", stripped)
    eligible, reason = engine.eligibility(
        _ctx(PrivacyClass.repo_sensitive, allow_rented=True), cfg, status=None
    )
    assert not eligible and reason == "rented_node_trust_policy_unmet"


def test_provisioner_lifecycle():
    prov = StubProvisioner(clock=1000.0)
    reg = ProviderRegistry.from_config(load_providers())
    spec = spec_from_provider(reg.get("rented_h100_pool"), model_id="qwen3.6-35b-a3b")
    h = prov.provision(spec)
    assert h.state.value == "ready" and h.manager_endpoint and h.started_at == 1000.0
    assert prov.terminate(h).state.value == "terminated"


def test_end_to_end_rented_dispatch_and_billing():
    mem = SharedMemory()
    # Inject an in-process client so the "rented" node runs the stub backend offline.
    factory = lambda cfg, handle: InProcessLocalClient(ResidencyManager())
    from agentconnect.common.authorization import AutoApproveSpendAuthorizer

    svc = RouterService.create(
        memory=mem, local_client=None,
        provisioner=StubProvisioner(), rented_client_factory=factory,
        authorizer=AutoApproveSpendAuthorizer(),  # auto-confirm charges in tests
    )
    svc.set_budget(50.0, "monthly")  # rented needs a configured budget
    sub = TaskSubmission(
        task="Reason over this large private design doc.",
        agent_type="repo_scout",
        constraints=TaskConstraints(privacy_class="repo_sensitive", allow_external=False, allow_rented=True),
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.COMPLETE
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] == "rented_h100_pool"
    # A rental window was billed against the daily budget.
    cfg = svc.registry.get("rented_h100_pool")
    assert svc.quota.rental_remaining_usd(cfg) < cfg.rental.max_daily_usd
