"""eligibility()/score()/route() for the real claude_cli/codex_cli entries
in config/providers.yaml -- exercised against the real loaded config, same
convention test_routing.py already uses."""

import pytest
from agentconnect.common.config import load_profiles, load_providers, load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.providers import ProviderRegistry
from agentconnect.common.quota import QuotaLedger
from agentconnect.common.schemas import PrivacyClass
from agentconnect.router.routing import RoutingContext, RoutingEngine


@pytest.fixture
def engine():
    mem = SharedMemory()
    reg = ProviderRegistry.from_config(load_providers())
    eng = RoutingEngine(reg, load_profiles(), load_routing(), QuotaLedger(memory=mem))
    return eng, reg


def _cli_cfg(reg, provider_id="claude_cli"):
    cfg = reg.get(provider_id)
    assert cfg is not None, f"{provider_id} missing from providers.yaml"
    return cfg


def test_claude_cli_is_registered_as_cli_subprocess_type(engine):
    _eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    assert cfg.type == "cli_subprocess"
    assert cfg.privacy == "external_paid"
    assert cfg.secret_ref is None
    assert cfg.cli is not None


def test_codex_cli_is_registered_as_cli_subprocess_type(engine):
    _eng, reg = engine
    cfg = _cli_cfg(reg, "codex_cli")
    assert cfg.type == "cli_subprocess"
    assert cfg.privacy == "external_paid"


def test_neither_cli_provider_grants_workspace_write_access(engine):
    _eng, reg = engine
    for pid in ("claude_cli", "codex_cli"):
        cfg = _cli_cfg(reg, pid)
        args = cfg.cli.args
        assert "--add-dir" not in args
        assert "workspace-write" not in args
        assert "danger-full-access" not in args


def test_secret_sensitive_never_reaches_cli_subprocess(engine):
    eng, _reg = engine
    cfg = _cli_cfg(_reg, "claude_cli")
    ctx = RoutingContext(task_id="t", privacy_class=PrivacyClass.secret_sensitive)
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "privacy_class_blocks_all_llm_routing"


def test_repo_sensitive_does_not_reach_cli_subprocess_by_default(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    ctx = RoutingContext(task_id="t", privacy_class=PrivacyClass.repo_sensitive, allow_external=True)
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "privacy_policy_blocks_provider_tier"


def test_public_task_with_allow_external_and_allow_paid_is_eligible(engine):
    # Requires an explicit budget too -- fail-closed default (bet #1)
    # applies to external_paid regardless of whether cost is metered.
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=True
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is True, reason


def test_allow_external_false_blocks_cli_subprocess(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=False, allow_paid=True
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "external_routing_not_allowed_for_task"


def test_allow_paid_false_blocks_external_paid_cli_subprocess(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=False
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "paid_routing_not_allowed_for_task"


def test_budget_not_configured_blocks_cli_subprocess_by_default(engine):
    # require_explicit is Aronnax's fail-closed default (bet #1) -- a paid
    # cli_subprocess provider must not be reachable until a budget is set,
    # same as any other external_paid provider, even with no per-call meter.
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    eng.set_budget_state(configured=False, remaining_usd=0.0, pressure=0.0, require_explicit=True)
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=True
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "budget_not_configured"


def test_budget_configured_allows_cli_subprocess(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=True
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is True, reason


def test_circuit_open_blocks_cli_subprocess(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    eng.set_circuit_state({"claude_cli"})
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=True
    )
    eligible, reason = eng.eligibility(ctx, cfg, status=None)
    assert eligible is False
    assert reason == "circuit_open"


def test_score_pins_cost_and_quota_scarcity_penalties_at_zero(engine):
    eng, reg = engine
    cfg = _cli_cfg(reg, "claude_cli")
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public, allow_external=True, allow_paid=True
    )
    breakdown = eng.score(ctx, cfg, status=None)
    assert breakdown.terms["cost_penalty"] == 0.0
    assert breakdown.terms["quota_scarcity_penalty"] == 0.0


def test_route_labels_cli_subprocess_decision_correctly(engine):
    eng, reg = engine
    eng.set_budget_state(configured=True, remaining_usd=100.0, pressure=0.0, require_explicit=True)
    ctx = RoutingContext(
        task_id="t",
        privacy_class=PrivacyClass.public,
        needed_capabilities=("reasoning",),
        allow_external=True,
        allow_paid=True,
        required_provider="claude_cli",
    )
    decision = eng.route(ctx, status=None)
    assert decision.selected_provider == "claude_cli"
    assert decision.decision == "route_to_cli_subprocess"
