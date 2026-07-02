"""Evaluation & learning (handoff §25 Phase 6): outcomes are recorded, scorecards
aggregate them, and the learned-quality signal (bounded) tilts routing."""

from agentconnect.common.config import load_profiles, load_providers, load_routing
from agentconnect.common.evaluation import Evaluator, Scorecard
from agentconnect.common.memory import SharedMemory
from agentconnect.common.providers import ProviderRegistry
from agentconnect.common.quota import QuotaLedger
from agentconnect.common.schemas import PrivacyClass, TaskConstraints, TaskState, TaskSubmission
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.routing import RoutingContext, RoutingEngine
from agentconnect.router.service import RouterService


def _svc():
    return RouterService.create(
        memory=SharedMemory(), local_client=InProcessLocalClient(ResidencyManager())
    )


def test_dispatch_records_evaluation():
    svc = _svc()
    svc.submit_task(
        TaskSubmission(task="do a thing", agent_type="patch_worker",
                       constraints=TaskConstraints(privacy_class="repo_sensitive"))
    )
    cards = svc.get_provider_scorecards()
    local = next(c for c in cards if c["provider"] == "local_r9700")
    assert local["samples"] == 1
    assert local["success_rate"] == 1.0
    assert local["avg_latency_ms"] >= 0.0


def test_scorecard_learned_quality_bounds():
    # Below min_samples -> zero signal.
    sc = Scorecard("p", samples=2, success_rate=1.0, avg_latency_ms=10, avg_cost_usd=0, avg_confidence=None)
    assert sc.learned_quality(min_samples=5) == 0.0
    # Enough samples, perfect success -> positive, clamped <= 1.
    sc2 = Scorecard("p", samples=10, success_rate=1.0, avg_latency_ms=10, avg_cost_usd=0, avg_confidence=None)
    assert 0.0 < sc2.learned_quality(min_samples=5) <= 1.0
    # All failures -> negative, clamped >= -1.
    sc3 = Scorecard("p", samples=10, success_rate=0.0, avg_latency_ms=10, avg_cost_usd=0, avg_confidence=None)
    assert -1.0 <= sc3.learned_quality(min_samples=5) < 0.0


def test_learned_signal_shifts_routing():
    mem = SharedMemory()
    reg = ProviderRegistry.from_config(load_providers())
    quota = QuotaLedger(memory=mem)
    engine = RoutingEngine(reg, load_profiles(), load_routing(), quota)
    ctx = RoutingContext(
        task_id="t", privacy_class=PrivacyClass.public,
        needed_capabilities=("classification",), profile="resident_ok",
        est_input_tokens=500, est_output_tokens=100,
    )
    # Baseline (no learning): note gemini's score.
    base = {s.provider: s.total for s in engine.route(ctx, None).scores}
    # Now inject a strong positive learned signal for groq and negative for gemini.
    engine.set_learned_quality({"groq_free": 1.0, "gemini_free": -1.0})
    after = {s.provider: s.total for s in engine.route(ctx, None).scores}
    assert after["groq_free"] > base["groq_free"]
    assert after["gemini_free"] < base["gemini_free"]


def test_failed_dispatch_recorded_as_failure():
    # Rented node selected but the client factory raises -> FAILED + eval record.
    def boom_factory(cfg, handle):
        raise RuntimeError("provision boom")

    from agentconnect.router.provisioning import StubProvisioner

    svc = RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=StubProvisioner(), rented_client_factory=boom_factory,
    )
    summary = svc.submit_task(
        TaskSubmission(task="huge private reasoning job", agent_type="repo_scout",
                       constraints=TaskConstraints(privacy_class="repo_sensitive",
                                                   allow_external=False, allow_rented=True))
    )
    assert summary.status == TaskState.FAILED
    cards = svc.get_provider_scorecards()
    rented = next(c for c in cards if c["provider"] == "rented_h100_pool")
    assert rented["samples"] == 1 and rented["success_rate"] == 0.0
