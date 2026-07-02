import pytest

from agentconnect.common.config import load_profiles, load_providers, load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.providers import ProviderRegistry
from agentconnect.common.quota import QuotaLedger
from agentconnect.common.schemas import (
    AvailableModel,
    LoadedModel,
    ManagerStatus,
    PrivacyClass,
    QueueStatus,
)
from agentconnect.router.routing import RoutingContext, RoutingEngine


def _status(loaded="qwen3.6-35b-a3b", waiting=0):
    models = [
        AvailableModel(model_id="qwen3.6-35b-a3b", profiles=["default_worker", "resident_ok"], max_model_len=16384),
        AvailableModel(model_id="ornith-1.0-35b", profiles=["coding_patch"], max_model_len=16384),
        AvailableModel(model_id="qwen3.6-27b", profiles=["coding_review"], max_model_len=16384),
    ]
    return ManagerStatus(
        node_id="test",
        loaded_model=LoadedModel(model_id=loaded, max_active_sequences=4, active_sequences=0),
        available_models=models,
        queue=QueueStatus(local_waiting=waiting),
    )


@pytest.fixture
def engine():
    mem = SharedMemory()
    reg = ProviderRegistry.from_config(load_providers())
    return RoutingEngine(reg, load_profiles(), load_routing(), QuotaLedger(memory=mem)), reg


def test_repo_sensitive_routes_local_only(engine):
    eng, _ = engine
    ctx = RoutingContext(
        task_id="t1", privacy_class=PrivacyClass.repo_sensitive,
        needed_capabilities=("coding",), profile="resident_ok",
        est_input_tokens=500, est_output_tokens=200,
    )
    decision = eng.route(ctx, _status())
    assert decision.selected_provider == "local_r9700"
    # every cloud provider rejected on privacy grounds
    cloud_rejections = {r.provider for r in decision.rejected_options}
    assert {"gemini_free", "groq_free", "openai_paid"} <= cloud_rejections


def test_secret_sensitive_blocks_all(engine):
    eng, _ = engine
    ctx = RoutingContext(
        task_id="t2", privacy_class=PrivacyClass.secret_sensitive,
        needed_capabilities=("coding",), est_input_tokens=100, est_output_tokens=100,
    )
    decision = eng.route(ctx, _status())
    assert decision.selected_provider is None
    assert decision.decision == "blocked_secret_sensitive"


def test_resident_model_preferred_over_switch(engine):
    eng, _ = engine
    # A patch task prefers ornith, but qwen is resident and queue is empty and no
    # batch => switch threshold not met => resident qwen should be chosen locally.
    ctx = RoutingContext(
        task_id="t3", privacy_class=PrivacyClass.repo_sensitive,
        needed_capabilities=("coding",), profile="coding_patch",
        est_input_tokens=500, est_output_tokens=200, pending_same_model_batch=0,
    )
    decision = eng.route(ctx, _status(loaded="qwen3.6-35b-a3b"))
    assert decision.selected_provider == "local_r9700"
    assert decision.selected_model == "qwen3.6-35b-a3b"


def test_context_over_cap_rejects_local(engine):
    eng, _ = engine
    ctx = RoutingContext(
        task_id="t4", privacy_class=PrivacyClass.repo_sensitive,
        needed_capabilities=("coding",), est_input_tokens=20000, est_output_tokens=1000,
    )
    decision = eng.route(ctx, _status())
    # local rejected on context cap; repo_sensitive can't go cloud => none
    assert decision.selected_provider is None
    reasons = {r.provider: r.reason for r in decision.rejected_options}
    assert "context_exceeds_max_model_len" in reasons["local_r9700"]


def test_public_task_can_use_cloud_when_local_absent(engine):
    eng, _ = engine
    ctx = RoutingContext(
        task_id="t5", privacy_class=PrivacyClass.public,
        needed_capabilities=("classification",), est_input_tokens=500, est_output_tokens=100,
        allow_external=True,
    )
    decision = eng.route(ctx, None)  # no local manager available
    assert decision.selected_provider in {"gemini_free", "groq_free"}


def test_decision_is_deterministic(engine):
    eng, _ = engine
    ctx = RoutingContext(
        task_id="t6", privacy_class=PrivacyClass.public,
        needed_capabilities=("coding",), profile="resident_ok",
        est_input_tokens=500, est_output_tokens=200,
    )
    d1 = eng.route(ctx, _status())
    d2 = eng.route(ctx, _status())
    assert d1.selected_provider == d2.selected_provider
    assert d1.selected_model == d2.selected_model
    assert [s.total for s in d1.scores] == [s.total for s in d2.scores]
