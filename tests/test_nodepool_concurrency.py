"""Rented-node warm reuse + idle reaping (Goal 4 amortization), and real
concurrency admission in the Model Manager (§18)."""

import threading
import time

from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import (
    CanAcceptRequest,
    GenerateRequest,
    TaskConstraints,
    TaskState,
    TaskSubmission,
)
from agentconnect.model_manager.backends import ModelBackend, StubBackend
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.provisioning import StubProvisioner
from agentconnect.router.service import RouterService


def _rented_svc():
    calls = {"provisions": 0}
    prov = StubProvisioner()
    orig = prov.provision

    def counting_provision(spec):
        calls["provisions"] += 1
        return orig(spec)

    prov.provision = counting_provision  # type: ignore[method-assign]
    factory = lambda cfg, handle: InProcessLocalClient(ResidencyManager())
    svc = RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=prov, rented_client_factory=factory,
    )
    return svc, calls


def _submit_rented(svc):
    return svc.submit_task(
        TaskSubmission(
            task="huge private reasoning job",
            agent_type="repo_scout",
            constraints=TaskConstraints(privacy_class="repo_sensitive",
                                        allow_external=False, allow_rented=True),
        )
    )


def test_warm_node_reused_across_tasks_and_billed_once():
    svc, calls = _rented_svc()
    cfg = svc.registry.get("rented_h100_pool")
    budget = cfg.rental.max_daily_usd

    assert _submit_rented(svc).status == TaskState.COMPLETE
    after_first = svc.quota.rental_remaining_usd(cfg)
    assert _submit_rented(svc).status == TaskState.COMPLETE
    after_second = svc.quota.rental_remaining_usd(cfg)

    assert calls["provisions"] == 1          # one spin-up serves both tasks
    assert after_first < budget              # billed the window once
    assert after_second == after_first       # reuse is free


def test_idle_reaper_terminates_warm_node():
    svc, _ = _rented_svc()
    _submit_rented(svc)
    assert "rented_h100_pool" in svc.node_pool.live_nodes()
    # Far past the idle window -> reaped.
    reaped = svc.reap_idle_nodes(now=10_000)
    assert "rented_h100_pool" in reaped
    assert "rented_h100_pool" not in svc.node_pool.live_nodes()


class _SlowBackend(StubBackend):
    """Blocks in generate until released, to observe concurrency limits."""

    def __init__(self, gate: threading.Event):
        super().__init__()
        self._gate = gate

    def generate(self, req):
        self._gate.wait(timeout=5)
        return super().generate(req)


def test_admission_caps_concurrent_sequences():
    gate = threading.Event()
    mgr = ResidencyManager(backend=_SlowBackend(gate), max_active_sequences=2)

    def worker():
        mgr.generate(GenerateRequest(request_id="r", task_id="t", model_id="qwen3.6-35b-a3b",
                                     messages=[{"role": "user", "content": "hi"}], max_output_tokens=8))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    # Only 2 slots -> exactly 2 active, the rest queued/waiting.
    st = mgr.status()
    assert st.loaded_model.active_sequences == 2
    gate.set()
    for t in threads:
        t.join(timeout=5)
    assert mgr.status().loaded_model.active_sequences == 0
