"""Rented/private-tier agentic carve-out (SECURITY + BILLING critical).

Agentic execution is permitted on an owned-local model OR a TRUSTED, opted-in
rented PRIVATE node — a box running your weights ephemerally with no external
logging. It must dispatch every step through the rented path (acquire ONCE,
reuse across steps, bill the rental window ONCE, release after), and stay
FAIL-CLOSED for cloud (external/external_paid) and for rented without opt-in or
without trust. All offline via StubProvisioner + an injected rented client."""

import re

from agentconnect.common.authorization import AutoApproveSpendAuthorizer
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import (
    GenerateResponse,
    TaskConstraints,
    TaskState,
    TaskSubmission,
)
from agentconnect.router.provisioning import StubProvisioner
from agentconnect.router.service import RouterService


class _TwoStepRentedClient:
    """A rented node's client that emits a tool action THEN a finish, so the
    agentic loop makes MULTIPLE generate() calls through the SINGLE acquired
    node — proving many generations bill exactly one rental window."""

    def __init__(self):
        self.calls = 0

    def generate(self, req) -> GenerateResponse:
        self.calls += 1
        if self.calls == 1:
            text = '{"action": "list_dir", "path": "."}'
        else:
            text = '{"action": "finish", "summary": "done on rented node", "confidence": 0.9}'
        return GenerateResponse(
            request_id=req.request_id, model_id=req.model_id, output_text=text,
            input_tokens=7, output_tokens=3, finish_reason="stop",
        )


def _rented_service(client_factory):
    return RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=StubProvisioner(), rented_client_factory=client_factory,
        authorizer=AutoApproveSpendAuthorizer(),
    )


def _count_rental_windows(svc, monkeypatch):
    counter = {"n": 0}
    orig = svc.quota.record_rental_window

    def counting(*a, **k):
        counter["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(svc.quota, "record_rental_window", counting)
    return counter


def test_trusted_rented_agentic_runs_bills_once_and_releases(monkeypatch):
    svc = _rented_service(lambda cfg, h: _TwoStepRentedClient())
    svc.set_budget(50.0, "monthly")
    windows = _count_rental_windows(svc, monkeypatch)

    sub = TaskSubmission(
        task="Reason over this large private design doc.",
        agent_type="repo_scout",
        constraints=TaskConstraints(
            privacy_class="repo_sensitive", allow_external=False,
            allow_rented=True, execution="agentic",
        ),
    )
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.COMPLETE
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] == "rented_h100_pool"
    # Structured WorkerResult stored (the loop ran), not a raw generation.
    chunk = svc.read_artifact_chunk(summary.artifacts["output"])
    assert '"status": "completed"' in chunk["content"]

    logs = svc.get_log_slice(summary.task_id, query="agentic")
    steps_lines = [ln for ln in logs if "steps=" in ln["message"]]
    assert steps_lines and "rented_node=" in steps_lines[0]["message"]
    steps = int(re.search(r"steps=(\d+)", steps_lines[0]["message"]).group(1))
    assert steps >= 2  # multiple generations...
    assert windows["n"] == 1  # ...but the rental window was billed EXACTLY once.

    cfg = svc.registry.get("rented_h100_pool")
    assert svc.quota.rental_remaining_usd(cfg) < cfg.rental.max_daily_usd
    # Node released back to the pool (available for the idle reaper).
    assert svc.node_pool.live_nodes().get("rented_h100_pool") is not None


def test_agentic_on_external_is_rejected_and_never_bills(monkeypatch):
    # No local client + public + allow_external -> routes to a cloud provider;
    # the guard rejects before any spend (tool observations must not reach an
    # untrusted external model).
    svc = _rented_service(lambda cfg, h: _TwoStepRentedClient())
    svc.set_budget(50.0, "monthly")
    windows = _count_rental_windows(svc, monkeypatch)

    sub = TaskSubmission(
        task="Summarize the public release notes for the 2.0 launch.",
        agent_type="log_summarizer",
        constraints=TaskConstraints(
            privacy_class="public", allow_external=True, execution="agentic",
        ),
    )
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.REJECTED
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] in {"gemini_free", "groq_free"}
    assert windows["n"] == 0


def test_agentic_on_rented_without_optin_is_rejected(monkeypatch):
    # Public task, external disabled -> routing still selects the rented node,
    # but allow_rented is False, so the guard rejects and nothing bills.
    svc = _rented_service(lambda cfg, h: _TwoStepRentedClient())
    svc.set_budget(50.0, "monthly")
    windows = _count_rental_windows(svc, monkeypatch)

    sub = TaskSubmission(
        task="Summarize the public release notes for the 2.0 launch.",
        agent_type="log_summarizer",
        constraints=TaskConstraints(
            privacy_class="public", allow_external=False,
            allow_rented=False, execution="agentic",
        ),
    )
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.REJECTED
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] == "rented_h100_pool"
    assert "allow_rented" in (summary.summary or "")
    assert windows["n"] == 0


def test_agentic_on_rented_without_trust_is_rejected(monkeypatch):
    # allow_rented=True, but the node's trust does not satisfy the repo_sensitive
    # policy -> the guard fails closed even though routing (public) picked it.
    svc = _rented_service(lambda cfg, h: _TwoStepRentedClient())
    svc.set_budget(50.0, "monthly")
    windows = _count_rental_windows(svc, monkeypatch)

    # NOTE: svc.registry wraps the lru_cached load_all() config, so this cfg
    # object is SHARED across every test in the process — the stripped trust
    # must be restored or later tests (e.g. the approval-web end-to-end, which
    # needs the rented node to stay trust-eligible) hang/fail on the mutation.
    cfg = svc.registry.get("rented_h100_pool")
    original_rental = cfg.rental
    stripped = cfg.rental.__class__(**{**cfg.rental.__dict__, "trust": {"ephemeral": True}})
    object.__setattr__(cfg, "rental", stripped)

    try:
        sub = TaskSubmission(
            task="Summarize the public release notes for the 2.0 launch.",
            agent_type="log_summarizer",
            constraints=TaskConstraints(
                privacy_class="public", allow_external=False,
                allow_rented=True, execution="agentic",
            ),
        )
        summary = svc.submit_task(sub)

        assert summary.status == TaskState.REJECTED
        decisions = svc.memory.get_routing_decisions(summary.task_id)
        assert decisions[-1]["selected_provider"] == "rented_h100_pool"
        assert windows["n"] == 0
    finally:
        object.__setattr__(cfg, "rental", original_rental)


class _NeverBootsProvisioner(StubProvisioner):
    """A rented node that provisions but never becomes ready — a realistic
    operational spin-up failure (wait_ready raises)."""

    def wait_ready(self, handle, timeout_seconds: int = 600):
        raise RuntimeError("rented node never booted")


def _rented_agentic_submission():
    return TaskSubmission(
        task="Reason over this large private design doc.",
        agent_type="repo_scout",
        constraints=TaskConstraints(
            privacy_class="repo_sensitive", allow_external=False,
            allow_rented=True, execution="agentic",
        ),
    )


def test_agentic_rented_spinup_failure_degrades_not_raises():
    # A trusted rented agentic task whose provisioner times out during spin-up
    # must degrade to a FAILED TaskSummary (the repo's hard contract), NOT
    # escape as a raw exception to the MCP caller — matching the one-shot path.
    svc = RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=_NeverBootsProvisioner(),
        rented_client_factory=lambda cfg, h: _TwoStepRentedClient(),
        authorizer=AutoApproveSpendAuthorizer(),
    )
    svc.set_budget(50.0, "monthly")

    summary = svc.submit_task(_rented_agentic_submission())  # must NOT raise

    assert summary.status == TaskState.FAILED
    # The node never came up, so acquire never populated the pool — nothing leaked.
    assert svc.node_pool.live_nodes().get("rented_h100_pool") is None


def test_agentic_rented_post_acquire_failure_releases_node(monkeypatch):
    # If a step AFTER a successful acquire raises (here: the rental-window billing
    # ledger is unavailable), the finally must still run so the just-provisioned
    # node is released for the reaper rather than leaked.
    svc = _rented_service(lambda cfg, h: _TwoStepRentedClient())
    svc.set_budget(50.0, "monthly")

    def _boom(*a, **k):
        raise RuntimeError("billing ledger unavailable")

    monkeypatch.setattr(svc.quota, "record_rental_window", _boom)

    summary = svc.submit_task(_rented_agentic_submission())  # must NOT raise

    assert summary.status == TaskState.FAILED
    # acquire succeeded before billing raised -> release() ran in finally, leaving
    # the node in the pool (timestamp bumped) for the idle reaper, not un-released.
    assert svc.node_pool.live_nodes().get("rented_h100_pool") is not None
