"""Router -> runtime dispatch wiring: a task submitted with execution="agentic"
runs through the worker runtime's act/tool loop (not a single generation), and
the guard that keeps agentic execution off external providers."""

from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import (
    GenerateRequest,
    TaskConstraints,
    TaskState,
    TaskSubmission,
)
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.gateway import GatewayResult
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.runtime_dispatch import GatewayModelSource
from agentconnect.router.service import RouterService


def _local_service():
    return RouterService.create(
        memory=SharedMemory(), local_client=InProcessLocalClient(ResidencyManager())
    )


def test_agentic_task_runs_through_the_runtime_loop():
    svc = _local_service()
    sub = TaskSubmission(
        task="Refactor the auth/session token refresh path in this private module.",
        agent_type="patch_worker",
        constraints=TaskConstraints(privacy_class="repo_sensitive", execution="agentic"),
    )
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.COMPLETE
    # The stored output is the structured WorkerResult, not a raw generation.
    chunk = svc.read_artifact_chunk(summary.artifacts["output"])
    assert '"status": "completed"' in chunk["content"]
    assert '"confidence"' in chunk["content"]
    # A distinctive agentic log line proves the runtime ran (one-shot never logs it).
    logs = svc.get_log_slice(summary.task_id, query="agentic")
    assert any("steps=" in ln["message"] for ln in logs)
    # An evaluation was recorded for the whole task.
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] == "local_r9700"


def test_oneshot_is_the_default_and_stays_a_single_generation():
    svc = _local_service()
    sub = TaskSubmission(
        task="Refactor the auth/session token refresh path in this private module.",
        agent_type="patch_worker",
        constraints=TaskConstraints(privacy_class="repo_sensitive"),
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.COMPLETE
    # One-shot stores the raw model text; no agentic loop, no agentic log.
    chunk = svc.read_artifact_chunk(summary.artifacts["output"])
    assert '"status": "completed"' not in chunk["content"]
    assert svc.get_log_slice(summary.task_id, query="agentic") == []


def test_agentic_on_an_external_provider_is_rejected():
    # No local client -> a public task routes to a cloud provider; the agentic
    # guard must reject it before any generation happens.
    svc = RouterService.create(memory=SharedMemory())
    sub = TaskSubmission(
        task="Summarize the public release notes for the 2.0 launch.",
        agent_type="summarizer",
        constraints=TaskConstraints(
            privacy_class="public", allow_external=True, execution="agentic"
        ),
    )
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.REJECTED
    # Cloud is neither owned-local nor a trusted rented node.
    assert "local or trusted rented node" in (summary.summary or "").lower()
    # It really did try to route to cloud — the guard, not some earlier gate, caught it.
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] in {"gemini_free", "groq_free"}


class _CountingGateway:
    """Fake gateway that returns fixed usage per call, to check accumulation."""

    def __init__(self):
        self.seen_models = []

    def call(self, cfg, req):
        self.seen_models.append(req.model_id)
        return GatewayResult(
            output_text="ok", input_tokens=10, output_tokens=3,
            provider="p", model=req.model_id,
        )


def test_gateway_model_source_pins_model_and_sums_usage():
    gw = _CountingGateway()
    src = GatewayModelSource(gw, cfg=object(), model_id="pinned-model")
    for i in range(3):
        req = GenerateRequest(
            request_id=f"r{i}", task_id="t", model_id="whatever-the-loop-said",
            messages=[{"role": "user", "content": "hi"}],
        )
        resp = src.generate(req)
        assert resp.model_id == "pinned-model"
    # The provider is pinned to the routing decision, not the loop's model_id.
    assert gw.seen_models == ["pinned-model"] * 3
    assert src.calls == 3
    assert src.total_input_tokens == 30
    assert src.total_output_tokens == 9
