"""End-to-end: submit a task through the RouterService with an in-process Local
Model Manager, verify context virtualization (compact summary + refs) and the
privacy fail-closed path."""

from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import TaskConstraints, TaskState, TaskSubmission
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.service import RouterService


def _service():
    mem = SharedMemory()
    local = InProcessLocalClient(ResidencyManager())
    return RouterService.create(memory=mem, local_client=local)


def test_repo_sensitive_task_completes_locally_with_refs():
    svc = _service()
    sub = TaskSubmission(
        task="Refactor the auth/session token refresh path in this private module.",
        agent_type="patch_worker",
        constraints=TaskConstraints(privacy_class="repo_sensitive"),
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.COMPLETE
    # Compact summary + refs, not full output.
    assert summary.summary
    assert "output" in summary.artifacts
    # The full output lives in shared memory and is read back in chunks.
    chunk = svc.read_artifact_chunk(summary.artifacts["output"])
    assert "content" in chunk
    # A routing decision was recorded and is explainable.
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions and decisions[-1]["selected_provider"] == "local_r9700"


def test_secret_sensitive_task_is_blocked():
    svc = _service()
    sub = TaskSubmission(
        task="Rotate this key sk-ABCD1234EFGH5678IJKL in the deploy script.",
        agent_type="patch_worker",
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.REJECTED
    assert "secret" in (summary.summary or "").lower()
    # No cloud provider was ever selected.
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions == []  # blocked before routing


def test_get_router_and_provider_status():
    svc = _service()
    rs = svc.get_router_status()
    assert rs["local_manager"]["loaded_model"]["model_id"]
    ps = svc.get_provider_status()
    assert any(p["provider"] == "local_r9700" for p in ps)


def test_cancel_task():
    svc = _service()
    tid = svc.memory.create_task({"task": "x"})
    svc.memory.update_task(tid, state=TaskState.QUEUED.value)
    out = svc.cancel_task(tid)
    assert out["state"] == TaskState.CANCELLED.value
