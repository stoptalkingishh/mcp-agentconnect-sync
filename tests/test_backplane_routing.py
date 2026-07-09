"""Worker routing MVP: echo worker execution, deterministic selection, route
explanation persistence, hard gates, and the approval path (spec §18-§21, §24).

No model, no network. `echo_worker` is the fixture that makes routing testable.
"""

import json

import pytest

from agentconnect.core import (
    AgentConnectService,
    ArtifactType,
    Conflict,
    CreateTaskRequest,
    EchoWorker,
    FilesystemAccess,
    PrivacyTier,
    RawModelWorker,
    RoutePolicy,
    SandboxSpec,
    SubtaskRequest,
    SubtaskStatus,
    TaskStatus,
    WorkerAdapter,
    WorkerCapabilities,
    WorkerHealth,
    WorkerLocation,
    WorkerResult,
    route,
)
from agentconnect.core.routing import WorkerRegistry


def make_service(tmp_path, workers, policy=None):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=workers, policy=policy,
    )


def cloud_worker(worker_id="cheap_cloud_deepseek", cost=1.0, tiers=None, tags=None):
    return RawModelWorker(
        worker_id, lambda prompt: f"cloud says: {prompt[:20]}", model="deepseek-v3",
        location=WorkerLocation.cloud,
        privacy_tiers=tiers or [PrivacyTier.public, PrivacyTier.public_redacted],
        capability_tags=tags or ["generate", "summarize"],
        cost_per_1k_tokens_usd=cost,
    )


def local_model_worker(worker_id="local_qwen_worker", tiers=None, tags=None):
    return RawModelWorker(
        worker_id, lambda prompt: "local output", model="qwen2.5-coder-14b",
        location=WorkerLocation.local,
        privacy_tiers=tiers or list(PrivacyTier),
        capability_tags=tags or ["generate", "inspect"],
        cost_per_1k_tokens_usd=0.0,
    )


class UnhealthyWorker(WorkerAdapter):
    @property
    def worker_id(self) -> str:
        return "sick_worker"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            worker_id="sick_worker", harness="stub", privacy_tiers=list(PrivacyTier),
        )

    def health(self) -> WorkerHealth:
        return WorkerHealth(available=False, detail="GPU fell over")

    def run(self, subtask, context) -> WorkerResult:  # pragma: no cover - never routed
        raise AssertionError("an unhealthy worker must never be selected")


class ExplodingWorker(WorkerAdapter):
    @property
    def worker_id(self) -> str:
        return "exploding_worker"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            worker_id="exploding_worker", harness="stub", privacy_tiers=list(PrivacyTier),
        )

    def run(self, subtask, context) -> WorkerResult:
        raise RuntimeError("harness segfaulted")


@pytest.fixture()
def task_svc(tmp_path):
    svc = make_service(tmp_path, [EchoWorker()])
    task = svc.create_task(CreateTaskRequest(title="Refactor auth", goal="dedupe expiry"))
    return svc, task


# ------------------------------------------------------- echo worker execution
def test_echo_worker_runs_and_produces_an_artifact(task_svc):
    svc, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="Find duplicated expiry checks",
        instructions="Inspect auth files and return file paths and line ranges only.",
        privacy_tier=PrivacyTier.repo_sensitive,
    ))
    assert subtask.status is SubtaskStatus.succeeded
    assert subtask.assigned_worker == "echo_worker"
    assert subtask.result_artifact_id

    body = svc.read_artifact_chunk(subtask.result_artifact_id, 0, 8000).content
    assert "Inspect auth files" in body
    assert svc.get_task(task.id).task.status is TaskStatus.in_progress


def test_worker_result_is_recorded_as_an_attempt(task_svc):
    svc, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    attempts = svc.get_task(task.id).attempts
    assert attempts[-1].actor_id == "echo_worker"
    assert attempts[-1].actor_type.value == "worker"
    assert attempts[-1].artifact_refs == [subtask.result_artifact_id]


def test_worker_run_record_is_persisted(task_svc):
    svc, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    runs = svc.get_subtask(subtask.id).runs
    assert len(runs) == 1
    assert runs[0].harness == "echo" and runs[0].status.value == "succeeded"
    assert runs[0].output_artifact_id == subtask.result_artifact_id


def test_a_crashing_worker_fails_the_subtask_not_the_service(tmp_path):
    svc = make_service(tmp_path, [ExplodingWorker()])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    assert subtask.status is SubtaskStatus.failed
    run = svc.get_subtask(subtask.id).runs[0]
    assert "segfaulted" in run.error


# --------------------------------------------------- deterministic route choice
def test_local_beats_cloud_even_when_both_are_eligible(tmp_path):
    svc = make_service(
        tmp_path, [cloud_worker(cost=0.0), local_model_worker()],
        policy=RoutePolicy(max_cost_usd=10.0),
    )
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))
    # The cloud worker is free here, so only privacy_fit (local-first) separates them.
    assert subtask.assigned_worker == "local_qwen_worker"


def test_route_selection_is_deterministic_across_registry_order(tmp_path):
    chosen = set()
    for workers in ([EchoWorker(), local_model_worker()], [local_model_worker(), EchoWorker()]):
        svc = make_service(tmp_path, workers)
        task = svc.create_task(CreateTaskRequest(title="t"))
        subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
        chosen.add(subtask.assigned_worker)
    assert len(chosen) == 1


def test_preferred_worker_breaks_the_tie(tmp_path):
    svc = make_service(tmp_path, [EchoWorker(), local_model_worker()])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", preferred_worker="local_qwen_worker"))
    assert subtask.assigned_worker == "local_qwen_worker"


# ------------------------------------------------------------------ hard gates
def test_repo_sensitive_task_cannot_use_a_public_cloud_worker(tmp_path):
    svc = make_service(tmp_path, [cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.repo_sensitive))

    assert subtask.status is SubtaskStatus.failed
    explanation = svc.explain_route(subtask.id)
    rejected = explanation.rejected_workers[0]
    assert rejected.gate == "privacy_allowed"
    assert "repo_sensitive" in rejected.reason and "cloud" in rejected.reason


def test_capability_mismatch_rejects_a_worker(tmp_path):
    svc = make_service(tmp_path, [local_model_worker(tags=["generate"])])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", required_capabilities=["run_tests"]))
    explanation = svc.explain_route(subtask.id)
    assert explanation.rejected_workers[0].gate == "capability_match"
    assert "run_tests" in explanation.rejected_workers[0].reason


def test_sandbox_demand_beyond_the_worker_offer_is_rejected(tmp_path):
    svc = make_service(tmp_path, [EchoWorker()])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i",
        sandbox=SandboxSpec(filesystem=FilesystemAccess.workspace_write, shell=True)))
    assert svc.explain_route(subtask.id).rejected_workers[0].gate == "sandbox_supported"


def test_unhealthy_workers_are_never_selected(tmp_path):
    svc = make_service(tmp_path, [UnhealthyWorker()])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    explanation = svc.explain_route(subtask.id)
    assert explanation.rejected_workers[0].gate == "healthy"
    assert "GPU fell over" in explanation.rejected_workers[0].reason


def test_budget_gate_rejects_a_worker_over_the_ceiling(tmp_path):
    # requires_approval=False isolates the budget gate from the approval gate.
    pricey = RawModelWorker(
        "pricey", lambda p: "x", model="big", location=WorkerLocation.local,
        privacy_tiers=list(PrivacyTier), cost_per_1k_tokens_usd=100.0,
        requires_approval=False,
    )
    svc = make_service(tmp_path, [pricey], policy=RoutePolicy(max_cost_usd=0.0001))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="a long instruction " * 20))
    explanation = svc.explain_route(subtask.id)
    assert explanation.rejected_workers[0].gate == "budget_allowed"
    assert subtask.status is SubtaskStatus.failed


# ------------------------------------------------ route explanation persistence
def test_route_explanation_is_persisted_inline_and_as_an_artifact(task_svc):
    svc, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))

    explanation = svc.explain_route(subtask.id)
    assert explanation.selected_worker == "echo_worker"
    assert explanation.selected_harness == "echo"
    assert set(explanation.score_terms) == {
        "privacy_fit", "cost", "availability", "capability", "preferred"
    }
    assert 0.0 < explanation.total_score <= 1.0

    artifacts = [a for a in svc.list_artifacts(task.id)
                 if a.type is ArtifactType.route_explanation]
    assert len(artifacts) == 1
    stored = json.loads(svc.read_artifact_chunk(artifacts[0].id, 0, 8000).content)
    assert stored["selected_worker"] == "echo_worker"
    assert stored["subtask_id"] == subtask.id


def test_explain_route_on_unknown_subtask_is_not_found(task_svc):
    svc, _ = task_svc
    with pytest.raises(Exception):
        svc.explain_route("subtask_nope")


# ------------------------------------------------------------ approval workflow
def test_cloud_worker_parks_the_subtask_for_human_approval(tmp_path):
    svc = make_service(tmp_path, [cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))

    assert subtask.status is SubtaskStatus.needs_approval
    assert svc.get_task(task.id).task.status is TaskStatus.needs_approval
    explanation = svc.explain_route(subtask.id)
    assert explanation.needs_approval
    assert explanation.approval_candidate == "cheap_cloud_deepseek"
    assert explanation.approval_location == "cloud"


def test_approval_reroutes_and_runs_the_subtask(tmp_path):
    svc = make_service(tmp_path, [cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))

    approved = svc.approve_subtask(subtask.id, "matthew", max_cost_usd=3.0)
    assert approved.status is SubtaskStatus.succeeded
    assert approved.assigned_worker == "cheap_cloud_deepseek"
    assert svc.get_task(task.id).task.status is TaskStatus.in_progress
    kinds = [e.kind for e in svc.list_events(task.id)]
    assert "subtask_approved" in kinds


def test_approval_ceiling_still_binds(tmp_path):
    svc = make_service(tmp_path, [cloud_worker(cost=50.0)], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="x" * 400, privacy_tier=PrivacyTier.public))
    # Approved, but with a ceiling below the estimate: the budget gate wins.
    result = svc.approve_subtask(subtask.id, "matthew", max_cost_usd=0.0001)
    assert result.status is SubtaskStatus.failed
    assert svc.explain_route(subtask.id).rejected_workers[0].gate == "budget_allowed"


def test_deny_fails_the_subtask_and_settles_the_task(tmp_path):
    svc = make_service(tmp_path, [cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))

    denied = svc.deny_subtask(subtask.id, "matthew", "too expensive")
    assert denied.status is SubtaskStatus.failed
    assert svc.get_task(task.id).task.status is TaskStatus.in_progress


def test_approving_a_running_subtask_is_a_conflict(task_svc):
    svc, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    with pytest.raises(Conflict):
        svc.approve_subtask(subtask.id, "matthew")


def test_local_free_worker_wins_without_any_approval(tmp_path):
    svc = make_service(
        tmp_path, [cloud_worker(), local_model_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))
    assert subtask.status is SubtaskStatus.succeeded
    assert subtask.assigned_worker == "local_qwen_worker"


# ------------------------------------------------------------------ subtask ops
def test_cancel_subtask_then_cancel_again_conflicts(tmp_path):
    svc = make_service(tmp_path, [cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))
    svc.cancel_subtask(subtask.id)
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.cancelled
    with pytest.raises(Conflict):
        svc.cancel_subtask(subtask.id)


def test_registry_route_is_pure_and_does_not_touch_the_ledger():
    registry = WorkerRegistry([EchoWorker()])
    from agentconnect.core.models import Subtask

    subtask = Subtask(id="subtask_x", parent_task_id="task_x", title="t", instructions="i")
    first = route(subtask, registry)
    second = route(subtask, registry)
    assert first.model_dump() == second.model_dump()
