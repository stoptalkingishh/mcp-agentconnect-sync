"""Memory adapter and external local-compute boundary (adapters spec, Parts A+B).

The acceptance criteria that matter most are negative ones: memory disabled must
work, a memory outage must not fail a subtask, capture must never promote, and a
dead local model manager must not take the process down.
"""

import pytest

from agentconnect.core import (
    AgentConnectService,
    CreateTaskRequest,
    EchoWorker,
    LocalModelManagerWorkerAdapter,
    NoopMemoryAdapter,
    PrivacyTier,
    RoutePolicy,
    StaticMemoryAdapter,
    SubtaskRequest,
    SubtaskStatus,
)
from agentconnect.core.local_compute import (
    HttpLocalComputeProvider,
    LocalComputeProvider,
    LocalEstimate,
    LocalRunResult,
)
from agentconnect.core.memory import (
    CaptureRequest,
    MemoryAdapter,
    MemoryFeedbackRequest,
    MemoryItem,
    RecallPack,
    RecallRequest,
)

PROMOTED = MemoryItem(
    text="Refresh token validation remains in auth/session.py because middleware depends on it.",
    status="promoted", confidence="verified", source_id="decision_004",
)
PENDING = MemoryItem(
    text="Qwen local worker performed poorly on auth patch review.",
    status="pending", confidence="low",
)
SUPERSEDED = MemoryItem(text="Old auth advice.", status="superseded", confidence="low")
REJECTED = MemoryItem(text="Rejected auth advice.", status="rejected", confidence="low")


def make_service(tmp_path, memory=None, workers=None, policy=None):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=workers if workers is not None else [EchoWorker()],
        memory=memory, policy=policy,
    )


# --------------------------------------------------------------- memory off
def test_backplane_runs_with_memory_disabled(tmp_path):
    svc = make_service(tmp_path)
    assert svc.memory.backend_name == "none"
    pack = svc.recall_memory(RecallRequest(query="anything"))
    assert pack.items == [] and pack.warnings
    assert svc.memory_health()["status"] == "disabled"

    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    assert subtask.status is SubtaskStatus.succeeded  # memory-off never blocks work


def test_capture_with_memory_disabled_is_a_noop_not_an_error(tmp_path):
    svc = make_service(tmp_path)
    result = svc.capture_memory_candidate(CaptureRequest(text="something"))
    assert result.accepted is False and result.status == "archived"


# ------------------------------------------------------------ visibility policy
def test_trusted_only_hides_pending_and_superseded(tmp_path):
    svc = make_service(tmp_path, memory=StaticMemoryAdapter(
        [PROMOTED, PENDING, SUPERSEDED, REJECTED]))
    pack = svc.recall_memory(RecallRequest(query=""))
    assert [i.status for i in pack.items] == ["promoted"]


def test_pending_is_returned_only_on_explicit_request_and_is_labeled(tmp_path):
    svc = make_service(tmp_path, memory=StaticMemoryAdapter([PROMOTED, PENDING]))
    pack = svc.recall_memory(RecallRequest(
        query="", trusted_only=False, include_pending=True))
    assert {i.status for i in pack.items} == {"promoted", "pending"}
    assert any("pending" in w for w in pack.warnings)


def test_rejected_and_archived_are_never_returned_even_untrusted(tmp_path):
    svc = make_service(tmp_path, memory=StaticMemoryAdapter([REJECTED]))
    pack = svc.recall_memory(RecallRequest(
        query="", trusted_only=False, include_pending=True, include_superseded=True))
    assert pack.items == []


def test_service_refilters_a_sloppy_backend_that_ignores_trusted_only(tmp_path):
    class LeakyAdapter(MemoryAdapter):
        backend_name = "leaky"

        def recall(self, request):  # ignores every flag it was given
            return RecallPack(profile=request.profile, query=request.query,
                              items=[PROMOTED, PENDING], backend="leaky")

        def capture_candidate(self, request):
            return None  # pragma: no cover

    svc = make_service(tmp_path, memory=LeakyAdapter())
    pack = svc.recall_memory(RecallRequest(query="", trusted_only=True))
    assert [i.status for i in pack.items] == ["promoted"]


def test_max_items_caps_the_pack(tmp_path):
    items = [MemoryItem(text=f"fact {n}", status="promoted", confidence="high")
             for n in range(20)]
    svc = make_service(tmp_path, memory=StaticMemoryAdapter(items))
    assert len(svc.recall_memory(RecallRequest(query="fact", max_items=3)).items) == 3


# ------------------------------------------------------------------- capture
def test_capture_never_promotes(tmp_path):
    static = StaticMemoryAdapter()
    svc = make_service(tmp_path, memory=static)
    task = svc.create_task(CreateTaskRequest(title="t"))
    result = svc.capture_memory_candidate(CaptureRequest(
        text="Qwen was weak on auth review.", task_id=task.id,
        origin_actor_id="claude-code", origin_actor_type="manager",
    ))
    assert result.accepted and result.status == "pending" and result.candidate_id
    assert [e.kind for e in svc.list_events(task.id)] == ["memory_candidate_captured"]


def test_a_backend_claiming_promotion_is_downgraded_to_pending(tmp_path):
    class PromotingAdapter(MemoryAdapter):
        backend_name = "eager"

        def recall(self, request):  # pragma: no cover
            return RecallPack(profile=request.profile, query=request.query, items=[])

        def capture_candidate(self, request):
            from agentconnect.core.memory import CaptureResult

            return CaptureResult(accepted=True, candidate_id="c1", status="promoted")

    svc = make_service(tmp_path, memory=PromotingAdapter())
    result = svc.capture_memory_candidate(CaptureRequest(text="x"))
    assert result.status == "pending"
    assert "promotion ignored" in (result.message or "")


# -------------------------------------------------------------- failure modes
class BrokenMemory(MemoryAdapter):
    backend_name = "broken"

    def recall(self, request):
        raise RuntimeError("memory backend on fire")

    def capture_candidate(self, request):
        raise RuntimeError("memory backend on fire")

    def health(self):
        raise RuntimeError("memory backend on fire")


def test_memory_failure_never_fails_a_subtask(tmp_path):
    svc = make_service(tmp_path, memory=BrokenMemory())
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    assert subtask.status is SubtaskStatus.succeeded

    pack = svc.recall_memory(RecallRequest(query="x"))
    assert pack.items == [] and "failed" in pack.warnings[0]
    assert svc.capture_memory_candidate(CaptureRequest(text="x")).accepted is False
    assert svc.memory_health()["status"] == "unreachable"


def test_feedback_reaches_the_backend_and_swallows_failure(tmp_path):
    static = StaticMemoryAdapter()
    svc = make_service(tmp_path, memory=static)
    svc.record_memory_feedback(MemoryFeedbackRequest(
        task_id=None, memory_item_id="m1", source_id=None, feedback="useful"))
    assert static.feedback[0].feedback == "useful"

    broken = make_service(tmp_path, memory=BrokenMemory())
    broken.record_memory_feedback(MemoryFeedbackRequest(
        task_id=None, memory_item_id=None, source_id=None, feedback="stale"))  # must not raise


# ---------------------------------------------------------------- context pack
def test_context_pack_combines_ledger_truth_and_labeled_memory(tmp_path):
    svc = make_service(tmp_path, memory=StaticMemoryAdapter([PROMOTED]))
    task = svc.create_task(CreateTaskRequest(
        title="Refactor auth", goal="dedupe expiry", constraints=["No schema changes"]))
    svc.claim_task(task.id, "claude-code")

    pack = svc.get_task_context_pack(task.id, query="refresh token")
    assert pack.memory_is_external_context is True
    assert pack.handoff.constraints == ["No schema changes"]
    assert [i.source_id for i in pack.memory.items] == ["decision_004"]


def test_context_pack_excludes_pending_by_default(tmp_path):
    svc = make_service(tmp_path, memory=StaticMemoryAdapter([PENDING]))
    task = svc.create_task(CreateTaskRequest(title="t", goal="qwen"))
    assert svc.get_task_context_pack(task.id, query="qwen").memory.items == []
    included = svc.get_task_context_pack(task.id, query="qwen", include_pending=True)
    assert [i.status for i in included.memory.items] == ["pending"]


# ------------------------------------------------- local compute provider (B)
class FakeLocalManager(LocalComputeProvider):
    """Stands in for a user's Ollama/vLLM rig behind the HTTP contract."""

    def __init__(self, eligible=True, healthy=True):
        self.eligible = eligible
        self.healthy = healthy
        self.runs = 0

    def inventory(self):
        return []

    def loaded(self):
        return []

    def estimate(self, request):
        return LocalEstimate(
            eligible=self.eligible, selected_model="qwen2.5-coder-14b-q4", runtime="vllm",
            loaded=True, estimated_queue_seconds=8, estimated_tokens_per_second=42,
            estimated_quality=0.72,
            reason={"hard_gates": ["context_fits", "capability_match"],
                    "score_terms": {"already_loaded": 1.0, "capability": 0.8}},
        )

    def run(self, request):
        self.runs += 1
        return LocalRunResult(
            status="succeeded", output="found duplicate expiry checks",
            model="qwen2.5-coder-14b-q4", runtime="vllm",
            metrics={"tokens_in": 100, "tokens_out": 20},
        )

    def health(self):
        return {"status": "ok"} if self.healthy else {"status": "unreachable"}


def test_backplane_runs_without_any_local_model_manager(tmp_path):
    svc = make_service(tmp_path)  # echo only
    task = svc.create_task(CreateTaskRequest(title="t"))
    assert svc.submit_subtask(
        task.id, SubtaskRequest(title="t", instructions="i")
    ).status is SubtaskStatus.succeeded


def test_local_manager_worker_runs_and_stores_an_artifact(tmp_path):
    manager = FakeLocalManager()
    svc = make_service(tmp_path, workers=[LocalModelManagerWorkerAdapter(manager)])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="find dupes", instructions="inspect auth",
        privacy_tier=PrivacyTier.repo_sensitive))

    assert subtask.status is SubtaskStatus.succeeded and manager.runs == 1
    body = svc.read_artifact_chunk(subtask.result_artifact_id, 0, 8000).content
    assert body == "found duplicate expiry checks"


def test_route_explanation_nests_the_local_manager_estimate(tmp_path):
    svc = make_service(tmp_path, workers=[LocalModelManagerWorkerAdapter(FakeLocalManager())])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))

    explanation = svc.explain_route(subtask.id)
    assert explanation.worker_type == "local_model_manager"
    assert explanation.local_estimate["selected_model"] == "qwen2.5-coder-14b-q4"
    assert explanation.local_estimate["runtime"] == "vllm"
    assert explanation.local_estimate["estimated_queue_seconds"] == 8


def test_local_manager_outage_rejects_the_worker_without_crashing(tmp_path):
    svc = make_service(tmp_path, workers=[LocalModelManagerWorkerAdapter(
        FakeLocalManager(healthy=False))])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))

    assert subtask.status is SubtaskStatus.failed
    rejected = svc.explain_route(subtask.id).rejected_workers[0]
    assert rejected.gate == "healthy" and "unreachable" in rejected.reason


def test_local_manager_declining_a_subtask_fails_the_run_not_the_process(tmp_path):
    svc = make_service(tmp_path, workers=[LocalModelManagerWorkerAdapter(
        FakeLocalManager(eligible=False))])
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    assert subtask.status is SubtaskStatus.failed
    assert "ineligible" in svc.get_subtask(subtask.id).runs[0].error


def test_http_local_provider_speaks_the_documented_surface():
    calls = []

    def transport(method, url, payload):
        calls.append((method, url))
        if url.endswith("/route/estimate"):
            return {"eligible": True, "selected_model": "qwen", "runtime": "vllm",
                    "loaded": True, "estimated_queue_seconds": 3}
        if url.endswith("/generate"):
            return {"status": "succeeded", "output": "hi", "model": "qwen", "runtime": "vllm"}
        if url.endswith("/models"):
            return {"models": [{"id": "qwen", "runtime": "vllm", "capabilities": ["code"],
                                "context_tokens": 32768, "loaded": True}]}
        return {"status": "ok"}

    provider = HttpLocalComputeProvider("http://localhost:8090", transport=transport)
    assert provider.health() == {"status": "ok"}
    assert provider.inventory()[0].id == "qwen"
    from agentconnect.core.local_compute import LocalEstimateRequest, LocalRunRequest

    estimate = provider.estimate(LocalEstimateRequest(
        task_type="code_search", privacy_tier="repo_sensitive",
        required_capabilities=["code"], context_tokens=12000, max_output_tokens=2000))
    assert estimate.eligible and estimate.runtime == "vllm"
    assert provider.run(LocalRunRequest(model="qwen", task_type="x", prompt="p")).output == "hi"
    assert ("POST", "http://localhost:8090/route/estimate") in calls


def test_unreachable_http_local_provider_reports_rather_than_raises():
    def transport(method, url, payload):
        raise ConnectionError("connection refused")

    provider = HttpLocalComputeProvider("http://localhost:9999", transport=transport)
    assert provider.health()["status"] == "unreachable"
