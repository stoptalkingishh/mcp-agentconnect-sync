"""Layered memory stack: ContextBuilder, MemoryRouter, MemoryRanker, three backends.

Covers memory-stack spec §16 test list and §17 acceptance. Every backend is a
fake HTTP transport — offline, deterministic.

The invariant under test throughout: **trust comes from promotion in WikiBrain,
never from a retrieval engine finding something.**
"""

import pytest

from agentconnect.core import (
    AgentConnectService,
    CogneeMemoryAdapter,
    ContextBuilder,
    CreateTaskRequest,
    EchoWorker,
    GraphitiMemoryAdapter,
    MemoryConfig,
    MemoryItem,
    MemoryRanker,
    MemoryRouter,
    RecallPack,
    RecordDecisionRequest,
    StaticMemoryAdapter,
    SubtaskRequest,
    WikiBrainMemoryAdapter,
)
from agentconnect.core.memory import label

PROMOTED_TEXT = "Refresh token validation stays in auth/session.py."
COGNEE_TEXT = "Auth session module handles token refresh."


def wikibrain_transport(items=None, pending=None, promoted=None):
    state = {"promoted": [], "captured": []}

    def transport(method, url, payload):
        if url.endswith("/recall"):
            return {"items": items if items is not None else [
                {"text": PROMOTED_TEXT, "status": "promoted", "trusted": True, "confidence": "verified",
                 "source_id": "claim_004", "trusted": True},
            ]}
        if url.endswith("/capture"):
            state["captured"].append(payload)
            return {"accepted": True, "candidate_id": "candidate_1", "status": "pending"}
        if "/promote" in url:
            state["promoted"].append(url)
            return {"claim_id": "claim_9", "text": PROMOTED_TEXT, "status": "promoted",
                    "promoted_by": payload["promoted_by"]}
        if "/candidates" in url:
            return {"candidates": pending or [{"candidate_id": "candidate_1",
                                               "text": "maybe true"}]}
        return {"status": "ok"}

    transport.state = state
    return transport


def cognee_transport(results=None):
    state = {"indexed": []}

    def transport(method, url, payload):
        if url.endswith("/search"):
            return {"results": results if results is not None else [
                {"text": COGNEE_TEXT, "source_id": "doc_17", "score": 0.81},
            ]}
        if url.endswith("/add"):
            state["indexed"].append(payload)
            return {}
        return {"status": "ok"}

    transport.state = state
    return transport


def graphiti_transport(facts=None):
    state = {"indexed": []}

    def transport(method, url, payload):
        if url.endswith("/search"):
            return {"facts": facts if facts is not None else [
                {"fact": "auth/session.py owns validation since v2.",
                 "source_id": "claim_004", "valid_from": "2026-01-01"},
            ]}
        if url.endswith("/episodes"):
            state["indexed"].append(payload)
            return {}
        return {"status": "ok"}

    transport.state = state
    return transport


def make_stack(tmp_path, wb=None, cog=None, gra=None, config=None, enabled=("wikibrain",)):
    adapters = {}
    if "wikibrain" in enabled:
        adapters["wikibrain"] = WikiBrainMemoryAdapter(transport=wb or wikibrain_transport())
    if "cognee" in enabled:
        adapters["cognee"] = CogneeMemoryAdapter(transport=cog or cognee_transport())
    if "graphiti" in enabled:
        adapters["graphiti"] = GraphitiMemoryAdapter(transport=gra or graphiti_transport())
    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"), workers=[EchoWorker()],
        memory_backends=adapters, memory_config=config or MemoryConfig(),
    )
    return svc, adapters


@pytest.fixture()
def task_svc(tmp_path):
    svc, adapters = make_stack(tmp_path, enabled=("wikibrain", "cognee", "graphiti"))
    task = svc.create_task(CreateTaskRequest(
        title="Refactor auth", goal="dedupe expiry", constraints=["No schema changes"]))
    return svc, adapters, task


# ------------------------------------------------------ acceptance 1, 2, 16.1-2
def test_runs_with_memory_disabled_and_returns_task_only_context(tmp_path):
    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "a"), workers=[EchoWorker()],
        memory_config=MemoryConfig(enabled=False),
    )
    task = svc.create_task(CreateTaskRequest(title="t", goal="g", constraints=["stay local"]))
    pack = svc.get_task_context_pack(task.id)

    assert pack.backends_queried == []
    assert any("memory is disabled" in w for w in pack.warnings)
    # Ledger truth still arrives — the constraint is task state, not memory.
    assert [i.text for i in pack.memory.items] == ["stay local"]
    assert pack.handoff is not None


def test_runs_with_any_single_backend_disabled(tmp_path):
    for enabled in (("wikibrain",), ("wikibrain", "cognee"), ("wikibrain", "graphiti")):
        svc, _ = make_stack(tmp_path, enabled=enabled)
        task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
        pack = svc.get_task_context_pack(task.id)
        assert set(pack.backends_queried) <= set(enabled)
        assert pack.memory.items  # wikibrain still answers


def test_a_dead_backend_degrades_the_pack_but_does_not_raise(tmp_path):
    def broken(method, url, payload):
        raise ConnectionError("cognee is down")

    svc, _ = make_stack(tmp_path, cog=broken, enabled=("wikibrain", "cognee"))
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id)

    assert any("cognee recall failed" in w for w in pack.warnings)
    assert [i.source_id for i in pack.memory.items] == ["claim_004"]  # wikibrain survived

    # And a subtask still runs to completion with memory broken.
    assert svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i")).status.value == "succeeded"


# ------------------------------------------------------------- trust semantics
def test_wikibrain_promoted_claim_appears_in_manager_brief_as_trusted(task_svc):
    svc, _, task = task_svc
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    claim = next(i for i in pack.memory.items if i.source_id == "claim_004")
    assert claim.metadata["backend"] == "wikibrain"
    assert claim.metadata["role"] == "trusted_authority"
    assert claim.metadata["trusted"] is True


def test_wikibrain_pending_claim_is_excluded_by_default(tmp_path):
    wb = wikibrain_transport(items=[
        {"text": PROMOTED_TEXT, "status": "promoted", "trusted": True, "confidence": "verified",
         "source_id": "claim_004", "trusted": True},
        {"text": "Qwen is weak at auth review.", "status": "pending", "confidence": "low",
         "source_id": "candidate_7"},
    ])
    svc, _ = make_stack(tmp_path, wb=wb)
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))

    default = svc.get_task_context_pack(task.id)
    assert [i.source_id for i in default.memory.items] == ["claim_004"]

    explicit = svc.get_task_context_pack(task.id, include_pending=True)
    assert "candidate_7" in [i.source_id for i in explicit.memory.items]
    assert any("pending" in w for w in explicit.warnings)


def test_superseded_wikibrain_claim_is_excluded_by_default(tmp_path):
    wb = wikibrain_transport(items=[
        {"text": "Old law.", "status": "superseded", "confidence": "high",
         "source_id": "claim_001", "superseded_by": "claim_004"},
    ])
    svc, _ = make_stack(tmp_path, wb=wb)
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    assert svc.get_task_context_pack(task.id).memory.items == []


def test_cognee_result_is_broad_retrieval_not_trusted_authority(task_svc):
    svc, _, task = task_svc
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    hit = next(i for i in pack.memory.items if i.source_id == "doc_17")
    assert hit.metadata["role"] == "broad_retrieval"
    assert hit.metadata["trusted"] is False
    assert hit.status == "unknown"


def test_graphiti_result_is_temporal_relationship_not_trusted_authority(task_svc):
    svc, _, task = task_svc
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    fact = next(i for i in pack.memory.items
                if i.metadata.get("backend") == "graphiti")
    assert fact.metadata["role"] == "temporal_graph"
    assert fact.metadata["trusted"] is False


def test_cognee_cannot_override_wikibrain_trust_state(tmp_path):
    """Cognee returning the same sentence does not make it a promoted claim."""
    cog = cognee_transport(results=[{"text": PROMOTED_TEXT, "source_id": "doc_1"}])
    svc, _ = make_stack(tmp_path, cog=cog, enabled=("cognee",))
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id)
    assert all(i.metadata["trusted"] is False for i in pack.memory.items)


# --------------------------------------------------------------- supersession
def test_graphiti_supersession_warning_appears_in_project_evolution(tmp_path):
    gra = graphiti_transport(facts=[
        {"fact": "Validation lived in middleware.", "source_id": "claim_001",
         "superseded_by": "claim_004"},
    ])
    svc, _ = make_stack(tmp_path, gra=gra, enabled=("wikibrain", "graphiti"))
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))

    evolution = svc.get_task_context_pack(task.id, profile="project_evolution")
    assert any("superseded by claim_004" in w for w in evolution.warnings)
    # project_evolution asks for superseded facts; manager_brief does not.
    assert any(i.status == "superseded" for i in evolution.memory.items)
    manager = svc.get_task_context_pack(task.id, profile="manager_brief")
    assert all(i.status != "superseded" for i in manager.memory.items)


# -------------------------------------------------------------------- ranking
def test_ranker_dedupes_the_same_fact_from_wikibrain_and_cognee(tmp_path):
    cog = cognee_transport(results=[{"text": PROMOTED_TEXT, "source_id": "doc_1"}])
    svc, _ = make_stack(tmp_path, cog=cog, enabled=("wikibrain", "cognee"))
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id)

    matches = [i for i in pack.memory.items if i.text == PROMOTED_TEXT]
    assert len(matches) == 1
    # The surviving copy is the trusted one, and it remembers who corroborated it.
    assert matches[0].metadata["backend"] == "wikibrain"
    assert "cognee" in matches[0].metadata["also_seen_in"]


def test_ranker_orders_ledger_then_wikibrain_then_graphiti_then_cognee(task_svc):
    svc, _, task = task_svc
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep validation in session.py.", locked=True))
    pack = svc.get_task_context_pack(task.id, profile="manager_brief", max_memory_items=10)
    roles = [i.metadata["role"] for i in pack.memory.items]
    assert roles == sorted(roles, key=lambda r: [
        "ledger", "trusted_authority", "temporal_graph", "broad_retrieval"].index(r))
    assert roles[0] == "ledger"


def test_ranker_prefers_a_promoted_fact_over_a_cognee_result():
    ranker = MemoryRanker()
    promoted = label(MemoryItem(text="x", status="promoted", confidence="verified",
                                source_id="c1"), "wikibrain", "trusted_authority")
    broad = label(MemoryItem(text="y", status="unknown", confidence="unknown",
                             source_id="d1"), "cognee", "broad_retrieval")
    merged = ranker.merge_and_rank(
        [RecallPack(profile="manager_brief", query="", items=[broad]),
         RecallPack(profile="manager_brief", query="", items=[promoted])],
        "manager_brief", 10,
    )
    assert [i.source_id for i in merged.items] == ["c1", "d1"]


def test_ranker_respects_the_context_budget(task_svc):
    svc, _, task = task_svc
    pack = svc.get_task_context_pack(task.id, max_memory_items=1)
    assert len(pack.memory.items) == 1


# -------------------------------------------------------------------- profiles
def test_worker_brief_is_smaller_than_manager_brief_and_carries_no_handoff(task_svc):
    svc, _, task = task_svc
    manager = svc.get_task_context_pack(task.id, profile="manager_brief")
    worker = svc.get_task_context_pack(task.id, profile="worker_brief")

    assert len(worker.memory.items) <= 5 < 8
    assert len(worker.memory.items) <= len(manager.memory.items)
    assert manager.handoff is not None
    assert worker.handoff is None  # no manager debate reaches a bounded worker
    assert "graphiti" not in worker.backends_queried


def test_implementation_constraints_returns_only_hard_constraints(task_svc):
    svc, _, task = task_svc
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep validation in session.py.", locked=True))
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Style: black.", locked=False))

    pack = svc.get_task_context_pack(task.id, profile="implementation_constraints")
    assert pack.backends_queried == ["wikibrain"]
    assert "cognee" not in pack.backends_queried and "graphiti" not in pack.backends_queried
    kinds = {i.metadata.get("kind") for i in pack.memory.items
             if i.metadata["role"] == "ledger"}
    assert kinds <= {"constraint", "locked_decision", "hard_policy"}
    assert all(i.metadata["trusted"] or i.metadata["role"] != "ledger"
               for i in pack.memory.items)
    assert "Style: black." not in [i.text for i in pack.memory.items]


def test_hard_policies_are_ledger_truth_and_soft_preferences_are_absent(tmp_path):
    config = MemoryConfig(hard_policies=["Repo-sensitive code must stay local."])
    svc, _ = make_stack(tmp_path, config=config)
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id, profile="hard_policy")

    policy = next(i for i in pack.memory.items if i.metadata.get("kind") == "hard_policy")
    assert policy.metadata["role"] == "ledger" and policy.metadata["trusted"] is True
    # Soft preferences are not modeled anywhere in the core path.
    assert not any("likes tables" in i.text or "concise" in i.text
                   for i in pack.memory.items)


def test_router_falls_back_to_available_adapters_for_unknown_backends(tmp_path):
    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "a"),
        memory=StaticMemoryAdapter([MemoryItem(
            text="a fact", status="promoted", confidence="high", source_id="s1")]),
    )
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id, query="a fact")
    assert pack.backends_queried == ["static"]
    assert [i.source_id for i in pack.memory.items] == ["s1"]


def test_memory_router_narrows_by_profile():
    config = MemoryConfig()
    adapters = {"wikibrain": object(), "cognee": object(), "graphiti": object()}
    router = MemoryRouter(config, adapters)  # type: ignore[arg-type]
    assert router.select_backends("manager_brief") == ["wikibrain", "cognee", "graphiti"]
    assert router.select_backends("worker_brief") == ["wikibrain", "cognee"]
    assert router.select_backends("implementation_constraints") == ["wikibrain"]
    assert router.select_backends("broad_project_rag") == ["cognee"]


# -------------------------------------------------------------- write path
def test_capture_creates_a_pending_candidate_and_never_promotes(task_svc):
    svc, adapters, task = task_svc
    from agentconnect.core.memory import CaptureRequest

    result = svc.capture_memory_candidate(CaptureRequest(
        text="local-qwen is fine for read-only search", task_id=task.id,
        origin_actor_id="claude-code", origin_actor_type="manager"))
    assert result.accepted and result.status == "pending"
    assert result.backend == "wikibrain"
    assert "memory_candidate_captured" in [e.kind for e in svc.list_events(task.id)]


def test_retrieval_backends_refuse_writes_from_agents(tmp_path):
    from agentconnect.core.memory import CaptureRequest

    cognee = CogneeMemoryAdapter(transport=cognee_transport())
    graphiti = GraphitiMemoryAdapter(transport=graphiti_transport())
    for adapter in (cognee, graphiti):
        result = adapter.capture_candidate(CaptureRequest(text="I decree this true"))
        assert result.accepted is False
        assert "trusted authority" in result.message


def test_promotion_is_human_only_and_fans_out_to_both_indexes(tmp_path):
    wb, cog, gra = wikibrain_transport(), cognee_transport(), graphiti_transport()
    svc, adapters = make_stack(tmp_path, wb=wb, cog=cog, gra=gra,
                               enabled=("wikibrain", "cognee", "graphiti"))
    claim = svc.promote_memory_candidate("candidate_1", promoted_by="matthew")

    assert claim["status"] == "promoted" and claim["promoted_by"] == "matthew"
    assert sorted(claim["indexed_into"]) == ["cognee", "graphiti"]
    assert cog.state["indexed"][0]["source_id"] == "claim_9"
    assert gra.state["indexed"][0]["name"] == "claim_9"


def test_promotion_survives_an_index_outage(tmp_path):
    def broken_cognee(method, url, payload):
        if url.endswith("/add"):
            raise ConnectionError("cognee down")
        return {"results": []}

    svc, _ = make_stack(tmp_path, cog=broken_cognee, enabled=("wikibrain", "cognee", "graphiti"))
    claim = svc.promote_memory_candidate("candidate_1", promoted_by="matthew")
    # The trusted authority is the record; the indexes are caches of it.
    assert claim["status"] == "promoted"
    assert claim["indexed_into"] == ["graphiti"] and claim["index_failures"] == ["cognee"]


def test_promotion_without_a_trusted_authority_is_refused(tmp_path):
    from agentconnect.core.errors import InvalidRequest

    svc, _ = make_stack(tmp_path, enabled=("cognee",))
    with pytest.raises(InvalidRequest, match="trusted memory authority"):
        svc.promote_memory_candidate("candidate_1", "matthew")


def test_pending_queue_is_readable_for_the_librarian(task_svc):
    svc, _, _ = task_svc
    assert svc.list_pending_memory()[0]["candidate_id"] == "candidate_1"


# ----------------------------------------------------------- worker push path
def test_worker_brief_is_attached_to_the_subtask_for_harnesses_without_mcp(task_svc):
    svc, _, task = task_svc
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    pack = svc.get_task_context_pack(task.id, profile="worker_brief")
    svc.attach_context_to_subtask(subtask.id, pack)

    stored = svc.get_subtask(subtask.id).subtask.metadata["context_pack"]
    assert stored["profile"] == "worker_brief"
    assert stored["memory_is_external_context"] is True
    assert all("trusted" in item for item in stored["items"])


def test_memory_health_reports_every_backend_and_the_trust_authority(task_svc):
    svc, _, _ = task_svc
    health = svc.memory_health()
    assert health["trusted_authority"] == "wikibrain"
    assert set(health["backends"]) == {"wikibrain", "cognee", "graphiti"}


def test_context_builder_is_directly_constructible(tmp_path):
    svc, adapters = make_stack(tmp_path)
    builder = ContextBuilder(svc, adapters, MemoryConfig())
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = builder.build_context_pack(task.id, profile="manager_brief", max_items=3)
    assert pack.profile == "manager_brief" and len(pack.memory.items) <= 3
