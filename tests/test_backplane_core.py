"""Core task ledger: tasks, constraints, claims, decisions, attempts, artifacts,
handoff (spec §24, first eleven items).

Everything here is offline and deterministic. The clock is injected so claim
expiry is tested by moving time, not by sleeping.
"""

import pytest

from agentconnect.core import (
    AgentConnectService,
    ArtifactType,
    ActorType,
    ClaimRole,
    Conflict,
    CreateArtifactRequest,
    CreateTaskRequest,
    EchoWorker,
    NotFound,
    PolicyViolation,
    Priority,
    RecordAttemptRequest,
    RecordDecisionRequest,
    TaskFilters,
    TaskStatus,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def svc(tmp_path, clock):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=[EchoWorker()], clock=clock,
    )


@pytest.fixture()
def task(svc):
    return svc.create_task(CreateTaskRequest(
        title="Refactor auth session handling",
        goal="Reduce duplicated token expiry logic without changing public login behavior.",
        created_by="human", priority=Priority.high,
        constraints=["No schema changes", "Preserve middleware contract"],
    ))


# ---------------------------------------------------------------- task basics
def test_create_task_records_constraints_and_starts_queued(svc, task):
    assert task.id.startswith("task_")
    assert task.status is TaskStatus.queued
    detail = svc.get_task(task.id)
    assert [c.text for c in detail.constraints] == [
        "No schema changes", "Preserve middleware contract"
    ]


def test_get_task_unknown_raises_not_found(svc):
    with pytest.raises(NotFound):
        svc.get_task("task_does_not_exist")


def test_list_tasks_filters_by_status_and_manager(svc, task):
    other = svc.create_task(CreateTaskRequest(title="Other"))
    svc.claim_task(task.id, "claude-code")

    assert {t.id for t in svc.list_tasks(TaskFilters())} == {task.id, other.id}
    in_progress = svc.list_tasks(TaskFilters(status=TaskStatus.in_progress))
    assert [t.id for t in in_progress] == [task.id]
    mine = svc.list_tasks(TaskFilters(current_manager="claude-code"))
    assert [t.id for t in mine] == [task.id]


def test_add_constraint_after_creation(svc, task):
    svc.add_constraint(task.id, "Do not touch the migration files", "claude-code")
    assert svc.get_task(task.id).constraints[-1].text == "Do not touch the migration files"


# --------------------------------------------------------------------- claims
def test_claim_sets_current_manager_and_moves_task_in_progress(svc, task):
    claim = svc.claim_task(task.id, "claude-code", ClaimRole.primary_manager.value)
    assert claim.id.startswith("claim_")
    detail = svc.get_task(task.id)
    assert detail.task.current_manager == "claude-code"
    assert detail.task.status is TaskStatus.in_progress


def test_primary_claim_is_exclusive(svc, task):
    svc.claim_task(task.id, "claude-code")
    with pytest.raises(Conflict, match="claude-code"):
        svc.claim_task(task.id, "codex")


def test_same_manager_reclaiming_is_a_renewal_not_a_conflict(svc, task, clock):
    first = svc.claim_task(task.id, "claude-code", ttl_seconds=60)
    clock.advance(30)
    second = svc.claim_task(task.id, "claude-code", ttl_seconds=60)
    assert second.expires_at > first.expires_at


def test_non_primary_roles_are_not_exclusive(svc, task):
    svc.claim_task(task.id, "claude-code", ClaimRole.primary_manager.value)
    svc.claim_task(task.id, "codex", ClaimRole.reviewer.value)
    svc.claim_task(task.id, "gemini", ClaimRole.observer.value)
    roles = {c.manager_id: c.role for c in svc.get_task(task.id).active_claims}
    assert roles == {
        "claude-code": ClaimRole.primary_manager,
        "codex": ClaimRole.reviewer,
        "gemini": ClaimRole.observer,
    }


def test_expired_claim_does_not_block_a_new_one(svc, task, clock):
    svc.claim_task(task.id, "claude-code", ttl_seconds=60)
    clock.advance(61)
    assert svc.get_task(task.id).active_claims == []
    claim = svc.claim_task(task.id, "codex")  # no Conflict: the lease lapsed
    assert claim.manager_id == "codex"


def test_release_clears_current_manager_and_frees_the_task(svc, task):
    svc.claim_task(task.id, "claude-code")
    svc.release_task(task.id, "claude-code")
    assert svc.get_task(task.id).task.current_manager is None
    svc.claim_task(task.id, "codex")  # must not raise


def test_release_without_a_claim_is_not_found(svc, task):
    with pytest.raises(NotFound):
        svc.release_task(task.id, "nobody")


def test_released_claims_remain_in_history(svc, task):
    svc.claim_task(task.id, "claude-code")
    svc.release_task(task.id, "claude-code")
    history = svc.list_claims(task.id)
    assert len(history) == 1 and history[0].released_at is not None


# ------------------------------------------------------------------ decisions
def test_record_decision_and_lock_it(svc, task):
    decision = svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep refresh token validation in auth/session.py.",
        rationale="Middleware assumes this location.", locked=True,
    ))
    assert decision.locked and decision.superseded_by is None


def test_superseding_an_unlocked_decision_is_ordinary_work(svc, task):
    first = svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Use a shared helper."))
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="codex", decision="Inline it instead.", supersedes=[first.id]))
    stored = {d.id: d for d in svc.list_decisions(task.id)}
    assert stored[first.id].superseded_by is not None


def test_superseding_a_locked_decision_without_authority_is_refused(svc, task):
    locked = svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep validation in auth/session.py.", locked=True))
    svc.claim_task(task.id, "codex")
    with pytest.raises(PolicyViolation, match="human_owner"):
        svc.record_decision(task.id, RecordDecisionRequest(
            made_by="codex", decision="Move it to middleware.", supersedes=[locked.id]))
    assert svc.list_decisions(task.id)[0].superseded_by is None


def test_human_owner_may_supersede_a_locked_decision(svc, task):
    locked = svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep validation in auth/session.py.", locked=True))
    svc.claim_task(task.id, "matthew", ClaimRole.human_owner.value)
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="matthew", decision="Move it to middleware.", supersedes=[locked.id]))
    assert svc.list_decisions(task.id)[0].superseded_by is not None


def test_superseding_an_unknown_decision_is_not_found(svc, task):
    with pytest.raises(NotFound):
        svc.record_decision(task.id, RecordDecisionRequest(
            made_by="claude-code", decision="x", supersedes=["decision_nope"]))


# ------------------------------------------------------------------- attempts
def test_record_attempt_with_artifact_refs(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.report, content="findings", summary="Auth flow map"))
    attempt = svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="claude-code", actor_type=ActorType.manager,
        summary="Mapped the auth flow", outcome="succeeded", artifact_refs=[artifact.id]))
    assert attempt.artifact_refs == [artifact.id]


def test_record_attempt_rejects_unknown_artifact(svc, task):
    with pytest.raises(NotFound):
        svc.record_attempt(task.id, RecordAttemptRequest(
            actor_id="claude-code", summary="x", artifact_refs=["artifact_nope"]))


# ------------------------------------------------------------------ artifacts
def test_artifact_write_read_and_list(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.report, content="line one\nline two\n", summary="A report"))
    assert artifact.size_bytes == len("line one\nline two\n")
    chunk = svc.read_artifact_chunk(artifact.id, 0, 8000)
    assert chunk.content == "line one\nline two\n" and chunk.eof and chunk.next_offset is None
    listed = svc.list_artifacts(task.id)
    assert [a.id for a in listed] == [artifact.id]


def test_artifact_chunking_pages_to_eof(svc, task):
    body = "abcdefghij" * 10  # 100 bytes
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(content=body))
    seen, offset, pages = "", 0, 0
    while offset is not None:
        chunk = svc.read_artifact_chunk(artifact.id, offset, 30)
        seen += chunk.content
        offset = chunk.next_offset
        pages += 1
    assert seen == body and pages == 4


def test_artifact_chunking_never_splits_a_multibyte_character(svc, task):
    body = "é" * 10  # 20 bytes, 2 bytes per char
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(content=body))
    chunk = svc.read_artifact_chunk(artifact.id, 0, 3)
    assert chunk.content == "é" and chunk.next_offset == 2

    # A limit narrower than one character must still make progress with a whole char.
    tiny = svc.read_artifact_chunk(artifact.id, 0, 1)
    assert tiny.content == "é" and tiny.next_offset == 2

    seen, offset = "", 0
    while offset is not None:
        page = svc.read_artifact_chunk(artifact.id, offset, 3)
        seen += page.content
        offset = page.next_offset
    assert seen == body


def test_artifact_chunk_offset_past_eof_is_empty(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(content="short"))
    chunk = svc.read_artifact_chunk(artifact.id, 999, 10)
    assert chunk.content == "" and chunk.eof


def test_read_unknown_artifact_is_not_found(svc):
    with pytest.raises(NotFound):
        svc.read_artifact_chunk("artifact_nope")


# -------------------------------------------------------------------- handoff
def test_handoff_surfaces_locked_decisions_and_constraints(svc, task):
    svc.claim_task(task.id, "claude-code")
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Keep refresh token validation in auth/session.py.",
        rationale="Middleware assumes this location.", locked=True))
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Style: black.", locked=False))

    summary = svc.get_handoff_summary(task.id, "claude-code")
    assert [d.decision for d in summary.locked_decisions] == [
        "Keep refresh token validation in auth/session.py."
    ]
    assert "Locked decisions:" in summary.text
    assert "Keep refresh token validation" in summary.text
    assert "Style: black." not in summary.text
    assert summary.viewer_holds_claim is True
    assert "No schema changes" in summary.text


def test_handoff_hides_superseded_locked_decisions(svc, task):
    locked = svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude-code", decision="Old law.", locked=True))
    svc.claim_task(task.id, "matthew", ClaimRole.human_owner.value)
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="matthew", decision="New law.", locked=True, supersedes=[locked.id]))
    summary = svc.get_handoff_summary(task.id)
    assert [d.decision for d in summary.locked_decisions] == ["New law."]


def test_handoff_next_step_asks_for_a_claim_when_unclaimed(svc, task):
    assert "Claim the task" in svc.get_handoff_summary(task.id).suggested_next_step


def test_handoff_is_deterministic(svc, task):
    svc.claim_task(task.id, "claude-code")
    assert svc.get_handoff_summary(task.id, "claude-code").text == \
        svc.get_handoff_summary(task.id, "claude-code").text


def test_handoff_is_persisted_for_the_linear_mirror(svc, task):
    summary = svc.get_handoff_summary(task.id)
    assert svc.get_task(task.id).task.handoff_summary == summary.text


def test_regenerate_handoff_matches_get(svc, task):
    svc.claim_task(task.id, "claude-code")
    assert svc.regenerate_handoff_summary(task.id).text == svc.get_handoff_summary(task.id).text
