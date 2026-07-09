"""Linear mirror: mapping, push sync, webhook ingest, approval commands (§14, §15, §21).

Offline throughout — the GraphQL transport is a fake. The load-bearing assertions
are the *withholding* ones: a secret_sensitive task's content must never appear
in anything we hand to Linear.
"""

import json

import pytest

from agentconnect.core import (
    AgentConnectService,
    CreateArtifactRequest,
    CreateTaskRequest,
    EchoWorker,
    PrivacyTier,
    RoutePolicy,
    SubtaskRequest,
    SubtaskStatus,
    WorkerLocation,
)
from agentconnect.core.workers import RawModelWorker
from agentconnect.linear import LinearClient, LinearSync, parse_command
from agentconnect.linear import mapping, webhooks


class FakeTransport:
    """Records GraphQL calls and answers them plausibly."""

    def __init__(self):
        self.calls = []
        self.comments = []
        self.issues = {}
        self._n = 0

    def __call__(self, query, variables):
        self.calls.append((query, variables))
        if "IssueCreate" in query:
            self._n += 1
            issue = {"id": f"lin-{self._n}", "identifier": f"AUTH-{self._n}",
                     "url": f"https://linear.app/x/issue/AUTH-{self._n}", "title": ""}
            self.issues[issue["id"]] = dict(variables["input"])
            return {"issueCreate": {"success": True, "issue": issue}}
        if "IssueUpdate" in query:
            issue_id = variables["id"]
            self.issues.setdefault(issue_id, {}).update(variables["input"])
            return {"issueUpdate": {"success": True, "issue": {
                "id": issue_id, "identifier": "AUTH-1",
                "url": "https://linear.app/x/issue/AUTH-1", "title": ""}}}
        if "CommentCreate" in query:
            self.comments.append(variables["input"]["body"])
            return {"commentCreate": {"success": True, "comment": {"id": "c1"}}}
        if "Labels" in query:
            return {"team": {"labels": {"nodes": [
                {"id": "l1", "name": "agentconnect"},
                {"id": "l2", "name": "privacy:repo-sensitive"},
                {"id": "l3", "name": "needs-approval"},
                {"id": "l4", "name": "manager:claude-code"},
            ]}}}
        return {}


def make_service(tmp_path, workers=None, policy=None):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=workers if workers is not None else [EchoWorker()], policy=policy,
    )


@pytest.fixture()
def wired(tmp_path):
    svc = make_service(tmp_path)
    transport = FakeTransport()
    sync = LinearSync(svc, LinearClient(transport=transport), team_id="team-1")
    return svc, sync, transport


# ------------------------------------------------------------------- mapping
def test_issue_body_carries_state_and_labels(wired):
    svc, _, _ = wired
    task = svc.create_task(CreateTaskRequest(
        title="Refactor auth", goal="dedupe expiry", constraints=["No schema changes"]))
    svc.claim_task(task.id, "claude-code")
    detail = svc.get_task(task.id)

    body = mapping.issue_body(detail, svc.get_handoff_summary(task.id).text)
    assert f"`{task.id}`" in body
    assert "dedupe expiry" in body and "No schema changes" in body
    assert "claude-code" in body

    labels = mapping.labels_for(detail)
    assert "agentconnect" in labels and "manager:claude-code" in labels


def test_secret_sensitive_task_withholds_every_bit_of_content(wired):
    svc, _, _ = wired
    task = svc.create_task(CreateTaskRequest(
        title="Rotate the deploy key", goal="Rotate sk-ABCD1234EFGH5678IJKL",
        constraints=["Never log the key"]))
    svc.submit_subtask(task.id, SubtaskRequest(
        title="rotate", instructions="the raw secret is sk-ABCD1234EFGH5678IJKL",
        privacy_tier=PrivacyTier.secret_sensitive))
    detail = svc.get_task(task.id)
    assert detail.effective_privacy is PrivacyTier.secret_sensitive
    assert mapping.is_withheld(detail)

    body = mapping.issue_body(detail, svc.get_handoff_summary(task.id).text)
    assert "sk-ABCD1234EFGH5678IJKL" not in body
    assert "Never log the key" not in body
    assert mapping.WITHHELD in body
    # The issue still exists and is actionable: id, status, privacy label survive.
    assert task.id in body and "secret_sensitive" in body
    assert "privacy:secret-sensitive" in mapping.labels_for(detail)


def test_artifact_comment_is_a_pointer_never_a_payload(wired):
    svc, _, _ = wired
    task = svc.create_task(CreateTaskRequest(title="t"))
    svc.create_artifact(task.id, CreateArtifactRequest(
        type="report", content="THE ENTIRE PATCH BODY", summary="A report"))
    summary = svc.list_artifacts(task.id)[0]
    comment = mapping.artifact_comment(summary)
    assert "THE ENTIRE PATCH BODY" not in comment
    assert summary.id in comment and "agentconnect artifacts read" in comment


# --------------------------------------------------------------- push sync
def test_sync_creates_then_updates_one_issue(wired):
    svc, sync, transport = wired
    task = svc.create_task(CreateTaskRequest(title="Refactor auth", goal="dedupe"))

    ref = sync.sync_task(task.id)
    assert ref.external_id == "lin-1"
    assert svc.get_task(task.id).task.linear_issue_url.endswith("AUTH-1")

    sync.sync_task(task.id)  # idempotent: updates, never a second issue
    creates = [q for q, _ in transport.calls if "IssueCreate" in q]
    updates = [q for q, _ in transport.calls if "IssueUpdate" in q]
    assert len(creates) == 1 and len(updates) == 1


def test_sync_skips_labels_the_team_does_not_have(wired):
    svc, sync, transport = wired
    task = svc.create_task(CreateTaskRequest(title="t"))
    svc.claim_task(task.id, "claude-code")
    sync.sync_task(task.id)
    create = next(v for q, v in transport.calls if "IssueCreate" in q)
    # privacy:public is not in the fake team's label set, so it is dropped, not created.
    assert set(create["input"]["labelIds"]) == {"l1", "l4"}


def test_comments_require_a_synced_task(wired):
    svc, sync, transport = wired
    task = svc.create_task(CreateTaskRequest(title="t"))
    decision = svc.record_decision(task.id, __import__(
        "agentconnect.core.models", fromlist=["RecordDecisionRequest"]
    ).RecordDecisionRequest(made_by="claude-code", decision="d", locked=True))
    assert sync.post_decision(task.id, decision.id) is False  # not synced yet
    sync.sync_task(task.id)
    assert sync.post_decision(task.id, decision.id) is True
    assert "**Decision:**" in transport.comments[-1] and "locked" in transport.comments[-1]


def test_approval_request_comment_quotes_price_and_why_local_lost(tmp_path):
    cloud = RawModelWorker("cheap_cloud_deepseek", lambda p: "x", model="deepseek",
                           location=WorkerLocation.cloud, privacy_tiers=[PrivacyTier.public],
                           cost_per_1k_tokens_usd=0.5)
    local = RawModelWorker("local_qwen", lambda p: "x", model="qwen",
                           location=WorkerLocation.local,
                           privacy_tiers=[PrivacyTier.repo_sensitive])
    svc = make_service(tmp_path, workers=[cloud, local], policy=RoutePolicy(max_cost_usd=10.0))
    transport = FakeTransport()
    sync = LinearSync(svc, LinearClient(transport=transport), team_id="team-1")

    task = svc.create_task(CreateTaskRequest(title="t"))
    sync.sync_task(task.id)
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))
    assert subtask.status is SubtaskStatus.needs_approval

    assert sync.post_approval_request(subtask.id) is True
    comment = transport.comments[-1]
    assert "cheap_cloud_deepseek" in comment and "cloud" in comment
    assert "/agentconnect approve cloud" in comment
    assert "local_qwen" in comment  # why the free worker was rejected
    assert "privacy_allowed" in comment


# --------------------------------------------------------- command parsing
@pytest.mark.parametrize("text,action,target,max_cost", [
    ("/agentconnect approve cloud", "approve", "cloud", None),
    ("/agentconnect approve rented-gpu max_cost=3.00", "approve", "rented", 3.0),
    ("/agentconnect deny", "deny", None, None),
    ("Sure, go ahead.\n/agentconnect approve cloud\nthanks", "approve", "cloud", None),
])
def test_parse_command(text, action, target, max_cost):
    command = parse_command(text)
    assert command.action == action
    assert command.target == target
    assert command.max_cost_usd == max_cost


def test_parse_command_ignores_chatter_and_unknown_verbs():
    assert parse_command("looks good to me") is None
    assert parse_command("/agentconnect yolo cloud") is None
    assert parse_command("") is None


def test_deny_captures_a_free_text_reason():
    command = parse_command("/agentconnect deny too expensive right now")
    assert command.action == "deny" and command.reason == "too expensive right now"


def test_bad_max_cost_is_ignored_rather_than_crashing():
    assert parse_command("/agentconnect approve cloud max_cost=lots").max_cost_usd is None


# ------------------------------------------------------------ webhook ingest
def _comment_payload(issue_id, body, author="matthew"):
    return {"action": "create", "type": "Comment",
            "data": {"body": body, "issue": {"id": issue_id}, "user": {"name": author}}}


@pytest.fixture()
def approval_wired(tmp_path):
    cloud = RawModelWorker("cloud", lambda p: "cloud output", model="gpt",
                           location=WorkerLocation.cloud, privacy_tiers=[PrivacyTier.public],
                           cost_per_1k_tokens_usd=0.5)
    svc = make_service(tmp_path, workers=[cloud], policy=RoutePolicy(max_cost_usd=10.0))
    transport = FakeTransport()
    sync = LinearSync(svc, LinearClient(transport=transport), team_id="team-1")
    task = svc.create_task(CreateTaskRequest(title="t"))
    sync.sync_task(task.id)
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="i", privacy_tier=PrivacyTier.public))
    return svc, task, subtask


def test_approve_comment_runs_the_parked_subtask(approval_wired):
    svc, task, subtask = approval_wired
    results = webhooks.handle_webhook(
        svc, _comment_payload("lin-1", "/agentconnect approve cloud max_cost=3.00"))

    assert results[0]["kind"] == "approved"
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.succeeded
    approval = svc.list_approvals(task.id)[0]
    assert approval.status.value == "granted" and approval.decided_by == "matthew"
    assert approval.max_cost_usd == 3.0


def test_deny_comment_fails_the_subtask(approval_wired):
    svc, task, subtask = approval_wired
    results = webhooks.handle_webhook(svc, _comment_payload("lin-1", "/agentconnect deny"))
    assert results[0]["kind"] == "denied"
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.failed
    assert svc.list_approvals(task.id)[0].status.value == "denied"


def test_approving_the_wrong_route_class_matches_nothing(approval_wired):
    svc, _, subtask = approval_wired
    results = webhooks.handle_webhook(
        svc, _comment_payload("lin-1", "/agentconnect approve rented-gpu"))
    assert results[0]["kind"] == "approval_no_match"
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.needs_approval


def test_webhook_for_an_unmapped_issue_is_ignored(approval_wired):
    svc, _, _ = approval_wired
    results = webhooks.handle_webhook(
        svc, _comment_payload("lin-999", "/agentconnect approve cloud"))
    assert results[0]["kind"] == "ignored"


def test_chatter_comment_is_ignored(approval_wired):
    svc, _, _ = approval_wired
    assert webhooks.handle_webhook(svc, _comment_payload("lin-1", "nice work"))[0]["kind"] \
        == "ignored"


def test_issue_status_change_is_recorded_but_never_overwrites_the_ledger(wired):
    svc, sync, _ = wired
    task = svc.create_task(CreateTaskRequest(title="t"))
    sync.sync_task(task.id)
    before = svc.get_task(task.id).task.status

    results = webhooks.handle_webhook(svc, {
        "action": "update", "type": "Issue",
        "data": {"id": "lin-1", "state": {"name": "Done"}, "user": {"name": "matthew"}},
        "updatedFrom": {"stateId": "old"},
    })
    assert results[0]["kind"] == "status_recorded"
    assert svc.get_task(task.id).task.status is before  # Linear is a mirror, not truth
    assert "linear_status_change" in [e.kind for e in svc.list_events(task.id)]


def test_issue_assignment_creates_a_manager_inbox_item(wired):
    svc, sync, _ = wired
    task = svc.create_task(CreateTaskRequest(title="Refactor auth"))
    sync.sync_task(task.id)

    results = webhooks.handle_webhook(svc, {
        "action": "update", "type": "Issue",
        "data": {"id": "lin-1", "assignee": {"name": "codex-manager"}},
        "updatedFrom": {"assigneeId": None},
    })
    assert results[0] == {"kind": "inbox_item", "task_id": task.id,
                          "manager_id": "codex-manager"}
    inbox = svc.get_manager_inbox("codex-manager")
    assert [i.ref_id for i in inbox] == [task.id]

    # Replaying the same webhook must not duplicate the item.
    webhooks.handle_webhook(svc, {
        "action": "update", "type": "Issue",
        "data": {"id": "lin-1", "assignee": {"name": "codex-manager"}},
        "updatedFrom": {"assigneeId": None},
    })
    assert len(svc.get_manager_inbox("codex-manager")) == 1


def test_unhandled_webhook_types_are_ignored_loudly(wired):
    svc, _, _ = wired
    results = webhooks.handle_webhook(svc, {"action": "remove", "type": "Project", "data": {}})
    assert results[0]["kind"] == "ignored" and "Project/remove" in results[0]["reason"]
