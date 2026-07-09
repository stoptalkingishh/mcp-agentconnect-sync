"""Adapters must be translation only (spec §5).

Each of MCP, HTTP, and CLI is driven against the *same* in-memory service, and
each is asserted to produce the same ledger effects. If an adapter ever grows its
own policy, one of these will disagree with the others.
"""

import json

import pytest

from agentconnect.core import (
    AgentConnectService,
    CreateTaskRequest,
    EchoWorker,
    MemoryItem,
    PrivacyTier,
    RoutePolicy,
    StaticMemoryAdapter,
    SubtaskRequest,
    WorkerLocation,
)
from agentconnect.core.workers import RawModelWorker

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from agentconnect.api.app import create_app  # noqa: E402
from agentconnect.cli.main import main as cli_main  # noqa: E402
from agentconnect.mcp.server import build_mcp_server  # noqa: E402

MEMORY = [
    MemoryItem(text="Refresh token validation stays in auth/session.py.",
               status="promoted", confidence="verified", source_id="decision_004"),
    MemoryItem(text="Qwen was weak on auth review.", status="pending", confidence="low"),
]


@pytest.fixture()
def svc(tmp_path):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=[EchoWorker()], memory=StaticMemoryAdapter(list(MEMORY)),
    )


@pytest.fixture()
def task(svc):
    return svc.create_task(CreateTaskRequest(title="Refactor auth", goal="dedupe expiry"))


def _call(mcp, name, **kwargs):
    return json.loads(mcp._tool_manager.get_tool(name).fn(**kwargs))


# ------------------------------------------------------------------------ MCP
def test_mcp_exposes_only_manager_tools(svc):
    mcp = build_mcp_server(service=svc)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    manager_tools = {
        "create_task", "open_task", "get_handoff_summary", "claim_task", "release_task",
        "record_decision", "record_attempt", "request_review", "submit_subtask",
        "get_status", "list_artifacts", "read_artifact_chunk", "explain_route",
    }
    memory_tools = {
        "recall_memory", "capture_memory_candidate", "record_memory_feedback",
        "get_task_context_pack",
    }
    assert manager_tools <= names
    assert memory_tools <= names
    # Administration stays off the manager surface: an agent must not approve its
    # own spend, drain another manager's inbox, or push to Linear.
    assert not names & {"approve_subtask", "deny_subtask", "get_manager_inbox",
                        "linear_sync", "cancel_subtask", "list_tasks"}


def test_mcp_round_trips_the_service(svc):
    mcp = build_mcp_server(service=svc)
    created = _call(mcp, "create_task", title="Refactor auth", goal="dedupe")
    task_id = created["task_id"]
    assert created["next_action"].startswith("claim_task")

    assert _call(mcp, "claim_task", task_id=task_id, manager_id="claude-code")["role"] \
        == "primary_manager"
    decision = _call(mcp, "record_decision", task_id=task_id, made_by="claude-code",
                     decision="Keep validation in auth/session.py.", locked=True)
    assert decision["locked"] is True

    subtask = _call(mcp, "submit_subtask", task_id=task_id, title="find dupes",
                    instructions="inspect auth files", privacy_tier="repo_sensitive")
    assert subtask["status"] == "succeeded"  # direct backend runs inline
    assert subtask["workflow_id"]  # a handle is always returned
    assert subtask["next_action"].startswith("read_artifact_chunk")

    chunk = _call(mcp, "read_artifact_chunk", artifact_id=subtask["result_artifact_id"])
    assert "inspect auth files" in chunk["content"] and chunk["eof"]

    handoff = _call(mcp, "get_handoff_summary", task_id=task_id, manager_id="claude-code")
    assert "Keep validation in auth/session.py." in handoff["summary"]


def test_mcp_never_inlines_artifact_bodies(svc, task):
    mcp = build_mcp_server(service=svc)
    svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="secret body text"))
    listed = _call(mcp, "list_artifacts", task_id=task.id)
    assert all("content" not in a for a in listed)
    opened = _call(mcp, "open_task", task_id=task.id)
    assert "secret body text" not in json.dumps(opened)
    assert opened["artifact_ids"]


def test_mcp_errors_are_data_not_exceptions(svc):
    mcp = build_mcp_server(service=svc)
    assert _call(mcp, "open_task", task_id="task_nope")["error"] == "not_found"


def test_mcp_route_explanation_is_bounded(tmp_path, svc, task):
    mcp = build_mcp_server(service=svc)
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    route = _call(mcp, "explain_route", subtask_id=subtask.id)
    assert route["selected_worker"] == "echo_worker"
    assert "rejected_count" in route and len(route["rejected"]) <= 3


def test_mcp_recall_is_bounded_and_labeled(svc, task):
    mcp = build_mcp_server(service=svc)
    pack = _call(mcp, "recall_memory", query="refresh token", task_id=task.id)
    assert [i["status"] for i in pack["items"]] == ["promoted"]
    assert "not ledger truth" in pack["note"]

    captured = _call(mcp, "capture_memory_candidate", text="qwen is weak", task_id=task.id)
    assert captured["status"] == "pending"

    context = _call(mcp, "get_task_context_pack", task_id=task.id)
    assert context["memory_is_external_context"] is True
    assert "handoff" in context and "memory" in context


# ----------------------------------------------------------------------- HTTP
def test_http_endpoints_drive_the_service(svc):
    client = TestClient(create_app(service=svc, linear_sync=None))

    assert client.get("/health").json()["execution_backend"] == "direct"

    task_id = client.post("/tasks", json={"title": "Refactor auth", "goal": "dedupe"}).json()["id"]
    assert client.post(f"/tasks/{task_id}/claim", json={"manager_id": "claude-code"}).status_code == 201
    assert client.post(f"/tasks/{task_id}/claim", json={"manager_id": "codex"}).status_code == 409

    decision = client.post(f"/tasks/{task_id}/decisions", json={
        "made_by": "claude-code", "decision": "Keep it in session.py", "locked": True})
    assert decision.status_code == 201 and decision.json()["locked"] is True

    subtask = client.post(f"/tasks/{task_id}/subtasks", json={
        "title": "find dupes", "instructions": "inspect auth"}).json()
    assert subtask["status"] == "succeeded"

    route = client.get(f"/subtasks/{subtask['id']}/route").json()
    assert route["selected_worker"] == "echo_worker"

    chunk = client.get(f"/artifacts/{subtask['result_artifact_id']}/chunk",
                       params={"offset": 0, "limit": 20}).json()
    assert chunk["content"] and chunk["eof"] is False

    handoff = client.get(f"/tasks/{task_id}/handoff").json()
    assert "Keep it in session.py" in handoff["text"]

    assert client.get("/tasks/task_missing").status_code == 404
    assert client.post(f"/tasks/{task_id}/release", json={"manager_id": "claude-code"}).status_code == 204


def test_http_review_and_inbox_flow(svc, task):
    client = TestClient(create_app(service=svc, linear_sync=None))
    artifact = client.post(f"/tasks/{task.id}/artifacts", json={
        "type": "patch", "content": "diff", "summary": "the patch"}).json()

    review = client.post(f"/tasks/{task.id}/reviews", json={
        "requested_by": "claude-code", "assigned_to": "codex",
        "artifact_refs": [artifact["id"]], "criteria": ["correctness"]}).json()

    inbox = client.get("/managers/codex/inbox").json()
    assert [i["ref_id"] for i in inbox] == [review["id"]]

    assert client.post(f"/reviews/{review['id']}/claim",
                       json={"manager_id": "claude-code"}).status_code == 403
    assert client.post(f"/reviews/{review['id']}/claim",
                       json={"manager_id": "codex"}).json()["status"] == "claimed"

    done = client.post(f"/reviews/{review['id']}/result", json={
        "completed_by": "codex", "summary": "looks fine", "content": "no issues"}).json()
    assert done["status"] == "completed" and done["result_artifact_id"]
    assert client.get("/managers/codex/inbox").json() == []


def test_http_memory_and_workflow_routes(svc, task):
    client = TestClient(create_app(service=svc, linear_sync=None))
    assert client.get("/memory/health").json()["backend"] == "static"

    pack = client.post("/memory/recall", json={"query": "refresh token"}).json()
    assert [i["status"] for i in pack["items"]] == ["promoted"]

    captured = client.post("/memory/capture", json={"text": "x", "task_id": task.id}).json()
    assert captured["status"] == "pending"
    assert client.post("/memory/feedback", json={"feedback": "useful"}).status_code == 202

    context = client.get(f"/tasks/{task.id}/context-pack").json()
    assert context["memory_is_external_context"] is True

    subtask = client.post(f"/tasks/{task.id}/subtasks", json={
        "title": "t", "instructions": "i"}).json()
    handle = svc.executions_for("subtask", subtask["id"])[0]
    workflow = client.get(f"/workflows/{handle.handle_id}").json()
    assert workflow["handle"]["entity_type"] == "subtask"
    assert client.get("/workflows/nope").status_code == 404


def test_http_approval_gate(tmp_path):
    cloud = RawModelWorker("cloud", lambda p: "out", model="gpt", location=WorkerLocation.cloud,
                           privacy_tiers=[PrivacyTier.public], cost_per_1k_tokens_usd=0.5)
    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "a"), workers=[cloud],
        policy=RoutePolicy(max_cost_usd=10.0))
    client = TestClient(create_app(service=svc, linear_sync=None))
    task_id = client.post("/tasks", json={"title": "t"}).json()["id"]
    subtask = client.post(f"/tasks/{task_id}/subtasks", json={
        "title": "t", "instructions": "i", "privacy_tier": "public"}).json()
    assert subtask["status"] == "needs_approval"

    approved = client.post(f"/subtasks/{subtask['id']}/approve", json={
        "approved_by": "matthew", "max_cost_usd": 3.0}).json()
    assert approved["status"] == "succeeded"


# ------------------------------------------------------------------------ CLI
def test_cli_smoke_covers_the_documented_commands(svc, capsys):
    def run(*argv):
        assert cli_main(list(argv), service=svc) == 0
        return json.loads(capsys.readouterr().out)

    task = run("tasks", "create", "--title", "Refactor auth", "--goal", "dedupe",
               "--constraint", "No schema changes")
    task_id = task["id"]

    run("tasks", "claim", task_id, "--manager", "claude-code")
    run("decisions", "add", task_id, "--by", "claude-code",
        "--decision", "Keep it in session.py", "--locked")
    run("attempts", "add", task_id, "--actor", "claude-code", "--summary", "mapped the flow")

    subtask = run("subtasks", "submit", task_id, "--title", "find dupes",
                  "--instructions", "inspect auth", "--privacy", "repo_sensitive")
    assert subtask["status"] == "succeeded"
    assert run("subtasks", "route", subtask["id"])["selected_worker"] == "echo_worker"

    artifact_id = subtask["result_artifact_id"]
    chunk = run("artifacts", "read", artifact_id, "--limit", "30")
    assert chunk["next_offset"] == 30
    assert run("artifacts", "list", task_id)

    review = run("reviews", "request", task_id, "--to", "codex", "--by", "claude-code",
                 "--artifact", artifact_id, "--criteria", "correctness")
    assert [i["ref_id"] for i in run("inbox", "codex")] == [review["id"]]
    run("reviews", "claim", review["id"], "--manager", "codex")
    # --by defaults to the assignee, so the spec's terse form works verbatim.
    assert run("reviews", "complete", review["id"], "--content", "lgtm")["status"] == "completed"

    assert run("memory", "recall", "--query", "refresh token")["items"]
    assert run("memory", "capture", "--task", task_id, "--text", "x")["status"] == "pending"
    assert run("memory", "health")["backend"] == "static"
    assert run("tasks", "context-pack", task_id)["memory_is_external_context"] is True

    run("tasks", "release", task_id, "--manager", "claude-code")


def test_cli_handoff_prints_text_not_json(svc, task, capsys):
    svc.claim_task(task.id, "claude-code")
    assert cli_main(["tasks", "handoff", task.id], service=svc) == 0
    out = capsys.readouterr().out
    assert out.startswith("Task: Refactor auth")
    assert "Suggested next step:" in out


def test_cli_refusal_exits_two(svc, task, capsys):
    svc.claim_task(task.id, "claude-code")
    assert cli_main(["tasks", "claim", task.id, "--manager", "codex"], service=svc) == 2
    assert "conflict" in capsys.readouterr().err


def test_cli_artifact_read_all_pages_to_eof(svc, task, capsys):
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="t", instructions="x" * 300))
    cli_main(["artifacts", "read", subtask.result_artifact_id, "--all", "--limit", "17"],
             service=svc)
    body = capsys.readouterr().out
    assert "x" * 300 in body
