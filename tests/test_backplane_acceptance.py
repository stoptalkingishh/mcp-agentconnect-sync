"""End-to-end acceptance (BACKPLANE_SPEC §25 and BACKPLANE_SPEC_TEMPORAL §Acceptance).

One task walks the whole system: CLI creates it, Linear mirrors it, MCP drives it,
a worker produces an artifact, a second manager reviews it, and a *third* process
picks the task up from nothing but the handoff summary.

The last assertion is the point of the entire project: after the manager switch,
Codex knows what Claude decided and why, without a line of Claude's chat history.
"""

import json

import pytest

from agentconnect.core import (
    AgentConnectService,
    CreateTaskRequest,
    EchoWorker,
    MemoryItem,
    StaticMemoryAdapter,
)

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from agentconnect.api.app import create_app  # noqa: E402
from agentconnect.cli.main import main as cli_main  # noqa: E402
from agentconnect.linear import LinearClient, LinearSync  # noqa: E402
from agentconnect.mcp.server import build_mcp_server  # noqa: E402

from test_backplane_linear import FakeTransport  # noqa: E402


def _mcp(mcp, name, **kwargs):
    return json.loads(mcp._tool_manager.get_tool(name).fn(**kwargs))


def test_full_manager_switch_flow(tmp_path, capsys):
    # A single shared ledger. In production these are separate processes pointed
    # at one AGENTCONNECT_DB_PATH; here they are one service object, which is the
    # same thing minus the sqlite file.
    svc = AgentConnectService.create(
        db_path=str(tmp_path / "ledger.db"), artifact_dir=str(tmp_path / "artifacts"),
        workers=[EchoWorker()],
        memory=StaticMemoryAdapter([MemoryItem(
            text="Middleware has historically assumed auth/session.py owns validation.",
            status="promoted", confidence="high", source_id="wiki_412")]),
    )
    transport = FakeTransport()
    sync = LinearSync(svc, LinearClient(transport=transport), team_id="team-1")
    api = TestClient(create_app(service=svc, linear_sync=sync))
    mcp = build_mcp_server(service=svc)

    def cli(*argv):
        assert cli_main(list(argv), service=svc) == 0
        return json.loads(capsys.readouterr().out)

    # 1-2. API is up; create the task through the CLI.
    assert api.get("/health").json()["status"] == "ok"
    task_id = cli("tasks", "create", "--title", "Refactor auth session handling",
                  "--goal", "Reduce duplicate token expiry logic without changing behavior",
                  "--constraint", "No schema changes")["id"]

    # 3. Sync it to Linear as an issue.
    ref = api.post("/linear/sync", json={"task_id": task_id}).json()
    assert ref["external_id"] == "lin-1"
    assert "Reduce duplicate token expiry logic" in transport.issues["lin-1"]["description"]

    # 4-5. Claude Code claims the task through MCP.
    assert _mcp(mcp, "claim_task", task_id=task_id,
                manager_id="claude-code")["role"] == "primary_manager"

    # 6. Claude records one locked decision.
    decision = _mcp(mcp, "record_decision", task_id=task_id, made_by="claude-code",
                    decision="Keep refresh token validation in auth/session.py.",
                    rationale="Middleware assumes this location.", locked=True)
    assert decision["locked"]
    sync.post_decision(task_id, decision["decision_id"])

    # 7-9. Claude submits a readonly subtask; the echo worker produces an artifact.
    subtask = _mcp(mcp, "submit_subtask", task_id=task_id,
                   title="Find duplicated expiry checks",
                   instructions="Inspect auth files and return file paths and line ranges "
                                "only. Do not edit files.",
                   privacy_tier="repo_sensitive", filesystem="none")
    assert subtask["status"] == "succeeded"
    assert subtask["workflow_id"]  # a durable handle exists either way
    worker_artifact = subtask["result_artifact_id"]

    # 10. The artifact reads back in chunks through MCP and the CLI.
    first = _mcp(mcp, "read_artifact_chunk", artifact_id=worker_artifact, offset=0, limit=40)
    assert first["eof"] is False and first["next_offset"] == 40
    rest = _mcp(mcp, "read_artifact_chunk", artifact_id=worker_artifact,
                offset=first["next_offset"], limit=4000)
    assert rest["eof"] and "Do not edit files." in (first["content"] + rest["content"])
    assert cli("artifacts", "read", worker_artifact, "--limit", "40")["next_offset"] == 40

    # 11. Linear gets a compact update with the artifact link — and no body.
    sync.post_subtask(subtask["subtask_id"])
    sync.post_artifact(task_id, worker_artifact)
    assert worker_artifact in transport.comments[-1]
    assert "Do not edit files." not in transport.comments[-1]

    # 12. Claude requests a Codex review.
    review = _mcp(mcp, "request_review", task_id=task_id, requested_by="claude-code",
                  assigned_to="codex", artifact_refs=[worker_artifact],
                  criteria=["Find correctness issues", "Ignore style-only changes"])
    review_id = review["review_id"]
    sync.post_review_request(review_id)
    assert "Assigned to: **codex**" in transport.comments[-1]

    # 13. The review appears in Codex's inbox through both the API and the CLI.
    assert [i["ref_id"] for i in api.get("/managers/codex/inbox").json()] == [review_id]
    assert [i["ref_id"] for i in cli("inbox", "codex")] == [review_id]

    # 14-15. Codex claims and completes it with a review artifact.
    cli("reviews", "claim", review_id, "--manager", "codex")
    completed = cli("reviews", "complete", review_id, "--summary", "Two real duplicates",
                    "--content", "auth/session.py:41 and auth/tokens.py:87 duplicate the check")
    assert completed["status"] == "completed"
    review_artifact = completed["result_artifact_id"]

    # 16. Linear receives the compact review result.
    sync.post_review_result(review_id)
    assert review_artifact in transport.comments[-1]

    # 17. The handoff carries state, locked decisions, artifacts, and the review.
    handoff = _mcp(mcp, "get_handoff_summary", task_id=task_id, manager_id="claude-code")
    text = handoff["summary"]
    assert "Keep refresh token validation in auth/session.py." in text
    assert "No schema changes" in text
    assert worker_artifact in text
    assert f"{review_id} from codex" in text
    assert handoff["suggested_next_step"].startswith("Read review result")

    # 18. Claude releases the task.
    _mcp(mcp, "release_task", task_id=task_id, manager_id="claude-code")
    assert svc.get_task(task_id).task.current_manager is None

    # 19-20. Codex claims it and continues from the same state — with no access to
    # anything Claude "thought", only what Claude recorded.
    cli("tasks", "claim", task_id, "--manager", "codex")
    resumed = _mcp(mcp, "get_handoff_summary", task_id=task_id, manager_id="codex")
    assert resumed["current_manager"] == "codex"
    assert resumed["viewer_holds_claim"] is True
    assert "Keep refresh token validation in auth/session.py." in resumed["summary"]
    assert "Middleware assumes this location." in resumed["summary"]

    # And the context pack hands Codex external memory, clearly separated from truth.
    context = _mcp(mcp, "get_task_context_pack", task_id=task_id, manager_id="codex")
    assert context["memory_is_external_context"] is True
    assert context["memory"]["items"] == [] or all(
        i["status"] == "promoted" for i in context["memory"]["items"]
    )

    # The whole run mirrored to exactly one Linear issue.
    creates = [q for q, _ in transport.calls if "IssueCreate" in q]
    assert len(creates) == 1


def test_a_second_manager_never_sees_the_first_managers_reasoning(tmp_path):
    """The durable boundary (§27): handoff summaries are built only from recorded
    facts. Nothing a manager merely 'thought' can leak into one."""
    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "a"), workers=[EchoWorker()])
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    svc.claim_task(task.id, "claude-code")
    summary = svc.get_handoff_summary(task.id, "codex")

    for field in (summary.locked_decisions, summary.recent_attempts,
                  summary.important_artifacts):
        assert field == []
    assert summary.viewer_holds_claim is False
    assert summary.current_manager == "claude-code"
