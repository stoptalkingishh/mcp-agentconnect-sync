"""The Level 4 compliance layer: launch, shell, workspace, token, audit, complete.

Covers compliance spec §19 (23 required tests) and §20 (14 acceptance criteria).

The rule under test throughout: *agents may think in their own harness, but
durable work must enter AgentConnect. If it is not recorded in AgentConnect, it
did not happen.*

`shell` is exercised against a real subprocess — the only honest way to prove an
environment variable is absent is to ask a child process to look for it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentconnect.core import (
    ActorType,
    AgentConnectService,
    ArtifactType,
    CreateArtifactRequest,
    CreateTaskRequest,
    EchoWorker,
    RecordAttemptRequest,
    RecordDecisionRequest,
    ReviewRequest,
    ReviewResultRequest,
    SessionMode,
    SessionStatus,
    StaticMemoryAdapter,
    SubtaskRequest,
    TaskStatus,
)
from agentconnect.core.errors import Conflict, PolicyViolation
from agentconnect.core.sessions import SECRET_DENYLIST, mint_token, sanitize_env
from agentconnect.core.workspace import DENIED_MCP_TOOLS, EXPOSED_MCP_TOOLS

CLI = [sys.executable, "-c",
       "import sys;sys.path[:0]=%r;from agentconnect.cli.main import main;sys.exit(main())"]


@pytest.fixture()
def svc(tmp_path):
    return AgentConnectService.create(
        db_path=str(tmp_path / "ledger.db"),
        artifact_dir=str(tmp_path / "artifacts"),
        workspace_dir=str(tmp_path / "workspaces"),
        workers=[EchoWorker()],
    )


@pytest.fixture()
def task(svc):
    return svc.create_task(CreateTaskRequest(
        title="Refactor auth expiry", goal="dedupe refresh-token expiry",
        constraints=["No schema changes"]))


def git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=path, capture_output=True, check=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (path / "auth.py").write_text("def validate():\n    pass\n")
    run("add", "-A")
    run("commit", "-qm", "init")
    return path


# ================================================================ 1. launch
def test_launch_verifies_the_task_exists(svc):
    from agentconnect.core.errors import NotFound

    with pytest.raises(NotFound):
        svc.launch_session("claude", task_id="task_missing")


def test_launch_needs_a_task_or_a_review(svc):
    from agentconnect.core.errors import InvalidRequest

    with pytest.raises(InvalidRequest, match="task_id or a review_id"):
        svc.launch_session("claude")


def test_launch_creates_a_workspace_with_the_documented_layout(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    base = Path(result["workspace"].path)

    assert base.name == task.id
    for entry in ("workspace.json", ".env.agentconnect", ".mcp.json", "AGENTCONNECT.md"):
        assert (base / entry).exists(), entry
    for directory in ("repo", "artifacts", "logs", "bin"):
        assert (base / directory).is_dir(), directory

    metadata = json.loads((base / "workspace.json").read_text())
    assert metadata["task_id"] == task.id
    assert metadata["session_id"] == result["session"].id
    assert metadata["manager_id"] == "claude"

    stored = svc.get_workspace(result["workspace"].id)
    assert stored.task_id == task.id and stored.destroyed_at is None


def test_launch_writes_env_file_with_the_session_block_and_0600(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    env_file = Path(result["workspace"].path) / ".env.agentconnect"

    assert oct(env_file.stat().st_mode)[-3:] == "600"  # it holds a token
    text = env_file.read_text()
    assert f"AGENTCONNECT_TASK_ID={task.id}" in text
    assert "AGENTCONNECT_MANAGER_ID=claude" in text
    assert "AGENTCONNECT_MODE=manager" in text
    assert "AGENTCONNECT_REVIEW_ID=" in text
    assert result["token"] in text
    assert result["token"].startswith("act_")


def test_launch_generates_agentconnect_md_with_the_compliance_rule(svc, task):
    svc.launch_session("claude", task_id=task.id)
    body = (Path(svc.workspace_for(task_id=task.id).path) / "AGENTCONNECT.md").read_text()

    assert "AgentConnect is the source of truth" in body
    assert "If it is not recorded in AgentConnect, it is not complete." in body
    for backend in ("Temporal", "WikiBrain", "Cognee", "Graphiti", "secrets manager"):
        assert backend in body


def test_launch_generates_claude_md_for_a_claude_manager(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    base = Path(result["workspace"].path)

    assert "CLAUDE.md" in result["files"]
    assert not (base / "CODEX.md").exists()
    body = (base / "CLAUDE.md").read_text()
    assert "MCP tools" in body
    assert "Do not rely on chat history as canonical task state." in body


def test_launch_generates_codex_md_for_a_codex_manager(svc, task):
    result = svc.launch_session("codex", task_id=task.id)
    base = Path(result["workspace"].path)

    assert "CODEX.md" in result["files"]
    assert not (base / "CLAUDE.md").exists()
    body = (base / "CODEX.md").read_text()
    assert "agentconnect tasks context-pack" in body
    assert "Do not produce final results only in chat" in body


def test_launch_generates_only_agentconnect_md_for_an_unknown_harness(svc, task):
    result = svc.launch_session("some-new-agent", task_id=task.id)
    assert result["files"][0] == "AGENTCONNECT.md"
    assert "CLAUDE.md" not in result["files"] and "CODEX.md" not in result["files"]


def test_launch_creates_a_manager_session(svc, task):
    result = svc.launch_session("claude", task_id=task.id, launch_command="launch claude")
    session = result["session"]

    assert session.status is SessionStatus.prepared
    assert session.mode is SessionMode.manager
    assert session.manager_id == "claude" and session.task_id == task.id
    assert svc.get_session(session.id).launch_command == "launch claude"
    assert "session_prepared" in [e.kind for e in svc.list_events(task.id)]
    assert svc.active_session_for(task_id=task.id).id == session.id


def test_launch_with_claim_claims_the_task(svc, task):
    result = svc.launch_session("claude", task_id=task.id, claim=True)
    assert result["claim_id"] is not None
    assert svc.get_task(task.id).task.current_manager == "claude"
    assert svc.get_task(task.id).task.status is TaskStatus.in_progress


def test_launch_with_claim_fails_when_the_task_is_already_claimed(svc, task, tmp_path):
    svc.claim_task(task.id, "codex")
    with pytest.raises(Conflict, match="codex"):
        svc.launch_session("claude", task_id=task.id, claim=True)

    # And nothing was left half-built on disk.
    assert svc.workspace_for(task_id=task.id) is None
    assert not (tmp_path / "workspaces" / task.id).exists()


def test_force_readonly_downgrades_instead_of_failing(svc, task):
    svc.claim_task(task.id, "codex")
    result = svc.launch_session("claude", task_id=task.id, claim=True, force_readonly=True)

    assert result["session"].mode is SessionMode.readonly
    assert result["claim_id"] is None
    assert svc.get_task(task.id).task.current_manager == "codex"  # still theirs
    assert "AGENTCONNECT_MODE=readonly" in (
        Path(result["workspace"].path) / ".env.agentconnect").read_text()


def test_a_review_launch_is_a_reviewer_session(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.report, content="findings", summary="scan", created_by="claude"))
    review = svc.request_review(task.id, ReviewRequest(
        requested_by="claude", assigned_to="codex", artifact_refs=[artifact.id]))

    result = svc.launch_session("codex", review_id=review.id, claim=True)
    session = result["session"]
    assert session.mode is SessionMode.reviewer
    assert session.review_id == review.id and session.task_id == task.id
    assert Path(result["workspace"].path).name == review.id
    assert f"--review {review.id}" in result["shell_command"]
    assert svc.get_review(review.id).status.value == "claimed"


def test_launch_materializes_a_git_worktree_on_its_own_branch(svc, task, tmp_path):
    source = git_repo(tmp_path / "src")
    result = svc.launch_session("claude", task_id=task.id, repo_source=str(source))
    workspace = result["workspace"]

    assert workspace.repo_mode.value == "git_worktree"
    repo = Path(workspace.repo_path)
    assert (repo / "auth.py").exists()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    assert branch == f"agentconnect/{task.id}/refactor-auth-expiry"
    # The source repo is untouched: the agent works on its own branch.
    assert subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=source,
                          capture_output=True, text=True).stdout.strip() == "main"


def test_launch_falls_back_to_an_empty_workspace_for_a_non_code_task(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    assert result["workspace"].repo_mode.value == "empty"
    assert Path(result["workspace"].repo_path).is_dir()


# ============================================================ 2. MCP config
def test_mcp_config_exposes_only_agentconnect_tools(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    config = json.loads((Path(result["workspace"].path) / ".mcp.json").read_text())

    # Exactly one server, and it is ours.
    assert list(config["mcpServers"]) == ["agentconnect"]
    assert config["mcpServers"]["agentconnect"]["command"] == "agentconnect-mcp"

    allowed = config["allowedTools"]
    assert len(allowed) == len(EXPOSED_MCP_TOOLS)
    assert all(t.startswith("mcp__agentconnect__") for t in allowed)
    for tool in ("get_task_context_pack", "record_decision", "submit_subtask"):
        assert f"mcp__agentconnect__{tool}" in allowed

    # No backend tool is reachable, and the denial is written down.
    for denied in DENIED_MCP_TOOLS:
        assert denied in config["deniedTools"]
        assert not any(denied in t for t in allowed)
    assert "promote_memory_candidate" not in json.dumps(config)

    # The server env carries the session, and nothing else.
    server_env = config["mcpServers"]["agentconnect"]["env"]
    assert server_env["AGENTCONNECT_TASK_ID"] == task.id
    assert all(k.startswith("AGENTCONNECT_") for k in server_env)


def test_the_mcp_surface_never_offers_promotion_or_approval(svc):
    """Acceptance 14, asserted against the real server rather than the config."""
    from agentconnect.mcp.server import build_mcp_server

    mcp = build_mcp_server(service=svc)
    import asyncio

    names = {t.name for t in asyncio.get_event_loop().run_until_complete(mcp.list_tools())}
    for forbidden in ("promote_memory_candidate", "grant_approval", "approve_subtask",
                      "deny_subtask", "complete_task", "launch_session"):
        assert forbidden not in names


def test_a_managed_agent_can_call_tools_without_typing_an_id(svc, task, monkeypatch):
    """Acceptance 9. `launch` puts the ids in the environment; the tools read them."""
    from agentconnect.mcp.server import build_mcp_server

    result = svc.launch_session("claude", task_id=task.id, claim=True)
    for key, value in result["env"].items():
        monkeypatch.setenv(key, value)

    mcp = build_mcp_server(service=svc)
    import asyncio

    def call(name, **kwargs):
        raw = asyncio.get_event_loop().run_until_complete(mcp.call_tool(name, kwargs))
        payload = raw[0] if isinstance(raw, tuple) else raw
        return json.loads(payload[0].text if isinstance(payload, list) else payload)

    pack = call("get_task_context_pack")  # no task_id
    assert pack["task_id"] == task.id

    assert call("record_attempt", summary="read the module")["attempt_id"]
    assert call("record_decision", decision="Keep it in session.py.", locked=True)["locked"]
    assert call("get_status")["task_id"] == task.id
    # The manager id was inferred too.
    assert svc.get_task(task.id).decisions[0].made_by == "claude"


def test_without_a_session_a_tool_says_how_to_get_one(svc, monkeypatch):
    from agentconnect.mcp.server import build_mcp_server
    import asyncio

    monkeypatch.delenv("AGENTCONNECT_TASK_ID", raising=False)
    mcp = build_mcp_server(service=svc)
    raw = asyncio.get_event_loop().run_until_complete(
        mcp.call_tool("get_task_context_pack", {}))
    payload = raw[0] if isinstance(raw, tuple) else raw
    text = payload[0].text if isinstance(payload, list) else payload
    assert "agentconnect launch" in text


# ========================================================== 3. session tokens
def test_a_token_is_scoped_to_its_mode_and_entity(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    scope = svc.authorize(result["token"], "record_decision")

    assert scope["mode"] == "manager"
    assert scope["task_id"] == task.id
    assert scope["session_id"] == result["session"].id
    assert "record_decision" in scope["actions"]


def test_a_manager_token_cannot_promote_memory_or_touch_temporal(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    for forbidden in ("promote_memory_candidate", "wikibrain_promote", "cognee_write",
                      "graphiti_write", "temporal_signal", "temporal_admin",
                      "secrets_read", "admin_settings", "grant_approval",
                      "approve_subtask", "local_model_generate"):
        with pytest.raises(PolicyViolation, match="not permitted"):
            svc.authorize(result["token"], forbidden)


def test_a_reviewer_token_cannot_decide_or_delegate(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.report, content="x", summary="s", created_by="claude"))
    review = svc.request_review(task.id, ReviewRequest(
        requested_by="claude", assigned_to="codex", artifact_refs=[artifact.id]))
    result = svc.launch_session("codex", review_id=review.id)

    assert svc.authorize(result["token"], "complete_review")["mode"] == "reviewer"
    for forbidden in ("record_decision", "submit_subtask", "request_review", "claim_task"):
        with pytest.raises(PolicyViolation):
            svc.authorize(result["token"], forbidden)


def test_a_readonly_token_can_look_but_not_touch(svc, task):
    result = svc.launch_session("claude", task_id=task.id, readonly=True)
    assert svc.authorize(result["token"], "get_task_context_pack")
    for forbidden in ("record_decision", "submit_subtask", "record_attempt", "claim_task"):
        with pytest.raises(PolicyViolation):
            svc.authorize(result["token"], forbidden)


def test_an_unknown_expired_or_revoked_token_is_refused(svc, task, monkeypatch):
    clock = {"now": 1000.0}
    svc._clock = lambda: clock["now"]

    with pytest.raises(PolicyViolation, match="unknown session token"):
        svc.authorize(mint_token(), "record_attempt")

    result = svc.launch_session("claude", task_id=task.id, token_ttl_seconds=60)
    assert svc.authorize(result["token"], "record_attempt")

    clock["now"] = 1061.0
    with pytest.raises(PolicyViolation, match="expired"):
        svc.authorize(result["token"], "record_attempt")

    clock["now"] = 1000.0
    svc.revoke_session_tokens(result["session"].id)
    with pytest.raises(PolicyViolation, match="revoked"):
        svc.authorize(result["token"], "record_attempt")


def test_only_the_hash_of_a_token_is_stored(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    rows = svc.storage._conn.execute("SELECT * FROM session_tokens").fetchall()
    dumped = " ".join(str(v) for row in rows for v in tuple(row))
    assert result["token"] not in dumped


def test_ending_a_shell_revokes_the_token(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    svc.start_shell(result["session"].id, "claude")
    svc.end_shell(result["session"].id, 0)
    with pytest.raises(PolicyViolation, match="revoked"):
        svc.authorize(result["token"], "record_attempt")


# ==================================================== 4. environment sanitizing
def test_sanitize_env_drops_every_denylisted_credential():
    dirty = {name: "secret" for name in SECRET_DENYLIST}
    dirty.update({"PATH": "/usr/bin", "HOME": "/home/a", "SOME_RANDOM_VAR": "x"})
    clean = sanitize_env(dirty, {"AGENTCONNECT_TASK_ID": "task_1"})

    assert not (SECRET_DENYLIST & set(clean))
    assert "SOME_RANDOM_VAR" not in clean  # allowlist: unknown means absent
    assert clean["PATH"] == "/usr/bin" and clean["HOME"] == "/home/a"
    assert clean["AGENTCONNECT_TASK_ID"] == "task_1"


def test_sanitize_env_refuses_to_readmit_a_credential_by_name():
    with pytest.raises(ValueError, match="looks like a credential"):
        sanitize_env({"MY_API_KEY": "x"}, {}, extra_allow=["MY_API_KEY"])
    with pytest.raises(ValueError):
        sanitize_env({"FOO_SECRET": "x"}, {}, extra_allow=["FOO_SECRET"])
    # A genuinely benign extra is allowed through.
    assert sanitize_env({"PYENV_ROOT": "/p"}, {}, extra_allow=["PYENV_ROOT"])["PYENV_ROOT"]


def test_sanitize_env_lets_the_session_win_over_the_ambient_environment():
    clean = sanitize_env({"AGENTCONNECT_TASK_ID": "task_wrong"},
                         {"AGENTCONNECT_TASK_ID": "task_right"})
    assert clean["AGENTCONNECT_TASK_ID"] == "task_right"


def test_sanitize_env_prepends_the_helper_bin_to_path():
    clean = sanitize_env({"PATH": "/usr/bin"}, {}, helper_bin="/ws/bin")
    assert clean["PATH"].split(os.pathsep)[0] == "/ws/bin"


def test_the_session_token_is_the_one_credential_that_survives():
    clean = sanitize_env(
        {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-real"},
        {"AGENTCONNECT_SESSION_TOKEN": "act_abc"},
    )
    assert clean["AGENTCONNECT_SESSION_TOKEN"] == "act_abc"
    assert "ANTHROPIC_API_KEY" not in clean


# =================================================================== 5. shell
def _cli(argv, cwd, env=None):
    src = [str(Path("packages") / p / "src") for p in (
        "agentconnect-cli", "agentconnect-core", "agentconnect-linear",
        "agentconnect-mcp", "agentconnect-temporal")]
    src = [str(Path(s).resolve()) for s in src]
    code = f"import sys;sys.path[:0]={src!r};from agentconnect.cli.main import main;sys.exit(main())"
    return subprocess.run([sys.executable, "-c", code, *argv], cwd=str(cwd), env=env,
                          capture_output=True, text=True)


@pytest.fixture()
def cli_env(tmp_path):
    env = dict(os.environ)
    env.update({
        "AGENTCONNECT_DB_PATH": str(tmp_path / "ledger.db"),
        "AGENTCONNECT_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "AGENTCONNECT_WORKSPACE_DIR": str(tmp_path / "workspaces"),
        "AGENTCONNECT_MEMORY_CONFIG": str(tmp_path / "no-memory.yaml"),
        # The credentials a real box would have lying around.
        "ANTHROPIC_API_KEY": "sk-ant-real", "OPENAI_API_KEY": "sk-oai-real",
        "LINEAR_API_KEY": "lin_real", "TEMPORAL_ADDRESS": "localhost:7233",
        "WIKIBRAIN_ADMIN_TOKEN": "wb-admin", "AWS_SECRET_ACCESS_KEY": "aws-real",
    })
    return env


def test_shell_sanitizes_secrets_and_runs_inside_the_workspace(tmp_path, cli_env):
    repo = str(Path(__file__).resolve())  # any path; we only need the CLI to work
    created = _cli(["tasks", "create", "--title", "Compliance run"], tmp_path, cli_env)
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    launched = _cli(["launch", "claude", "--task", task_id, "--claim"], tmp_path, cli_env)
    assert launched.returncode == 0, launched.stderr
    assert "Prepared AgentConnect session." in launched.stdout
    assert f"agentconnect shell --task {task_id} -- claude" in launched.stdout

    # The child reports its cwd and whether it can see any backend credential.
    probe = (
        "import os,json;"
        "print(json.dumps({"
        "'cwd': os.getcwd(),"
        "'leaked': sorted(k for k in os.environ if k in "
        "('ANTHROPIC_API_KEY','OPENAI_API_KEY','LINEAR_API_KEY','TEMPORAL_ADDRESS',"
        "'WIKIBRAIN_ADMIN_TOKEN','AWS_SECRET_ACCESS_KEY')),"
        "'task': os.environ.get('AGENTCONNECT_TASK_ID'),"
        "'manager': os.environ.get('AGENTCONNECT_MANAGER_ID'),"
        "'mode': os.environ.get('AGENTCONNECT_MODE'),"
        "'token': bool(os.environ.get('AGENTCONNECT_SESSION_TOKEN')),"
        "'path0': os.environ['PATH'].split(os.pathsep)[0],"
        "}))"
    )
    ran = _cli(["shell", "--task", task_id, "--", sys.executable, "-c", probe],
               tmp_path, cli_env)
    assert ran.returncode == 0, ran.stderr
    seen = json.loads(ran.stdout)

    assert seen["leaked"] == []                                     # acceptance 7
    assert seen["task"] == task_id and seen["manager"] == "claude"  # acceptance 6
    assert seen["mode"] == "manager"
    assert seen["token"] is True                                    # acceptance 8
    assert seen["cwd"].endswith(f"workspaces/{task_id}/repo")
    assert seen["path0"].endswith(f"workspaces/{task_id}/bin")


def test_shell_records_the_session_start_and_end(tmp_path, cli_env):
    created = _cli(["tasks", "create", "--title", "t"], tmp_path, cli_env)
    task_id = json.loads(created.stdout)["id"]
    _cli(["launch", "claude", "--task", task_id], tmp_path, cli_env)
    _cli(["shell", "--task", task_id, "--", sys.executable, "-c", "pass"], tmp_path, cli_env)

    sessions = json.loads(_cli(["sessions", "list", "--task", task_id], tmp_path,
                               cli_env).stdout)
    assert len(sessions) == 1
    assert sessions[0]["status"] == "ended"
    assert sessions[0]["ended_at"] is not None
    assert "-c" in sessions[0]["shell_command"]


def test_shell_propagates_the_agents_exit_code_and_marks_the_session_failed(
    tmp_path, cli_env
):
    created = _cli(["tasks", "create", "--title", "t"], tmp_path, cli_env)
    task_id = json.loads(created.stdout)["id"]
    _cli(["launch", "claude", "--task", task_id], tmp_path, cli_env)

    ran = _cli(["shell", "--task", task_id, "--", sys.executable, "-c",
                "import sys;sys.exit(3)"], tmp_path, cli_env)
    assert ran.returncode == 3

    sessions = json.loads(_cli(["sessions", "list", "--task", task_id], tmp_path,
                               cli_env).stdout)
    assert sessions[0]["status"] == "failed"


def test_shell_without_a_launch_says_to_launch(tmp_path, cli_env):
    created = _cli(["tasks", "create", "--title", "t"], tmp_path, cli_env)
    task_id = json.loads(created.stdout)["id"]
    ran = _cli(["shell", "--task", task_id, "--", "true"], tmp_path, cli_env)
    assert ran.returncode == 2
    assert "agentconnect launch" in ran.stderr


# =================================================================== 6. audit
def _work(svc, task, manager="claude"):
    """A well-behaved session: claim, attempt, decision, artifact."""
    result = svc.launch_session(manager, task_id=task.id, claim=True)
    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id=manager, actor_type=ActorType.manager, summary="did the work"))
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by=manager, decision="Keep validation in session.py.", locked=True))
    svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.patch, content="diff", summary="rewrote auth.py",
        created_by=manager, metadata={"files": ["auth.py"]}))
    return result


def test_audit_passes_when_attempts_artifacts_and_reviews_are_complete(svc, task):
    _work(svc, task)
    svc.get_handoff_summary(task.id)  # a manager that hands off keeps it fresh
    report = svc.audit_task(task.id)

    assert report.passed, report.problems
    assert "PASS" in report.render()
    names = {c.name for c in report.checks}
    assert {"task_claimed", "attempt_recorded", "changed_files_registered",
            "subtasks_resolved", "reviews_completed", "decisions_recorded",
            "handoff_fresh", "linear_sync_current", "status_consistent"} <= names


def test_audit_fails_when_no_attempt_was_recorded(svc, task):
    svc.launch_session("claude", task_id=task.id, claim=True)
    report = svc.audit_task(task.id)

    assert not report.passed
    assert "No record_attempt was made during this session." in report.problems
    assert "Task cannot be marked complete." in report.render()


def test_audit_fails_when_the_attempt_predates_the_session(svc, task):
    """A stale attempt from a previous session does not excuse this one."""
    clock = {"now": 1000.0}
    svc._clock = lambda: clock["now"]
    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="claude", actor_type=ActorType.manager, summary="last week"))

    clock["now"] = 9000.0
    svc.launch_session("claude", task_id=task.id, claim=True)
    assert "No record_attempt was made during this session." in svc.audit_task(task.id).problems


def test_audit_fails_when_changed_files_are_unregistered(svc, task, tmp_path):
    source = git_repo(tmp_path / "src")
    svc.launch_session("claude", task_id=task.id, claim=True, repo_source=str(source))
    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="claude", actor_type=ActorType.manager, summary="edited"))

    repo = Path(svc.workspace_for(task_id=task.id).repo_path)
    (repo / "auth.py").write_text("def validate():\n    return True\n")
    (repo / "secrets.env").write_text("KEY=1\n")

    report = svc.audit_task(task.id)
    assert not report.passed
    problem = next(p for p in report.problems if "not registered as artifacts" in p)
    assert "auth.py" in problem and "secrets.env" in problem
    assert "Files changed but no artifact was registered." in report.problems


def test_registering_the_changed_files_clears_the_audit(svc, task, tmp_path):
    source = git_repo(tmp_path / "src")
    svc.launch_session("claude", task_id=task.id, claim=True, repo_source=str(source))
    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="claude", actor_type=ActorType.manager, summary="edited"))
    svc.record_decision(task.id, RecordDecisionRequest(
        made_by="claude", decision="Validate eagerly.", locked=True))
    repo = Path(svc.workspace_for(task_id=task.id).repo_path)
    (repo / "auth.py").write_text("def validate():\n    return True\n")

    # Structurally, via metadata["files"] ...
    svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.patch, content="diff", summary="tightened validation",
        created_by="claude", metadata={"files": ["auth.py"]}))
    svc.get_handoff_summary(task.id)
    assert svc.audit_task(task.id).passed

    # ... or in prose, because telling the next manager is what actually matters.
    (repo / "tokens.py").write_text("x = 1\n")
    assert not svc.audit_task(task.id).passed
    svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.summary, content="notes", created_by="claude",
        summary="Also added tokens.py for the expiry helper."))
    svc.get_handoff_summary(task.id)
    assert svc.audit_task(task.id).passed


def test_audit_fails_when_a_required_review_is_incomplete(svc, task):
    _work(svc, task)
    artifact = svc.get_task(task.id).artifacts[0]
    review = svc.request_review(task.id, ReviewRequest(
        requested_by="claude", assigned_to="codex", artifact_refs=[artifact.id]))
    svc.get_handoff_summary(task.id)

    report = svc.audit_task(task.id)
    assert not report.passed
    assert any(review.id in p and "no completed review exists" in p for p in report.problems)

    svc.claim_review(review.id, "codex")
    svc.complete_review(review.id, ReviewResultRequest(
        completed_by="codex", summary="looks right"))
    svc.get_handoff_summary(task.id)
    assert svc.audit_task(task.id).passed


def test_audit_fails_when_a_subtask_is_still_open(svc, task):
    from agentconnect.core import ids
    from agentconnect.core.models import Subtask

    _work(svc, task)
    svc.storage.insert_subtask(Subtask(
        id=ids.new_id(ids.SUBTASK), parent_task_id=task.id, title="t", instructions="i"))
    svc.get_handoff_summary(task.id)

    report = svc.audit_task(task.id)
    assert not report.passed
    assert any("subtasks are still open" in p for p in report.problems)


def test_audit_fails_when_durable_work_has_no_recorded_decision(svc, task):
    svc.launch_session("claude", task_id=task.id, claim=True)
    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="claude", actor_type=ActorType.manager, summary="did it"))
    svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.patch, content="diff", summary="rewrote things",
        created_by="claude"))

    report = svc.audit_task(task.id)
    assert "Durable changes were made but no decision was recorded." in report.problems


def test_audit_fails_when_the_handoff_is_stale(svc, task):
    _work(svc, task)
    report = svc.audit_task(task.id)
    assert "Handoff summary is stale; regenerate it before handing off." in report.problems

    # The audit must not have repaired what it reported: it is read-only, so a
    # second run says exactly the same thing.
    assert svc.audit_task(task.id).problems == report.problems

    svc.get_handoff_summary(task.id)  # what a departing manager does
    assert svc.audit_task(task.id).passed


def test_complete_regenerates_the_handoff_before_auditing(svc, task):
    """A manager who did the work but never re-read the handoff is not punished;
    completing the task is what makes the handoff current."""
    _work(svc, task)
    assert not svc.audit_task(task.id).passed  # stale handoff

    result = svc.complete_task(task.id, "claude")
    assert result["status"] == "succeeded"
    assert svc.get_task(task.id).task.handoff_summary


def test_audit_fails_when_linear_says_done_and_the_ledger_does_not(svc, task):
    _work(svc, task)
    svc.set_external_ref("task", task.id, "linear", "issue_1", "https://linear.app/i/1")
    svc.get_handoff_summary(task.id)  # the Linear url is part of the handoff
    assert svc.audit_task(task.id).passed

    svc.record_event(task.id, "linear_status_change", "human", {"state": "Done"})
    report = svc.audit_task(task.id)
    assert not report.passed
    assert any("Linear issue is marked Done" in p and "in_progress" in p
               for p in report.problems)


def test_missing_memory_capture_is_a_warning_not_a_failure(svc, task, tmp_path):
    svc.bind_memory(StaticMemoryAdapter([]))
    _work(svc, task)
    svc.get_handoff_summary(task.id)

    report = svc.audit_task(task.id)
    assert report.passed  # advisory checks never block
    assert any("reusable lessons may be lost" in w for w in report.warnings)
    assert "Warnings:" in report.render()


def test_audit_of_a_task_never_worked_through_agentconnect_names_every_gap(svc, task):
    report = svc.audit_task(task.id)
    assert not report.passed
    assert "No AgentConnect workspace: the task was worked outside a managed session." \
        in report.problems
    assert "No manager session was launched for this task." in report.problems
    assert "Task was never claimed; no manager is accountable for it." in report.problems


def test_audit_review_requires_a_claim_and_an_attempt(svc, task):
    artifact = svc.create_artifact(task.id, CreateArtifactRequest(
        type=ArtifactType.report, content="x", summary="s", created_by="claude"))
    review = svc.request_review(task.id, ReviewRequest(
        requested_by="claude", assigned_to="codex", artifact_refs=[artifact.id]))

    assert not svc.audit_review(review.id).passed  # never launched

    svc.launch_session("codex", review_id=review.id, claim=True)
    report = svc.audit_review(review.id)
    assert "No record_attempt was made during this session." in report.problems

    svc.record_attempt(task.id, RecordAttemptRequest(
        actor_id="codex", actor_type=ActorType.manager, summary="read the artifact"))
    assert svc.audit_review(review.id).passed


# ================================================================ 7. complete
def test_complete_refuses_a_task_whose_audit_fails(svc, task):
    svc.launch_session("claude", task_id=task.id, claim=True)
    with pytest.raises(PolicyViolation) as exc:
        svc.complete_task(task.id, "claude")

    assert "audit failed" in str(exc.value)
    assert "No record_attempt" in str(exc.value)
    assert svc.get_task(task.id).task.status is not TaskStatus.succeeded
    assert "completion_refused" in [e.kind for e in svc.list_events(task.id)]


def test_complete_marks_the_ledger_succeeded_then_updates_linear(svc, task):
    order: list[str] = []
    original = svc._touch

    def spy_touch(task_id, **fields):
        if fields.get("status") == "succeeded":
            order.append("ledger")
        original(task_id, **fields)

    svc._touch = spy_touch

    def linear_hook(task_id: str) -> None:
        # Acceptance 13: by the time the tracker hears, the ledger already says so.
        assert svc.get_task(task_id).task.status is TaskStatus.succeeded
        order.append("linear")

    linear_hook.__name__ = "linear_post_completion"
    svc.bind_completion_hook(linear_hook)

    _work(svc, task)
    svc.get_handoff_summary(task.id)
    result = svc.complete_task(task.id, "claude")

    assert order == ["ledger", "linear"]
    assert result["status"] == "succeeded" and result["audit"]["status"] == "PASS"
    assert result["mirrored"] == ["linear_post_completion"]
    assert svc.get_task(task.id).task.status is TaskStatus.succeeded


def test_a_linear_outage_does_not_undo_a_completion(svc, task):
    def broken(task_id: str) -> None:
        raise ConnectionError("linear is down")

    svc.bind_completion_hook(broken)
    _work(svc, task)
    svc.get_handoff_summary(task.id)
    result = svc.complete_task(task.id, "claude")

    assert result["status"] == "succeeded" and result["mirrored"] == []
    assert svc.get_task(task.id).task.status is TaskStatus.succeeded


def test_force_completes_but_records_the_problems(svc, task):
    svc.launch_session("claude", task_id=task.id, claim=True)
    result = svc.complete_task(task.id, "matthew", force=True)

    assert result["forced"] is True
    assert svc.get_task(task.id).task.status is TaskStatus.succeeded
    event = next(e for e in svc.list_events(task.id) if e.kind == "task_completed")
    assert event.payload["forced"] is True
    assert event.payload["problems"]  # the audit's objections are on the record


def test_a_task_cannot_be_completed_twice(svc, task):
    _work(svc, task)
    svc.get_handoff_summary(task.id)
    svc.complete_task(task.id, "claude")
    with pytest.raises(Conflict, match="already succeeded"):
        svc.complete_task(task.id, "claude")


def test_completion_is_not_reachable_from_a_session_token(svc, task):
    result = svc.launch_session("claude", task_id=task.id, claim=True)
    with pytest.raises(PolicyViolation):
        svc.authorize(result["token"], "complete_task")


# =========================================== 8. Linear asks, AgentConnect decides
def test_linear_complete_command_runs_the_audit_first(svc, task):
    from agentconnect.linear.webhooks import handle_webhook

    svc.set_external_ref("task", task.id, "linear", "issue_1")
    svc.launch_session("claude", task_id=task.id, claim=True)

    payload = {"type": "Comment", "action": "create",
               "data": {"body": "/agentconnect complete", "issue": {"id": "issue_1"},
                        "user": {"name": "matthew"}}}
    refused = handle_webhook(svc, payload)[0]
    assert refused["kind"] == "completion_refused"
    assert any("record_attempt" in p for p in refused["problems"])
    assert svc.get_task(task.id).task.status is not TaskStatus.succeeded

    _work(svc, task)
    svc.get_handoff_summary(task.id)
    accepted = handle_webhook(svc, payload)[0]
    assert accepted["kind"] == "completed" and accepted["audit"] == "PASS"
    assert svc.get_task(task.id).task.status is TaskStatus.succeeded


def test_linear_status_command_reports_the_audit(svc, task):
    from agentconnect.linear.webhooks import handle_webhook

    svc.set_external_ref("task", task.id, "linear", "issue_1")
    result = handle_webhook(svc, {
        "type": "Comment", "action": "create",
        "data": {"body": "/agentconnect status", "issue": {"id": "issue_1"}}})[0]
    assert result["kind"] == "status" and result["audit"] == "FAIL"
    assert result["problems"]


def test_linear_request_review_command_creates_a_review(svc, task):
    from agentconnect.linear.webhooks import handle_webhook

    svc.set_external_ref("task", task.id, "linear", "issue_1")
    result = handle_webhook(svc, {
        "type": "Comment", "action": "create",
        "data": {"body": "/agentconnect request-review codex",
                 "issue": {"id": "issue_1"}, "user": {"name": "matthew"}}})[0]

    assert result["kind"] == "review_requested" and result["assigned_to"] == "codex"
    assert svc.get_review(result["review_id"]).assigned_to == "codex"


def test_moving_the_linear_issue_never_completes_the_task(svc, task):
    from agentconnect.linear.webhooks import handle_webhook

    svc.set_external_ref("task", task.id, "linear", "issue_1")
    results = handle_webhook(svc, {
        "type": "Issue", "action": "update",
        "data": {"id": "issue_1", "state": {"name": "Done"}},
        "updatedFrom": {"stateId": "x"}})

    assert results[0]["kind"] == "status_recorded"
    assert svc.get_task(task.id).task.status is not TaskStatus.succeeded


def test_the_linear_issue_body_says_agentconnect_is_canonical(svc, task):
    from agentconnect.linear import mapping

    body = mapping.issue_body(svc.get_task(task.id), "handoff text")
    assert "**Canonical status:** AgentConnect-managed" in body
    assert "moving this issue does not change task state" in body


# ============================================== 9. sessions, workspaces, cleanup
def test_sessions_and_workspaces_are_listable(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    assert [s.id for s in svc.list_sessions(task_id=task.id)] == [result["session"].id]
    assert [s.id for s in svc.list_sessions(manager_id="claude")] == [result["session"].id]
    assert [w.id for w in svc.list_workspaces()] == [result["workspace"].id]


def test_cleanup_marks_a_workspace_destroyed_and_hides_it(svc, task):
    result = svc.launch_session("claude", task_id=task.id)
    svc.cleanup_workspace(result["workspace"].id, actor="matthew")

    assert svc.list_workspaces() == []
    assert len(svc.list_workspaces(include_destroyed=True)) == 1
    assert svc.workspace_for(task_id=task.id) is None
    assert svc.get_workspace(result["workspace"].id).destroyed_at is not None


def test_a_stale_session_is_abandoned_and_its_token_revoked(svc, task):
    clock = {"now": 1000.0}
    svc._clock = lambda: clock["now"]
    result = svc.launch_session("claude", task_id=task.id)

    clock["now"] = 1000.0 + 48 * 3600
    assert svc.abandon_stale_sessions(older_than_seconds=24 * 3600) == [result["session"].id]
    assert svc.get_session(result["session"].id).status is SessionStatus.abandoned
    with pytest.raises(PolicyViolation, match="revoked"):
        svc.authorize(result["token"], "record_attempt")
    assert svc.active_session_for(task_id=task.id) is None


def test_relaunching_reuses_the_workspace_directory(svc, task):
    first = svc.launch_session("claude", task_id=task.id)
    (Path(first["workspace"].path) / "repo" / "note.txt").write_text("work in progress")
    svc.end_shell(first["session"].id, 0)

    second = svc.launch_session("claude", task_id=task.id)
    assert second["workspace"].path == first["workspace"].path
    assert (Path(second["workspace"].path) / "repo" / "note.txt").read_text() == \
        "work in progress"
    # A fresh session, and a fresh token: the old one died with the old shell.
    assert second["session"].id != first["session"].id
    assert second["token"] != first["token"]


# ================================================================ 10. HTTP API
@pytest.fixture()
def client(svc):
    from fastapi.testclient import TestClient

    from agentconnect.api.app import create_app

    return TestClient(create_app(service=svc, linear_sync=None))


def test_api_launch_audit_and_complete(client, svc, task):
    launched = client.post("/sessions/launch", json={
        "manager_id": "claude", "task_id": task.id, "claim": True})
    assert launched.status_code == 201
    body = launched.json()
    assert body["token"].startswith("act_")
    assert "AGENTCONNECT.md" in body["files"] and "CLAUDE.md" in body["files"]

    audit = client.get(f"/tasks/{task.id}/audit").json()
    assert audit["status"] == "FAIL"
    assert any("record_attempt" in p for p in audit["problems"])
    # The audit is read-only: asking twice gives the same answer.
    assert client.get(f"/tasks/{task.id}/audit").json()["problems"] == audit["problems"]

    refused = client.post(f"/tasks/{task.id}/complete", json={"completed_by": "matthew"})
    assert refused.status_code == 403
    assert refused.json()["error"] == "policy_violation"

    _work(svc, task)
    completed = client.post(f"/tasks/{task.id}/complete", json={"completed_by": "matthew"})
    assert completed.status_code == 200
    assert completed.json()["status"] == "succeeded"
    assert completed.json()["audit"]["status"] == "PASS"


def test_api_lists_sessions_and_workspaces(client, task):
    client.post("/sessions/launch", json={"manager_id": "claude", "task_id": task.id})
    sessions = client.get("/sessions", params={"task_id": task.id}).json()
    assert len(sessions) == 1 and sessions[0]["mode"] == "manager"

    assert client.get(f"/sessions/{sessions[0]['id']}").json()["manager_id"] == "claude"
    workspaces = client.get("/workspaces").json()
    assert len(workspaces) == 1 and workspaces[0]["task_id"] == task.id

    ended = client.post(f"/sessions/{sessions[0]['id']}/end").json()
    assert ended["status"] == "ended"


def test_api_completing_twice_is_a_conflict(client, svc, task):
    client.post("/sessions/launch", json={"manager_id": "claude", "task_id": task.id,
                                          "claim": True})
    _work(svc, task)
    assert client.post(f"/tasks/{task.id}/complete", json={}).status_code == 200
    again = client.post(f"/tasks/{task.id}/complete", json={})
    assert again.status_code == 409 and again.json()["error"] == "conflict"
