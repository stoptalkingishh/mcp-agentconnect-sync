"""`agentconnect` — the CLI adapter (spec §12).

Runs in local mode: it builds the service directly against
`AGENTCONNECT_DB_PATH`. That is deliberate — a human debugging the ledger, and a
Codex-style manager driving it from a shell, should not need an HTTP server
running. Point it at the same DB the API and MCP server use and all three see
one ledger.

Output is JSON on stdout so it composes with `jq`; `tasks handoff` prints the
rendered summary instead, because that text *is* the deliverable.
Errors go to stderr; exit code 2 means the backplane refused, 1 means it broke.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from agentconnect.core import sessions as sessions_mod
from agentconnect.core.bootstrap import service_from_env
from agentconnect.core.context import PROFILES
from agentconnect.core.errors import AgentConnectError
from agentconnect.core.models import (
    ActorType,
    ArtifactType,
    ClaimRole,
    CreateArtifactRequest,
    CreateTaskRequest,
    FilesystemAccess,
    Priority,
    PrivacyTier,
    RecordAttemptRequest,
    RecordDecisionRequest,
    RepoMode,
    ReviewRequest,
    ReviewResultRequest,
    ReviewStatus,
    SandboxSpec,
    SubtaskRequest,
    TaskFilters,
    TaskStatus,
)
from agentconnect.core.service import DEFAULT_CLAIM_TTL_SECONDS, AgentConnectService

EXIT_REFUSED = 2
EXIT_ERROR = 1


def _emit(obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    elif isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
        obj = [o.model_dump(mode="json") for o in obj]
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _read_body(file: Optional[str], content: Optional[str]) -> str:
    if file:
        return Path(file).read_text(encoding="utf-8")
    return content or ""


# --------------------------------------------------------------------- tasks
def _cmd_tasks_create(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.create_task(CreateTaskRequest(
        title=a.title, goal=a.goal or "", priority=Priority(a.priority),
        created_by=a.by, constraints=a.constraint or [],
    )))


def _cmd_tasks_list(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.list_tasks(TaskFilters(
        status=TaskStatus(a.status) if a.status else None,
        current_manager=a.manager, limit=a.limit,
    )))


def _cmd_tasks_show(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.get_task(a.task_id))


def _cmd_tasks_handoff(svc: AgentConnectService, a: argparse.Namespace) -> None:
    summary = svc.get_handoff_summary(a.task_id, a.manager)
    if a.json:
        _emit(summary)
    else:
        print(summary.text)


def _cmd_tasks_claim(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.claim_task(a.task_id, a.manager, a.role, a.ttl))


def _cmd_tasks_release(svc: AgentConnectService, a: argparse.Namespace) -> None:
    svc.release_task(a.task_id, a.manager)
    _emit({"released": a.task_id, "manager": a.manager})


# ----------------------------------------------------- decisions / attempts
def _cmd_decisions_add(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.record_decision(a.task_id, RecordDecisionRequest(
        made_by=a.by, decision=a.decision, rationale=a.rationale or "",
        locked=a.locked, supersedes=a.supersedes or [],
    )))


def _cmd_attempts_add(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.record_attempt(a.task_id, RecordAttemptRequest(
        actor_id=a.actor, actor_type=ActorType(a.actor_type), summary=a.summary,
        outcome=a.outcome or "", artifact_refs=a.artifact or [],
    )))


# ----------------------------------------------------------------- artifacts
def _cmd_artifacts_add(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.create_artifact(a.task_id, CreateArtifactRequest(
        type=ArtifactType(a.type), content=_read_body(a.file, a.content),
        summary=a.summary or "", created_by=a.by,
    )))


def _cmd_artifacts_list(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.list_artifacts(a.task_id))


def _cmd_artifacts_read(svc: AgentConnectService, a: argparse.Namespace) -> None:
    if not a.all:
        _emit(svc.read_artifact_chunk(a.artifact_id, a.offset, a.limit))
        return
    offset: Optional[int] = a.offset
    while offset is not None:
        chunk = svc.read_artifact_chunk(a.artifact_id, offset, a.limit)
        sys.stdout.write(chunk.content)
        offset = chunk.next_offset
    sys.stdout.write("\n")


# ------------------------------------------------------------------- reviews
def _cmd_reviews_request(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.request_review(a.task_id, ReviewRequest(
        requested_by=a.by, assigned_to=a.to,
        criteria=a.criteria or [], artifact_refs=a.artifact or [],
    )))


def _cmd_reviews_claim(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.claim_review(a.review_id, a.manager))


def _cmd_reviews_complete(svc: AgentConnectService, a: argparse.Namespace) -> None:
    # Default the author to the assignee so the spec's terse form works verbatim.
    completed_by = a.by or svc.get_review(a.review_id).assigned_to
    _emit(svc.complete_review(a.review_id, ReviewResultRequest(
        completed_by=completed_by, status=ReviewStatus(a.status),
        summary=a.summary or "", content=_read_body(a.file, a.content),
    )))


def _cmd_inbox(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.get_manager_inbox(a.manager_id))


# ------------------------------------------------------------------ subtasks
def _cmd_subtasks_submit(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.submit_subtask(a.task_id, SubtaskRequest(
        title=a.title, instructions=a.instructions,
        privacy_tier=PrivacyTier(a.privacy), preferred_worker=a.worker,
        sandbox=SandboxSpec(
            filesystem=FilesystemAccess(a.filesystem), network=a.network, shell=a.shell
        ),
        required_capabilities=a.capability or [],
    )))


def _cmd_subtasks_show(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.get_subtask(a.subtask_id))


def _cmd_subtasks_route(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.explain_route(a.subtask_id))


def _cmd_subtasks_cancel(svc: AgentConnectService, a: argparse.Namespace) -> None:
    svc.cancel_subtask(a.subtask_id)
    _emit({"cancelled": a.subtask_id})


def _cmd_subtasks_approve(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.approve_subtask(a.subtask_id, a.by, a.max_cost))


def _cmd_subtasks_deny(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.deny_subtask(a.subtask_id, a.by, a.reason or ""))


# -------------------------------------------------------------------- linear
def _linear_sync(svc: AgentConnectService):
    import os

    from agentconnect.linear import LinearClient, LinearSync

    team_id = os.environ.get("LINEAR_TEAM_ID")
    if not team_id:
        raise AgentConnectError("LINEAR_TEAM_ID is not set")
    return LinearSync(svc, LinearClient(), team_id,
                      artifact_base_url=os.environ.get("AGENTCONNECT_BASE_URL"))


def _cmd_linear_sync(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(_linear_sync(svc).sync_task(a.task_id))


def _cmd_linear_issue(svc: AgentConnectService, a: argparse.Namespace) -> None:
    ref = svc.get_external_ref("task", a.task_id, "linear")
    if ref is None:
        raise AgentConnectError(f"task {a.task_id} is not synced to Linear")
    _emit(ref)


# -------------------------------------------------------------------- memory
def _cmd_memory_recall(svc: AgentConnectService, a: argparse.Namespace) -> None:
    from agentconnect.core.memory import MemoryScope, RecallRequest

    pack = svc.recall_memory(RecallRequest(
        query=a.query, task_id=a.task, profile=a.profile, max_items=a.max_items,
        trusted_only=not a.include_pending, include_pending=a.include_pending,
        scopes=[MemoryScope("task", a.task)] if a.task else [],
    ))
    _emit({
        "backend": pack.backend, "profile": pack.profile, "warnings": pack.warnings,
        "items": [
            {"text": i.text, "status": i.status, "confidence": i.confidence,
             "source_id": i.source_id}
            for i in pack.items
        ],
        "memory_is_external_context": True,
    })


def _cmd_memory_capture(svc: AgentConnectService, a: argparse.Namespace) -> None:
    from agentconnect.core.memory import CaptureRequest

    result = svc.capture_memory_candidate(CaptureRequest(
        text=_read_body(a.file, a.text), task_id=a.task, origin_actor_id=a.by,
        origin_actor_type=a.actor_type, tags=a.tag or [],
    ))
    _emit({
        "accepted": result.accepted, "candidate_id": result.candidate_id,
        "status": result.status, "message": result.message, "backend": result.backend,
    })


def _cmd_memory_feedback(svc: AgentConnectService, a: argparse.Namespace) -> None:
    from agentconnect.core.memory import MemoryFeedbackRequest

    svc.record_memory_feedback(MemoryFeedbackRequest(
        task_id=a.task, memory_item_id=a.item, source_id=a.source, feedback=a.feedback,
        actor_id=a.by, note=a.note,
    ))
    _emit({"recorded": True})


def _cmd_memory_health(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.memory_health())


def _cmd_memory_pending(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit({"candidates": svc.list_pending_memory(a.limit)})


def _cmd_memory_promote(svc: AgentConnectService, a: argparse.Namespace) -> None:
    """Human/librarian only. There is no MCP tool for this, on purpose."""
    _emit(svc.promote_memory_candidate(a.candidate_id, a.by))


def _cmd_tasks_context_pack(svc: AgentConnectService, a: argparse.Namespace) -> None:
    pack = svc.get_task_context_pack(
        a.task_id, profile=a.profile, max_memory_items=a.max_items, manager_id=a.manager
    )
    _emit({
        "task_id": pack.task_id, "profile": pack.profile,
        "handoff": pack.handoff.model_dump(mode="json") if pack.handoff else None,
        "backends_queried": pack.backends_queried,
        "memory": {
            "backend": pack.memory.backend, "warnings": pack.warnings,
            "items": [
                {"text": i.text, "status": i.status, "confidence": i.confidence,
                 "source_id": i.source_id,
                 "backend": (i.metadata or {}).get("backend"),
                 "trusted": (i.metadata or {}).get("trusted", False)}
                for i in pack.memory.items
            ],
        },
        "memory_is_external_context": pack.memory_is_external_context,
    })


# ---------------------------------------------------- compliance: launch/shell
def _cmd_launch(svc: AgentConnectService, a: argparse.Namespace) -> None:
    """Prepare a managed session (compliance §3.1). Prints the shell command."""
    result = svc.launch_session(
        manager_id=a.manager, task_id=a.task, review_id=a.review, claim=a.claim,
        readonly=a.readonly, force_readonly=a.force_readonly,
        repo_source=a.repo, repo_mode=a.repo_mode,
        launch_command=" ".join(sys.argv[1:]),
    )
    if a.json:
        _emit({
            "session": result["session"].model_dump(mode="json"),
            "workspace": result["workspace"].model_dump(mode="json"),
            "claim_id": result["claim_id"], "files": result["files"],
            "shell_command": result["shell_command"],
            # The token is written to `.env.agentconnect` (0600) and never printed:
            # a token in a terminal scrollback is a token in a log.
            "token": "(written to .env.agentconnect)",
        })
        return
    session, workspace = result["session"], result["workspace"]
    print("Prepared AgentConnect session.")
    if session.task_id:
        print(f"Task: {session.task_id}")
    if session.review_id:
        print(f"Review: {session.review_id}")
    print(f"Manager: {session.manager_id}")
    print(f"Mode: {session.mode.value}")
    print(f"Workspace: {workspace.path}")
    print(f"Repo: {workspace.repo_path} ({workspace.repo_mode.value})")
    print(f"Claim: {result['claim_id'] or '(none)'}")
    print(f"Wrote: {', '.join(result['files'])}")
    print("Run:")
    print(f"  {result['shell_command']}")


def _cmd_shell(svc: AgentConnectService, a: argparse.Namespace) -> None:
    """Run a command inside the managed workspace (compliance §3.2).

    The agent inherits an allowlisted environment plus its session vars. Backend
    credentials are not removed from the environment so much as never copied into
    it — see `core.sessions.sanitize_env`.
    """
    session = svc.active_session_for(task_id=a.task, review_id=a.review)
    if session is None:
        raise AgentConnectError(
            f"no prepared session for {a.review or a.task}; run `agentconnect launch` first"
        )
    workspace = svc.get_workspace(session.workspace_id) if session.workspace_id else None
    if workspace is None:
        raise AgentConnectError(f"session {session.id} has no workspace")

    base = Path(workspace.path)
    stored = sessions_mod.parse_env_file((base / ".env.agentconnect").read_text("utf-8"))
    cwd = Path(workspace.repo_path) if workspace.repo_path else base
    if not cwd.exists():
        cwd = base

    try:
        env = sessions_mod.sanitize_env(dict(os.environ), stored, helper_bin=str(base / "bin"))
    except ValueError as exc:
        raise AgentConnectError(str(exc)) from None

    if a.print_env:
        _emit({"cwd": str(cwd), "env": {k: ("***" if "TOKEN" in k else v)
                                        for k, v in sorted(env.items())}})
        return

    # argparse.REMAINDER hands back the `--` separator too.
    argv = a.command[1:] if a.command and a.command[0] == "--" else list(a.command)
    if not argv:
        raise AgentConnectError("nothing to run: `agentconnect shell --task T -- <command>`")

    svc.start_shell(session.id, " ".join(argv))
    exit_code = 1
    try:
        exit_code = subprocess.run(argv, cwd=str(cwd), env=env, check=False).returncode
    except FileNotFoundError:
        print(f"error: {argv[0]}: not found", file=sys.stderr)
    finally:
        svc.end_shell(session.id, exit_code)

    if a.audit and session.task_id:
        report = svc.audit_task(session.task_id)
        print("", file=sys.stderr)
        print(report.render(), file=sys.stderr)
    raise SystemExit(exit_code)


def _cmd_audit(svc: AgentConnectService, a: argparse.Namespace) -> None:
    report = svc.audit_review(a.review) if a.review else svc.audit_task(a.task_id)
    if a.json:
        _emit(report.to_dict())
    else:
        print(report.render())
    if not report.passed:
        raise SystemExit(EXIT_REFUSED)


def _cmd_complete(svc: AgentConnectService, a: argparse.Namespace) -> None:
    if a.review:
        result = svc.complete_review_audited(
            a.review,
            ReviewResultRequest(completed_by=a.by, summary=a.summary or "Reviewed.",
                                content=_read_body(a.file, a.content)),
            force=a.force,
        )
        _emit({"review": result["review"].model_dump(mode="json"),
               "audit": result["audit"], "forced": result["forced"]})
        return
    _emit(svc.complete_task(a.task_id, completed_by=a.by, force=a.force))


def _cmd_sessions_list(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.list_sessions(task_id=a.task, manager_id=a.manager, status=a.status,
                            limit=a.limit))


def _cmd_sessions_show(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.get_session(a.session_id))


def _cmd_workspaces_list(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.list_workspaces(include_destroyed=a.all))


def _cmd_workspaces_show(svc: AgentConnectService, a: argparse.Namespace) -> None:
    _emit(svc.get_workspace(a.workspace_id))


def _cmd_cleanup(svc: AgentConnectService, a: argparse.Namespace) -> None:
    if a.abandoned:
        _emit({"abandoned_sessions": svc.abandon_stale_sessions(a.older_than)})
        return
    if not a.task_id:
        raise AgentConnectError("cleanup needs a TASK_ID or --abandoned")
    workspace = svc.workspace_for(task_id=a.task_id)
    if workspace is None:
        raise AgentConnectError(f"no live workspace for {a.task_id}")
    svc.cleanup_workspace(workspace.id, actor=a.by)
    removed = False
    if a.remove_files:
        # A git worktree must be unregistered, not just deleted, or `git worktree
        # list` keeps a dangling entry forever.
        source = (workspace.metadata or {}).get("repo_source")
        if workspace.repo_mode is RepoMode.git_worktree and source:
            subprocess.run(["git", "worktree", "remove", "--force", workspace.repo_path],
                           cwd=source, capture_output=True, check=False)
        shutil.rmtree(workspace.path, ignore_errors=True)
        removed = True
    _emit({"workspace_id": workspace.id, "destroyed": True, "files_removed": removed})


def _cmd_linear_webhook_test(svc: AgentConnectService, a: argparse.Namespace) -> None:
    """Apply a saved webhook payload to the ledger. No network, no credentials."""
    from agentconnect.linear.webhooks import handle_webhook

    payload = json.loads(Path(a.payload_file).read_text(encoding="utf-8"))
    _emit(handle_webhook(svc, payload))


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentconnect",
        description="Local-first task backplane for interchangeable agent managers and workers.",
    )
    top = parser.add_subparsers(dest="group", required=True)

    # ------------------------------------------------- compliance layer (§18)
    p = top.add_parser(
        "launch",
        help="prepare a managed agent session: workspace, instructions, scoped token",
    )
    p.add_argument("manager", help="claude | codex | any manager id")
    p.add_argument("--task")
    p.add_argument("--review")
    p.add_argument("--claim", action="store_true", help="claim the task/review")
    p.add_argument("--readonly", action="store_true",
                   help="context inspection only; no decisions, subtasks, or completion")
    p.add_argument("--force-readonly", action="store_true",
                   help="downgrade to readonly instead of failing when the claim is held")
    p.add_argument("--repo", help="source repo to materialize a workspace from")
    p.add_argument("--repo-mode", default="auto",
                   choices=["auto", "git_worktree", "copy", "bind", "empty"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_launch)

    p = top.add_parser("shell", help="run a command inside the managed workspace")
    p.add_argument("--task")
    p.add_argument("--review")
    p.add_argument("--audit", action="store_true", help="audit the task on exit")
    p.add_argument("--print-env", action="store_true",
                   help="show the sanitized environment instead of running")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="the agent command, after `--`")
    p.set_defaults(func=_cmd_shell)

    p = top.add_parser("audit", help="can this task be completed?")
    p.add_argument("task_id", nargs="?")
    p.add_argument("--review")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_audit)

    p = top.add_parser("complete", help="mark complete — only if the audit passes")
    p.add_argument("task_id", nargs="?")
    p.add_argument("--review")
    p.add_argument("--by", default="human")
    p.add_argument("--summary")
    p.add_argument("--content")
    p.add_argument("--file")
    p.add_argument("--force", action="store_true",
                   help="human override; the audit problems are recorded anyway")
    p.set_defaults(func=_cmd_complete)

    sessions = top.add_parser("sessions", help="managed agent sessions").add_subparsers(
        dest="cmd", required=True)
    p = sessions.add_parser("list")
    p.add_argument("--task")
    p.add_argument("--manager")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=_cmd_sessions_list)

    p = sessions.add_parser("show")
    p.add_argument("session_id")
    p.set_defaults(func=_cmd_sessions_show)

    workspaces = top.add_parser("workspaces", help="task workspaces").add_subparsers(
        dest="cmd", required=True)
    p = workspaces.add_parser("list")
    p.add_argument("--all", action="store_true", help="include destroyed workspaces")
    p.set_defaults(func=_cmd_workspaces_list)

    p = workspaces.add_parser("show")
    p.add_argument("workspace_id")
    p.set_defaults(func=_cmd_workspaces_show)

    p = top.add_parser("cleanup", help="retire a workspace, or sweep abandoned sessions")
    p.add_argument("task_id", nargs="?")
    p.add_argument("--abandoned", action="store_true")
    p.add_argument("--older-than", type=float, default=24 * 3600)
    p.add_argument("--remove-files", action="store_true",
                   help="also delete the directory and unregister the git worktree")
    p.add_argument("--by", default="human")
    p.set_defaults(func=_cmd_cleanup)

    # tasks
    tasks = top.add_parser("tasks", help="task ledger").add_subparsers(dest="cmd", required=True)
    p = tasks.add_parser("create", help="create a task")
    p.add_argument("--title", required=True)
    p.add_argument("--goal", default="")
    p.add_argument("--by", default="human")
    p.add_argument("--priority", default=Priority.normal.value,
                   choices=[x.value for x in Priority])
    p.add_argument("--constraint", action="append", help="repeatable")
    p.set_defaults(func=_cmd_tasks_create)

    p = tasks.add_parser("list", help="list tasks")
    p.add_argument("--status", choices=[x.value for x in TaskStatus])
    p.add_argument("--manager")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=_cmd_tasks_list)

    p = tasks.add_parser("show", help="full task detail")
    p.add_argument("task_id")
    p.set_defaults(func=_cmd_tasks_show)

    p = tasks.add_parser("handoff", help="deterministic handoff summary")
    p.add_argument("task_id")
    p.add_argument("--manager")
    p.add_argument("--json", action="store_true", help="emit the structured summary")
    p.set_defaults(func=_cmd_tasks_handoff)

    p = tasks.add_parser("claim", help="claim a task")
    p.add_argument("task_id")
    p.add_argument("--manager", required=True)
    p.add_argument("--role", default=ClaimRole.primary_manager.value,
                   choices=[x.value for x in ClaimRole])
    p.add_argument("--ttl", type=int, default=DEFAULT_CLAIM_TTL_SECONDS)
    p.set_defaults(func=_cmd_tasks_claim)

    p = tasks.add_parser("release", help="release a claim")
    p.add_argument("task_id")
    p.add_argument("--manager", required=True)
    p.set_defaults(func=_cmd_tasks_release)

    p = tasks.add_parser("context-pack", help="handoff + labeled external memory")
    p.add_argument("task_id")
    p.add_argument("--profile", default="manager_brief", choices=sorted(PROFILES))
    p.add_argument("--max-items", dest="max_items", type=int, default=None)
    p.add_argument("--manager")
    p.set_defaults(func=_cmd_tasks_context_pack)

    # decisions
    decisions = top.add_parser("decisions", help="decision log").add_subparsers(
        dest="cmd", required=True)
    p = decisions.add_parser("add")
    p.add_argument("task_id")
    p.add_argument("--by", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--rationale", default="")
    p.add_argument("--locked", action="store_true")
    p.add_argument("--supersedes", action="append")
    p.set_defaults(func=_cmd_decisions_add)

    # attempts
    attempts = top.add_parser("attempts", help="attempt log").add_subparsers(
        dest="cmd", required=True)
    p = attempts.add_parser("add")
    p.add_argument("task_id")
    p.add_argument("--actor", required=True)
    p.add_argument("--actor-type", dest="actor_type", default=ActorType.manager.value,
                   choices=[x.value for x in ActorType])
    p.add_argument("--summary", required=True)
    p.add_argument("--outcome", default="")
    p.add_argument("--artifact", action="append")
    p.set_defaults(func=_cmd_attempts_add)

    # artifacts
    artifacts = top.add_parser("artifacts", help="artifact registry").add_subparsers(
        dest="cmd", required=True)
    p = artifacts.add_parser("add")
    p.add_argument("task_id")
    p.add_argument("--type", default=ArtifactType.other.value,
                   choices=[x.value for x in ArtifactType])
    p.add_argument("--file")
    p.add_argument("--content")
    p.add_argument("--summary", default="")
    p.add_argument("--by", default="human")
    p.set_defaults(func=_cmd_artifacts_add)

    p = artifacts.add_parser("list")
    p.add_argument("task_id")
    p.set_defaults(func=_cmd_artifacts_list)

    p = artifacts.add_parser("read")
    p.add_argument("artifact_id")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--all", action="store_true", help="page to EOF and print the body")
    p.set_defaults(func=_cmd_artifacts_read)

    # reviews
    reviews = top.add_parser("reviews", help="review tickets").add_subparsers(
        dest="cmd", required=True)
    p = reviews.add_parser("request")
    p.add_argument("task_id")
    p.add_argument("--to", required=True, help="assignee manager id")
    p.add_argument("--by", required=True, help="requesting manager id")
    p.add_argument("--artifact", action="append")
    p.add_argument("--criteria", action="append")
    p.set_defaults(func=_cmd_reviews_request)

    p = reviews.add_parser("claim")
    p.add_argument("review_id")
    p.add_argument("--manager", required=True)
    p.set_defaults(func=_cmd_reviews_claim)

    p = reviews.add_parser("complete")
    p.add_argument("review_id")
    p.add_argument("--by", help="defaults to the review's assignee")
    p.add_argument("--file")
    p.add_argument("--content")
    p.add_argument("--summary", default="")
    p.add_argument("--status", default=ReviewStatus.completed.value,
                   choices=[ReviewStatus.completed.value, ReviewStatus.rejected.value])
    p.set_defaults(func=_cmd_reviews_complete)

    # inbox
    p = top.add_parser("inbox", help="a manager's pending work")
    p.add_argument("manager_id")
    p.set_defaults(func=_cmd_inbox)

    # subtasks
    subtasks = top.add_parser("subtasks", help="worker delegation").add_subparsers(
        dest="cmd", required=True)
    p = subtasks.add_parser("submit")
    p.add_argument("task_id")
    p.add_argument("--title", required=True)
    p.add_argument("--instructions", required=True)
    p.add_argument("--privacy", default=PrivacyTier.repo_sensitive.value,
                   choices=[x.value for x in PrivacyTier])
    p.add_argument("--worker", help="preferred worker id, harness, or location")
    p.add_argument("--filesystem", default=FilesystemAccess.none.value,
                   choices=[x.value for x in FilesystemAccess])
    p.add_argument("--network", action="store_true")
    p.add_argument("--shell", action="store_true")
    p.add_argument("--capability", action="append")
    p.set_defaults(func=_cmd_subtasks_submit)

    p = subtasks.add_parser("show")
    p.add_argument("subtask_id")
    p.set_defaults(func=_cmd_subtasks_show)

    p = subtasks.add_parser("route", help="why this worker was chosen")
    p.add_argument("subtask_id")
    p.set_defaults(func=_cmd_subtasks_route)

    p = subtasks.add_parser("cancel")
    p.add_argument("subtask_id")
    p.set_defaults(func=_cmd_subtasks_cancel)

    p = subtasks.add_parser("approve", help="release a needs_approval subtask")
    p.add_argument("subtask_id")
    p.add_argument("--by", required=True)
    p.add_argument("--max-cost", dest="max_cost", type=float)
    p.set_defaults(func=_cmd_subtasks_approve)

    p = subtasks.add_parser("deny")
    p.add_argument("subtask_id")
    p.add_argument("--by", required=True)
    p.add_argument("--reason", default="")
    p.set_defaults(func=_cmd_subtasks_deny)

    # memory
    memory = top.add_parser("memory", help="external memory layer").add_subparsers(
        dest="cmd", required=True)
    p = memory.add_parser("recall", help="bounded, scoped recall")
    p.add_argument("--task", help="task id to scope the recall to")
    p.add_argument("--query", required=True)
    p.add_argument("--profile", default="manager_brief")
    p.add_argument("--max-items", dest="max_items", type=int, default=8)
    p.add_argument("--include-pending", dest="include_pending", action="store_true",
                   help="also return unpromoted candidates (they will be labeled)")
    p.set_defaults(func=_cmd_memory_recall)

    p = memory.add_parser("capture", help="offer a candidate (never promotes)")
    p.add_argument("--task")
    p.add_argument("--text")
    p.add_argument("--file")
    p.add_argument("--by", default="human")
    p.add_argument("--actor-type", dest="actor_type", default="human",
                   choices=["manager", "worker", "human", "system"])
    p.add_argument("--tag", action="append")
    p.set_defaults(func=_cmd_memory_capture)

    p = memory.add_parser("feedback", help="rate a recalled item")
    p.add_argument("--feedback", required=True,
                   choices=["useful", "irrelevant", "stale", "wrong", "too_broad",
                            "missing_context"])
    p.add_argument("--item", dest="item")
    p.add_argument("--source", dest="source")
    p.add_argument("--task")
    p.add_argument("--by", default="human")
    p.add_argument("--note")
    p.set_defaults(func=_cmd_memory_feedback)

    p = memory.add_parser("health")
    p.set_defaults(func=_cmd_memory_health)

    p = memory.add_parser("pending", help="candidates awaiting a human promotion decision")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=_cmd_memory_pending)

    p = memory.add_parser(
        "promote", help="promote a candidate to a trusted claim (human/librarian only)")
    p.add_argument("candidate_id")
    p.add_argument("--by", required=True, help="the human or librarian promoting it")
    p.set_defaults(func=_cmd_memory_promote)

    # linear
    linear = top.add_parser("linear", help="Linear mirror").add_subparsers(
        dest="cmd", required=True)
    p = linear.add_parser("sync", help="push the task to its Linear issue")
    p.add_argument("task_id")
    p.set_defaults(func=_cmd_linear_sync)

    p = linear.add_parser("issue", help="show the stored Linear mapping")
    p.add_argument("task_id")
    p.set_defaults(func=_cmd_linear_issue)

    p = linear.add_parser("webhook-test", help="apply a saved webhook payload offline")
    p.add_argument("payload_file")
    p.set_defaults(func=_cmd_linear_webhook_test)

    return parser


def main(argv: Optional[list[str]] = None,
         service: Optional[AgentConnectService] = None) -> int:
    args = build_parser().parse_args(argv)
    svc = service or service_from_env()
    try:
        args.func(svc, args)
    except AgentConnectError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
