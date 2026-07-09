"""Sessions, workspaces, audit, completion (compliance spec §12–§13, §18).

These are *operator* routes, not agent routes. A launched agent holds a scoped
session token that buys `get_task_context_pack`, `record_attempt`, and friends —
never `complete`. Completion is a decision about whether work was recorded, and
an agent grading its own homework is the failure this whole layer exists to stop.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agentconnect.core.models import ReviewResultRequest

from .routes_tasks import service

router = APIRouter(tags=["compliance"])


class LaunchBody(BaseModel):
    manager_id: str
    task_id: Optional[str] = None
    review_id: Optional[str] = None
    claim: bool = False
    readonly: bool = False
    force_readonly: bool = False
    repo_source: Optional[str] = None
    repo_mode: str = "auto"


class CompleteBody(BaseModel):
    completed_by: str = "human"
    force: bool = False
    summary: str = ""
    content: str = ""


@router.post("/sessions/launch", status_code=201)
def launch(body: LaunchBody, request: Request) -> dict[str, Any]:
    """Prepare a managed session. The token is returned **once** and never again."""
    result = service(request).launch_session(
        manager_id=body.manager_id, task_id=body.task_id, review_id=body.review_id,
        claim=body.claim, readonly=body.readonly, force_readonly=body.force_readonly,
        repo_source=body.repo_source, repo_mode=body.repo_mode,
        launch_command="POST /sessions/launch",
    )
    return {
        "session": result["session"].model_dump(mode="json"),
        "workspace": result["workspace"].model_dump(mode="json"),
        "claim_id": result["claim_id"], "files": result["files"],
        "token": result["token"], "shell_command": result["shell_command"],
    }


@router.get("/sessions")
def list_sessions(
    request: Request, task_id: Optional[str] = None, manager_id: Optional[str] = None,
    status: Optional[str] = None, limit: int = 50,
) -> list[dict[str, Any]]:
    return [
        s.model_dump(mode="json")
        for s in service(request).list_sessions(task_id, manager_id, status, limit)
    ]


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> dict[str, Any]:
    return service(request).get_session(session_id).model_dump(mode="json")


@router.post("/sessions/{session_id}/end")
def end_session(session_id: str, request: Request, exit_code: int = 0) -> dict[str, Any]:
    """Ends the session and revokes its token: a leaked env file becomes inert."""
    return service(request).end_shell(session_id, exit_code).model_dump(mode="json")


@router.get("/workspaces")
def list_workspaces(request: Request, include_destroyed: bool = False) -> list[dict[str, Any]]:
    return [
        w.model_dump(mode="json")
        for w in service(request).list_workspaces(include_destroyed)
    ]


@router.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str, request: Request) -> dict[str, Any]:
    return service(request).get_workspace(workspace_id).model_dump(mode="json")


@router.get("/tasks/{task_id}/audit")
def audit_task(task_id: str, request: Request) -> dict[str, Any]:
    """Read-only and idempotent: asking twice gives the same answer."""
    return service(request).audit_task(task_id).to_dict()


@router.get("/reviews/{review_id}/audit")
def audit_review(review_id: str, request: Request) -> dict[str, Any]:
    return service(request).audit_review(review_id).to_dict()


@router.post("/tasks/{task_id}/complete")
def complete_task(task_id: str, body: CompleteBody, request: Request) -> dict[str, Any]:
    """Audit first. A failing audit is a `policy_violation` (409), not a 500."""
    return service(request).complete_task(task_id, body.completed_by, force=body.force)


@router.post("/reviews/{review_id}/complete")
def complete_review(review_id: str, body: CompleteBody, request: Request) -> dict[str, Any]:
    result = service(request).complete_review_audited(
        review_id,
        ReviewResultRequest(completed_by=body.completed_by, summary=body.summary,
                            content=body.content),
        force=body.force,
    )
    return {"review": result["review"].model_dump(mode="json"),
            "audit": result["audit"], "forced": result["forced"]}
