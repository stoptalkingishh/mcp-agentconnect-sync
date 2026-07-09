"""Task, claim, decision, and attempt routes (spec §11)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from agentconnect.core.models import (
    Attempt,
    Claim,
    ClaimRole,
    CreateTaskRequest,
    Decision,
    HandoffSummary,
    RecordAttemptRequest,
    RecordDecisionRequest,
    Task,
    TaskDetail,
    TaskFilters,
    TaskStatus,
    TaskSummary,
)
from agentconnect.core.service import DEFAULT_CLAIM_TTL_SECONDS, AgentConnectService

router = APIRouter(tags=["tasks"])


def service(request: Request) -> AgentConnectService:
    return request.app.state.service


class ClaimBody(BaseModel):
    manager_id: str
    role: str = ClaimRole.primary_manager.value
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS


class ReleaseBody(BaseModel):
    manager_id: str


class ConstraintBody(BaseModel):
    text: str
    created_by: str = "unknown"


@router.post("/tasks", response_model=Task, status_code=201)
def create_task(body: CreateTaskRequest, request: Request) -> Task:
    return service(request).create_task(body)


@router.get("/tasks", response_model=list[TaskSummary])
def list_tasks(
    request: Request,
    status: Optional[TaskStatus] = None,
    current_manager: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[TaskSummary]:
    return service(request).list_tasks(
        TaskFilters(status=status, current_manager=current_manager, limit=limit, offset=offset)
    )


@router.get("/tasks/{task_id}", response_model=TaskDetail)
def get_task(task_id: str, request: Request) -> TaskDetail:
    return service(request).get_task(task_id)


@router.get("/tasks/{task_id}/handoff", response_model=HandoffSummary)
def get_handoff(
    task_id: str, request: Request, manager_id: Optional[str] = None
) -> HandoffSummary:
    return service(request).get_handoff_summary(task_id, manager_id)


@router.post("/tasks/{task_id}/handoff/regenerate", response_model=HandoffSummary)
def regenerate_handoff(task_id: str, request: Request) -> HandoffSummary:
    return service(request).regenerate_handoff_summary(task_id)


@router.post("/tasks/{task_id}/claim", response_model=Claim, status_code=201)
def claim_task(task_id: str, body: ClaimBody, request: Request) -> Claim:
    return service(request).claim_task(task_id, body.manager_id, body.role, body.ttl_seconds)


@router.post("/tasks/{task_id}/release", status_code=204, response_class=Response)
def release_task(task_id: str, body: ReleaseBody, request: Request) -> Response:
    service(request).release_task(task_id, body.manager_id)
    return Response(status_code=204)


@router.post("/tasks/{task_id}/constraints", status_code=201)
def add_constraint(task_id: str, body: ConstraintBody, request: Request) -> dict:
    constraint = service(request).add_constraint(task_id, body.text, body.created_by)
    return constraint.model_dump(mode="json")


@router.post("/tasks/{task_id}/decisions", response_model=Decision, status_code=201)
def record_decision(task_id: str, body: RecordDecisionRequest, request: Request) -> Decision:
    return service(request).record_decision(task_id, body)


@router.post("/tasks/{task_id}/attempts", response_model=Attempt, status_code=201)
def record_attempt(task_id: str, body: RecordAttemptRequest, request: Request) -> Attempt:
    return service(request).record_attempt(task_id, body)
