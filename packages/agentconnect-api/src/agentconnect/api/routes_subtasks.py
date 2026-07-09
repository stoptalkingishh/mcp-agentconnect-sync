"""Subtask routes: delegation, routing transparency, and the approval gate (§11, §15)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from agentconnect.core.models import Subtask, SubtaskDetail, SubtaskRequest
from agentconnect.core.routing import RouteExplanation

from .routes_tasks import service

router = APIRouter(tags=["subtasks"])


class ApproveBody(BaseModel):
    approved_by: str
    max_cost_usd: Optional[float] = None


class DenyBody(BaseModel):
    denied_by: str
    reason: str = ""


@router.post("/tasks/{task_id}/subtasks", response_model=Subtask, status_code=201)
def submit_subtask(task_id: str, body: SubtaskRequest, request: Request) -> Subtask:
    return service(request).submit_subtask(task_id, body)


@router.get("/subtasks/{subtask_id}", response_model=SubtaskDetail)
def get_subtask(subtask_id: str, request: Request) -> SubtaskDetail:
    return service(request).get_subtask(subtask_id)


@router.post("/subtasks/{subtask_id}/cancel", status_code=204, response_class=Response)
def cancel_subtask(subtask_id: str, request: Request) -> Response:
    service(request).cancel_subtask(subtask_id)
    return Response(status_code=204)


@router.get("/subtasks/{subtask_id}/route", response_model=RouteExplanation)
def explain_route(subtask_id: str, request: Request) -> RouteExplanation:
    return service(request).explain_route(subtask_id)


@router.post("/subtasks/{subtask_id}/approve", response_model=Subtask)
def approve_subtask(subtask_id: str, body: ApproveBody, request: Request) -> Subtask:
    return service(request).approve_subtask(subtask_id, body.approved_by, body.max_cost_usd)


@router.post("/subtasks/{subtask_id}/deny", response_model=Subtask)
def deny_subtask(subtask_id: str, body: DenyBody, request: Request) -> Subtask:
    return service(request).deny_subtask(subtask_id, body.denied_by, body.reason)
