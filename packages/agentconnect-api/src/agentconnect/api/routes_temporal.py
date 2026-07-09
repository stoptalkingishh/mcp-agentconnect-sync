"""Workflow inspection and signalling (Temporal spec, HTTP API).

`workflow_id` is the execution handle id, so these routes work for the direct
backend too — you just get a handle that was never a workflow.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .routes_tasks import service

router = APIRouter(prefix="/workflows", tags=["workflows"])


class SignalBody(BaseModel):
    name: str
    payload: dict[str, Any] = {}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str, request: Request) -> dict[str, Any]:
    svc = service(request)
    handle = svc.get_execution(workflow_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_id}")
    status = svc.execution.get_status(workflow_id)
    return {
        "handle": handle.model_dump(mode="json"),
        "status": status.model_dump(mode="json"),
    }


@router.post("/{workflow_id}/signal", status_code=202)
def signal_workflow(workflow_id: str, body: SignalBody, request: Request) -> dict[str, Any]:
    svc = service(request)
    if svc.get_execution(workflow_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_id}")
    svc.execution.signal(workflow_id, body.name, body.payload)
    return {"signalled": workflow_id, "name": body.name}
