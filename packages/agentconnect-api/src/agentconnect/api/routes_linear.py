"""Linear mirror routes (spec §11, §14).

`/linear/webhook` needs no Linear credentials — it only *reads* an inbound
payload and applies it to the ledger. `/linear/sync` does need them, and returns
503 when the deployment has not configured a team.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .routes_tasks import service

router = APIRouter(prefix="/linear", tags=["linear"])


class SyncBody(BaseModel):
    task_id: str


def _sync(request: Request):
    sync = getattr(request.app.state, "linear_sync", None)
    if sync is None:
        raise HTTPException(
            status_code=503,
            detail="Linear sync is not configured (set LINEAR_API_KEY and LINEAR_TEAM_ID)",
        )
    return sync


@router.post("/sync")
def sync_task(body: SyncBody, request: Request) -> dict[str, Any]:
    ref = _sync(request).sync_task(body.task_id)
    return ref.model_dump(mode="json")


@router.get("/tasks/{task_id}")
def get_linear_ref(task_id: str, request: Request) -> dict[str, Any]:
    ref = service(request).get_external_ref("task", task_id, "linear")
    if ref is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} is not synced to Linear")
    return ref.model_dump(mode="json")


@router.post("/webhook")
def webhook(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    from agentconnect.linear.webhooks import handle_webhook

    return {"results": handle_webhook(service(request), payload)}
