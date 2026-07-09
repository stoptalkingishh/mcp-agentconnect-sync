"""Manager inbox (spec §11, §14.4 mode 3)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from agentconnect.core.models import InboxItem

from .routes_tasks import service

router = APIRouter(tags=["managers"])


@router.get("/managers/{manager_id}/inbox", response_model=list[InboxItem])
def get_manager_inbox(manager_id: str, request: Request) -> list[InboxItem]:
    return service(request).get_manager_inbox(manager_id)
