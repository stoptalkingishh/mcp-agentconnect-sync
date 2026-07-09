"""Memory routes (adapters spec, Part A).

These call `AgentConnectService`, never a memory backend directly — that is what
keeps visibility policy (trusted_only, pending labelling, item caps) in one place
instead of scattered across callers.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agentconnect.core.memory import (
    CaptureRequest,
    MemoryFeedbackRequest,
    MemoryScope,
    RecallRequest,
)

from .routes_tasks import service

router = APIRouter(tags=["memory"])


class ScopeBody(BaseModel):
    scope_type: str
    scope_id: str


class RecallBody(BaseModel):
    query: str
    task_id: Optional[str] = None
    profile: str = "manager_brief"
    scopes: list[ScopeBody] = []
    max_items: int = 8
    trusted_only: bool = True
    include_pending: bool = False
    include_superseded: bool = False


class CaptureBody(BaseModel):
    text: str
    task_id: Optional[str] = None
    origin_actor_id: Optional[str] = None
    origin_actor_type: Optional[str] = None
    source_ref: Optional[str] = None
    tags: list[str] = []


class FeedbackBody(BaseModel):
    feedback: str
    task_id: Optional[str] = None
    memory_item_id: Optional[str] = None
    source_id: Optional[str] = None
    actor_id: Optional[str] = None
    note: Optional[str] = None


def _pack(pack) -> dict[str, Any]:
    return {
        "backend": pack.backend, "profile": pack.profile, "query": pack.query,
        "warnings": pack.warnings,
        "items": [
            {"text": i.text, "status": i.status, "confidence": i.confidence,
             "source_id": i.source_id, "source_url": i.source_url,
             "superseded_by": i.superseded_by}
            for i in pack.items
        ],
    }


@router.post("/memory/recall")
def recall(body: RecallBody, request: Request) -> dict[str, Any]:
    pack = service(request).recall_memory(RecallRequest(
        query=body.query, task_id=body.task_id, profile=body.profile,  # type: ignore[arg-type]
        scopes=[MemoryScope(s.scope_type, s.scope_id) for s in body.scopes],
        max_items=body.max_items, trusted_only=body.trusted_only,
        include_pending=body.include_pending, include_superseded=body.include_superseded,
    ))
    return _pack(pack)


@router.post("/memory/capture")
def capture(body: CaptureBody, request: Request) -> dict[str, Any]:
    result = service(request).capture_memory_candidate(CaptureRequest(
        text=body.text, task_id=body.task_id, origin_actor_id=body.origin_actor_id,
        origin_actor_type=body.origin_actor_type, source_ref=body.source_ref, tags=body.tags,
    ))
    return {
        "accepted": result.accepted, "candidate_id": result.candidate_id,
        "status": result.status, "message": result.message, "backend": result.backend,
    }


@router.post("/memory/feedback", status_code=202)
def feedback(body: FeedbackBody, request: Request) -> dict[str, Any]:
    service(request).record_memory_feedback(MemoryFeedbackRequest(
        task_id=body.task_id, memory_item_id=body.memory_item_id, source_id=body.source_id,
        feedback=body.feedback, actor_id=body.actor_id, note=body.note,
    ))
    return {"recorded": True}


@router.get("/memory/health")
def health(request: Request) -> dict[str, Any]:
    return service(request).memory_health()


@router.get("/tasks/{task_id}/context-pack")
def context_pack(
    task_id: str, request: Request, profile: str = "manager_brief",
    max_memory_items: int = 8, manager_id: Optional[str] = None,
    include_pending: bool = False,
) -> dict[str, Any]:
    pack = service(request).get_task_context_pack(
        task_id, profile=profile, max_memory_items=max_memory_items,
        manager_id=manager_id, include_pending=include_pending,
    )
    return {
        "task_id": pack.task_id, "profile": pack.profile,
        "handoff": pack.handoff.model_dump(mode="json"),
        "memory": _pack(pack.memory),
        "memory_is_external_context": pack.memory_is_external_context,
    }
