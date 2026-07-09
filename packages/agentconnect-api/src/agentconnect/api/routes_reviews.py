"""Review routes — the manager-to-manager coordination surface (spec §11, §17)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agentconnect.core.models import Review, ReviewRequest, ReviewResultRequest

from .routes_tasks import service

router = APIRouter(tags=["reviews"])


class ClaimReviewBody(BaseModel):
    manager_id: str


@router.post("/tasks/{task_id}/reviews", response_model=Review, status_code=201)
def request_review(task_id: str, body: ReviewRequest, request: Request) -> Review:
    return service(request).request_review(task_id, body)


@router.get("/reviews/{review_id}", response_model=Review)
def get_review(review_id: str, request: Request) -> Review:
    return service(request).get_review(review_id)


@router.post("/reviews/{review_id}/claim", response_model=Review)
def claim_review(review_id: str, body: ClaimReviewBody, request: Request) -> Review:
    return service(request).claim_review(review_id, body.manager_id)


@router.post("/reviews/{review_id}/result", response_model=Review)
def complete_review(review_id: str, body: ReviewResultRequest, request: Request) -> Review:
    return service(request).complete_review(review_id, body)
