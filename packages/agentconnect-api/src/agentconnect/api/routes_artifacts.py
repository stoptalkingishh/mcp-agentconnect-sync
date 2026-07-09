"""Artifact routes (spec §11). Bodies leave only through /chunk."""

from __future__ import annotations

from fastapi import APIRouter, Request

from agentconnect.core.models import Artifact, ArtifactChunk, ArtifactSummary, CreateArtifactRequest

from .routes_tasks import service

router = APIRouter(tags=["artifacts"])


@router.post("/tasks/{task_id}/artifacts", response_model=Artifact, status_code=201)
def create_artifact(task_id: str, body: CreateArtifactRequest, request: Request) -> Artifact:
    return service(request).create_artifact(task_id, body)


@router.get("/tasks/{task_id}/artifacts", response_model=list[ArtifactSummary])
def list_artifacts(task_id: str, request: Request) -> list[ArtifactSummary]:
    return service(request).list_artifacts(task_id)


@router.get("/artifacts/{artifact_id}", response_model=Artifact)
def get_artifact(artifact_id: str, request: Request) -> Artifact:
    return service(request).get_artifact(artifact_id)


@router.get("/artifacts/{artifact_id}/chunk", response_model=ArtifactChunk)
def read_artifact_chunk(
    artifact_id: str, request: Request, offset: int = 0, limit: int = 8000
) -> ArtifactChunk:
    return service(request).read_artifact_chunk(artifact_id, offset, limit)
