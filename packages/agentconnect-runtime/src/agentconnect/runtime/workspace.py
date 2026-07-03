"""Workspace helpers for the worker runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceHandle:
    task_id: str
    path: Path
    branch: str | None = None
