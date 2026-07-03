"""Typed state carried through the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    task_id: str = ""
    step: str = "init"
    iteration: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    changed_artifacts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
