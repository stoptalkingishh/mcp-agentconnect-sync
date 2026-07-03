"""Workspace management for the worker runtime.

Every task executes inside one workspace directory. All tool file access goes
through :meth:`Workspace.resolve`, which confines paths to the workspace root —
a tool-supplied path may not escape via ``..`` or absolute segments.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class WorkspaceError(Exception):
    """A tool asked for a path outside the workspace, or an invalid one."""


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.changed_files: list[str] = []
        self._ephemeral = False

    @classmethod
    def create(cls, root: str | Path | None = None, task_id: str = "") -> "Workspace":
        """Open `root`, or make a fresh temp workspace when none is given."""
        if root:
            return cls(root)
        suffix = f"-{task_id}" if task_id else ""
        ws = cls(tempfile.mkdtemp(prefix=f"agentconnect-ws{suffix}-"))
        ws._ephemeral = True
        return ws

    def resolve(self, relative: str) -> Path:
        """Resolve a tool-supplied path, confined to the workspace root."""
        if not relative or not str(relative).strip():
            raise WorkspaceError("empty path")
        try:
            candidate = (self.root / relative).resolve()
        except (ValueError, OSError) as exc:  # e.g. embedded null byte
            raise WorkspaceError(f"invalid path {relative!r}: {exc}") from exc
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"path escapes the workspace: {relative!r}")
        return candidate

    def cleanup(self) -> None:
        """Remove an ephemeral (runtime-created temp) workspace. No-op for a
        caller-supplied root — the caller owns that directory and its artifacts."""
        if self._ephemeral:
            shutil.rmtree(self.root, ignore_errors=True)

    def record_change(self, relative: str) -> None:
        if relative not in self.changed_files:
            self.changed_files.append(relative)
