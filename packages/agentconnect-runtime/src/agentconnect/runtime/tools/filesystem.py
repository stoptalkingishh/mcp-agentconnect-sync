"""Filesystem tools: read, write, and list inside the task workspace.

Each function returns an observation string for the model. Errors — missing
files, escaping paths, and OS-level failures like writing over a directory —
come back as ``ERROR: ...`` observations rather than exceptions, so the loop
can show them to the model and continue.
"""

from __future__ import annotations

from ..workspace import Workspace, WorkspaceError


def read_file(ws: Workspace, path: str, max_chars: int | None = None) -> str:
    """Read a file, loading at most ``max_chars + 1`` characters so a huge file
    cannot balloon memory; the graph adds the truncation marker."""
    try:
        target = ws.resolve(path)
        if not target.is_file():
            return f"ERROR: file not found: {path}"
        with target.open(encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars + 1 if max_chars is not None else -1)
    except (WorkspaceError, OSError) as exc:
        return f"ERROR: {exc}"


def write_file(ws: Workspace, path: str, content: str) -> str:
    try:
        target = ws.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except (WorkspaceError, OSError) as exc:
        return f"ERROR: {exc}"
    ws.record_change(path)
    return f"OK: wrote {len(content)} chars to {path}"


def list_dir(ws: Workspace, path: str = ".") -> str:
    try:
        target = ws.resolve(path or ".")
        if not target.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in target.iterdir())
    except (WorkspaceError, OSError) as exc:
        return f"ERROR: {exc}"
    return "\n".join(entries) if entries else "(empty)"
