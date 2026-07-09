"""Filesystem artifact bodies with bounded, resumable reads (spec §8, §9.6, §21).

Bodies never travel inline through MCP or Linear — adapters hand back an
``artifact_id`` and callers page with :meth:`read_chunk`.

Offsets are **byte** offsets, but chunks always end on a UTF-8 character
boundary: a chunk that would split a multi-byte sequence is trimmed and
``next_offset`` reports the aligned position. So paging with the returned
``next_offset`` never corrupts a character, and never silently drops one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")

#: Max continuation bytes to walk back over when trimming a partial UTF-8 char.
_MAX_UTF8_TAIL = 3


def default_artifact_dir() -> str:
    env = os.environ.get("AGENTCONNECT_ARTIFACT_DIR")
    if env:
        return env
    return str(Path.home() / ".agentconnect" / "artifacts")


def _safe(component: str) -> str:
    cleaned = _SAFE.sub("_", component).strip("._") or "unnamed"
    return cleaned[:120]


def _valid_prefix_len(chunk: bytes) -> int:
    """Length of the longest prefix of ``chunk`` that is whole UTF-8.

    Returns 0 when the chunk is nothing but the start of a character (a 1-byte
    read of a 2-byte char); the caller must then read further rather than emit a
    replacement char. Returns ``len(chunk)`` when the damage is further back than
    a character boundary could explain — that is real corruption, and paging must
    still make progress.
    """
    for length in range(len(chunk), max(-1, len(chunk) - _MAX_UTF8_TAIL - 1), -1):
        try:
            chunk[:length].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return length
    return len(chunk)


def _first_char_len(chunk: bytes) -> int:
    """Bytes in the first whole UTF-8 character. Used when the caller's limit is
    narrower than one character: we return exactly one char, never two."""
    for length in range(1, min(len(chunk), 4) + 1):
        try:
            chunk[:length].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return length
    return len(chunk)


def _align_start(fh, offset: int, size: int) -> int:
    """Advance a caller-supplied offset off a continuation byte (0b10xxxxxx)."""
    pos = offset
    while pos < size and pos - offset <= _MAX_UTF8_TAIL:
        fh.seek(pos)
        byte = fh.read(1)
        if not byte or (byte[0] & 0xC0) != 0x80:
            return pos
        pos += 1
    return pos


class FilesystemArtifactStore:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or default_artifact_dir())
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, task_id: str, artifact_id: str, content: str) -> tuple[str, int]:
        """Write a body. Returns ``(relative_path, size_bytes)``."""
        rel = Path(_safe(task_id)) / f"{_safe(artifact_id)}.txt"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        return str(rel), len(data)

    def size(self, rel_path: str) -> int:
        path = self.root / rel_path
        return path.stat().st_size if path.exists() else 0

    def read_all(self, rel_path: str) -> str:
        path = self.root / rel_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def read_chunk(
        self, rel_path: str, offset: int = 0, limit: int = 8000
    ) -> tuple[str, Optional[int], bool, int]:
        """Return ``(content, next_offset, eof, size_bytes)``."""
        path = self.root / rel_path
        if not path.exists():
            return "", None, True, 0
        size = path.stat().st_size
        offset = max(0, offset)
        limit = max(1, limit)
        if offset >= size:
            return "", None, True, size
        with path.open("rb") as fh:
            start = _align_start(fh, offset, size)
            fh.seek(start)
            raw = fh.read(limit)
            at_eof = start + len(raw) >= size
            if not at_eof:
                keep = _valid_prefix_len(raw)
                if keep == 0:
                    # The whole chunk is the head of one character (limit < char
                    # width). Over-read just far enough to finish that one
                    # character — never far enough to return a second.
                    raw += fh.read(_MAX_UTF8_TAIL)
                    keep = _first_char_len(raw)
                raw = raw[:keep]
        text = raw.decode("utf-8", errors="replace")
        next_offset = start + len(raw)
        eof = next_offset >= size
        return text, (None if eof else next_offset), eof, size
