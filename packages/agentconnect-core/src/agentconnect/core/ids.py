"""Stable public ID minting (spec §11).

Every entity gets a prefixed opaque ID so a bare string is self-describing in a
log line, an MCP response, or a Linear comment.
"""

from __future__ import annotations

import uuid

TASK = "task"
CLAIM = "claim"
DECISION = "decision"
ATTEMPT = "attempt"
ARTIFACT = "artifact"
REVIEW = "review"
SUBTASK = "subtask"
RUN = "run"
EXTERNAL = "external"
CONSTRAINT = "constraint"
INBOX = "inbox"
EVENT = "event"
APPROVAL = "approval"
SESSION = "session"
WORKSPACE = "workspace"
TOKEN = "token"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def is_id(value: str, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(f"{prefix}_")
