"""Service wiring and error translation for the HTTP adapter.

The app holds one `AgentConnectService` on `app.state`; routes read it through
`get_service`. Tests override it with an in-memory service — the same seam the
CLI and MCP adapters use, just spelled FastAPI-style.
"""

from __future__ import annotations

import os
from typing import Optional

from agentconnect.core.errors import (
    AgentConnectError,
    Conflict,
    InvalidRequest,
    NotFound,
    PolicyViolation,
)
from agentconnect.core.service import AgentConnectService

#: Backplane error -> HTTP status. Adapters translate; the service never knows.
STATUS_FOR: dict[type[AgentConnectError], int] = {
    NotFound: 404,
    Conflict: 409,
    PolicyViolation: 403,
    InvalidRequest: 400,
}


def status_for(exc: AgentConnectError) -> int:
    for kind, status in STATUS_FOR.items():
        if isinstance(exc, kind):
            return status
    return 500


def linear_sync_from_env(service: AgentConnectService) -> Optional[object]:
    """Build a `LinearSync` when the deployment is configured for it.

    Returns None (never raises) when the Linear extra is absent or unconfigured,
    so `/linear/*` degrades to a clear 503 instead of breaking app startup.
    """
    team_id = os.environ.get("LINEAR_TEAM_ID")
    if not team_id:
        return None
    try:
        from agentconnect.linear import LinearClient, LinearSync
    except ImportError:
        return None
    try:
        client = LinearClient()
    except Exception:  # missing/invalid credentials
        return None
    return LinearSync(
        service, client, team_id, artifact_base_url=os.environ.get("AGENTCONNECT_BASE_URL")
    )
