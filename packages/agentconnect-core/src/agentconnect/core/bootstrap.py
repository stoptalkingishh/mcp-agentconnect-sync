"""One way to build the service from the environment (spec §5, §8).

Every adapter — MCP, HTTP, CLI — constructs its `AgentConnectService` here, so a
CLI command and an MCP tool run against the same database, the same artifact
directory, and the same worker registry. Point two adapters at the same
`AGENTCONNECT_DB_PATH` and they are looking at one ledger.

Env:
  AGENTCONNECT_DB_PATH        sqlite file (default ~/.agentconnect/agentconnect.db)
  AGENTCONNECT_ARTIFACT_DIR   artifact bodies (default ~/.agentconnect/artifacts)
  AGENTCONNECT_MAX_COST_USD   standing budget ceiling for routing (default 0.0)
  AGENTCONNECT_WORKERS        comma-separated built-ins to register (default "echo")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .routing import RoutePolicy
from .service import AgentConnectService
from .workers import EchoWorker, WorkerAdapter

_log = logging.getLogger(__name__)

#: Built-in, dependency-free workers. Real harnesses (LiteLLM, local model
#: manager, Deep Agents, sandboxed shell) register themselves at runtime — the
#: core never imports them (§3: this is not a model gateway).
_BUILTIN_WORKERS = {"echo": EchoWorker}


def workers_from_env() -> list[WorkerAdapter]:
    names = os.environ.get("AGENTCONNECT_WORKERS", "echo")
    workers: list[WorkerAdapter] = []
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        factory = _BUILTIN_WORKERS.get(name)
        if factory is None:
            _log.warning("unknown built-in worker %r in AGENTCONNECT_WORKERS; skipping", name)
            continue
        workers.append(factory())
    return workers


def policy_from_env() -> RoutePolicy:
    raw = os.environ.get("AGENTCONNECT_MAX_COST_USD", "0")
    try:
        return RoutePolicy(max_cost_usd=float(raw))
    except ValueError:
        _log.warning("AGENTCONNECT_MAX_COST_USD=%r is not a number; defaulting to 0", raw)
        return RoutePolicy()


def service_from_env(
    workers: Optional[list[WorkerAdapter]] = None,
    db_path: Optional[str] = None,
    artifact_dir: Optional[str] = None,
) -> AgentConnectService:
    return AgentConnectService.create(
        db_path=db_path or os.environ.get("AGENTCONNECT_DB_PATH"),
        artifact_dir=artifact_dir or os.environ.get("AGENTCONNECT_ARTIFACT_DIR"),
        workers=workers if workers is not None else workers_from_env(),
        policy=policy_from_env(),
    )
