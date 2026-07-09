"""`agentconnect-temporal-worker` — the process that actually runs the work.

The API and MCP servers *start* workflows; this process executes them. Run at
least one of these, pointed at the same `AGENTCONNECT_DB_PATH`, or subtasks will
sit in `running` forever with nobody to serve the task queue.

    TEMPORAL_ADDRESS=localhost:7233 agentconnect-temporal-worker
"""

from __future__ import annotations

import asyncio
import logging
import os

from agentconnect.core.bootstrap import service_from_env

from .client import build_worker, connect, task_queue_from_env

_log = logging.getLogger(__name__)


def _linear_sync(service):
    team_id = os.environ.get("LINEAR_TEAM_ID")
    if not team_id:
        return None
    try:
        from agentconnect.linear import LinearClient, LinearSync

        return LinearSync(service, LinearClient(), team_id)
    except Exception as exc:  # missing extra or credentials: mirror silently off
        _log.info("Linear mirror disabled (%s)", exc)
        return None


async def _run() -> None:
    service = service_from_env()
    client = await connect()
    worker = build_worker(client, service, _linear_sync(service))
    _log.info(
        "agentconnect temporal worker serving task queue %r (db=%s)",
        task_queue_from_env(), service.storage.path,
    )
    await worker.run()


def main() -> None:
    logging.basicConfig(level=os.environ.get("AGENTCONNECT_LOG_LEVEL", "INFO"))
    asyncio.run(_run())


if __name__ == "__main__":
    main()
