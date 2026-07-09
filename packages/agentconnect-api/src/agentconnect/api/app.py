"""FastAPI adapter over `AgentConnectService` (spec §5, §11).

Every route is a translation: HTTP in, service call, model out. No route holds
policy, touches storage, or knows what a worker is.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentconnect.core.bootstrap import service_from_env
from agentconnect.core.errors import AgentConnectError
from agentconnect.core.service import AgentConnectService

from . import (
    routes_artifacts,
    routes_compliance,
    routes_linear,
    routes_managers,
    routes_memory,
    routes_reviews,
    routes_subtasks,
    routes_tasks,
    routes_temporal,
)
from .deps import linear_sync_from_env, status_for


def create_app(
    service: Optional[AgentConnectService] = None,
    linear_sync: Optional[object] = None,
) -> FastAPI:
    app = FastAPI(
        title="AgentConnect",
        description="Local-first task backplane for interchangeable agent managers and workers.",
        version="0.1.0",
    )
    svc = service or service_from_env()
    app.state.service = svc
    app.state.linear_sync = linear_sync if linear_sync is not None else linear_sync_from_env(svc)

    @app.exception_handler(AgentConnectError)
    async def _backplane_error(_: Request, exc: AgentConnectError) -> JSONResponse:
        return JSONResponse(
            status_code=status_for(exc), content={"error": exc.code, "detail": str(exc)}
        )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "workers": [w.worker_id for w in svc.registry.all()],
            "linear_sync": app.state.linear_sync is not None,
            "execution_backend": svc.execution.name,
            "memory_backend": svc.memory.backend_name,
        }

    for module in (
        routes_tasks, routes_artifacts, routes_reviews, routes_managers,
        routes_subtasks, routes_linear, routes_memory, routes_temporal,
        routes_compliance,
    ):
        app.include_router(module.router)
    return app


def main() -> None:
    import uvicorn

    import os

    uvicorn.run(
        create_app(),
        host=os.environ.get("AGENTCONNECT_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENTCONNECT_API_PORT", "8790")),
    )


if __name__ == "__main__":
    main()
