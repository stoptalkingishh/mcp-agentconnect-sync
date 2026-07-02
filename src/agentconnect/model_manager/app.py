"""Local Model Manager HTTP service (handoff §4.2, §22).

Exposes the internal API the Agent Router polls and calls:

    GET  /status      -> ManagerStatus
    GET  /models      -> available models
    GET  /queue       -> queue depth
    GET  /metrics     -> lightweight metrics
    POST /can_accept  -> admission decision
    POST /generate    -> run a local generation
    POST /load        -> make a model resident
    POST /unload      -> evict a model

This service is intentionally an appliance: it knows nothing about global routing
policy, quota, cloud providers, or secrets. Auth is via a shared bearer token
(the router resolves the token from the secrets manager; see §7).
"""

from __future__ import annotations

import os
from typing import Optional

from ..common.schemas import (
    CanAcceptRequest,
    GenerateRequest,
    LoadRequest,
)
from .residency import ResidencyManager


def create_app(manager: Optional[ResidencyManager] = None):
    """Build the FastAPI app. Imported lazily so the core has no FastAPI dep."""
    from fastapi import Depends, FastAPI, Header, HTTPException

    mgr = manager or ResidencyManager()
    app = FastAPI(title="Local Model Manager", version="0.1.0")

    expected_token = os.environ.get("LOCAL_R9700_API_TOKEN")

    def auth(authorization: str | None = Header(default=None)) -> None:
        # If no token is configured, run open (dev). Otherwise require a match.
        if expected_token is None:
            return
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/status")
    def status(_: None = Depends(auth)):
        return mgr.status().model_dump()

    @app.get("/models")
    def models(_: None = Depends(auth)):
        return [m.model_dump() for m in mgr.inventory()]

    @app.get("/queue")
    def queue(_: None = Depends(auth)):
        return mgr.status().queue.model_dump()

    @app.get("/metrics")
    def metrics(_: None = Depends(auth)):
        s = mgr.status()
        return {
            "node_id": s.node_id,
            "status": s.status,
            "loaded_model": s.loaded_model.model_id if s.loaded_model else None,
            "active_sequences": s.loaded_model.active_sequences if s.loaded_model else 0,
            "max_active_sequences": mgr.max_active_sequences,
            "vram_free_gb": s.gpu.vram_free_gb if s.gpu else None,
            "queue_depth": s.queue.local_waiting,
        }

    @app.post("/can_accept")
    def can_accept(req: CanAcceptRequest, _: None = Depends(auth)):
        return mgr.can_accept(req).model_dump()

    @app.post("/generate")
    def generate(req: GenerateRequest, _: None = Depends(auth)):
        return mgr.generate(req).model_dump()

    @app.post("/load")
    def load(req: LoadRequest, _: None = Depends(auth)):
        return mgr.load(req).model_dump()

    @app.post("/unload")
    def unload(req: LoadRequest, _: None = Depends(auth)):
        return mgr.unload(req.target_model).model_dump()

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("MODEL_MANAGER_HOST", "0.0.0.0")
    port = int(os.environ.get("MODEL_MANAGER_PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
