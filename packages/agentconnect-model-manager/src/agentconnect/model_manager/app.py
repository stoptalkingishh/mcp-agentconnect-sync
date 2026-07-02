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
policy, quota, cloud providers, or secrets. Authentication is **mutual TLS** — the
manager requires a client certificate signed by a trusted internal CA (there is no
shared bearer token). See ``tls.py`` and handoff §7.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..common.schemas import (
    CanAcceptRequest,
    GenerateRequest,
    LoadRequest,
)
from .residency import ResidencyManager
from .tls import (
    ClientIdentityMiddleware,
    allowed_clients_from_env,
    build_ssl_kwargs,
    manager_tls_from_env,
)

_log = logging.getLogger(__name__)


def create_app(
    manager: Optional[ResidencyManager] = None,
    allowed_clients: Optional[set[str]] = None,
):
    """Build the FastAPI app. Imported lazily so the core has no FastAPI dep.

    Transport authentication (client-cert verification) is configured on the
    uvicorn server in :func:`main`. ``allowed_clients`` adds an optional
    application-layer per-identity allowlist on top (defense in depth)."""
    from fastapi import FastAPI

    mgr = manager or ResidencyManager()
    app = FastAPI(title="Local Model Manager", version="0.2.0")
    if allowed_clients:
        app.add_middleware(ClientIdentityMiddleware, allowed=allowed_clients)

    @app.get("/status")
    def status():
        return mgr.status().model_dump()

    @app.get("/models")
    def models():
        return [m.model_dump() for m in mgr.inventory()]

    @app.get("/queue")
    def queue():
        return mgr.status().queue.model_dump()

    @app.get("/metrics")
    def metrics():
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
    def can_accept(req: CanAcceptRequest):
        return mgr.can_accept(req).model_dump()

    @app.post("/generate")
    def generate(req: GenerateRequest):
        return mgr.generate(req).model_dump()

    @app.post("/load")
    def load(req: LoadRequest):
        return mgr.load(req).model_dump()

    @app.post("/unload")
    def unload(req: LoadRequest):
        return mgr.unload(req.target_model).model_dump()

    return app


def main() -> None:
    import uvicorn

    from .backends import backend_from_env

    tls = manager_tls_from_env()
    allowed = allowed_clients_from_env()
    app = create_app(manager=ResidencyManager(backend=backend_from_env()), allowed_clients=allowed)
    port = int(os.environ.get("MODEL_MANAGER_PORT", "8443"))

    if tls.mode == "mutual":
        ssl_kwargs = build_ssl_kwargs(tls)
        host = os.environ.get("MODEL_MANAGER_HOST", "0.0.0.0")
        _log.info("Model Manager serving mTLS on %s:%s (client cert required).", host, port)
        uvicorn.run(app, host=host, port=port, **ssl_kwargs)
    else:
        # insecure_localhost: never bind a public interface without TLS.
        host = "127.0.0.1"
        _log.warning(
            "Model Manager running WITHOUT TLS on loopback %s:%s (dev only — do not "
            "expose). Set MODEL_MANAGER_TLS_MODE=mutual with certs for real use.",
            host,
            port,
        )
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
