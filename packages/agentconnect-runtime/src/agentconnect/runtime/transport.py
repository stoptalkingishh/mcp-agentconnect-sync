"""Worker-over-HTTP transport: serve an ``AgentRuntime``, and call one remotely.

The app factory configures **no TLS** — transport authentication is mutual TLS
at the server launcher (uvicorn ``ssl_cert_reqs=CERT_REQUIRED`` +
``ssl_ca_certs``, see ``agentconnect.model_manager.tls.build_ssl_kwargs``).
Never bind a non-loopback interface without it; there is no shared secret on
this wire ever — identity is the certificate.

``RuntimeConfig`` is worker-side only: the wire carries ``task_id`` plus
``TaskSubmission`` and nothing else, so a router can never relax
``allow_shell``/``allow_tests``/``allow_browser`` or workspace policy remotely.
Workspace confinement, observation truncation, and the ERROR-string
conventions all run server-side inside the runtime exactly as they do locally.

Capacity is a lock-guarded counter because FastAPI sync endpoints run in a
threadpool. fastapi and httpx are imported lazily (the ``remote`` extra), so
importing this module — and re-exporting from the package — needs neither.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from pydantic import BaseModel

from agentconnect.common.schemas import CanAcceptResponse, TaskSubmission, WorkerResult

if TYPE_CHECKING:
    import httpx
    from fastapi import FastAPI

    from agentconnect.common.config import TlsClientConfig
    from agentconnect.common.workqueue import WorkQueue

    from .agent import AgentRuntime


@dataclass(frozen=True)
class RuntimeEndpoint:
    kind: str  # local | http
    address: str


class RunTaskRequest(BaseModel):
    """Wire contract for POST /run. Defined here rather than in core schemas:
    both ends of this wire live in this module (precedent: approval_web's
    BudgetBody)."""

    task_id: str
    submission: TaskSubmission


def create_worker_app(
    runtime: "AgentRuntime",
    *,
    worker_id: str = "worker_local",
    max_concurrent_tasks: int = 1,
    allowed_clients: Optional[set[str]] = None,
) -> "FastAPI":
    """FastAPI app serving ``runtime``: ``POST /run`` and ``GET /can_accept``.

    ``max_concurrent_tasks`` defaults to 1 — ``LangGraphAgentRuntime`` is
    documented single-task; raising it requires a thread-safe model source.
    Each ``run()`` creates and cleans a fresh workspace, so one runtime
    instance serves many sequential tasks. ``allowed_clients`` mounts the same
    defense-in-depth identity allowlist as ``model_manager.app.create_app``.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"AgentConnect Worker {worker_id}", version="0.1.0")
    if allowed_clients:
        from agentconnect.common.asgi_identity import ClientIdentityMiddleware

        app.add_middleware(ClientIdentityMiddleware, allowed=allowed_clients)

    lock = threading.Lock()
    running = [0]

    @app.post("/run")
    def run_task(req: RunTaskRequest) -> dict:
        with lock:
            if running[0] >= max_concurrent_tasks:
                raise HTTPException(status_code=503, detail="worker at capacity")
            running[0] += 1
        try:
            try:
                result = runtime.run(req.submission, task_id=req.task_id)
            except Exception as exc:  # noqa: BLE001 — the endpoint stays total:
                # TaskSubmission in -> WorkerResult out; a completed exchange
                # carrying a failed contract beats a 500 for the router.
                result = WorkerResult(
                    status="failed",
                    summary=f"ERROR: worker exception: {exc}"[:400],
                    confidence=0.0,
                    risks=["worker_exception"],
                )
            return result.model_dump()
        finally:
            with lock:
                running[0] -= 1

    @app.get("/can_accept")
    def can_accept() -> dict:
        with lock:
            busy = running[0] >= max_concurrent_tasks
        if busy:
            return CanAcceptResponse(can_accept=False, reason="worker at capacity").model_dump()
        return CanAcceptResponse(can_accept=True).model_dump()

    return app


class QueueReportBody(BaseModel):
    """Wire body for POST /queue/{ticket_id}/report: the fencing token plus the
    same fields as ``WorkerResult`` (defined here, not core, mirroring
    ``RunTaskRequest`` — both ends of this wire live in this module)."""

    lease_token: str
    status: str = "completed"
    summary: str = ""
    confidence: float = 0.0
    changed_artifacts: list[str] = []
    evidence_refs: list[str] = []
    risks: list[str] = []
    recommended_next_action: Optional[str] = None


class QueueHeartbeatBody(BaseModel):
    lease_token: str
    extend_seconds: int = 120


def add_pull_routes(
    app: "FastAPI",
    queue: Optional["WorkQueue"],
    tier_resolver: Callable[[str], Optional[str]],
    *,
    trust_proxy_headers: bool = False,
) -> "FastAPI":
    """Mount the federated pull surface on ``app``: ``GET /queue/next``,
    ``POST /queue/{ticket_id}/report``, ``POST /queue/{ticket_id}/heartbeat``.

    Additive and trimmable: if ``queue`` is None, nothing is mounted and
    ``app`` is returned unchanged — the core worker app (``/run``,
    ``/can_accept``) is untouched by this function's existence.

    Identity, never the request body, decides the trust tier — and that tier is
    the SOLE gate on which privacy_class a caller may claim and whether its
    report is auto-accepted. Because identity IS authorization here, the trust
    anchor must be real: the identity is taken from the ASGI-TLS extension (the
    terminating server's view of the client certificate, not settable by the
    remote peer). The client-settable ``X-Client-Cert-DN`` / ``X-SPIFFE-ID``
    header is trusted ONLY when the operator passes ``trust_proxy_headers=True``,
    asserting that a header-stripping, mTLS-terminating reverse proxy sits in
    front (it overwrites, never forwards, those headers). With the default
    ``trust_proxy_headers=False`` a header-only identity is treated as no
    identity -> 403, so a peer connecting directly over plain HTTP cannot spoof a
    trusted tier. No identity, or an identity ``tier_resolver`` does not
    recognize, is a 403 before any queue call — fail-closed, same contract as
    ``queue_next``'s ``unknown_worker_identity`` on the MCP surface. The body
    never carries a tier (mirrors ``RunTaskRequest`` carrying only task content,
    never a policy override).

    Business-logic outcomes (``lease_lost``, ``not_authorized``, ...) are the
    ``WorkQueue`` methods' own typed dict returns, passed through as 200 JSON —
    never a raw 500. Only identity/authorization failures raise HTTP errors.
    """
    if queue is None:
        return app

    from fastapi import HTTPException, Request

    # ``from __future__ import annotations`` makes every annotation below a
    # string; FastAPI resolves it via ``typing.get_type_hints(fn)`` against
    # the route functions' ``__globals__`` (this module's globals), not this
    # factory's locals. Publish the lazily-imported names there so `Request`
    # is recognized as the special inject-the-request type, not a query param.
    globals().setdefault("Request", Request)
    globals().setdefault("HTTPException", HTTPException)

    from agentconnect.common.asgi_identity import (
        _forwarded_header_identity,
        _tls_extension_identity,
    )

    def _identity_and_tier(request: "Request") -> tuple[str, str]:
        # The certificate as surfaced by the TLS-terminating server is always a
        # trustworthy anchor. A plain proxy header is trusted only under the
        # operator's explicit trust_proxy_headers opt-in (header-stripping proxy).
        identity = _tls_extension_identity(request.scope)
        if identity is None and trust_proxy_headers:
            identity = _forwarded_header_identity(request.scope)
        if identity is None:
            raise HTTPException(status_code=403, detail="no_client_identity")
        tier = tier_resolver(identity)
        if tier is None:
            raise HTTPException(status_code=403, detail="unknown_or_unauthorized_identity")
        return identity, tier

    @app.get("/queue/next")
    def queue_next(request: Request, capabilities: str = "", max: int = 1) -> dict:
        identity, tier = _identity_and_tier(request)
        caps = [c for c in capabilities.split(",") if c]
        tickets = queue.claim_next(identity, tier, capabilities=caps, max=max)
        # Deliver the redacted body inline so a remote worker (which has no access
        # to the broker's artifact store) can actually run the task in one round
        # trip. Lease-gated resolution: the token was just minted for this holder.
        # If delivery fails (e.g. a lease race between claim and this call), surface
        # the error explicitly rather than handing back an empty payload the worker
        # cannot distinguish from a genuinely empty task — see PullWorker.run_once.
        for t in tickets:
            resolved = queue.payload_for(identity, t["ticket_id"], t["lease_token"], tier)
            if "error" in resolved:
                t["payload"] = None
                t["payload_error"] = resolved["error"]
            else:
                t["payload"] = resolved.get("payload", "")
        return {"tickets": tickets}

    @app.get("/queue/{ticket_id}/payload")
    def queue_payload(ticket_id: str, request: Request, lease_token: str = "") -> dict:
        # Standalone re-fetch of the redacted body (e.g. after a worker restart
        # that kept the lease). Same authorized, lease-gated seam as the inline
        # delivery on /queue/next.
        identity, tier = _identity_and_tier(request)
        return queue.payload_for(identity, ticket_id, lease_token, tier)

    @app.post("/queue/{ticket_id}/report")
    def queue_report(ticket_id: str, body: QueueReportBody, request: Request) -> dict:
        identity, tier = _identity_and_tier(request)
        result = WorkerResult(
            status=body.status,
            summary=body.summary,
            confidence=body.confidence,
            changed_artifacts=body.changed_artifacts,
            evidence_refs=body.evidence_refs,
            risks=body.risks,
            recommended_next_action=body.recommended_next_action,
        )
        return queue.report(identity, tier, ticket_id, body.lease_token, result)

    @app.post("/queue/{ticket_id}/heartbeat")
    def queue_heartbeat(ticket_id: str, body: QueueHeartbeatBody, request: Request) -> dict:
        identity, _tier = _identity_and_tier(request)
        return queue.renew(
            identity, ticket_id, body.lease_token, extend_seconds=body.extend_seconds
        )

    return app


class HttpAgentRuntime:
    """``AgentRuntime`` over HTTP. Authentication is the X.509 client
    certificate (``TlsClientConfig``, ``mode="mutual"``); plain HTTP is for
    ``insecure_localhost`` and injected test clients only.

    Transport/HTTP failures raise (httpx exceptions propagate), matching
    ``HttpLocalClient`` — the router's dispatch path already converts dispatch
    exceptions to FAILED.
    """

    def __init__(
        self,
        base_url: str,
        *,
        tls: Optional["TlsClientConfig"] = None,
        timeout: float = 600.0,  # runs can take max_steps * shell_timeout
        client: Optional["httpx.Client"] = None,  # test seam: starlette TestClient
    ):
        if client is not None:
            self._client = client
            return
        import httpx

        from agentconnect.common.config import client_ssl_context

        base = base_url.rstrip("/")
        ctx = client_ssl_context(tls)
        if ctx is not None:
            self._client = httpx.Client(base_url=base, verify=ctx, timeout=timeout)
        else:
            # insecure_localhost / no TLS material — plain HTTP, loopback only.
            self._client = httpx.Client(base_url=base, timeout=timeout)

    def run(self, task: TaskSubmission, task_id: str = "task_remote") -> WorkerResult:
        r = self._client.post(
            "/run", json={"task_id": task_id, "submission": task.model_dump(mode="json")}
        )
        r.raise_for_status()
        return WorkerResult.model_validate(r.json())

    def can_accept(self) -> CanAcceptResponse:
        r = self._client.get("/can_accept")
        r.raise_for_status()
        return CanAcceptResponse.model_validate(r.json())
