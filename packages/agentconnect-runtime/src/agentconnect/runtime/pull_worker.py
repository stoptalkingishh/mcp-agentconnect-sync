"""Worker-side pull loop — the program a compute contributor runs on their box.

This is the client half of the federation. The broker (``add_pull_routes``)
brokers tickets but never executes; the compute happens HERE, on the
contributor's machine, with a LOCAL runtime. The loop:

  1. claims work its attested tier is authorized for (``GET /queue/next``),
     receiving the redacted task body inline,
  2. runs it through its own ``AgentRuntime`` (LangGraph act/tool loop),
  3. reports the ``WorkerResult`` back under the lease (``POST .../report``).

Identity — and therefore which privacy classes it may claim — is the mTLS
client certificate the httpx client presents (``TlsClientConfig`` mode
``mutual``); the broker derives the tier from the cert, never from anything the
worker sends. ``identity_headers`` exists only for the header-stripping-proxy /
TestClient seam (matches ``add_pull_routes(trust_proxy_headers=True)``); a real
deployment leaves it empty and authenticates by cert.

Crash safety is the broker's: if this worker dies mid-task, its lease expires
and the reaper requeues the ticket (attempts++). A task that outlives its lease
should ``heartbeat`` to renew; otherwise its ``report`` returns ``lease_lost``
(refused, not corrupting) and the reaper will have requeued it.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable, Optional

from agentconnect.common.schemas import TaskSubmission, WorkerResult

from .agent import AgentRuntime


class PullWorker:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        base_url: Optional[str] = None,
        tls: Optional[Any] = None,  # TlsClientConfig
        client: Optional[Any] = None,  # test seam: starlette TestClient / injected httpx.Client
        capabilities: Iterable[str] = (),
        identity_headers: Optional[dict[str, str]] = None,
        poll_interval: float = 2.0,
        heartbeat_interval: float = 0.0,
        heartbeat_extend_seconds: int = 120,
        timeout: float = 600.0,
        task_id_prefix: str = "pull",
    ):
        self.runtime = runtime
        self.capabilities = list(capabilities)
        self.poll_interval = poll_interval
        # >0 renews the lease every `heartbeat_interval`s while a task runs, so a
        # task that outlives its lease is not reaped mid-flight. 0 disables it.
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_extend_seconds = heartbeat_extend_seconds
        self._headers = dict(identity_headers or {})
        self._prefix = task_id_prefix
        if client is not None:
            self._client = client
            return
        import httpx

        from agentconnect.common.config import client_ssl_context

        base = (base_url or "").rstrip("/")
        ctx = client_ssl_context(tls)
        if ctx is not None:
            self._client = httpx.Client(base_url=base, verify=ctx, timeout=timeout)
        else:
            # insecure_localhost / no TLS material — plain HTTP, loopback only.
            self._client = httpx.Client(base_url=base, timeout=timeout)

    # ------------------------------------------------------------- broker calls
    def claim(self, max_tickets: int = 1) -> list[dict[str, Any]]:
        r = self._client.get(
            "/queue/next",
            params={"capabilities": ",".join(self.capabilities), "max": max_tickets},
            headers=self._headers,
        )
        r.raise_for_status()
        return r.json().get("tickets", [])

    def heartbeat(self, ticket_id: str, lease_token: str) -> dict[str, Any]:
        r = self._client.post(
            f"/queue/{ticket_id}/heartbeat",
            json={"lease_token": lease_token, "extend_seconds": self.heartbeat_extend_seconds},
            headers=self._headers,
        )
        r.raise_for_status()
        return r.json()

    def report(self, ticket_id: str, lease_token: str, result: WorkerResult) -> dict[str, Any]:
        r = self._client.post(
            f"/queue/{ticket_id}/report",
            json={
                "lease_token": lease_token,
                "status": result.status,
                "summary": result.summary,
                "confidence": result.confidence,
                "changed_artifacts": list(result.changed_artifacts),
                "evidence_refs": list(result.evidence_refs),
                "risks": list(result.risks),
                "recommended_next_action": result.recommended_next_action,
            },
            headers=self._headers,
        )
        r.raise_for_status()
        return r.json()

    # --------------------------------------------------------------- execution
    def execute(self, ticket: dict[str, Any]) -> WorkerResult:
        """Run one claimed ticket's redacted payload through the local runtime."""
        submission = TaskSubmission(task=ticket.get("payload", "") or "")
        return self.runtime.run(submission, task_id=f"{self._prefix}_{ticket['ticket_id']}")

    def _execute_with_heartbeat(self, ticket: dict[str, Any]) -> WorkerResult:
        """Run the task while a background thread renews the lease, so a slow run
        is not reaped and re-handed to another worker mid-flight. Heartbeat
        failures (e.g. the lease was already lost) are swallowed here — ``report``
        is the authoritative fence and will surface ``lease_lost`` if so."""
        if self.heartbeat_interval <= 0:
            return self.execute(ticket)
        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(self.heartbeat_interval):
                try:
                    self.heartbeat(ticket["ticket_id"], ticket["lease_token"])
                except Exception:  # transport hiccup or lost lease — report() decides.
                    pass

        beater = threading.Thread(target=_beat, name="pull-heartbeat", daemon=True)
        beater.start()
        try:
            return self.execute(ticket)
        finally:
            stop.set()
            beater.join(timeout=self.heartbeat_interval + 1.0)

    def run_once(self) -> Optional[dict[str, Any]]:
        """Claim → execute → report a single ticket. Returns the outcome, or
        ``None`` when nothing is authorized/available (the caller should back off)."""
        tickets = self.claim(max_tickets=1)
        if not tickets:
            return None
        ticket = tickets[0]
        # Refuse to "run" a ticket whose redacted body never arrived (delivery
        # error, or a null payload): executing an empty task would report a bogus
        # success. Report a failure instead so the broker requeues it (attempts
        # remaining) or fails it terminally — never a silent empty completion.
        if ticket.get("payload_error") or ticket.get("payload") is None:
            reason = ticket.get("payload_error", "payload_missing")
            result = WorkerResult(status="failed", summary=f"payload not delivered: {reason}")
            outcome = self.report(ticket["ticket_id"], ticket["lease_token"], result)
            return {"ticket_id": ticket["ticket_id"], "result": result, "report": outcome}
        result = self._execute_with_heartbeat(ticket)
        outcome = self.report(ticket["ticket_id"], ticket["lease_token"], result)
        return {"ticket_id": ticket["ticket_id"], "result": result, "report": outcome}

    def run_forever(
        self,
        *,
        max_iterations: Optional[int] = None,
        stop: Optional[Callable[[], bool]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> int:
        """Poll-claim-run-report until ``stop()`` is true or ``max_iterations``
        claim attempts are made. Sleeps ``poll_interval`` on an empty claim.
        Returns the number of tickets actually processed."""
        processed = 0
        attempts = 0
        while max_iterations is None or attempts < max_iterations:
            if stop is not None and stop():
                break
            outcome = self.run_once()
            attempts += 1
            if outcome is None:
                sleep(self.poll_interval)
            else:
                processed += 1
        return processed
