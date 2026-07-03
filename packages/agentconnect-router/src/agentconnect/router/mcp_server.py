"""Agent Router MCP server (handoff §4.1, §23).

Exposes the router as MCP tools so Claude Code (or another manager agent) can
delegate work. The manager talks to THIS server, not to model providers directly.

Tools return compact summaries + artifact references; large data is read back in
bounded chunks via ``read_artifact_chunk`` / ``get_log_slice`` (context
virtualization, §9). The MCP SDK is imported lazily so the core stays framework
free.

Run with:  ``agentconnect-router``  (stdio transport)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from ..common import privacy as privacy_mod
from ..common.config import TlsClientConfig, load_workers
from ..common.memory import SharedMemory
from ..common.privacy import ClassificationHints
from ..common.schemas import TaskConstraints, TaskSubmission
from .service import RouterService

_log = logging.getLogger(__name__)


def _default_memory() -> SharedMemory:
    db = os.environ.get("AGENTCONNECT_DB", "")
    if db:
        return SharedMemory(db)
    data_dir = Path(os.environ.get("AGENTCONNECT_DATA_DIR", "data"))
    return SharedMemory(data_dir / "shared_memory.sqlite")


def _local_tls_from_env() -> Optional[TlsClientConfig]:
    """Build the router's client-side mTLS config from env, if provided."""
    mode = os.environ.get("AGENTCONNECT_LOCAL_TLS_MODE", "mutual")
    ca = os.environ.get("AGENTCONNECT_LOCAL_CA")
    cert = os.environ.get("AGENTCONNECT_LOCAL_CLIENT_CERT")
    key = os.environ.get("AGENTCONNECT_LOCAL_CLIENT_KEY")
    if mode == "insecure_localhost":
        return TlsClientConfig(mode="insecure_localhost")
    if not (ca or cert or key):
        return None  # no material configured; HttpLocalClient falls back to plain HTTP
    return TlsClientConfig(
        mode="mutual", ca_cert=ca, client_cert=cert, client_key=key,
        server_name=os.environ.get("AGENTCONNECT_LOCAL_SERVER_NAME"),
    )


def _try_embedded_manager():
    """Soft-optional: embed an in-process Model Manager for single-box convenience.

    The router library never hard-imports the manager; this bootstrap is the ONLY
    place that references it, guarded by try/except so the router runs fine when
    ``agentconnect-model-manager`` is not installed."""
    try:
        from ..model_manager.residency import ResidencyManager  # optional dependency
        from .local_client import InProcessLocalClient
    except ImportError:
        _log.info(
            "agentconnect-model-manager is not installed; running cloud-only "
            "standalone. Local-only/repo-sensitive tasks will report no local node. "
            "Install the 'embedded' extra or set MODEL_MANAGER_URL to add local inference."
        )
        return None
    return InProcessLocalClient(ResidencyManager())


def _build_service() -> RouterService:
    """Wire the router.

    Local inference is optional and never a hard dependency:
      * MODEL_MANAGER_URL set   -> talk to a remote manager over mutual TLS
      * else manager installed  -> embed an in-process manager (single-box convenience)
      * else                    -> cloud-only standalone (the Router is the product)
    """
    manager_url = os.environ.get("MODEL_MANAGER_URL")
    if manager_url:
        from .local_client import HttpLocalClient

        local_client = HttpLocalClient(manager_url, tls=_local_tls_from_env())
    else:
        local_client = _try_embedded_manager()
    return RouterService.create(
        memory=_default_memory(), local_client=local_client, authorizer=_spend_authorizer_from_env()
    )


def _spend_authorizer_from_env():
    """Select the direct-to-user spend authorizer (deterministic money gate).

    Default DENY (fail-closed) — a real deployment should wire a CallbackSpendAuthorizer
    to its own user-facing confirmation UI. `console` prompts on the terminal (only safe
    when NOT running the MCP stdio server on that terminal); `auto` approves everything
    (trusted automation only)."""
    mode = os.environ.get("AGENTCONNECT_SPEND_AUTHORIZER", "deny").lower()
    if mode == "auto":
        from ..common.authorization import AutoApproveSpendAuthorizer

        return AutoApproveSpendAuthorizer()
    if mode == "console":
        from ..common.authorization import ConsoleSpendAuthorizer

        return ConsoleSpendAuthorizer()
    if mode == "web":
        try:
            from ..common.approval import ApprovalQueue, WebApprovalAuthorizer
            from .approval_web import start_web_approval
        except ImportError as exc:
            _log.warning(
                "AGENTCONNECT_SPEND_AUTHORIZER=web but the 'web' extra is not installed "
                "(%s); falling back to deny. Install agentconnect-router[web].", exc,
            )
            from ..common.authorization import DenyingSpendAuthorizer

            return DenyingSpendAuthorizer()
        host = os.environ.get("AGENTCONNECT_APPROVAL_HOST", "127.0.0.1")
        port = int(os.environ.get("AGENTCONNECT_APPROVAL_PORT", "8770"))
        base_url = os.environ.get("AGENTCONNECT_APPROVAL_URL", f"http://{host}:{port}")
        token = os.environ.get("AGENTCONNECT_APPROVAL_TOKEN")
        timeout = float(os.environ.get("AGENTCONNECT_APPROVAL_TIMEOUT", "300"))
        from ..common.notifiers import notifier_from_env

        queue = ApprovalQueue()
        start_web_approval(queue, host=host, port=port, token=token)
        return WebApprovalAuthorizer(
            queue, base_url=base_url, notifier=notifier_from_env(), timeout_seconds=timeout,
        )
    from ..common.authorization import DenyingSpendAuthorizer

    return DenyingSpendAuthorizer()


def _load_worker_tiers() -> dict[str, str]:
    """Worker/reviewer identity -> ATTESTED tier map for the queue_* MCP tools.

    ``AGENTCONNECT_WORKER_TIERS`` (a JSON object) wins when set, for quick local
    wiring/tests; otherwise ``config/workers.yaml``. This is the ONLY source of
    truth for a caller's tier on this surface — the queue_* tools never accept a
    tier in the request body. An identity absent here resolves to ``None``,
    which the work-queue treats as an empty admissible-class set: fail-closed,
    never silently granted access.
    """
    override = os.environ.get("AGENTCONNECT_WORKER_TIERS")
    if override:
        try:
            return {str(k): str(v) for k, v in json.loads(override).items()}
        except (ValueError, AttributeError, TypeError) as exc:
            _log.warning("AGENTCONNECT_WORKER_TIERS is not a valid JSON object (%s); ignoring", exc)
    return load_workers()


def build_mcp_server(service: Optional[RouterService] = None, worker_tiers: Optional[dict[str, str]] = None):
    """Construct the FastMCP server bound to a RouterService."""
    from mcp.server.fastmcp import FastMCP

    svc = service or _build_service()
    workers = worker_tiers if worker_tiers is not None else _load_worker_tiers()
    mcp = FastMCP("agentconnect-router")

    @mcp.tool()
    def submit_task(
        task: str,
        agent_type: Optional[str] = None,
        profile: Optional[str] = None,
        privacy_class: Optional[str] = None,
        allow_external: bool = True,
        allow_paid: bool = False,
        allow_rented: bool = False,
        priority: str = "normal",
        quality: str = "standard",
        max_output_tokens: Optional[int] = None,
        execution: str = "oneshot",
        refs: Optional[list[str]] = None,
    ) -> str:
        """Submit a task for routing. Returns a COMPACT summary + artifact refs
        (never full output). Use read_artifact_chunk / get_log_slice for detail.

        Set allow_rented=True to permit a repo_sensitive task to run on a trusted
        rented GPU node (for very large private models).

        Set execution="agentic" to run the task through the worker runtime's
        act/tool loop (filesystem/shell/tests/browser tools in a confined
        workspace) instead of a single generation. Agentic runs local-only."""
        constraints = TaskConstraints(
            profile=profile,
            privacy_class=privacy_class,  # type: ignore[arg-type]
            allow_external=allow_external,
            allow_paid=allow_paid,
            allow_rented=allow_rented,
            priority=priority,  # type: ignore[arg-type]
            quality=quality,
            max_output_tokens=max_output_tokens,
            execution=execution,
        )
        submission = TaskSubmission(
            task=task, agent_type=agent_type, constraints=constraints, refs=refs or []
        )
        return _json(svc.submit_task(submission).model_dump(mode="json"))

    @mcp.tool()
    def get_task_status(task_id: str) -> str:
        """Get a task's compact status summary + artifact references."""
        res = svc.get_task_status(task_id)
        return _json(res.model_dump(mode="json") if res else {"error": "task_not_found"})

    @mcp.tool()
    def get_task_artifacts(task_id: str) -> str:
        """List a task's artifacts as {kind: artifact_id}."""
        return _json(svc.get_task_artifacts(task_id))

    @mcp.tool()
    def read_artifact_chunk(artifact_id: str, offset: int = 0, max_chars: int = 4000) -> str:
        """Read a bounded chunk of an artifact. Follow `next_offset` to page."""
        return _json(svc.read_artifact_chunk(artifact_id, offset, max_chars))

    @mcp.tool()
    def get_log_slice(
        task_id: str, level: Optional[str] = None, query: Optional[str] = None, max_lines: int = 100
    ) -> str:
        """Read a bounded slice of a task's logs (optionally filtered)."""
        return _json(svc.get_log_slice(task_id, level=level, query=query, max_lines=max_lines))

    @mcp.tool()
    def search_memory(query: str, scope: str = "all", limit: int = 20) -> str:
        """Substring-search shared memory (tasks/artifacts/logs). Returns snippets."""
        return _json(svc.search_memory(query, scope=scope, limit=limit))

    @mcp.tool()
    def get_router_status() -> str:
        """Router policy version, providers, live local-manager status, output policy."""
        return _json(svc.get_router_status())

    @mcp.tool()
    def get_provider_status() -> str:
        """Per-provider type/privacy/health/capabilities and remaining quota."""
        return _json(svc.get_provider_status())

    @mcp.tool()
    def get_provider_scorecards() -> str:
        """Learned per-provider scorecards (success rate, latency, cost, sample
        count) and the current learned-quality signal folded into routing (Phase 6)."""
        return _json(svc.get_provider_scorecards())

    @mcp.tool()
    def set_budget(amount_usd: float, period: str = "monthly") -> str:
        """Set the global spend budget: a dollar amount over a period
        ('daily' | 'weekly' | 'monthly'). The router paces spend to it and hard-stops
        paid/rented routes when it is exhausted. This is the ONLY way to set it — there
        is no default. Returns the resulting budget status."""
        return _json(svc.set_budget(amount_usd, period))

    @mcp.tool()
    def get_budget_status() -> str:
        """Current spend-budget status: amount, period, window, spent, remaining,
        pace, projection, and on_track. If it reports configured=false with
        action_required='set_budget', you MUST ask the user for a budget (amount +
        period) and call set_budget before any paid-cloud or rented-GPU work — the
        router keeps those disabled until then."""
        return _json(svc.get_budget_status())

    @mcp.tool()
    def promote_task(task_id: str) -> str:
        """Raise a task's priority to urgent."""
        return _json(svc.promote_task(task_id))

    @mcp.tool()
    def cancel_task(task_id: str) -> str:
        """Cancel a non-terminal task."""
        return _json(svc.cancel_task(task_id))

    # ---------------------------------------------------------- federated queue
    def _wq():
        if svc.workqueue is None:
            return None
        return svc.workqueue

    @mcp.tool()
    def queue_add(
        task: str,
        agent_type: Optional[str] = None,
        privacy_class: Optional[str] = None,
        required_capabilities: Optional[list[str]] = None,
        priority: str = "normal",
        dedup_key: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        refs: Optional[list[str]] = None,
    ) -> str:
        """Enqueue a ticket on the pull-based federated work queue.

        Classifies + redacts `task` exactly like submit_task: a claimer only
        ever receives the redacted, worker-visible payload, never the raw text.
        secret_sensitive (or anything require_redaction blocks that isn't
        cloud_safe) is stored 'parked' and is never leasable by any tier.
        Idempotent on dedup_key: re-adding an existing key returns the existing
        ticket unchanged (a done ticket is never reopened)."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        hints = ClassificationHints(file_paths=tuple(refs or []), declared=privacy_class)
        pc = privacy_mod.classify(task, hints)
        redaction, redacted_text = privacy_mod.redact(task, pc)
        ticket = wq.add(
            task=task,
            origin=f"mcp:{agent_type or 'unknown'}",
            privacy_class=pc,
            payload=redacted_text,
            required_capabilities=required_capabilities,
            priority=priority,
            dedup_key=dedup_key,
            depends_on=depends_on,
            cloud_safe=redaction.cloud_safe,
        )
        return _json(ticket)

    @mcp.tool()
    def queue_next(worker_id: str, capabilities: Optional[list[str]] = None, max: int = 1) -> str:
        """Atomically claim up to `max` tickets this worker is authorized to see:
        `worker_id`'s tier is resolved ONLY from the server's attested
        identity->tier map (never from any tier a caller might pass), and a
        ticket's privacy_class must admit that tier live from routing.yaml. An
        unknown worker_id is refused; returns {"tickets": []} when nothing is
        authorized/available/unblocked."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        tier = workers.get(worker_id)
        if tier is None:
            return _json({"error": "unknown_worker_identity"})
        got = wq.claim_next(worker_id, tier, capabilities=capabilities, max=max)
        return _json({"tickets": got})

    @mcp.tool()
    def queue_claim(worker_id: str, ticket_id: str) -> str:
        """Targeted atomic claim of one ticket, gated by the same attested
        tier x privacy_class authorization as queue_next."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        tier = workers.get(worker_id)
        if tier is None:
            return _json({"error": "unknown_worker_identity"})
        return _json(wq.claim(worker_id, tier, ticket_id))

    @mcp.tool()
    def queue_update(worker_id: str, ticket_id: str, lease_token: str, extend_seconds: int = 120) -> str:
        """Heartbeat/renew a held lease. Refused if `worker_id`/`lease_token`
        doesn't match the current lease holder or the lease already expired."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        return _json(wq.renew(worker_id, ticket_id, lease_token, extend_seconds=extend_seconds))

    @mcp.tool()
    def queue_report(
        worker_id: str,
        ticket_id: str,
        lease_token: str,
        status: str = "completed",
        summary: str = "",
        confidence: float = 0.0,
        changed_artifacts: Optional[list[str]] = None,
        risks: Optional[list[str]] = None,
    ) -> str:
        """Report a result under the fencing lease_token. Idempotent: a second
        report (or a stale token after a reaper requeue+reclaim) is refused.
        Only a local_only-attested worker's report auto-accepts (done); every
        other tier lands in_review/pending until a local_only reviewer calls
        queue_approve — a federated report can never silently become truth."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        tier = workers.get(worker_id)
        if tier is None:
            return _json({"error": "unknown_worker_identity"})
        result = {
            "status": status, "summary": summary, "confidence": confidence,
            "changed_artifacts": changed_artifacts or [], "risks": risks or [],
        }
        return _json(wq.report(worker_id, tier, ticket_id, lease_token, result))

    @mcp.tool()
    def queue_link(ticket_id: str, depends_on: str) -> str:
        """Add a dependency edge. Enforces privacy monotonicity: the child's
        admissible-tier set must be a subset of the parent's, so sensitive
        output cannot be laundered down to a lower class through the edge."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        return _json(wq.link(ticket_id, depends_on))

    @mcp.tool()
    def queue_status(
        ticket_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> str:
        """Redacted ticket rows for audit/UX — NEVER the payload content itself,
        only its artifact ref and queue metadata. 'blocked' is derived (an open
        ticket with an unsatisfied dependency), never a stored state."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        return _json(wq.status(ticket_id=ticket_id, status=status, limit=limit))

    @mcp.tool()
    def queue_approve(reviewer_id: str, ticket_id: str) -> str:
        """Promote an in_review ticket to done/approved. The reviewer's tier is
        resolved from the same attested identity map and must be local_only —
        the human-spot-check analogue for federated results."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        tier = workers.get(reviewer_id)
        if tier is None:
            return _json({"error": "unknown_worker_identity"})
        return _json(wq.approve(reviewer_id, tier, ticket_id))

    @mcp.tool()
    def queue_reject(reviewer_id: str, ticket_id: str, reason: str = "") -> str:
        """Reject an in_review ticket (local_only reviewer only). Requeues for
        another attempt while attempts remain, otherwise the ticket (and any
        linked task) fails."""
        wq = _wq()
        if wq is None:
            return _json({"error": "workqueue_unavailable"})
        tier = workers.get(reviewer_id)
        if tier is None:
            return _json({"error": "unknown_worker_identity"})
        return _json(wq.reject(reviewer_id, tier, ticket_id, reason=reason))

    @mcp.tool()
    def enqueue_task(
        task: str,
        agent_type: Optional[str] = None,
        privacy_class: Optional[str] = None,
        allow_external: bool = True,
        allow_paid: bool = False,
        allow_rented: bool = False,
        priority: str = "normal",
        quality: str = "standard",
        required_capabilities: Optional[list[str]] = None,
        dedup_key: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        refs: Optional[list[str]] = None,
    ) -> str:
        """Router-as-assigner tie-in: submit a task through the full
        classify/redact pipeline AND place it on the pull work-queue in one
        call, with a routing decision recorded as an ADVISORY assignee hint.
        The hint never gates a claim — pull workers still self-select within
        their own attested-tier authorization via queue_next/queue_claim."""
        constraints = TaskConstraints(
            profile=None,
            privacy_class=privacy_class,  # type: ignore[arg-type]
            allow_external=allow_external,
            allow_paid=allow_paid,
            allow_rented=allow_rented,
            priority=priority,  # type: ignore[arg-type]
            quality=quality,
        )
        submission = TaskSubmission(
            task=task, agent_type=agent_type, constraints=constraints, refs=refs or []
        )
        ticket = svc.enqueue_task(
            submission,
            dedup_key=dedup_key,
            required_capabilities=required_capabilities,
            priority=priority,
            depends_on=depends_on,
        )
        return _json(ticket)

    return mcp


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> None:
    build_mcp_server().run()


if __name__ == "__main__":
    main()
