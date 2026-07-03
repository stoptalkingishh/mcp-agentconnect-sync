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

from ..common.config import TlsClientConfig
from ..common.memory import SharedMemory
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


def build_mcp_server(service: Optional[RouterService] = None):
    """Construct the FastMCP server bound to a RouterService."""
    from mcp.server.fastmcp import FastMCP

    svc = service or _build_service()
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

    return mcp


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> None:
    build_mcp_server().run()


if __name__ == "__main__":
    main()
