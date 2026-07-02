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
import os
from pathlib import Path
from typing import Any, Optional

from ..common.memory import SharedMemory
from ..common.schemas import TaskConstraints, TaskSubmission
from .service import RouterService


def _default_memory() -> SharedMemory:
    db = os.environ.get("AGENTCONNECT_DB", "")
    if db:
        return SharedMemory(db)
    data_dir = Path(os.environ.get("AGENTCONNECT_DATA_DIR", "data"))
    return SharedMemory(data_dir / "shared_memory.sqlite")


def _build_service() -> RouterService:
    """Wire the router. If a Local Model Manager endpoint is configured we use the
    HTTP client; otherwise we fall back to an in-process manager so the server is
    usable on a single box out of the box."""
    local_client = None
    manager_url = os.environ.get("MODEL_MANAGER_URL")
    if manager_url:
        from .local_client import HttpLocalClient

        token = os.environ.get("LOCAL_R9700_API_TOKEN")
        local_client = HttpLocalClient(manager_url, token=token)
    else:
        from ..model_manager.residency import ResidencyManager
        from .local_client import InProcessLocalClient

        local_client = InProcessLocalClient(ResidencyManager())
    return RouterService.create(memory=_default_memory(), local_client=local_client)


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
        priority: str = "normal",
        quality: str = "standard",
        max_output_tokens: Optional[int] = None,
        refs: Optional[list[str]] = None,
    ) -> str:
        """Submit a task for routing. Returns a COMPACT summary + artifact refs
        (never full output). Use read_artifact_chunk / get_log_slice for detail."""
        constraints = TaskConstraints(
            profile=profile,
            privacy_class=privacy_class,  # type: ignore[arg-type]
            allow_external=allow_external,
            allow_paid=allow_paid,
            priority=priority,  # type: ignore[arg-type]
            quality=quality,
            max_output_tokens=max_output_tokens,
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
