"""AgentConnect MCP server — the manager-facing adapter (spec §13).

Exactly the thirteen tools a *manager* needs. Administration (approve/deny spend,
inbox drain, Linear sync, task listing) deliberately lives on the HTTP API and
CLI: a manager agent should not be able to approve its own spend.

Run with ``agentconnect-mcp`` (stdio), or set ``AGENTCONNECT_MCP_TRANSPORT`` to
``sse``/``streamable-http`` to serve many harnesses from one process — point them
all at the same ``AGENTCONNECT_DB_PATH`` and they share one ledger.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agentconnect.core.bootstrap import service_from_env
from agentconnect.core.errors import AgentConnectError, InvalidRequest
from agentconnect.core.memory import (
    CaptureRequest,
    MemoryFeedbackRequest,
    MemoryScope,
    RecallRequest,
)
from agentconnect.core.models import (
    ActorType,
    ClaimRole,
    CreateTaskRequest,
    FilesystemAccess,
    Priority,
    PrivacyTier,
    RecordAttemptRequest,
    RecordDecisionRequest,
    ReviewRequest,
    SandboxSpec,
    SubtaskRequest,
)
from agentconnect.core.service import DEFAULT_CLAIM_TTL_SECONDS, AgentConnectService

from . import tools

_log = logging.getLogger(__name__)

_VALID_TRANSPORTS = {"stdio", "sse", "streamable-http"}


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def _task(task_id: Optional[str]) -> str:
    """Infer the task from `AGENTCONNECT_TASK_ID` when the caller omits it.

    `agentconnect launch` puts the id in the environment precisely so an agent
    never has to type one, and therefore can never record its work against the
    wrong task (compliance §7).
    """
    resolved = task_id or _env("AGENTCONNECT_TASK_ID")
    if not resolved:
        raise InvalidRequest(
            "no task_id given and AGENTCONNECT_TASK_ID is unset; "
            "launch this session with `agentconnect launch <manager> --task <id>`"
        )
    return resolved


def _review(review_id: Optional[str]) -> str:
    resolved = review_id or _env("AGENTCONNECT_REVIEW_ID")
    if not resolved:
        raise InvalidRequest("no review_id given and AGENTCONNECT_REVIEW_ID is unset")
    return resolved


def _actor(actor_id: Optional[str], default: str = "manager") -> str:
    return actor_id or _env("AGENTCONNECT_MANAGER_ID") or default


def build_mcp_server(
    service: Optional[AgentConnectService] = None,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
):
    from mcp.server.fastmcp import FastMCP

    svc = service or service_from_env()
    kwargs: dict[str, Any] = {}
    if host is not None:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port
    mcp = FastMCP("agentconnect", **kwargs)

    @mcp.tool()
    def create_task(
        title: str,
        goal: str = "",
        created_by: str = "manager",
        priority: str = Priority.normal.value,
        constraints: Optional[list[str]] = None,
    ) -> str:
        """Create a task in the ledger. Returns its id — claim it before working."""
        try:
            task = svc.create_task(CreateTaskRequest(
                title=title, goal=goal, created_by=created_by,
                priority=Priority(priority), constraints=constraints or [],
            ))
        except (AgentConnectError, ValueError) as exc:
            return _err(exc)
        return tools.dumps({
            "task_id": task.id, "status": task.status.value,
            "next_action": f"claim_task({task.id})",
        })

    @mcp.tool()
    def open_task(task_id: Optional[str] = None) -> str:
        """Compact task state: locked decisions, artifact ids, open reviews and
        subtasks. Bodies are never inlined — page them with read_artifact_chunk.
        Omit task_id inside a managed session."""
        try:
            return tools.dumps(tools.compact_task(svc.get_task(_task(task_id))))
        except AgentConnectError as exc:
            return _err(exc)

    @mcp.tool()
    def get_handoff_summary(
        task_id: Optional[str] = None, manager_id: Optional[str] = None
    ) -> str:
        """The deterministic handoff (§16): goal, constraints, locked decisions,
        recent attempts, artifacts, open items, suggested next step. Read this
        FIRST when picking up a task somebody else was working on."""
        try:
            return tools.dumps(tools.compact_handoff(
                svc.get_handoff_summary(_task(task_id), manager_id)))
        except AgentConnectError as exc:
            return _err(exc)

    @mcp.tool()
    def claim_task(
        task_id: Optional[str] = None,
        manager_id: Optional[str] = None,
        role: str = ClaimRole.primary_manager.value,
        ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    ) -> str:
        """Take a lease on a task. Only one primary_manager may hold a task at a
        time; the lease expires on its own so a dead manager never strands work."""
        try:
            task_id = _task(task_id)
            claim = svc.claim_task(task_id, _actor(manager_id), role, ttl_seconds)
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({
            "claim_id": claim.id, "task_id": task_id, "role": claim.role.value,
            "expires_at": claim.expires_at,
            "next_action": f"get_handoff_summary({task_id})",
        })

    @mcp.tool()
    def release_task(
        task_id: Optional[str] = None, manager_id: Optional[str] = None
    ) -> str:
        """Give up every claim you hold on a task so another manager can take it."""
        try:
            task_id, manager_id = _task(task_id), _actor(manager_id)
            svc.release_task(task_id, manager_id)
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({"released": task_id, "manager_id": manager_id})

    @mcp.tool()
    def record_decision(
        decision: str,
        task_id: Optional[str] = None,
        made_by: Optional[str] = None,
        rationale: str = "",
        locked: bool = False,
        supersedes: Optional[list[str]] = None,
    ) -> str:
        """Record a decision future managers must see. `locked=True` binds them:
        overturning it later requires naming it in `supersedes` AND holding a
        human_owner claim."""
        try:
            result = svc.record_decision(_task(task_id), RecordDecisionRequest(
                made_by=_actor(made_by), decision=decision, rationale=rationale,
                locked=locked, supersedes=supersedes or [],
            ))
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({
            "decision_id": result.id, "locked": result.locked,
            "note": "locked decisions appear in every future handoff summary"
            if result.locked else "",
        })

    @mcp.tool()
    def record_attempt(
        summary: str,
        task_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        outcome: str = "",
        actor_type: str = ActorType.manager.value,
        artifact_refs: Optional[list[str]] = None,
    ) -> str:
        """Record what you tried and what happened. This is what a replacement
        manager reads instead of your chat history — write it for them."""
        try:
            attempt = svc.record_attempt(_task(task_id), RecordAttemptRequest(
                actor_id=_actor(actor_id), actor_type=ActorType(actor_type), summary=summary,
                outcome=outcome, artifact_refs=artifact_refs or [],
            ))
        except (AgentConnectError, ValueError) as exc:
            return _err(exc)
        return tools.dumps({"attempt_id": attempt.id})

    @mcp.tool()
    def request_review(
        assigned_to: str,
        task_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        criteria: Optional[list[str]] = None,
        artifact_refs: Optional[list[str]] = None,
    ) -> str:
        """Ask another manager to review artifacts against criteria. This is how
        managers coordinate — by durable ticket, not by dumping context at each
        other. The reviewer answers with an artifact you can read back."""
        try:
            task_id = _task(task_id)
            review = svc.request_review(task_id, ReviewRequest(
                requested_by=_actor(requested_by), assigned_to=assigned_to,
                criteria=criteria or [], artifact_refs=artifact_refs or [],
            ))
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({
            "review_id": review.id, "assigned_to": review.assigned_to,
            "status": review.status.value,
            "next_action": f"poll get_status({task_id}) until the review completes",
        })

    @mcp.tool()
    def submit_subtask(
        title: str,
        instructions: str,
        task_id: Optional[str] = None,
        privacy_tier: str = PrivacyTier.repo_sensitive.value,
        preferred_worker: Optional[str] = None,
        filesystem: str = FilesystemAccess.none.value,
        network: bool = False,
        shell: bool = False,
        required_capabilities: Optional[list[str]] = None,
    ) -> str:
        """Delegate bounded work to a worker. Routing is deterministic and runs
        immediately; the response tells you which worker took it and where the
        output landed. If the only eligible worker costs money, the subtask parks
        in needs_approval until a human approves it."""
        try:
            subtask = svc.submit_subtask(_task(task_id), SubtaskRequest(
                title=title, instructions=instructions,
                privacy_tier=PrivacyTier(privacy_tier), preferred_worker=preferred_worker,
                sandbox=SandboxSpec(
                    filesystem=FilesystemAccess(filesystem), network=network, shell=shell
                ),
                required_capabilities=required_capabilities or [],
            ))
        except (AgentConnectError, ValueError) as exc:
            return _err(exc)
        # Under Temporal the worker has NOT run yet — return the handle and let
        # the manager poll. Never block an MCP request on a worker.
        handles = svc.executions_for("subtask", subtask.id)
        return tools.dumps(tools.compact_subtask(subtask, handles[-1] if handles else None))

    @mcp.tool()
    def get_status(task_id: Optional[str] = None) -> str:
        """One-line task state: status, manager, open review/subtask counts, and
        anything awaiting human approval."""
        try:
            return tools.dumps(tools.compact_status(svc.get_task(_task(task_id))))
        except AgentConnectError as exc:
            return _err(exc)

    @mcp.tool()
    def list_artifacts(task_id: Optional[str] = None) -> str:
        """Artifact ids, types, sizes, and one-line summaries. Never bodies."""
        try:
            artifacts = svc.list_artifacts(_task(task_id))
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps([
            {"artifact_id": a.id, "type": a.type.value, "summary": a.summary,
             "size_bytes": a.size_bytes, "created_by": a.created_by}
            for a in artifacts
        ])

    @mcp.tool()
    def read_artifact_chunk(artifact_id: str, offset: int = 0, limit: int = 8000) -> str:
        """Read a bounded slice of an artifact body. Follow `next_offset` to page;
        `eof: true` means you have the whole thing."""
        try:
            chunk = svc.read_artifact_chunk(artifact_id, offset, limit)
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({
            "artifact_id": chunk.artifact_id, "content": chunk.content,
            "next_offset": chunk.next_offset, "eof": chunk.eof,
            "size_bytes": chunk.size_bytes,
        })

    @mcp.tool()
    def explain_route(subtask_id: str) -> str:
        """Why a subtask went where it went: the winning worker's score terms and
        the workers that were rejected, with the gate each one failed."""
        try:
            return tools.dumps(tools.compact_route(svc.explain_route(subtask_id)))
        except AgentConnectError as exc:
            return _err(exc)

    # ---------------------------------------------------------------- memory
    @mcp.tool()
    def recall_memory(
        query: str,
        task_id: Optional[str] = None,
        profile: str = "manager_brief",
        max_items: int = 8,
        trusted_only: bool = True,
        include_pending: bool = False,
    ) -> str:
        """Recall bounded external context. This is NOT ledger truth — it is a
        hint that may be stale or wrong. Unpromoted ("pending") memory is
        withheld unless you ask for it explicitly, and is labeled when returned."""
        task_id = task_id or _env("AGENTCONNECT_TASK_ID")
        pack = svc.recall_memory(RecallRequest(
            query=query, task_id=task_id, profile=profile,  # type: ignore[arg-type]
            max_items=max_items, trusted_only=trusted_only, include_pending=include_pending,
            scopes=[MemoryScope("task", task_id)] if task_id else [],
        ))
        return tools.dumps(tools.compact_recall(pack))

    @mcp.tool()
    def capture_memory_candidate(
        text: str,
        task_id: Optional[str] = None,
        origin_actor_id: Optional[str] = None,
        origin_actor_type: str = "manager",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Offer a reusable insight to the memory layer. It is stored as a
        *pending candidate* for later human review — capture never promotes."""
        result = svc.capture_memory_candidate(CaptureRequest(
            text=text, task_id=task_id or _env("AGENTCONNECT_TASK_ID"),
            origin_actor_id=_actor(origin_actor_id),
            origin_actor_type=origin_actor_type, tags=tags or [],  # noqa: E501
        ))
        return tools.dumps({
            "accepted": result.accepted, "candidate_id": result.candidate_id,
            "status": result.status, "message": result.message, "backend": result.backend,
        })

    @mcp.tool()
    def record_memory_feedback(
        feedback: str,
        task_id: Optional[str] = None,
        memory_item_id: Optional[str] = None,
        source_id: Optional[str] = None,
        actor_id: str = "manager",
        note: Optional[str] = None,
    ) -> str:
        """Tell the memory layer whether a recalled item helped:
        useful | irrelevant | stale | wrong | too_broad | missing_context."""
        svc.record_memory_feedback(MemoryFeedbackRequest(
            task_id=task_id, memory_item_id=memory_item_id, source_id=source_id,
            feedback=feedback, actor_id=actor_id, note=note,
        ))
        return tools.dumps({"recorded": True})

    @mcp.tool()
    def get_task_context_pack(
        task_id: Optional[str] = None,
        profile: str = "manager_brief",
        max_memory_items: int = 8,
        manager_id: Optional[str] = None,
    ) -> str:
        """Everything you need to pick up a task: the deterministic handoff
        (ledger truth) plus clearly-labeled recalled memory (external context).

        Call this FIRST. Inside a managed session every argument is optional —
        the task and your manager id come from the environment."""
        try:
            pack = svc.get_task_context_pack(
                _task(task_id), profile=profile, max_memory_items=max_memory_items,
                manager_id=_actor(manager_id, None) or None,
            )
        except AgentConnectError as exc:
            return _err(exc)
        return tools.dumps({
            "task_id": pack.task_id,
            "profile": pack.profile,
            "handoff": tools.compact_handoff(pack.handoff) if pack.handoff else None,
            "memory": tools.compact_recall(pack.memory),
            "backends_queried": pack.backends_queried,
            "warnings": pack.warnings,
            "memory_is_external_context": pack.memory_is_external_context,
        })

    return mcp


def _err(exc: Exception) -> str:
    if isinstance(exc, AgentConnectError):
        return tools.error(exc)
    return tools.dumps({"error": "invalid_request", "detail": str(exc)})


def transport_from_env() -> tuple[str, str, int]:
    transport = os.environ.get("AGENTCONNECT_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise SystemExit(
            f"AGENTCONNECT_MCP_TRANSPORT must be one of {sorted(_VALID_TRANSPORTS)}, "
            f"got {transport!r}"
        )
    host = os.environ.get("AGENTCONNECT_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("AGENTCONNECT_MCP_PORT", "8765"))
    return transport, host, port


def main() -> None:
    transport, host, port = transport_from_env()
    if transport == "stdio":
        build_mcp_server().run()
        return
    logging.basicConfig(level=logging.INFO)  # stderr; stdout is not the channel here
    _log.info("agentconnect MCP serving over %s on %s:%s", transport, host, port)
    build_mcp_server(host=host, port=port).run(transport=transport)


if __name__ == "__main__":
    main()
