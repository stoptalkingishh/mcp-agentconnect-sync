"""Compact response shaping for the MCP adapter (spec §13).

A manager's context window is the scarcest resource in the system. Every shape
here answers three questions and stops: *what is the state*, *what are the
handles*, *what should I do next*. Bodies, logs, traces, and full route JSON are
reachable — by ID, on demand — and never volunteered.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agentconnect.core.errors import AgentConnectError
from agentconnect.core.models import (
    ArtifactType,
    HandoffSummary,
    ReviewStatus,
    Subtask,
    SubtaskStatus,
    TaskDetail,
)
from agentconnect.core.routing import RouteExplanation

MAX_REJECTIONS_SHOWN = 3
MAX_SUMMARY_CHARS = 200

_OPEN_REVIEW_STATUSES = (ReviewStatus.open, ReviewStatus.claimed, ReviewStatus.in_progress)
_OPEN_SUBTASK_STATUSES = (
    SubtaskStatus.queued, SubtaskStatus.running, SubtaskStatus.needs_approval,
)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def error(exc: AgentConnectError) -> str:
    return dumps({"error": exc.code, "detail": str(exc)})


def _clip(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= MAX_SUMMARY_CHARS else text[: MAX_SUMMARY_CHARS - 1] + "…"


def compact_task(detail: TaskDetail, next_action: str = "") -> dict[str, Any]:
    open_reviews = [r for r in detail.reviews if r.status in _OPEN_REVIEW_STATUSES]
    open_subtasks = [s for s in detail.subtasks if s.status in _OPEN_SUBTASK_STATUSES]
    return {
        "task_id": detail.task.id,
        "title": detail.task.title,
        "status": detail.task.status.value,
        "current_manager": detail.task.current_manager,
        "privacy": detail.effective_privacy.value,
        "locked_decisions": [
            _clip(d.decision) for d in detail.decisions if d.locked and not d.superseded_by
        ],
        "artifact_ids": [
            a.id for a in detail.artifacts if a.type is not ArtifactType.route_explanation
        ],
        "open_reviews": [
            {"review_id": r.id, "assigned_to": r.assigned_to, "status": r.status.value}
            for r in open_reviews
        ],
        "open_subtasks": [
            {"subtask_id": s.id, "status": s.status.value, "title": _clip(s.title)}
            for s in open_subtasks
        ],
        "linear": detail.task.linear_issue_url,
        "next_action": next_action or "call get_handoff_summary for the full state",
        "hint": "artifact bodies are not inlined; page them with read_artifact_chunk",
    }


def compact_status(detail: TaskDetail) -> dict[str, Any]:
    return {
        "task_id": detail.task.id,
        "status": detail.task.status.value,
        "current_manager": detail.task.current_manager,
        "open_reviews": sum(1 for r in detail.reviews if r.status in _OPEN_REVIEW_STATUSES),
        "open_subtasks": sum(1 for s in detail.subtasks if s.status in _OPEN_SUBTASK_STATUSES),
        "awaiting_approval": [
            s.id for s in detail.subtasks if s.status is SubtaskStatus.needs_approval
        ],
    }


def compact_handoff(summary: HandoffSummary) -> dict[str, Any]:
    return {
        "task_id": summary.task_id,
        "status": summary.status.value,
        "current_manager": summary.current_manager,
        "viewer_holds_claim": summary.viewer_holds_claim,
        "suggested_next_step": summary.suggested_next_step,
        "summary": summary.text,
    }


def compact_subtask(subtask: Subtask, handle: Optional[Any] = None) -> dict[str, Any]:
    reason = subtask.route_reason or {}
    route: dict[str, Any] = {}
    if reason.get("selected_worker"):
        route = {
            "worker": reason["selected_worker"],
            "harness": reason.get("selected_harness"),
            "model": reason.get("selected_model"),
            "score": reason.get("total_score"),
        }
        if reason.get("local_estimate"):
            route["local_estimate"] = reason["local_estimate"]
    elif reason.get("needs_approval"):
        route = {
            "needs_approval": True,
            "candidate": reason.get("approval_candidate"),
            "estimated_cost_usd": reason.get("estimated_cost_usd"),
        }

    if subtask.status is SubtaskStatus.needs_approval:
        next_action = "a human must approve this route in Linear or via the CLI/API"
    elif subtask.status is SubtaskStatus.succeeded and subtask.result_artifact_id:
        next_action = f"read_artifact_chunk({subtask.result_artifact_id})"
    elif subtask.status is SubtaskStatus.failed:
        next_action = "call explain_route for why, then resubmit or escalate"
    else:
        next_action = f"poll get_status({subtask.parent_task_id}) — the worker runs asynchronously"

    payload = {
        "subtask_id": subtask.id,
        "status": subtask.status.value,
        "privacy_tier": subtask.privacy_tier.value,
        "assigned_worker": subtask.assigned_worker,
        "result_artifact_id": subtask.result_artifact_id,
        "route": route,
        "next_action": next_action,
    }
    if handle is not None:
        payload["workflow_id"] = handle.workflow_id or handle.handle_id
        payload["execution_state"] = handle.state.value
    return payload


def compact_recall(pack: Any) -> dict[str, Any]:
    """Bounded. A raw backend search dump must never reach a manager (§MCP)."""
    return {
        "backend": pack.backend,
        "profile": pack.profile,
        "items": [
            {"text": _clip(i.text), "status": i.status, "confidence": i.confidence,
             "source_id": i.source_id}
            for i in pack.items
        ],
        "warnings": pack.warnings,
        "note": "external recalled context — not ledger truth; verify before relying on it",
    }


def compact_route(explanation: RouteExplanation) -> dict[str, Any]:
    """The route in one screen. The full JSON lives in a `route_explanation`
    artifact for anyone who wants every rejected worker."""
    return {
        "subtask_id": explanation.subtask_id,
        "selected_worker": explanation.selected_worker,
        "selected_harness": explanation.selected_harness,
        "selected_model": explanation.selected_model,
        "total_score": explanation.total_score,
        "score_terms": explanation.score_terms,
        "estimated_cost_usd": explanation.estimated_cost_usd,
        "needs_approval": explanation.needs_approval,
        "approval_candidate": explanation.approval_candidate,
        "rejected": [
            {"worker": r.worker, "gate": r.gate, "reason": _clip(r.reason)}
            for r in explanation.rejected_workers[:MAX_REJECTIONS_SHOWN]
        ],
        "rejected_count": len(explanation.rejected_workers),
    }
