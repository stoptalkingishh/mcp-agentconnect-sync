"""Deterministic handoff summaries (spec §16).

**No LLM.** A handoff is derived from the ledger by pure functions, so it works
offline, costs nothing, never hallucinates a decision that was not recorded, and
produces byte-identical output for identical state. An LLM-assisted summary may
be layered on later; this one must always work.

The summary is what a *replacement* manager reads instead of the previous
manager's chat history (§27).
"""

from __future__ import annotations

from typing import Optional

from .claims import active_primary
from .decisions import locked_decisions
from .models import (
    ArtifactType,
    HandoffSummary,
    ReviewStatus,
    SubtaskStatus,
    TERMINAL_TASK_STATUSES,
    TaskDetail,
)

MAX_RECENT_ATTEMPTS = 5
MAX_IMPORTANT_ARTIFACTS = 10

#: Bookkeeping, not evidence — never worth a slot in a bounded summary.
_UNIMPORTANT_ARTIFACT_TYPES = frozenset({ArtifactType.route_explanation})


def _important_artifacts(detail: TaskDetail):
    kept = [a for a in detail.artifacts if a.type not in _UNIMPORTANT_ARTIFACT_TYPES]
    return kept[-MAX_IMPORTANT_ARTIFACTS:]


def _suggested_next_step(detail: TaskDetail, manager_id: Optional[str], at: float) -> Optional[str]:
    """First rule that fires wins. Ordered by who is blocked on whom."""
    if detail.task.status in TERMINAL_TASK_STATUSES:
        return None

    pending_approval = [s for s in detail.subtasks if s.status is SubtaskStatus.needs_approval]
    if pending_approval:
        s = pending_approval[0]
        candidate = s.route_reason.get("approval_candidate") or "a paid worker"
        cost = s.route_reason.get("estimated_cost_usd", 0.0)
        return (
            f"Approve or deny subtask {s.id} ({s.title!r}): routing wants {candidate}, "
            f"estimated ${cost:.4f}."
        )

    open_reviews = [r for r in detail.reviews if r.status is ReviewStatus.open]
    mine = [r for r in open_reviews if manager_id and r.assigned_to == manager_id]
    if mine:
        return f"Claim review {mine[0].id} assigned to you, then complete it with an artifact."
    if open_reviews:
        r = open_reviews[0]
        return f"Await review {r.id} — assigned to {r.assigned_to}, not yet claimed."

    in_flight = [
        r for r in detail.reviews
        if r.status in (ReviewStatus.claimed, ReviewStatus.in_progress)
    ]
    if in_flight:
        r = in_flight[0]
        return f"Await review {r.id} — {r.assigned_to} is working on it."

    last_attempt_at = max((a.created_at for a in detail.attempts), default=0.0)
    unread = [
        r for r in detail.reviews
        if r.status is ReviewStatus.completed
        and r.result_artifact_id
        and r.updated_at > last_attempt_at
    ]
    if unread:
        r = unread[0]
        return (
            f"Read review result {r.result_artifact_id} (review {r.id} from {r.assigned_to}) "
            f"and record an attempt."
        )

    running = [
        s for s in detail.subtasks
        if s.status in (SubtaskStatus.queued, SubtaskStatus.running)
    ]
    if running:
        s = running[0]
        return f"Await subtask {s.id} ({s.title!r}) — currently {s.status.value}."

    if active_primary(detail.active_claims, at) is None:
        return "Claim the task as primary_manager before doing any work."

    return "No open reviews or subtasks: record the next attempt, or close the task."


def _bullets(lines: list[str], empty: str = "(none)") -> str:
    if not lines:
        return empty
    return "\n".join(f"- {line}" for line in lines)


def render(summary: HandoffSummary) -> str:
    """The §16 layout. Stable ordering, stable wording."""
    parts = [
        f"Task: {summary.title}",
        f"Status: {summary.status.value}",
        f"Current manager: {summary.current_manager or '(unclaimed)'}",
    ]
    if summary.linear_issue_url:
        parts.append(f"Linear: {summary.linear_issue_url}")
    parts.append(f"Goal:\n{summary.goal or '(no goal recorded)'}")
    parts.append(f"Constraints:\n{_bullets(summary.constraints)}")
    parts.append(
        "Locked decisions:\n"
        + _bullets(
            [
                f"{d.decision}" + (f" — {d.rationale}" if d.rationale else "")
                for d in summary.locked_decisions
            ]
        )
    )
    parts.append(
        "Recent attempts:\n"
        + _bullets(
            [
                f"{a.actor_id}: {a.summary}" + (f" [{a.outcome}]" if a.outcome else "")
                for a in summary.recent_attempts
            ]
        )
    )
    parts.append(
        "Important artifacts:\n"
        + _bullets([f"{a.id}: {a.summary or a.type.value}" for a in summary.important_artifacts])
    )
    open_items: list[str] = []
    for review in summary.open_reviews:
        open_items.append(f"review {review.id} → {review.assigned_to} ({review.status.value})")
    for subtask in summary.open_subtasks:
        open_items.append(f"subtask {subtask.id}: {subtask.title} ({subtask.status.value})")
    parts.append("Open items:\n" + _bullets(open_items))
    if summary.running_workflows:
        parts.append(
            "Running workflows:\n"
            + _bullets(
                [
                    f"{w.get('workflow_id') or w.get('handle_id')} "
                    f"({w.get('entity_type')} {w.get('entity_id')}) — {w.get('state')}"
                    for w in summary.running_workflows
                ]
            )
        )
    if summary.waiting_approvals:
        parts.append(
            "Waiting approvals:\n"
            + _bullets(
                [
                    f"{a.id}: {a.requested_worker or 'a worker'} "
                    f"({a.requested_location or 'unknown'}) ~${a.estimated_cost_usd:.4f}"
                    for a in summary.waiting_approvals
                ]
            )
        )
    if summary.completed_reviews:
        parts.append(
            "Completed reviews:\n"
            + _bullets(
                [
                    f"{r.id} from {r.assigned_to}"
                    + (f" → {r.result_artifact_id}" if r.result_artifact_id else "")
                    for r in summary.completed_reviews
                ]
            )
        )
    if summary.suggested_next_step:
        parts.append(f"Suggested next step:\n{summary.suggested_next_step}")
    return "\n".join(parts)


def build(
    detail: TaskDetail,
    manager_id: Optional[str],
    at: float,
    running_workflows: Optional[list[dict]] = None,
    waiting_approvals: Optional[list] = None,
) -> HandoffSummary:
    holder = active_primary(detail.active_claims, at)
    open_reviews = [
        r for r in detail.reviews
        if r.status in (ReviewStatus.open, ReviewStatus.claimed, ReviewStatus.in_progress)
    ]
    summary = HandoffSummary(
        task_id=detail.task.id,
        title=detail.task.title,
        goal=detail.task.goal,
        status=detail.task.status,
        current_manager=holder.manager_id if holder else None,
        viewer_holds_claim=(
            None if manager_id is None else bool(holder and holder.manager_id == manager_id)
        ),
        linear_issue_url=detail.task.linear_issue_url,
        constraints=[c.text for c in detail.constraints],
        locked_decisions=locked_decisions(detail.decisions),
        recent_attempts=detail.attempts[-MAX_RECENT_ATTEMPTS:],
        important_artifacts=_important_artifacts(detail),
        open_reviews=open_reviews,
        completed_reviews=[r for r in detail.reviews if r.status is ReviewStatus.completed],
        open_subtasks=[
            s for s in detail.subtasks
            if s.status in (SubtaskStatus.queued, SubtaskStatus.running,
                            SubtaskStatus.needs_approval)
        ],
        running_workflows=running_workflows or [],
        waiting_approvals=waiting_approvals or [],
        suggested_next_step=_suggested_next_step(detail, manager_id, at),
    )
    summary.text = render(summary)
    return summary
