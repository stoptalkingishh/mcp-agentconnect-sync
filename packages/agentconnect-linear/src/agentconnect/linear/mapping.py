"""AgentConnect → Linear rendering (spec §14.1–14.3, §21).

Two rules govern everything here:

1. **Compact.** Linear gets IDs, statuses, and one-line summaries. Artifact
   bodies, worker traces, full route JSON, and logs never cross this boundary.
2. **Withholding is by tier, not by field.** A task whose effective privacy is
   ``secret_sensitive`` syncs its *existence* and its *workflow state* and
   nothing else — no goal, no constraints, no handoff, no artifact summaries.
   The issue still exists so a human can act on it; the content stays on the box.
"""

from __future__ import annotations

from typing import Any, Optional

from agentconnect.core.models import (
    ArtifactSummary,
    Decision,
    PrivacyTier,
    Review,
    Subtask,
    SubtaskStatus,
    TaskDetail,
)
from agentconnect.core.routing import RouteExplanation

BASE_LABEL = "agentconnect"
WITHHELD = "_[withheld: secret_sensitive — content stays on the local box]_"

_PRIVACY_LABELS = {
    PrivacyTier.public: "privacy:public",
    PrivacyTier.public_redacted: "privacy:public",
    PrivacyTier.repo_sensitive: "privacy:repo-sensitive",
    PrivacyTier.secret_sensitive: "privacy:secret-sensitive",
    PrivacyTier.local_only: "privacy:local-only",
}

_LOCATION_LABELS = {"local": "worker:local", "cloud": "worker:cloud", "rented": "worker:rented"}


def is_withheld(detail: TaskDetail) -> bool:
    return detail.effective_privacy is PrivacyTier.secret_sensitive


def labels_for(detail: TaskDetail) -> list[str]:
    labels = [BASE_LABEL, _PRIVACY_LABELS[detail.effective_privacy]]
    if detail.task.current_manager:
        labels.append(f"manager:{detail.task.current_manager}")
    for subtask in detail.subtasks:
        location = subtask.route_reason.get("approval_location") or (
            "local" if subtask.assigned_worker else None
        )
        label = _LOCATION_LABELS.get(location or "")
        if label and label not in labels:
            labels.append(label)
    status = detail.task.status.value
    if status == "needs_review":
        labels.append("needs-review")
    elif status == "needs_approval":
        labels.append("needs-approval")
    elif status == "blocked":
        labels.append("blocked")
    return labels


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else "- (none)"


def issue_title(detail: TaskDetail) -> str:
    return detail.task.title


def issue_body(detail: TaskDetail, handoff_text: Optional[str] = None) -> str:
    """The §14.2 description block."""
    withheld = is_withheld(detail)
    task = detail.task
    open_review = next(
        (r for r in detail.reviews
         if r.status.value in ("open", "claimed", "in_progress")), None,
    )
    lines = [
        f"**AgentConnect Task:** `{task.id}`",
        "**Canonical status:** AgentConnect-managed — "
        "moving this issue does not change task state.",
        f"**Status:** {task.status.value}",
        f"**Current manager:** {task.current_manager or '(unclaimed)'}",
        f"**Current review:** "
        f"{f'`{open_review.id}` → {open_review.assigned_to}' if open_review else '(none)'}",
        f"**Privacy:** {detail.effective_privacy.value}",
        f"**Priority:** {task.priority.value}",
        "",
        "## Goal",
        WITHHELD if withheld else (task.goal or "(no goal recorded)"),
        "",
        "## Constraints",
        WITHHELD if withheld else _bullets([c.text for c in detail.constraints]),
        "",
        "## Current handoff",
    ]
    if withheld:
        lines.append(WITHHELD)
    else:
        lines.append(f"```\n{handoff_text or '(no handoff yet)'}\n```")

    lines += ["", "## Important artifacts"]
    if withheld:
        lines.append(WITHHELD)
    else:
        lines.append(
            _bullets(
                [
                    f"`{a.id}` ({a.type.value}): {a.summary or '—'}"
                    for a in detail.artifacts
                    if a.type.value != "route_explanation"
                ][-10:]
            )
        )

    open_reviews = [r for r in detail.reviews if r.status.value in ("open", "claimed", "in_progress")]
    lines += ["", "## Open reviews",
              _bullets([f"`{r.id}` → {r.assigned_to} ({r.status.value})" for r in open_reviews])]

    open_subtasks = [
        s for s in detail.subtasks
        if s.status in (SubtaskStatus.queued, SubtaskStatus.running, SubtaskStatus.needs_approval)
    ]
    lines += ["", "## Open subtasks",
              _bullets(
                  [
                      f"`{s.id}` {'(withheld)' if withheld else s.title} "
                      f"— {s.status.value} [{s.privacy_tier.value}]"
                      for s in open_subtasks
                  ]
              )]
    return "\n".join(lines)


def decision_comment(decision: Decision, withheld: bool = False) -> str:
    if withheld:
        return f"**Decision:** `{decision.id}` recorded by {decision.made_by}. {WITHHELD}"
    lock = " 🔒 **locked**" if decision.locked else ""
    body = f"**Decision:**{lock} {decision.decision}\n\n_by {decision.made_by}_"
    if decision.rationale:
        body += f"\n\n**Rationale:** {decision.rationale}"
    return body


def artifact_comment(
    artifact: ArtifactSummary, withheld: bool = False, base_url: Optional[str] = None
) -> str:
    """A pointer, never a payload (§21)."""
    ref = f"`{artifact.id}`"
    if base_url:
        ref = f"[{artifact.id}]({base_url.rstrip('/')}/artifacts/{artifact.id})"
    if withheld:
        return f"**Artifact:** {ref} ({artifact.type.value}, {artifact.size_bytes} bytes). {WITHHELD}"
    return (
        f"**Artifact:** {ref} ({artifact.type.value}, {artifact.size_bytes} bytes)\n\n"
        f"{artifact.summary or '—'}\n\n"
        f"_Read it with_ `agentconnect artifacts read {artifact.id}`"
    )


def review_request_comment(review: Review, withheld: bool = False) -> str:
    body = (
        f"**Review requested:** `{review.id}`\n\n"
        f"- Requested by: {review.requested_by}\n"
        f"- Assigned to: **{review.assigned_to}**\n"
        f"- Artifacts: {', '.join(f'`{a}`' for a in review.artifact_refs) or '(none)'}\n"
    )
    if not withheld and review.criteria:
        body += "- Criteria:\n" + "\n".join(f"  - {c}" for c in review.criteria)
    return body


def review_result_comment(review: Review, withheld: bool = False) -> str:
    verdict = "completed" if review.status.value == "completed" else review.status.value
    body = (
        f"**Review {verdict}:** `{review.id}` by {review.assigned_to}\n\n"
        f"Result artifact: `{review.result_artifact_id or '(none)'}`"
    )
    if withheld:
        body += f"\n\n{WITHHELD}"
    else:
        body += f"\n\n_Read it with_ `agentconnect artifacts read {review.result_artifact_id}`"
    return body


def subtask_comment(subtask: Subtask, withheld: bool = False) -> str:
    title = "(withheld)" if withheld else subtask.title
    body = (
        f"**Subtask {subtask.status.value}:** `{subtask.id}` — {title}\n\n"
        f"- Privacy: {subtask.privacy_tier.value}\n"
        f"- Worker: {subtask.assigned_worker or '(unassigned)'}\n"
    )
    if subtask.result_artifact_id:
        body += f"- Result artifact: `{subtask.result_artifact_id}`\n"
    return body


def memory_comment(kind: str, **detail: object) -> str:
    """One line, never a backend dump (memory-stack §13).

    Linear shows a human that *something happened to memory*; the pack itself
    lives behind AgentConnect, where trust and scope are enforced.
    """
    if kind == "captured":
        return "**Memory candidate captured** for review — not yet trusted."
    if kind == "promoted":
        return f"**Promoted memory:** {detail.get('text', '(claim)')}"
    if kind == "conflict":
        return (
            f"**Potential contradiction detected** between promoted claims "
            f"`{detail.get('a')}` and `{detail.get('b')}`."
        )
    return f"**Memory update:** {kind}"


def audit_comment(report: Any) -> str:
    """What `/agentconnect status` and a refused `/agentconnect complete` return.

    A human in Linear should be able to see *why* the backplane refused, without
    reading the ledger. The problems are the whole message.
    """
    if report.passed:
        checks = "\n".join(f"- {c.detail}" for c in report.checks if c.passed and c.detail)
        return f"**AgentConnect audit: PASS**\n\n{checks}"
    problems = "\n".join(f"- {p}" for p in report.problems)
    return (
        f"**AgentConnect audit: FAIL**\n\n{problems}\n\n"
        f"_Task cannot be marked complete. Linear status is a mirror; "
        f"AgentConnect decides completion._"
    )


def completion_comment(task_id: str, completed_by: str) -> str:
    return (
        f"**Task completed** `{task_id}` by {completed_by}.\n\n"
        f"_Audit passed. AgentConnect marked the task succeeded, then updated Linear._"
    )


def approval_request_comment(subtask: Subtask, explanation: RouteExplanation) -> str:
    """§15 step 5: the human must see the price, the tier, and *why local lost*."""
    rejected = "\n".join(
        f"  - `{r.worker}` — {r.reason} (gate: {r.gate})" for r in explanation.rejected_workers
    ) or "  - (no other workers registered)"
    return (
        f"**Approval required** for subtask `{subtask.id}`\n\n"
        f"- Requested route: **{explanation.approval_candidate}** "
        f"({explanation.approval_location})\n"
        f"- Estimated cost: **${explanation.estimated_cost_usd:.4f}**\n"
        f"- Privacy tier: `{subtask.privacy_tier.value}`\n"
        f"- Why the free/local workers were rejected:\n{rejected}\n\n"
        f"Reply with one of:\n"
        f"```\n"
        f"/agentconnect approve {explanation.approval_location}\n"
        f"/agentconnect approve {explanation.approval_location} max_cost=1.00\n"
        f"/agentconnect deny\n"
        f"```"
    )
