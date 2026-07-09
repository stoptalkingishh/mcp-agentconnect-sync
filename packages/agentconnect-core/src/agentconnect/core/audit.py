"""Completion audit (compliance spec §12–§13).

The audit is the teeth behind the compliance rule. Instructions in `CLAUDE.md`
ask an agent to record its work; the audit is what happens when it doesn't.

It answers one question in several ways: **you did something — where is it in the
ledger?** Files changed but no artifact registered. A session ran but no attempt
recorded. A review requested but never completed. Each is a way for durable work
to escape the backplane, and each fails the audit.

Two kinds of check:

* **Required** — a failure blocks completion. These are the ones that mean work
  was lost.
* **Advisory** — reported, never blocking. Memory capture is advisory because the
  memory stack is optional and no core flow may depend on it.

Nothing here writes. The audit reads the ledger and the worktree, and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .models import (
    ManagerSession,
    ReviewStatus,
    SubtaskStatus,
    TaskDetail,
    TaskStatus,
    Workspace,
)
from .workspace import changed_files

#: Linear state names that mean a human believes the work is finished.
_DONE_STATES = frozenset({"done", "completed", "closed", "shipped", "merged"})

#: Reviews that are still owed an answer.
_OPEN_REVIEW_STATUSES = frozenset(
    {ReviewStatus.open, ReviewStatus.claimed, ReviewStatus.in_progress}
)
_OPEN_SUBTASK_STATUSES = frozenset(
    {SubtaskStatus.queued, SubtaskStatus.running, SubtaskStatus.needs_approval}
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    #: An advisory check reports but never blocks completion.
    required: bool = True


@dataclass
class AuditReport:
    entity_type: str  # task | review
    entity_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    @property
    def problems(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed and c.required]

    @property
    def warnings(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed and not c.required]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type, "entity_id": self.entity_id,
            "status": "PASS" if self.passed else "FAIL",
            "problems": self.problems, "warnings": self.warnings,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail,
                 "required": c.required}
                for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = [f"{self.entity_type.capitalize()} audit: {self.entity_id}",
                 f"Status: {'PASS' if self.passed else 'FAIL'}"]
        if self.problems:
            lines.append("Problems:")
            lines += [f"- {p}" for p in self.problems]
        else:
            lines.append("Checks:")
            lines += [f"- {c.detail}" for c in self.checks if c.passed and c.detail]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"- {w}" for w in self.warnings]
        if not self.passed:
            noun = "Task" if self.entity_type == "task" else "Review"
            lines.append(f"{noun} cannot be marked complete.")
        return "\n".join(lines)


def _repo_path(workspace: Optional[Workspace]) -> Optional[Path]:
    if workspace is None or not workspace.repo_path:
        return None
    path = Path(workspace.repo_path)
    return path if path.exists() else None


def _registered(detail: TaskDetail) -> tuple[set[str], str]:
    """What the artifacts claim to cover: exact paths, and all summary prose.

    An artifact registers files either structurally (``metadata["files"]``) or by
    naming them in its summary. The second is looser on purpose: a manager that
    writes "rewrote auth/session.py and auth/tokens.py" has, in fact, told the
    next manager what changed.
    """
    paths: set[str] = set()
    prose: list[str] = []
    for artifact in detail.artifacts:
        for path in artifact.metadata.get("files", []) or []:
            paths.add(str(path))
        prose.append(artifact.summary or "")
    return paths, "\n".join(prose)


def _unregistered(changed: list[str], detail: TaskDetail) -> list[str]:
    paths, prose = _registered(detail)
    return [
        path for path in changed
        if path not in paths and path not in prose and Path(path).name not in prose
    ]


def _attempt_during(detail: TaskDetail, session: Optional[ManagerSession]) -> bool:
    if not detail.attempts:
        return False
    if session is None:
        return True
    return any(a.created_at >= session.started_at for a in detail.attempts)


def audit_task(
    detail: TaskDetail,
    workspace: Optional[Workspace],
    session: Optional[ManagerSession],
    handoff_text: str,
    stored_handoff: Optional[str],
    linear_ref: Optional[Any] = None,
    linear_state: Optional[str] = None,
    memory_captured: bool = False,
    memory_enabled: bool = False,
) -> AuditReport:
    """Every check from §12, in the order a human would ask them."""
    report = AuditReport("task", detail.task.id)
    add = report.checks.append
    task = detail.task

    add(Check("task_exists", True, f"Task {task.id} exists."))

    add(Check(
        "workspace_exists", workspace is not None,
        f"Workspace {workspace.id} at {workspace.path}." if workspace
        else "No AgentConnect workspace: the task was worked outside a managed session.",
    ))
    add(Check(
        "manager_session_exists", session is not None,
        f"Session {session.id} for {session.manager_id}." if session
        else "No manager session was launched for this task.",
    ))

    claimed = task.current_manager is not None or bool(detail.active_claims)
    add(Check(
        "task_claimed", claimed,
        f"Task claimed by {task.current_manager}." if claimed
        else "Task was never claimed; no manager is accountable for it.",
    ))

    attempted = _attempt_during(detail, session)
    add(Check(
        "attempt_recorded", attempted,
        f"{len(detail.attempts)} attempts recorded."
        if attempted else "No record_attempt was made during this session.",
    ))

    repo = _repo_path(workspace)
    changed = changed_files(repo) if repo else []
    unregistered = _unregistered(changed, detail)
    add(Check(
        "changed_files_registered", not unregistered,
        f"{len(changed)} changed files, all registered."
        if changed and not unregistered else
        (f"Git worktree has modified files not registered as artifacts: "
         f"{', '.join(unregistered[:5])}"
         f"{f' (+{len(unregistered) - 5} more)' if len(unregistered) > 5 else ''}."
         if unregistered else "No unregistered file changes."),
    ))

    # "Important artifacts registered" only bites once there is something to show
    # for the work: a task that changed files but produced no artifact at all.
    add(Check(
        "artifacts_registered", bool(detail.artifacts) or not changed,
        f"{len(detail.artifacts)} artifacts registered." if detail.artifacts
        else "Files changed but no artifact was registered.",
    ))

    open_subtasks = [s for s in detail.subtasks if s.status in _OPEN_SUBTASK_STATUSES]
    add(Check(
        "subtasks_resolved", not open_subtasks,
        "All subtasks resolved." if not open_subtasks else
        f"{len(open_subtasks)} subtasks are still open: "
        f"{', '.join(s.id for s in open_subtasks[:3])}.",
    ))

    open_reviews = [r for r in detail.reviews if r.status in _OPEN_REVIEW_STATUSES]
    add(Check(
        "reviews_completed", not open_reviews,
        f"{len(detail.reviews)} reviews, all complete."
        if detail.reviews and not open_reviews else
        (f"Review is required but no completed review exists: "
         f"{', '.join(r.id for r in open_reviews)}." if open_reviews
         else "No reviews were required."),
    ))

    # A durable change with no recorded decision is exactly the failure mode the
    # backplane exists to prevent: work that only the departed manager understood.
    durable = bool(changed) or bool(detail.artifacts)
    has_decision = bool(detail.decisions)
    add(Check(
        "decisions_recorded", has_decision or not durable,
        f"{len(detail.decisions)} decisions recorded." if has_decision
        else "Durable changes were made but no decision was recorded.",
    ))

    fresh = stored_handoff == handoff_text
    add(Check(
        "handoff_fresh", fresh,
        "Handoff summary fresh." if fresh
        else "Handoff summary is stale; regenerate it before handing off.",
    ))

    if linear_ref is None:
        add(Check("linear_sync_current", True, "Not mirrored to Linear."))
    else:
        conflict = (
            linear_state is not None
            and linear_state.strip().lower() in _DONE_STATES
            and task.status is not TaskStatus.succeeded
        )
        add(Check(
            "linear_sync_current", not conflict,
            "Linear sync current." if not conflict else
            f"Linear issue is marked {linear_state} but AgentConnect task is "
            f"{task.status.value}.",
        ))

    add(Check(
        "memory_captured", memory_captured or not memory_enabled,
        "Memory candidates captured." if memory_captured else
        "No memory candidate was captured; reusable lessons may be lost.",
        required=False,
    ))

    consistent = task.status not in (TaskStatus.cancelled, TaskStatus.failed)
    add(Check(
        "status_consistent", consistent,
        f"Task status is {task.status.value}." if consistent
        else f"Task is already {task.status.value} and cannot be completed.",
    ))
    return report


def audit_review(
    review: Any,
    detail: TaskDetail,
    workspace: Optional[Workspace],
    session: Optional[ManagerSession],
) -> AuditReport:
    """A reviewer's obligations are narrower: read, judge, record (§12)."""
    report = AuditReport("review", review.id)
    add = report.checks.append

    add(Check("review_exists", True, f"Review {review.id} exists."))
    add(Check(
        "workspace_exists", workspace is not None,
        f"Workspace {workspace.id}." if workspace else "No workspace for this review.",
    ))
    add(Check(
        "manager_session_exists", session is not None,
        f"Session {session.id} for {session.manager_id}." if session
        else "No reviewer session was launched for this review.",
    ))

    claimed = review.status is not ReviewStatus.open
    add(Check(
        "review_claimed", claimed,
        f"Review claimed by {review.assigned_to}." if claimed
        else "Review was never claimed.",
    ))

    attempted = _attempt_during(detail, session)
    add(Check(
        "attempt_recorded", attempted,
        f"{len(detail.attempts)} attempts recorded."
        if attempted else "No record_attempt was made during this session.",
    ))

    repo = _repo_path(workspace)
    changed = changed_files(repo) if repo else []
    unregistered = _unregistered(changed, detail)
    add(Check(
        "changed_files_registered", not unregistered,
        "No unregistered file changes." if not unregistered else
        f"Reviewer changed files without registering them: {', '.join(unregistered[:5])}.",
    ))

    # The audit runs *before* `complete_review`, so a missing result artifact is
    # expected. What must not happen is completing a review twice.
    decidable = review.status in _OPEN_REVIEW_STATUSES
    add(Check(
        "review_is_decidable", decidable,
        f"Review is {review.status.value} and awaiting a verdict." if decidable
        else f"Review is already {review.status.value}.",
    ))
    add(Check(
        "review_has_result", bool(review.result_artifact_id),
        f"Review result artifact {review.result_artifact_id}."
        if review.result_artifact_id else "Review has no result artifact yet.",
        required=False,
    ))
    return report
