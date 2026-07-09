"""Linear → AgentConnect ingest (spec §14.4 modes 2-3, §15).

Linear is a control surface, never a source of truth. So an inbound webhook can
do exactly three things:

* **approve or deny** a subtask that routing already parked in `needs_approval`
  (a comment command). It cannot invent a route or raise a budget by itself —
  the ceiling it carries is applied at re-route time by the service.
* **record an event** for a status or label change a human made in Linear. The
  task's own status is *not* overwritten: the ledger decides status from work,
  not from what somebody dragged across a board.
* **create an inbox item** when a human assigns the issue to a manager.

Everything else is ignored, loudly enough to debug.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional

from agentconnect.core.errors import Conflict, PolicyViolation
from agentconnect.core.models import InboxKind, ReviewRequest, SubtaskStatus
from agentconnect.core.service import AgentConnectService

_log = logging.getLogger(__name__)

COMMAND_PREFIX = "/agentconnect"

#: Human-facing route names → the `approval_location` recorded by routing.
_TARGET_ALIASES = {
    "cloud": "cloud",
    "rented": "rented",
    "rented-gpu": "rented",
    "rented_gpu": "rented",
    "gpu": "rented",
    "local": "local",
}

_COMMAND_RE = re.compile(rf"^\s*{re.escape(COMMAND_PREFIX)}\s+(?P<rest>.+)$", re.MULTILINE)


#: Commands a human may type in a Linear comment. `complete` is audit-gated:
#: Linear asks, AgentConnect decides (compliance §13-§14).
_ACTIONS = ("approve", "deny", "status", "complete", "request-review")


@dataclass
class ApprovalCommand:
    action: str  # approve | deny | status | complete | request-review
    target: Optional[str] = None  # normalized approval_location, or None = any
    params: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def max_cost_usd(self) -> Optional[float]:
        raw = self.params.get("max_cost")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            _log.warning("ignoring non-numeric max_cost=%r", raw)
            return None

    @property
    def reason(self) -> str:
        return self.params.get("reason", "")


@dataclass
class WebhookEvent:
    type: str
    action: str
    data: dict[str, Any]
    updated_from: dict[str, Any] = field(default_factory=dict)


def parse_command(text: str) -> Optional[ApprovalCommand]:
    """Parse the first ``/agentconnect ...`` line in a comment body."""
    if not text:
        return None
    match = _COMMAND_RE.search(text)
    if not match:
        return None
    rest = match.group("rest").strip()
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    if not tokens:
        return None

    action = tokens[0].lower()
    if action not in _ACTIONS:
        _log.info("unknown agentconnect command %r", action)
        return None

    target: Optional[str] = None
    params: dict[str, str] = {}
    free_text: list[str] = []
    for token in tokens[1:]:
        if "=" in token:
            key, _, value = token.partition("=")
            params[key.strip().lower()] = value.strip()
        elif target is None and token.lower() in _TARGET_ALIASES:
            target = _TARGET_ALIASES[token.lower()]
        else:
            free_text.append(token)
    if free_text and "reason" not in params:
        params["reason"] = " ".join(free_text)
    return ApprovalCommand(action=action, target=target, params=params, raw=rest)


def parse_webhook(payload: dict[str, Any]) -> WebhookEvent:
    return WebhookEvent(
        type=str(payload.get("type") or ""),
        action=str(payload.get("action") or ""),
        data=payload.get("data") or {},
        updated_from=payload.get("updatedFrom") or {},
    )


def _issue_id(data: dict[str, Any]) -> Optional[str]:
    issue = data.get("issue")
    if isinstance(issue, dict) and issue.get("id"):
        return str(issue["id"])
    return str(data["issueId"]) if data.get("issueId") else None


def _actor(data: dict[str, Any]) -> str:
    user = data.get("user") or data.get("actor") or {}
    if isinstance(user, dict):
        return str(user.get("name") or user.get("displayName") or "linear")
    return "linear"


def _assignee(data: dict[str, Any]) -> Optional[str]:
    assignee = data.get("assignee")
    if isinstance(assignee, dict):
        name = assignee.get("name") or assignee.get("displayName")
        return str(name) if name else None
    return None


def _pending_subtasks(service: AgentConnectService, task_id: str, target: Optional[str]):
    pending = [
        s for s in service.get_task(task_id).subtasks
        if s.status is SubtaskStatus.needs_approval
    ]
    if target is None:
        return pending
    return [s for s in pending if s.route_reason.get("approval_location") == target]


def _handle_comment(
    service: AgentConnectService, event: WebhookEvent
) -> list[dict[str, Any]]:
    issue_id = _issue_id(event.data)
    if not issue_id:
        return [{"kind": "ignored", "reason": "comment carries no issue id"}]
    task = service.find_task_by_external_id(issue_id)
    if task is None:
        return [{"kind": "ignored", "reason": f"no task mapped to Linear issue {issue_id}"}]

    command = parse_command(str(event.data.get("body") or ""))
    if command is None:
        return [{"kind": "ignored", "reason": "no /agentconnect command in comment"}]

    actor = _actor(event.data)
    service.record_event(
        task.id, "linear_command", actor,
        {"action": command.action, "target": command.target, "params": command.params},
    )

    if command.action in ("status", "complete", "request-review"):
        return [_handle_task_command(service, task, command, actor)]

    pending = _pending_subtasks(service, task.id, command.target)
    if not pending:
        return [{
            "kind": "approval_no_match", "task_id": task.id, "action": command.action,
            "target": command.target,
            "reason": "no subtask is awaiting approval for that route",
        }]

    results: list[dict[str, Any]] = []
    for subtask in pending:
        if command.action == "approve":
            updated = service.approve_subtask(subtask.id, actor, command.max_cost_usd)
            results.append({
                "kind": "approved", "task_id": task.id, "subtask_id": subtask.id,
                "status": updated.status.value, "max_cost_usd": command.max_cost_usd,
            })
        else:
            updated = service.deny_subtask(subtask.id, actor, command.reason)
            results.append({
                "kind": "denied", "task_id": task.id, "subtask_id": subtask.id,
                "status": updated.status.value, "reason": command.reason,
            })
    return results


def _handle_task_command(
    service: AgentConnectService, task: Any, command: ApprovalCommand, actor: str,
) -> dict[str, Any]:
    """`status`, `complete`, and `request-review` from a Linear comment (§14).

    `complete` runs the audit first. A human typing "done" in a tracker does not
    make work complete; the ledger does. On failure the problems come back so the
    human sees exactly what was never recorded.
    """
    if command.action == "status":
        report = service.audit_task(task.id)
        return {"kind": "status", "task_id": task.id, "status": task.status.value,
                "audit": "PASS" if report.passed else "FAIL",
                "problems": report.problems}

    if command.action == "request-review":
        reviewer = command.target or command.params.get("reason") or ""
        reviewer = reviewer.strip()
        if not reviewer:
            return {"kind": "review_no_assignee", "task_id": task.id,
                    "reason": "usage: /agentconnect request-review <manager-id>"}
        review = service.request_review(task.id, ReviewRequest(
            requested_by=actor, assigned_to=reviewer,
            criteria=[c.text for c in service.get_task(task.id).constraints],
        ))
        return {"kind": "review_requested", "task_id": task.id,
                "review_id": review.id, "assigned_to": reviewer}

    try:
        result = service.complete_task(task.id, completed_by=actor)
    except PolicyViolation as exc:
        report = service.audit_task(task.id)
        return {"kind": "completion_refused", "task_id": task.id,
                "problems": report.problems, "reason": str(exc)}
    except Conflict as exc:
        return {"kind": "completion_refused", "task_id": task.id, "reason": str(exc)}
    return {"kind": "completed", "task_id": task.id, "status": result["status"],
            "audit": "PASS"}


def _handle_issue_update(
    service: AgentConnectService, event: WebhookEvent
) -> list[dict[str, Any]]:
    issue_id = str(event.data.get("id") or "")
    task = service.find_task_by_external_id(issue_id) if issue_id else None
    if task is None:
        return [{"kind": "ignored", "reason": f"no task mapped to Linear issue {issue_id}"}]

    results: list[dict[str, Any]] = []
    changed = event.updated_from
    actor = _actor(event.data)

    # When Linear tells us which fields changed, respect it; when it does not
    # (some payload shapes omit updatedFrom), fall back to what is present.
    state_changed = ("stateId" in changed) if changed else ("state" in event.data)
    assignee_changed = ("assigneeId" in changed) if changed else ("assignee" in event.data)

    if state_changed:
        state = event.data.get("state") or {}
        name = state.get("name") if isinstance(state, dict) else None
        service.record_event(
            task.id, "linear_status_change", actor, {"state": name, "issue_id": issue_id}
        )
        # Deliberately NOT mirrored onto task.status: the ledger owns status (§6).
        results.append({"kind": "status_recorded", "task_id": task.id, "state": name})

    if assignee_changed:
        manager_id = _assignee(event.data)
        if manager_id:
            service.add_inbox_item(
                manager_id=manager_id, kind=InboxKind.task, ref_id=task.id, task_id=task.id,
                title=f"Assigned in Linear: {task.title}",
            )
            service.record_event(
                task.id, "linear_assigned", actor, {"manager_id": manager_id}
            )
            results.append({
                "kind": "inbox_item", "task_id": task.id, "manager_id": manager_id,
            })
        else:
            results.append({"kind": "unassigned", "task_id": task.id})

    return results or [{"kind": "ignored", "reason": "no actionable field changed"}]


def handle_webhook(
    service: AgentConnectService, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply a Linear webhook to the ledger. Returns one result per action taken."""
    event = parse_webhook(payload)
    if event.type == "Comment" and event.action in ("create", "update"):
        return _handle_comment(service, event)
    if event.type == "Issue" and event.action == "update":
        return _handle_issue_update(service, event)
    return [{"kind": "ignored", "reason": f"unhandled {event.type}/{event.action}"}]
