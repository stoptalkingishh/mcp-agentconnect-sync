"""Push-only sync: AgentConnect → Linear (spec §14.4 mode 1).

Sync is idempotent on the `ExternalRef`: the first call creates an issue and
stores the mapping, later calls update that issue in place. AgentConnect never
reads state back from Linear to decide anything — Linear is a mirror (§6).
"""

from __future__ import annotations

import logging
from typing import Optional

from agentconnect.core.errors import NotFound
from agentconnect.core.models import ExternalRef
from agentconnect.core.routing import RouteExplanation
from agentconnect.core.service import AgentConnectService

from . import mapping
from .client import LinearClient

_log = logging.getLogger(__name__)

PROVIDER = "linear"


class LinearSync:
    def __init__(
        self,
        service: AgentConnectService,
        client: LinearClient,
        team_id: str,
        artifact_base_url: Optional[str] = None,
    ) -> None:
        self.service = service
        self.client = client
        self.team_id = team_id
        self.artifact_base_url = artifact_base_url
        self._label_cache: Optional[dict[str, str]] = None

    # ------------------------------------------------------------- internals
    def _label_ids(self, names: list[str]) -> list[str]:
        """Resolve label names to IDs, skipping any that do not exist in the
        team. The backplane mirrors workflow; it does not administer Linear."""
        if self._label_cache is None:
            try:
                self._label_cache = self.client.list_labels(self.team_id)
            except Exception as exc:  # a mirror must not break the ledger
                _log.warning("could not list Linear labels (%s); syncing without labels", exc)
                self._label_cache = {}
        missing = [n for n in names if n not in self._label_cache]
        if missing:
            _log.info("Linear labels not present in team, skipping: %s", ", ".join(missing))
        return [self._label_cache[n] for n in names if n in self._label_cache]

    def _issue_id(self, task_id: str) -> Optional[str]:
        ref = self.service.get_external_ref("task", task_id, PROVIDER)
        return ref.external_id if ref and ref.sync_enabled else None

    def _comment(self, task_id: str, body: str) -> bool:
        issue_id = self._issue_id(task_id)
        if not issue_id:
            _log.info("task %s is not synced to Linear; skipping comment", task_id)
            return False
        self.client.create_comment(issue_id, body)
        return True

    # ----------------------------------------------------------------- push
    def sync_task(self, task_id: str) -> ExternalRef:
        """Create or update the mirror issue. Returns the stored `ExternalRef`."""
        detail = self.service.get_task(task_id)
        handoff = self.service.get_handoff_summary(task_id)
        title = mapping.issue_title(detail)
        body = mapping.issue_body(detail, handoff.text)
        label_ids = self._label_ids(mapping.labels_for(detail))

        existing = self.service.get_external_ref("task", task_id, PROVIDER)
        if existing is None:
            issue = self.client.create_issue(self.team_id, title, body, label_ids)
        else:
            issue = self.client.update_issue(existing.external_id, title, body, label_ids)

        return self.service.set_external_ref(
            entity_type="task", entity_id=task_id, provider=PROVIDER,
            external_id=issue["id"], external_url=issue.get("url"),
            metadata={"identifier": issue.get("identifier")},
        )

    def post_decision(self, task_id: str, decision_id: str) -> bool:
        detail = self.service.get_task(task_id)
        found = next((d for d in detail.decisions if d.id == decision_id), None)
        if found is None:
            raise NotFound(f"unknown decision {decision_id!r} on task {task_id}")
        return self._comment(
            task_id, mapping.decision_comment(found, mapping.is_withheld(detail))
        )

    def post_artifact(self, task_id: str, artifact_id: str) -> bool:
        detail = self.service.get_task(task_id)
        found = next((a for a in detail.artifacts if a.id == artifact_id), None)
        if found is None:
            raise NotFound(f"unknown artifact {artifact_id!r} on task {task_id}")
        return self._comment(
            task_id,
            mapping.artifact_comment(
                found, mapping.is_withheld(detail), self.artifact_base_url
            ),
        )

    def post_review_request(self, review_id: str) -> bool:
        review = self.service.get_review(review_id)
        detail = self.service.get_task(review.task_id)
        return self._comment(
            review.task_id, mapping.review_request_comment(review, mapping.is_withheld(detail))
        )

    def post_review_result(self, review_id: str) -> bool:
        review = self.service.get_review(review_id)
        detail = self.service.get_task(review.task_id)
        return self._comment(
            review.task_id, mapping.review_result_comment(review, mapping.is_withheld(detail))
        )

    def post_subtask(self, subtask_id: str) -> bool:
        subtask = self.service.get_subtask(subtask_id).subtask
        detail = self.service.get_task(subtask.parent_task_id)
        return self._comment(
            subtask.parent_task_id,
            mapping.subtask_comment(subtask, mapping.is_withheld(detail)),
        )

    def post_approval_request(self, subtask_id: str) -> bool:
        """§15 steps 4-5. Also re-syncs the issue so the `needs-approval` label lands."""
        subtask = self.service.get_subtask(subtask_id).subtask
        explanation = RouteExplanation(**subtask.route_reason)
        self.sync_task(subtask.parent_task_id)
        return self._comment(
            subtask.parent_task_id, mapping.approval_request_comment(subtask, explanation)
        )
