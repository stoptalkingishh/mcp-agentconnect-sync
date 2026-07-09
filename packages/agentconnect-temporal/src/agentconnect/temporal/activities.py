"""Activities: every side effect a workflow needs, and nothing else.

The spec sketches one module per activity. They are gathered on one class here
because each is a two-line call into `AgentConnectService` and they all need the
same injected service, worker registry, and (optional) Linear sync — a bound
instance is the idiomatic way temporalio carries that dependency. The activity
*names* are exactly the spec's, and workflows reference them by name only.

**Idempotency.** Temporal retries activities. Each one below is safe to run
twice:

* ``route_subtask`` — routing is deterministic; re-routing recomputes the same
  answer and rewrites the same rows.
* ``run_worker`` — a subtask that already reached a terminal state is returned
  as-is rather than re-executed, so a retry after a lost heartbeat never
  double-runs a worker's side effects.
* ``request_approval`` — reuses the pending approval record if one exists.
* ``update_linear`` — posts against the stored `ExternalRef`; a re-post is a
  duplicate comment at worst, never a second issue.
* ``record_approval_decision`` — a decided approval is left alone.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from temporalio import activity

from agentconnect.core.errors import AgentConnectError, Conflict
from agentconnect.core.memory import (
    CaptureRequest,
    MemoryFeedbackRequest,
    MemoryScope,
    RecallRequest,
)
from agentconnect.core.models import ActorType, InboxKind, RecordAttemptRequest
from agentconnect.core.service import AgentConnectService

_log = logging.getLogger(__name__)


class BackplaneActivities:
    def __init__(
        self, service: AgentConnectService, linear_sync: Optional[Any] = None
    ) -> None:
        self.service = service
        self.linear_sync = linear_sync

    def all(self) -> list[Any]:
        return [
            self.route_subtask, self.run_worker, self.request_approval,
            self.record_approval_decision, self.cancel_subtask, self.deny_subtask,
            self.save_artifact, self.record_attempt, self.update_linear,
            self.load_review, self.complete_review, self.load_approval,
            self.notify_manager, self.recall_memory, self.capture_memory_candidate,
            self.record_memory_feedback,
        ]

    # ------------------------------------------------------------- decisions
    @activity.defn(name="route_subtask")
    async def route_subtask(self, subtask_id: str) -> dict[str, Any]:
        explanation = self.service.route_subtask(subtask_id)
        payload = explanation.model_dump(mode="json")
        subtask = self.service.get_subtask(subtask_id).subtask
        payload["task_id"] = subtask.parent_task_id
        return payload

    @activity.defn(name="request_approval")
    async def request_approval(self, subtask_id: str) -> dict[str, Any]:
        record = self.service.pending_approval_for(subtask_id)
        if record is None:
            raise ValueError(f"subtask {subtask_id} has no pending approval to request")
        if self.linear_sync is not None:
            try:
                self.linear_sync.post_approval_request(subtask_id)
            except Exception as exc:  # the human can still approve via CLI/API
                _log.warning("Linear approval request failed for %s: %s", subtask_id, exc)
        return record.model_dump(mode="json")

    @activity.defn(name="record_approval_decision")
    async def record_approval_decision(
        self, approval_id: str, decision: Optional[str], decided_by: Optional[str],
        max_cost_usd: Optional[float], reason: str,
    ) -> dict[str, Any]:
        record = self.service.get_approval(approval_id)
        if record.status.value != "pending":
            return record.model_dump(mode="json")  # already decided; replay-safe
        if decision == "granted":
            self.service.grant_approval(record.subtask_id, decided_by or "human", max_cost_usd)
        elif decision == "denied":
            self.service.deny_subtask(record.subtask_id, decided_by or "human", reason)
        else:
            self.service.storage.update_approval(
                approval_id, status="expired", decided_at=self.service._now()
            )
        return self.service.get_approval(approval_id).model_dump(mode="json")

    # ------------------------------------------------------------ execution
    @activity.defn(name="run_worker")
    async def run_worker(self, subtask_id: str) -> dict[str, Any]:
        try:
            subtask = self.service.run_subtask(subtask_id)
        except Conflict:
            # Already terminal (a prior attempt succeeded before the heartbeat
            # was lost). Report what the ledger says rather than re-running.
            subtask = self.service.get_subtask(subtask_id).subtask
        return {
            "subtask_id": subtask.id, "task_id": subtask.parent_task_id,
            "status": subtask.status.value, "assigned_worker": subtask.assigned_worker,
            "result_artifact_id": subtask.result_artifact_id,
        }

    @activity.defn(name="cancel_subtask")
    async def cancel_subtask(self, subtask_id: str) -> dict[str, Any]:
        try:
            self.service.cancel_subtask(subtask_id)
        except Conflict:
            pass  # already terminal
        return {"subtask_id": subtask_id, "status": "cancelled"}

    @activity.defn(name="deny_subtask")
    async def deny_subtask(self, subtask_id: str, denied_by: str, reason: str) -> dict[str, Any]:
        subtask = self.service.deny_subtask(subtask_id, denied_by, reason)
        return {"subtask_id": subtask.id, "status": subtask.status.value}

    # ------------------------------------------------------------- ledger IO
    @activity.defn(name="save_artifact")
    async def save_artifact(
        self, task_id: str, artifact_type: str, content: str, summary: str, created_by: str,
    ) -> dict[str, Any]:
        from agentconnect.core.models import ArtifactType, CreateArtifactRequest

        artifact = self.service.create_artifact(task_id, CreateArtifactRequest(
            type=ArtifactType(artifact_type), content=content, summary=summary,
            created_by=created_by,
        ))
        return {"artifact_id": artifact.id, "size_bytes": artifact.size_bytes}

    @activity.defn(name="record_attempt")
    async def record_attempt(
        self, task_id: str, actor_id: str, actor_type: str, summary: str, outcome: str,
        artifact_refs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        attempt = self.service.record_attempt(task_id, RecordAttemptRequest(
            actor_id=actor_id, actor_type=ActorType(actor_type), summary=summary,
            outcome=outcome, artifact_refs=artifact_refs or [],
        ))
        return {"attempt_id": attempt.id}

    @activity.defn(name="load_review")
    async def load_review(self, review_id: str) -> dict[str, Any]:
        return self.service.get_review(review_id).model_dump(mode="json")

    @activity.defn(name="complete_review")
    async def complete_review(
        self, review_id: str, completed_by: str, summary: str, content: str,
    ) -> dict[str, Any]:
        from agentconnect.core.models import ReviewResultRequest

        review = self.service.complete_review(review_id, ReviewResultRequest(
            completed_by=completed_by, summary=summary, content=content,
        ))
        return review.model_dump(mode="json")

    @activity.defn(name="load_approval")
    async def load_approval(self, approval_id: str) -> dict[str, Any]:
        return self.service.get_approval(approval_id).model_dump(mode="json")

    # -------------------------------------------------------------- outward
    @activity.defn(name="update_linear")
    async def update_linear(self, task_id: str, kind: str, ref_id: str) -> None:
        """Mirror one change outward. A tracker outage is logged, never raised —
        the ledger is already correct and the workflow must not fail on it."""
        if self.linear_sync is None or not task_id:
            return None
        try:
            if kind == "subtask":
                self.linear_sync.post_subtask(ref_id)
            elif kind == "review_request":
                self.linear_sync.post_review_request(ref_id)
            elif kind == "review_result":
                self.linear_sync.post_review_result(ref_id)
            elif kind == "approval_request":
                self.linear_sync.post_approval_request(ref_id)
            else:
                self.linear_sync.sync_task(task_id)
        except Exception as exc:
            _log.warning("Linear update (%s, %s) failed: %s", kind, ref_id, exc)
        return None

    @activity.defn(name="notify_manager")
    async def notify_manager(self, manager_id: str, message: str) -> None:
        if not manager_id:
            return None
        # A content digest, not `hash()`: PYTHONHASHSEED varies per process, so
        # `hash()` would mint a fresh ref_id on every restart and defeat the
        # inbox's (manager_id, kind, ref_id) uniqueness on retry.
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
        self.service.add_inbox_item(
            manager_id=manager_id, kind=InboxKind.task,
            ref_id=f"notice:{digest}", title=message,
        )
        return None

    # --------------------------------------------------------------- memory
    @activity.defn(name="recall_memory")
    async def recall_memory(
        self, task_id: Optional[str], query: str, profile: str = "manager_brief",
        max_items: int = 8,
    ) -> dict[str, Any]:
        """Memory lives in an activity, never in workflow code. A backend failure
        yields an empty pack with a warning — it never fails the workflow."""
        pack = self.service.recall_memory(RecallRequest(
            query=query, task_id=task_id, profile=profile,  # type: ignore[arg-type]
            max_items=max_items,
            scopes=[MemoryScope("task", task_id)] if task_id else [],
        ))
        return {
            "backend": pack.backend, "profile": pack.profile, "query": pack.query,
            "warnings": pack.warnings,
            "items": [
                {"text": i.text, "status": i.status, "confidence": i.confidence,
                 "source_id": i.source_id}
                for i in pack.items
            ],
        }

    @activity.defn(name="capture_memory_candidate")
    async def capture_memory_candidate(
        self, task_id: Optional[str], text: str, origin_actor_id: str, origin_actor_type: str,
    ) -> dict[str, Any]:
        result = self.service.capture_memory_candidate(CaptureRequest(
            text=text, task_id=task_id, origin_actor_id=origin_actor_id,
            origin_actor_type=origin_actor_type,
        ))
        return {
            "accepted": result.accepted, "candidate_id": result.candidate_id,
            "status": result.status, "backend": result.backend, "message": result.message,
        }

    @activity.defn(name="record_memory_feedback")
    async def record_memory_feedback(
        self, task_id: Optional[str], memory_item_id: Optional[str], source_id: Optional[str],
        feedback: str, actor_id: Optional[str] = None, note: Optional[str] = None,
    ) -> None:
        self.service.record_memory_feedback(MemoryFeedbackRequest(
            task_id=task_id, memory_item_id=memory_item_id, source_id=source_id,
            feedback=feedback, actor_id=actor_id, note=note,
        ))
        return None


__all__ = ["AgentConnectError", "BackplaneActivities"]
