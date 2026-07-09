"""`AgentConnectService` — the one place work state changes (spec §5, §10).

MCP tools, HTTP routes, CLI commands, and Linear webhooks are *adapters*: they
translate a protocol into a call on this class and translate the result back.
None of them may touch storage, the artifact store, or the router directly. That
is the whole reason a manager can be swapped mid-task without the ledger
noticing.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

from . import audit as audit_mod
from . import claims as claims_policy
from . import decisions as decisions_policy
from . import handoff as handoff_mod
from . import ids
from . import memory as memory_mod
from . import reviews as reviews_policy
from . import sessions as sessions_mod
from . import subtasks as subtasks_policy
from .context import ContextBuilder, ContextPack, MemoryConfig
from .memory import (
    TRUSTED_AUTHORITY,
    CaptureRequest,
    CaptureResult,
    IndexingMemoryAdapter,
    MemoryAdapter,
    MemoryFeedbackRequest,
    NoopMemoryAdapter,
    RecallPack,
    RecallRequest,
    TrustedMemoryAdapter,
)
from .artifacts import FilesystemArtifactStore
from .audit import AuditReport
from .errors import Conflict, InvalidRequest, NotFound, PolicyViolation
from .execution import DirectExecutionBackend, ExecutionBackend, ExecutionHandle, ExecutionState
from .workspace import WorkspaceBuilder, mode_for
from .models import (
    ApprovalRecord,
    ApprovalStatus,
    Artifact,
    ArtifactChunk,
    ArtifactSummary,
    ArtifactType,
    ActorType,
    Attempt,
    Claim,
    ClaimRole,
    Constraint,
    CreateArtifactRequest,
    CreateTaskRequest,
    Decision,
    Event,
    ExternalRef,
    HandoffSummary,
    InboxItem,
    InboxKind,
    ManagerSession,
    PrivacyTier,
    RecordAttemptRequest,
    RecordDecisionRequest,
    Review,
    ReviewRequest,
    ReviewResultRequest,
    ReviewStatus,
    RunStatus,
    SessionMode,
    SessionStatus,
    SessionToken,
    Subtask,
    SubtaskDetail,
    SubtaskRequest,
    SubtaskStatus,
    Task,
    TaskDetail,
    TaskFilters,
    TaskStatus,
    TaskSummary,
    TERMINAL_TASK_STATUSES,
    WorkerRun,
    Workspace,
)
from .routing import RouteExplanation, RoutePolicy, WorkerRegistry, route
from .storage import SqliteStorage, default_db_path
from .workers import WorkerContext, WorkerResult

_log = logging.getLogger(__name__)

DEFAULT_CLAIM_TTL_SECONDS = 3600


#: `ContextPack` is defined in `core.context`; re-exported here because it is the
#: return type of a service method and callers should not need two imports.
TaskContextPack = ContextPack

#: Which statuses each transition is allowed to leave. Anything else is a no-op,
#: so a running subtask never clears a `needs_review` task and a terminal task is
#: never resurrected.
_TO_IN_PROGRESS = frozenset({TaskStatus.queued, TaskStatus.needs_approval})
_TO_NEEDS_REVIEW = frozenset({TaskStatus.queued, TaskStatus.in_progress})
_TO_NEEDS_APPROVAL = frozenset({TaskStatus.queued, TaskStatus.in_progress})

#: Executions that can still receive a signal.
_LIVE_EXECUTION_STATES = frozenset({
    ExecutionState.running, ExecutionState.waiting_approval, ExecutionState.waiting_review,
})


class AgentConnectService:
    def __init__(
        self,
        storage: SqliteStorage,
        artifact_store: FilesystemArtifactStore,
        registry: Optional[WorkerRegistry] = None,
        policy: Optional[RoutePolicy] = None,
        clock: Callable[[], float] = time.time,
        execution: Optional[ExecutionBackend] = None,
        memory: Optional[MemoryAdapter] = None,
        memory_backends: Optional[dict[str, MemoryAdapter]] = None,
        memory_config: Optional[MemoryConfig] = None,
        workspace_dir: Optional[str] = None,
        api_url: str = "http://localhost:8790",
    ) -> None:
        self.storage = storage
        self.artifacts = artifact_store
        self.workspaces = WorkspaceBuilder(workspace_dir)
        self.api_url = api_url
        #: Called with a task id *after* the ledger marks it succeeded. Linear
        #: registers here, which is what makes AgentConnect the upstream of the
        #: tracker rather than the other way round (compliance §13).
        self.completion_hooks: list[Callable[[str], None]] = []
        self.registry = registry or WorkerRegistry()
        self.policy = policy or RoutePolicy()
        self._clock = clock
        # Memory off by default: the backplane is a task ledger, not a brain.
        self.memory_config = memory_config or MemoryConfig()
        self.memory_backends: dict[str, MemoryAdapter] = dict(memory_backends or {})
        if not self.memory_backends and memory is not None:
            self.memory_backends = {memory.backend_name: memory}
        # The single-adapter path (`recall_memory`, `capture_memory_candidate`)
        # answers from the trusted authority whenever one is configured, so a
        # bare recall never returns a search hit dressed up as a claim.
        self.memory: MemoryAdapter = (
            memory
            or self.memory_backends.get(self.memory_config.trusted_authority)
            or next(iter(self.memory_backends.values()), None)
            or NoopMemoryAdapter()
        )
        self.context_builder = ContextBuilder(
            self, self.memory_backends, self.memory_config
        )
        # Default to inline execution so `pip install agentconnect-core` is a
        # working backplane with no workflow server. Adapters that want
        # durability call `bind_execution(TemporalExecutionBackend(...))`.
        self.execution: ExecutionBackend = execution or DirectExecutionBackend(self)

    def bind_execution(self, backend: ExecutionBackend) -> None:
        self.execution = backend

    def bind_completion_hook(self, hook: Callable[[str], None]) -> None:
        """Run after a task is marked succeeded. Never before: the ledger decides."""
        self.completion_hooks.append(hook)

    def bind_memory(self, adapter: MemoryAdapter) -> None:
        """Single-backend convenience. Prefer `bind_memory_stack` for the real stack."""
        self.memory = adapter
        self.bind_memory_stack({adapter.backend_name: adapter}, self.memory_config)

    def bind_memory_stack(
        self, adapters: dict[str, MemoryAdapter], config: Optional[MemoryConfig] = None
    ) -> None:
        """Bind WikiBrain + Cognee + Graphiti (or any subset — each is optional)."""
        self.memory_backends = dict(adapters)
        if config is not None:
            self.memory_config = config
        authority = self.memory_config.trusted_authority
        # `self.memory` remains the single-adapter recall path; point it at the
        # trusted authority when one is present, so a bare `recall_memory` call
        # never accidentally answers from a retrieval engine.
        self.memory = adapters.get(authority) or next(
            iter(adapters.values()), NoopMemoryAdapter()
        )
        self.context_builder = ContextBuilder(self, self.memory_backends, self.memory_config)

    def trusted_authority(self) -> Optional[TrustedMemoryAdapter]:
        adapter = self.memory_backends.get(self.memory_config.trusted_authority)
        return adapter if isinstance(adapter, TrustedMemoryAdapter) else None

    @classmethod
    def create(
        cls,
        db_path: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        workers: Optional[list[Any]] = None,
        policy: Optional[RoutePolicy] = None,
        clock: Callable[[], float] = time.time,
        execution: Optional[ExecutionBackend] = None,
        memory: Optional[MemoryAdapter] = None,
        memory_backends: Optional[dict[str, MemoryAdapter]] = None,
        memory_config: Optional[MemoryConfig] = None,
        workspace_dir: Optional[str] = None,
        api_url: str = "http://localhost:8790",
    ) -> "AgentConnectService":
        return cls(
            storage=SqliteStorage(db_path or default_db_path()),
            artifact_store=FilesystemArtifactStore(artifact_dir),
            registry=WorkerRegistry(workers or []),
            policy=policy,
            clock=clock,
            execution=execution,
            memory=memory,
            memory_backends=memory_backends,
            memory_config=memory_config,
            workspace_dir=workspace_dir,
            api_url=api_url,
        )

    # ------------------------------------------------------------ internals
    def _now(self) -> float:
        return self._clock()

    def _require_task(self, task_id: str) -> Task:
        task = self.storage.get_task(task_id)
        if task is None:
            raise NotFound(f"unknown task {task_id!r}")
        return task

    def _require_subtask(self, subtask_id: str) -> Subtask:
        subtask = self.storage.get_subtask(subtask_id)
        if subtask is None:
            raise NotFound(f"unknown subtask {subtask_id!r}")
        return subtask

    def _require_review(self, review_id: str) -> Review:
        review = self.storage.get_review(review_id)
        if review is None:
            raise NotFound(f"unknown review {review_id!r}")
        return review

    def _touch(self, task_id: str, **fields: Any) -> None:
        self.storage.update_task(task_id, updated_at=self._now(), **fields)

    def _advance_task(
        self, task_id: str, to: TaskStatus, only_from: frozenset[TaskStatus]
    ) -> None:
        """Move a task's status, but only out of the states named in ``only_from``.

        The guard matters: a subtask starting to run must not silently clear a
        ``needs_review`` task, and a terminal task must never be resurrected by a
        late worker result or a replayed webhook.
        """
        task = self._require_task(task_id)
        if task.status in TERMINAL_TASK_STATUSES or task.status is to:
            return
        if task.status not in only_from:
            return
        self._touch(task_id, status=to.value)

    # ----------------------------------------------------------------- tasks
    def create_task(self, request: CreateTaskRequest) -> Task:
        now = self._now()
        task = Task(
            id=ids.new_id(ids.TASK), title=request.title, goal=request.goal,
            status=TaskStatus.queued, priority=request.priority, created_by=request.created_by,
            created_at=now, updated_at=now, metadata=request.metadata,
        )
        self.storage.insert_task(task)
        for text in request.constraints:
            self.add_constraint(task.id, text, request.created_by)
        return task

    def add_constraint(self, task_id: str, text: str, created_by: str = "unknown") -> Constraint:
        self._require_task(task_id)
        constraint = Constraint(
            id=ids.new_id(ids.CONSTRAINT), task_id=task_id, text=text,
            created_by=created_by, created_at=self._now(),
        )
        self.storage.insert_constraint(constraint)
        self._touch(task_id)
        return constraint

    def get_task(self, task_id: str) -> TaskDetail:
        task = self._require_task(task_id)
        now = self._now()
        return TaskDetail(
            task=task,
            constraints=self.storage.list_constraints(task_id),
            active_claims=self.storage.active_claims(task_id, now),
            decisions=self.storage.list_decisions(task_id),
            attempts=self.storage.list_attempts(task_id),
            artifacts=self.storage.list_artifacts(task_id),
            reviews=self.storage.list_reviews(task_id),
            subtasks=self.storage.list_subtasks(task_id),
        )

    def list_tasks(self, filters: Optional[TaskFilters] = None) -> list[TaskSummary]:
        return self.storage.list_tasks(filters or TaskFilters())

    def cancel_task(self, task_id: str, actor: str = "system") -> Task:
        task = self._require_task(task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            raise Conflict(f"task {task_id} is already {task.status.value}")
        self._touch(task_id, status=TaskStatus.cancelled.value)
        self.record_event(task_id, "task_cancelled", actor, {})
        return self._require_task(task_id)

    # ---------------------------------------------------------------- claims
    def claim_task(
        self,
        task_id: str,
        manager_id: str,
        role: str = ClaimRole.primary_manager.value,
        ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    ) -> Claim:
        self._require_task(task_id)
        try:
            role_enum = ClaimRole(role)
        except ValueError:
            raise InvalidRequest(f"unknown claim role {role!r}") from None
        ttl = claims_policy.validate_ttl(ttl_seconds)
        now = self._now()

        # Check-and-insert must be one serialized span or two managers racing to
        # claim can both observe "no holder" and both insert.
        with self.storage.transaction() as conn:
            existing = self.storage.active_claims(task_id, now, conn=conn)
            if role_enum is ClaimRole.primary_manager:
                claims_policy.check_primary_exclusivity(existing, manager_id, now)
            claim = Claim(
                id=ids.new_id(ids.CLAIM), task_id=task_id, manager_id=manager_id,
                role=role_enum, expires_at=now + ttl, created_at=now,
            )
            self.storage.insert_claim(claim, conn=conn)
            if role_enum is ClaimRole.primary_manager:
                task = self._require_task(task_id)
                status = (
                    task.status.value
                    if task.status not in (TaskStatus.queued,)
                    else TaskStatus.in_progress.value
                )
                conn.execute(
                    "UPDATE tasks SET current_manager=?, status=?, updated_at=? WHERE id=?",
                    (manager_id, status, now, task_id),
                )
        self.record_event(task_id, "task_claimed", manager_id, {"role": role_enum.value})
        return claim

    def release_task(self, task_id: str, manager_id: str) -> None:
        task = self._require_task(task_id)
        now = self._now()
        released = self.storage.release_claims(task_id, manager_id, now)
        if not released:
            raise NotFound(f"{manager_id!r} holds no active claim on {task_id}")
        if task.current_manager == manager_id:
            self._touch(task_id, current_manager=None)
        self.record_event(task_id, "task_released", manager_id, {})

    def list_claims(self, task_id: str) -> list[Claim]:
        self._require_task(task_id)
        return self.storage.list_claims(task_id)

    # ------------------------------------------------------------- decisions
    def record_decision(self, task_id: str, request: RecordDecisionRequest) -> Decision:
        self._require_task(task_id)
        now = self._now()
        with self.storage.transaction() as conn:
            targets, missing = [], []
            for decision_id in request.supersedes:
                found = self.storage.get_decision(decision_id, conn=conn)
                (targets if found else missing).append(found or decision_id)
            if request.supersedes:
                active = self.storage.active_claims(task_id, now, conn=conn)
                decisions_policy.check_supersede_allowed(
                    targets, missing, request.made_by, active, now
                )
            decision = Decision(
                id=ids.new_id(ids.DECISION), task_id=task_id, made_by=request.made_by,
                decision=request.decision, rationale=request.rationale, locked=request.locked,
                created_at=now,
            )
            self.storage.insert_decision(decision, conn=conn)
            for target in targets:
                self.storage.mark_superseded(target.id, decision.id, conn=conn)
            conn.execute("UPDATE tasks SET updated_at=? WHERE id=?", (now, task_id))
        return decision

    def list_decisions(self, task_id: str) -> list[Decision]:
        self._require_task(task_id)
        return self.storage.list_decisions(task_id)

    # -------------------------------------------------------------- attempts
    def record_attempt(self, task_id: str, request: RecordAttemptRequest) -> Attempt:
        self._require_task(task_id)
        for ref in request.artifact_refs:
            if self.storage.get_artifact(ref) is None:
                raise NotFound(f"unknown artifact {ref!r}")
        attempt = Attempt(
            id=ids.new_id(ids.ATTEMPT), task_id=task_id, actor_id=request.actor_id,
            actor_type=request.actor_type, summary=request.summary, outcome=request.outcome,
            created_at=self._now(), artifact_refs=request.artifact_refs,
        )
        self.storage.insert_attempt(attempt)
        self._touch(task_id)
        return attempt

    # ------------------------------------------------------------- artifacts
    def create_artifact(self, task_id: str, request: CreateArtifactRequest) -> Artifact:
        self._require_task(task_id)
        artifact_id = ids.new_id(ids.ARTIFACT)
        rel_path, size = self.artifacts.write(task_id, artifact_id, request.content)
        artifact = Artifact(
            id=artifact_id, task_id=task_id, type=request.type, path=rel_path,
            summary=request.summary, created_by=request.created_by, created_at=self._now(),
            size_bytes=size, metadata=request.metadata,
        )
        self.storage.insert_artifact(artifact)
        self._touch(task_id)
        return artifact

    def get_artifact(self, artifact_id: str) -> Artifact:
        artifact = self.storage.get_artifact(artifact_id)
        if artifact is None:
            raise NotFound(f"unknown artifact {artifact_id!r}")
        return artifact

    def list_artifacts(self, task_id: str) -> list[ArtifactSummary]:
        self._require_task(task_id)
        return self.storage.list_artifacts(task_id)

    def read_artifact_chunk(
        self, artifact_id: str, offset: int = 0, limit: int = 8000
    ) -> ArtifactChunk:
        artifact = self.get_artifact(artifact_id)
        content, next_offset, eof, size = self.artifacts.read_chunk(artifact.path, offset, limit)
        return ArtifactChunk(
            artifact_id=artifact_id, offset=offset, limit=limit, content=content,
            next_offset=next_offset, eof=eof, size_bytes=size,
        )

    # --------------------------------------------------------------- reviews
    def request_review(self, task_id: str, request: ReviewRequest) -> Review:
        self._require_task(task_id)
        if request.assigned_to == request.requested_by:
            raise InvalidRequest("a manager cannot assign a review to itself")
        for ref in request.artifact_refs:
            artifact = self.storage.get_artifact(ref)
            if artifact is None:
                raise NotFound(f"unknown artifact {ref!r}")
            if artifact.task_id != task_id:
                raise InvalidRequest(f"artifact {ref!r} does not belong to task {task_id}")
        now = self._now()
        review = Review(
            id=ids.new_id(ids.REVIEW), task_id=task_id, requested_by=request.requested_by,
            assigned_to=request.assigned_to, status=ReviewStatus.open,
            criteria=request.criteria, artifact_refs=request.artifact_refs,
            created_at=now, updated_at=now,
        )
        self.storage.insert_review(review)
        self.storage.insert_inbox_item(
            InboxItem(
                id=ids.new_id(ids.INBOX), manager_id=request.assigned_to, kind=InboxKind.review,
                ref_id=review.id, task_id=task_id,
                title=f"Review requested by {request.requested_by}", created_at=now,
            )
        )
        self._advance_task(task_id, TaskStatus.needs_review, _TO_NEEDS_REVIEW)
        self.record_event(task_id, "review_requested", request.requested_by,
                          {"review_id": review.id, "assigned_to": request.assigned_to})
        self.execution.start_review(review.id)
        return review

    def get_review(self, review_id: str) -> Review:
        return self._require_review(review_id)

    def claim_review(self, review_id: str, manager_id: str) -> Review:
        now = self._now()
        with self.storage.transaction() as conn:
            review = self.storage.get_review(review_id, conn=conn)
            if review is None:
                raise NotFound(f"unknown review {review_id!r}")
            reviews_policy.check_claimable(review, manager_id)
            self.storage.update_review(
                review_id, conn=conn, status=ReviewStatus.claimed.value, updated_at=now
            )
        return self._require_review(review_id)

    def complete_review(self, review_id: str, request: ReviewResultRequest) -> Review:
        review = self._require_review(review_id)
        reviews_policy.check_completable(review, request.completed_by)
        if request.status not in (ReviewStatus.completed, ReviewStatus.rejected):
            raise InvalidRequest(
                f"a review result must be completed or rejected, got {request.status.value}"
            )

        result_artifact_id = request.result_artifact_id
        if result_artifact_id is None:
            artifact = self.create_artifact(
                review.task_id,
                CreateArtifactRequest(
                    type=ArtifactType.review,
                    content=request.content or request.summary,
                    summary=request.summary or f"Review result for {review_id}",
                    created_by=request.completed_by,
                    metadata={"review_id": review_id},
                ),
            )
            result_artifact_id = artifact.id
        else:
            artifact = self.get_artifact(result_artifact_id)
            if artifact.task_id != review.task_id:
                raise InvalidRequest(
                    f"artifact {result_artifact_id!r} does not belong to task {review.task_id}"
                )

        now = self._now()
        self.storage.update_review(
            review_id, status=request.status.value, result_artifact_id=result_artifact_id,
            updated_at=now,
        )
        self.storage.dismiss_inbox_items(review_id, now)

        # The task leaves needs_review only when nothing is outstanding.
        remaining = [
            r for r in self.storage.list_reviews(review.task_id)
            if r.status in (ReviewStatus.open, ReviewStatus.claimed, ReviewStatus.in_progress)
        ]
        if not remaining:
            task = self._require_task(review.task_id)
            if task.status is TaskStatus.needs_review:
                self._touch(review.task_id, status=TaskStatus.in_progress.value)
        self.record_event(
            review.task_id, "review_completed", request.completed_by,
            {"review_id": review_id, "status": request.status.value,
             "result_artifact_id": result_artifact_id},
        )
        self._signal_entity("review", review_id, "review_completed",
                            {"status": request.status.value,
                             "result_artifact_id": result_artifact_id})
        return self._require_review(review_id)

    # ----------------------------------------------------------------- inbox
    def get_manager_inbox(self, manager_id: str) -> list[InboxItem]:
        """Union of review assignments, externally-created items (Linear
        assignment, §14.4 mode 3), and approvals this manager can grant."""
        items: list[InboxItem] = []
        for review in self.storage.reviews_for_manager(
            manager_id, (ReviewStatus.open.value, ReviewStatus.claimed.value,
                         ReviewStatus.in_progress.value)
        ):
            items.append(
                InboxItem(
                    id=f"{ids.INBOX}_{review.id}", manager_id=manager_id, kind=InboxKind.review,
                    ref_id=review.id, task_id=review.task_id,
                    title=f"Review [{review.status.value}] requested by {review.requested_by}",
                    created_at=review.created_at,
                )
            )
        seen = {(i.kind, i.ref_id) for i in items}
        for item in self.storage.list_inbox_items(manager_id):
            if (item.kind, item.ref_id) not in seen:
                items.append(item)
                seen.add((item.kind, item.ref_id))
        for summary in self.storage.list_tasks(
            TaskFilters(current_manager=manager_id, status=TaskStatus.needs_approval, limit=100)
        ):
            for subtask in self.storage.list_subtasks(summary.id):
                if subtask.status is SubtaskStatus.needs_approval:
                    key = (InboxKind.approval, subtask.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(
                        InboxItem(
                            id=f"{ids.INBOX}_{subtask.id}", manager_id=manager_id,
                            kind=InboxKind.approval, ref_id=subtask.id, task_id=summary.id,
                            title=f"Approval needed: {subtask.title}",
                            created_at=subtask.created_at,
                        )
                    )
        items.sort(key=lambda i: i.created_at)
        return items

    def add_inbox_item(
        self, manager_id: str, kind: InboxKind, ref_id: str,
        task_id: Optional[str] = None, title: str = "",
    ) -> InboxItem:
        item = InboxItem(
            id=ids.new_id(ids.INBOX), manager_id=manager_id, kind=kind, ref_id=ref_id,
            task_id=task_id, title=title, created_at=self._now(),
        )
        self.storage.insert_inbox_item(item)
        return item

    # -------------------------------------------------------------- subtasks
    def submit_subtask(self, task_id: str, request: SubtaskRequest) -> Subtask:
        """Record the subtask and hand it to the execution backend.

        This returns as soon as the backend has a handle. Under Temporal that is
        immediately after `start_workflow` — the worker has not run yet and the
        caller must poll. Under the direct backend the work has already happened
        by the time we return, because there is nothing to wait on."""
        task = self._require_task(task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            raise Conflict(f"task {task_id} is {task.status.value}; cannot accept subtasks")
        now = self._now()
        subtask = Subtask(
            id=ids.new_id(ids.SUBTASK), parent_task_id=task_id, title=request.title,
            instructions=request.instructions, status=SubtaskStatus.queued,
            privacy_tier=request.privacy_tier, preferred_worker=request.preferred_worker,
            created_at=now, updated_at=now, sandbox=request.sandbox,
            required_capabilities=request.required_capabilities, metadata=request.metadata,
        )
        self.storage.insert_subtask(subtask)
        self.execution.start_subtask(subtask.id)
        return self._require_subtask(subtask.id)

    def grant_approval(
        self, subtask_id: str, approved_by: str, max_cost_usd: Optional[float] = None
    ) -> ApprovalRecord:
        """Record a human's YES. Does **not** run anything — resuming the work is
        the execution backend's job, reached via a signal. Split this way so a
        Temporal activity can record the decision durably before the workflow
        wakes, and a replay never re-asks the human."""
        subtask = self._require_subtask(subtask_id)
        if subtask.status is not SubtaskStatus.needs_approval:
            raise Conflict(
                f"subtask {subtask_id} is {subtask.status.value}, not needs_approval"
            )
        now = self._now()
        approval = self.storage.pending_approval_for(subtask_id) or self._create_approval(
            subtask, RouteExplanation(**subtask.route_reason)
        )
        self.storage.update_approval(
            approval.id, status=ApprovalStatus.granted.value, decided_by=approved_by,
            max_cost_usd=max_cost_usd, decided_at=now,
        )
        self.storage.update_subtask(
            subtask_id, status=SubtaskStatus.queued.value, approved_by=approved_by,
            approved_max_cost_usd=max_cost_usd, updated_at=now,
        )
        self.storage.dismiss_inbox_items(subtask_id, now)
        self.record_event(
            subtask.parent_task_id, "subtask_approved", approved_by,
            {"subtask_id": subtask_id, "approval_id": approval.id, "max_cost_usd": max_cost_usd},
        )
        return self.storage.get_approval(approval.id) or approval

    def approve_subtask(
        self, subtask_id: str, approved_by: str, max_cost_usd: Optional[float] = None
    ) -> Subtask:
        """Human unblocks a paid/rented route (§15 steps 6-8): record the grant,
        then signal the running execution to resume. The resumed run re-routes
        rather than blindly using the previously-named candidate, because the
        registry may have changed since the approval was requested."""
        subtask = self._require_subtask(subtask_id)
        self.grant_approval(subtask_id, approved_by, max_cost_usd)
        self._signal_entity("subtask", subtask_id, "approval_granted",
                            {"approved_by": approved_by, "max_cost_usd": max_cost_usd})
        return self._require_subtask(subtask.id)

    def deny_subtask(self, subtask_id: str, denied_by: str, reason: str = "") -> Subtask:
        subtask = self._require_subtask(subtask_id)
        if subtask.status is not SubtaskStatus.needs_approval:
            raise Conflict(
                f"subtask {subtask_id} is {subtask.status.value}, not needs_approval"
            )
        subtasks_policy.check_transition(subtask, SubtaskStatus.failed)
        now = self._now()
        pending = self.storage.pending_approval_for(subtask_id)
        if pending:
            self.storage.update_approval(
                pending.id, status=ApprovalStatus.denied.value, decided_by=denied_by,
                reason=reason, decided_at=now,
            )
        self.storage.update_subtask(
            subtask_id, status=SubtaskStatus.failed.value, updated_at=now
        )
        self.storage.dismiss_inbox_items(subtask_id, now)
        self.record_event(
            subtask.parent_task_id, "subtask_denied", denied_by,
            {"subtask_id": subtask_id, "reason": reason},
        )
        self._signal_entity("subtask", subtask_id, "approval_denied", {"reason": reason})
        self._settle_parent_after_subtask(subtask.parent_task_id)
        return self._require_subtask(subtask_id)

    # ------------------------------------------------------- approvals/executions
    def _create_approval(self, subtask: Subtask, explanation: RouteExplanation) -> ApprovalRecord:
        record = ApprovalRecord(
            id=ids.new_id(ids.APPROVAL), task_id=subtask.parent_task_id, subtask_id=subtask.id,
            status=ApprovalStatus.pending, requested_worker=explanation.approval_candidate,
            requested_location=explanation.approval_location,
            estimated_cost_usd=explanation.estimated_cost_usd, created_at=self._now(),
        )
        return self.storage.insert_approval(record)

    def get_approval(self, approval_id: str) -> ApprovalRecord:
        record = self.storage.get_approval(approval_id)
        if record is None:
            raise NotFound(f"unknown approval {approval_id!r}")
        return record

    def pending_approval_for(self, subtask_id: str) -> Optional[ApprovalRecord]:
        return self.storage.pending_approval_for(subtask_id)

    def list_approvals(self, task_id: str) -> list[ApprovalRecord]:
        return self.storage.list_approvals(task_id)

    def put_execution(self, handle: ExecutionHandle) -> ExecutionHandle:
        return self.storage.upsert_execution(handle)

    def get_execution(self, handle_id: str) -> Optional[ExecutionHandle]:
        return self.storage.get_execution(handle_id)

    def executions_for(self, entity_type: str, entity_id: str) -> list[ExecutionHandle]:
        return self.storage.executions_for(entity_type, entity_id)

    def update_execution(self, handle_id: str, **fields: Any) -> None:
        if "state" in fields and hasattr(fields["state"], "value"):
            fields["state"] = fields["state"].value
        self.storage.update_execution(handle_id, updated_at=self._now(), **fields)

    def _signal_entity(
        self, entity_type: str, entity_id: str, name: str, payload: dict[str, Any]
    ) -> None:
        """Deliver a signal to whatever execution is driving this entity.

        A missing handle is not an error: an entity created before a backend was
        bound (or by a plain service call in a test) simply has nothing running.
        """
        for handle in self.storage.executions_for(entity_type, entity_id):
            if handle.state in _LIVE_EXECUTION_STATES:
                self.execution.signal(handle.handle_id, name, payload)

    def get_subtask(self, subtask_id: str) -> SubtaskDetail:
        subtask = self._require_subtask(subtask_id)
        return SubtaskDetail(subtask=subtask, runs=self.storage.list_runs(subtask_id))

    def cancel_subtask(self, subtask_id: str) -> None:
        subtask = self._require_subtask(subtask_id)
        if subtask.status in subtasks_policy.TERMINAL:
            raise Conflict(f"subtask {subtask_id} is already {subtask.status.value}")
        now = self._now()
        live = [
            h for h in self.storage.executions_for("subtask", subtask_id)
            if h.state in _LIVE_EXECUTION_STATES
        ]
        for run in self.storage.list_runs(subtask_id):
            if run.status is RunStatus.running:
                if subtask.assigned_worker:
                    try:
                        self.registry.get(subtask.assigned_worker).cancel(run.id)
                    except NotFound:
                        pass
                self.storage.update_run(
                    run.id, status=RunStatus.cancelled.value, finished_at=now
                )
        # Mark the ledger terminal FIRST, then tell the backend. Ordering matters:
        # a backend whose cancel() calls back into the service must find a
        # terminal subtask and stop, rather than recurse.
        self.storage.update_subtask(
            subtask_id, status=SubtaskStatus.cancelled.value, updated_at=now
        )
        for handle in live:
            self.storage.update_execution(
                handle.handle_id, state=ExecutionState.cancelled.value, updated_at=now
            )
            try:
                self.execution.cancel(handle.handle_id)
            except Exception as exc:  # a dead workflow server must not strand the ledger
                _log.warning("execution cancel(%s) failed: %s", handle.handle_id, exc)
        self._settle_parent_after_subtask(subtask.parent_task_id)

    def explain_route(self, subtask_id: str) -> RouteExplanation:
        subtask = self._require_subtask(subtask_id)
        if not subtask.route_reason:
            raise NotFound(f"subtask {subtask_id} has no recorded route")
        return RouteExplanation(**subtask.route_reason)

    # ----------------------------------------------------- routing/execution
    def route_subtask(self, subtask_id: str) -> RouteExplanation:
        """The `route_subtask` activity. Pure decision + persistence, no worker.

        Idempotent: routing an already-routed subtask recomputes the same answer
        (routing is deterministic) and rewrites the same rows, so a workflow
        replay costs an artifact row, not a wrong decision.
        """
        subtask = self._require_subtask(subtask_id)
        if subtask.status not in (SubtaskStatus.queued, SubtaskStatus.needs_approval):
            raise Conflict(
                f"subtask {subtask_id} is {subtask.status.value}; cannot be routed"
            )
        explanation = route(subtask, self.registry, self.policy)
        now = self._now()

        # The explanation is durable twice over: inline for programmatic reads,
        # and as an artifact so a human can page through it from Linear (§20).
        self.create_artifact(
            subtask.parent_task_id,
            CreateArtifactRequest(
                type=ArtifactType.route_explanation,
                content=json.dumps(explanation.model_dump(mode="json"), indent=2),
                summary=f"Route explanation for subtask {subtask.id}",
                created_by="router",
                metadata={"subtask_id": subtask.id},
            ),
        )
        self.storage.update_subtask(
            subtask.id, route_reason=explanation.model_dump(mode="json"), updated_at=now
        )

        if explanation.selected_worker is not None:
            self.storage.update_subtask(
                subtask.id, status=SubtaskStatus.queued.value, updated_at=now
            )
            return explanation

        if explanation.needs_approval:
            self.storage.update_subtask(
                subtask.id, status=SubtaskStatus.needs_approval.value, updated_at=now
            )
            self._advance_task(
                subtask.parent_task_id, TaskStatus.needs_approval, _TO_NEEDS_APPROVAL
            )
            if self.storage.pending_approval_for(subtask.id) is None:
                self._create_approval(subtask, explanation)
            return explanation

        self.storage.update_subtask(
            subtask.id, status=SubtaskStatus.failed.value, updated_at=now
        )
        self.record_attempt(
            subtask.parent_task_id,
            RecordAttemptRequest(
                actor_id="router", actor_type=ActorType.system,
                summary=f"No eligible worker for subtask {subtask.id}", outcome="failed",
            ),
        )
        self._settle_parent_after_subtask(subtask.parent_task_id)
        return explanation

    def run_subtask(self, subtask_id: str) -> Subtask:
        """The `run_worker` activity. Requires an already-routed, runnable subtask.

        Idempotent on retry: a subtask that already reached a terminal state is
        returned untouched rather than re-run, so a Temporal activity retry after
        a lost heartbeat does not double-execute a worker's side effects.
        """
        subtask = self._require_subtask(subtask_id)
        if subtask.status in subtasks_policy.TERMINAL:
            return subtask
        if subtask.status is not SubtaskStatus.queued:
            raise Conflict(f"subtask {subtask_id} is {subtask.status.value}; cannot be run")
        if not subtask.route_reason:
            raise Conflict(f"subtask {subtask_id} has not been routed")
        explanation = RouteExplanation(**subtask.route_reason)
        if explanation.selected_worker is None:
            raise Conflict(f"subtask {subtask_id} has no selected worker; route it first")
        return self._execute(subtask, explanation)

    def _execute(self, subtask: Subtask, explanation: RouteExplanation) -> Subtask:
        worker = self.registry.get(explanation.selected_worker or "")
        caps = worker.capabilities()
        now = self._now()
        run = WorkerRun(
            id=ids.new_id(ids.RUN), subtask_id=subtask.id, worker_id=caps.worker_id,
            harness=caps.harness, model=caps.model, status=RunStatus.running,
            route_reason=explanation.model_dump(mode="json"), started_at=now,
        )
        self.storage.insert_run(run)
        self.storage.update_subtask(
            subtask.id, status=SubtaskStatus.running.value,
            assigned_worker=caps.worker_id, updated_at=now,
        )
        self._advance_task(subtask.parent_task_id, TaskStatus.in_progress, _TO_IN_PROGRESS)

        task = self._require_task(subtask.parent_task_id)
        subtask = self._require_subtask(subtask.id)
        context = WorkerContext(
            task=task,
            subtask=subtask,
            run_id=run.id,
            create_artifact=lambda type, content, summary="": self.create_artifact(
                task.id,
                CreateArtifactRequest(
                    type=type, content=content, summary=summary,
                    created_by=caps.worker_id,
                    metadata={"subtask_id": subtask.id, "run_id": run.id},
                ),
            ),
            read_artifact_chunk=self.read_artifact_chunk,
        )

        try:
            result = worker.run(subtask, context)
        except Exception as exc:  # a harness crash is a failed run, never ours
            result = WorkerResult(
                status="failed", summary="Worker raised", error=f"{type(exc).__name__}: {exc}"
            )
        return self._record_result(subtask, run, result, caps.worker_id)

    def _record_result(
        self, subtask: Subtask, run: WorkerRun, result: WorkerResult, worker_id: str
    ) -> Subtask:
        now = self._now()
        artifact_ids = [a.artifact_id for a in result.artifacts]
        succeeded = result.status == "succeeded"
        self.storage.update_run(
            run.id,
            status=(RunStatus.succeeded if succeeded else RunStatus.failed).value,
            finished_at=now,
            output_artifact_id=artifact_ids[0] if artifact_ids else None,
            metrics=result.metrics,
            error=result.error,
        )
        self.storage.update_subtask(
            subtask.id,
            status=(SubtaskStatus.succeeded if succeeded else SubtaskStatus.failed).value,
            result_artifact_id=artifact_ids[0] if artifact_ids else None,
            updated_at=now,
        )
        summary = result.summary or ("Worker succeeded" if succeeded else "Worker failed")
        if result.warnings:
            summary = f"{summary} (warnings: {'; '.join(result.warnings)})"
        self.record_attempt(
            subtask.parent_task_id,
            RecordAttemptRequest(
                actor_id=worker_id, actor_type=ActorType.worker, summary=summary,
                outcome=result.status, artifact_refs=artifact_ids,
            ),
        )
        self._settle_parent_after_subtask(subtask.parent_task_id)
        return self._require_subtask(subtask.id)

    def _settle_parent_after_subtask(self, task_id: str) -> None:
        """A task sitting in needs_approval returns to in_progress once no
        subtask is still waiting on a human."""
        task = self._require_task(task_id)
        if task.status is not TaskStatus.needs_approval:
            return
        waiting = [
            s for s in self.storage.list_subtasks(task_id)
            if s.status is SubtaskStatus.needs_approval
        ]
        if not waiting:
            self._touch(task_id, status=TaskStatus.in_progress.value)

    # -------------------------------------------------------------- handoff
    def get_handoff_summary(
        self, task_id: str, manager_id: Optional[str] = None
    ) -> HandoffSummary:
        """Always recomputed from the ledger — a cached handoff is a stale one.

        The rendered text is also written back to `tasks.handoff_summary` so the
        Linear mirror has something to display without recomputing."""
        detail = self.get_task(task_id)
        summary = handoff_mod.build(
            detail, manager_id, self._now(),
            running_workflows=self._running_workflows(detail),
            waiting_approvals=[
                a for a in self.storage.list_approvals(task_id)
                if a.status is ApprovalStatus.pending
            ],
        )
        if detail.task.handoff_summary != summary.text:
            self.storage.update_task(task_id, handoff_summary=summary.text)
        return summary

    def _running_workflows(self, detail: TaskDetail) -> list[dict[str, Any]]:
        """Durable executions still in flight for this task's subtasks and reviews."""
        rows: list[dict[str, Any]] = []
        entities = [("subtask", s.id) for s in detail.subtasks]
        entities += [("review", r.id) for r in detail.reviews]
        for entity_type, entity_id in entities:
            for handle in self.storage.executions_for(entity_type, entity_id):
                if handle.state not in _LIVE_EXECUTION_STATES:
                    continue
                rows.append({
                    "handle_id": handle.handle_id, "backend": handle.backend,
                    "workflow_id": handle.workflow_id, "entity_type": entity_type,
                    "entity_id": entity_id, "state": handle.state.value,
                    "detail": handle.detail,
                })
        return rows

    def regenerate_handoff_summary(self, task_id: str) -> HandoffSummary:
        """Explicit invalidation entry point. Identical today; the seam exists so
        an optional LLM-assisted summarizer can be added without changing callers."""
        return self.get_handoff_summary(task_id)

    # -------------------------------------------------------- external refs
    def set_external_ref(
        self, entity_type: str, entity_id: str, provider: str, external_id: str,
        external_url: Optional[str] = None, sync_enabled: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExternalRef:
        now = self._now()
        ref = ExternalRef(
            id=ids.new_id(ids.EXTERNAL), entity_type=entity_type, entity_id=entity_id,
            provider=provider, external_id=external_id, external_url=external_url,
            sync_enabled=sync_enabled, created_at=now, updated_at=now, metadata=metadata or {},
        )
        stored = self.storage.upsert_external_ref(ref)
        if entity_type == "task" and provider == "linear":
            self._touch(entity_id, linear_issue_id=external_id, linear_issue_url=external_url)
        return stored

    def get_external_ref(
        self, entity_type: str, entity_id: str, provider: str = "linear"
    ) -> Optional[ExternalRef]:
        return self.storage.get_external_ref(entity_type, entity_id, provider)

    def find_task_by_external_id(
        self, external_id: str, provider: str = "linear"
    ) -> Optional[Task]:
        ref = self.storage.find_by_external_id(provider, external_id)
        if ref is None or ref.entity_type != "task":
            return None
        return self.storage.get_task(ref.entity_id)

    # --------------------------------------------------------------- events
    def record_event(
        self, task_id: Optional[str], kind: str, actor: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> Event:
        event = Event(
            id=ids.new_id(ids.EVENT), task_id=task_id, kind=kind, actor=actor,
            payload=payload or {}, created_at=self._now(),
        )
        return self.storage.insert_event(event)

    def list_events(self, task_id: Optional[str] = None, limit: int = 100) -> list[Event]:
        return self.storage.list_events(task_id, limit)

    # -------------------------------------------------------------- privacy
    def effective_privacy(self, task_id: str) -> PrivacyTier:
        return self.get_task(task_id).effective_privacy

    # --------------------------------------------------------------- memory
    def recall_memory(self, request: RecallRequest) -> RecallPack:
        """Scoped, bounded recall. Managers call this instead of touching a
        memory backend, so visibility policy stays ours.

        A backend that raises is a warning, never an exception: no core flow may
        fail because memory is unavailable.
        """
        try:
            pack = self.memory.recall(request)
        except Exception as exc:
            _log.warning("memory backend %r recall failed: %s", self.memory.backend_name, exc)
            return RecallPack(
                profile=request.profile, query=request.query, items=[],
                backend=self.memory.backend_name,
                warnings=[f"memory recall failed: {exc}"],
            )
        # Re-apply visibility even if the adapter already did: a backend must not
        # be able to smuggle pending items into a manager's context.
        pack.items = memory_mod.apply_visibility(pack.items, request)
        return pack

    def capture_memory_candidate(self, request: CaptureRequest) -> CaptureResult:
        """Never promotes. Whatever an agent volunteers arrives as `pending`."""
        try:
            result = self.memory.capture_candidate(request)
        except Exception as exc:
            _log.warning("memory backend %r capture failed: %s", self.memory.backend_name, exc)
            return CaptureResult(
                accepted=False, status="pending", backend=self.memory.backend_name,
                message=f"memory capture failed: {exc}",
            )
        if result.status == "promoted":
            result.status = "pending"
            result.message = (
                (result.message or "") + " (promotion ignored: capture never promotes)"
            ).strip()
        if request.task_id:
            self.record_event(
                request.task_id, "memory_candidate_captured",
                request.origin_actor_id or "unknown",
                {"candidate_id": result.candidate_id, "status": result.status,
                 "backend": result.backend},
            )
        return result

    def record_memory_feedback(self, request: MemoryFeedbackRequest) -> None:
        try:
            self.memory.record_feedback(request)
        except Exception as exc:
            _log.warning("memory backend %r feedback failed: %s", self.memory.backend_name, exc)

    def memory_health(self) -> dict[str, Any]:
        """Health of every configured backend, plus which one confers trust."""
        backends: dict[str, Any] = {}
        for name, adapter in self.memory_backends.items():
            try:
                backends[name] = adapter.health()
            except Exception as exc:
                backends[name] = {"backend": name, "status": "unreachable", "detail": str(exc)}
        if not backends:
            try:
                return self.memory.health()
            except Exception as exc:
                return {"backend": self.memory.backend_name, "status": "unreachable",
                        "detail": str(exc)}
        primary = backends.get(self.memory_config.trusted_authority)
        return {
            "backend": self.memory.backend_name,
            "status": (primary or next(iter(backends.values()))).get("status", "unknown"),
            "trusted_authority": self.memory_config.trusted_authority,
            "enabled": self.memory_config.enabled,
            "backends": backends,
        }

    def get_task_context_pack(
        self,
        task_id: str,
        profile: str = "manager_brief",
        max_memory_items: Optional[int] = None,
        query: Optional[str] = None,
        manager_id: Optional[str] = None,
        include_pending: bool = False,
    ) -> ContextPack:
        """The one way a manager or worker gets context (memory-stack §5).

        Ledger truth and recalled memory are never merged into one list: a locked
        decision is law, a Cognee hit is a lead. The `ContextBuilder` bounds,
        dedupes, orders by authority, and labels every item with its source.
        """
        return self.context_builder.build_context_pack(
            task_id, profile=profile, query=query, max_items=max_memory_items,
            manager_id=manager_id, include_pending=include_pending,
        )

    def attach_context_to_subtask(self, subtask_id: str, pack: ContextPack) -> None:
        """Push a bounded pack down to a worker that cannot call MCP.

        Managers *pull* context through the MCP tool; a bounded worker harness
        often has no MCP client at all, so the `recall_context` activity attaches
        the pack here and the harness reads `subtask.metadata["context_pack"]`.
        """
        subtask = self._require_subtask(subtask_id)
        metadata = dict(subtask.metadata)
        metadata["context_pack"] = {
            "profile": pack.profile,
            "warnings": pack.warnings,
            "memory_is_external_context": True,
            "items": [
                {"text": i.text, "status": i.status, "confidence": i.confidence,
                 "source_id": i.source_id, "backend": (i.metadata or {}).get("backend"),
                 "trusted": (i.metadata or {}).get("trusted", False)}
                for i in pack.memory.items
            ],
        }
        self.storage.update_subtask(subtask_id, metadata=metadata, updated_at=self._now())

    # ------------------------------------------------------ memory promotion
    def promote_memory_candidate(
        self, candidate_id: str, promoted_by: str,
        confidence: Optional[str] = None, scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Human/librarian only (memory-stack §4). Never exposed as an MCP tool.

        Promotion is the single act that turns an agent's suggestion into a
        trusted claim, and it is the moment the claim is fanned out to the
        retrieval indexes. Indexing failures do not undo the promotion — the
        trusted authority is the record; Cognee and Graphiti are caches of it.
        """
        authority = self.trusted_authority()
        if authority is None:
            raise InvalidRequest(
                f"no trusted memory authority configured "
                f"(expected {self.memory_config.trusted_authority!r})"
            )
        # Forwarded only when supplied, so a TrustedMemoryAdapter with the older
        # two-argument signature still works. The authority decides whether it can
        # promote without them; it must never guess.
        extra: dict[str, Any] = {}
        if confidence is not None:
            extra["confidence"] = confidence
        if scope is not None:
            extra["scope"] = scope
        claim = authority.promote_candidate(candidate_id, promoted_by, **extra)
        claim.setdefault("claim_id", candidate_id)
        indexed, failed = [], []
        for name, adapter in self.memory_backends.items():
            if name == self.memory_config.trusted_authority:
                continue
            if not isinstance(adapter, IndexingMemoryAdapter):
                continue
            try:
                adapter.index_claim(claim)
                indexed.append(name)
            except Exception as exc:
                _log.warning("indexing promoted claim into %s failed: %s", name, exc)
                failed.append(name)
        claim["indexed_into"] = indexed
        if failed:
            claim["index_failures"] = failed
        self.record_event(
            claim.get("task_id"), "memory_promoted", promoted_by,
            {"claim_id": claim.get("claim_id"), "indexed_into": indexed,
             "index_failures": failed},
        )
        return claim

    def list_pending_memory(self, limit: int = 50) -> list[dict[str, Any]]:
        authority = self.trusted_authority()
        if authority is None:
            return []
        try:
            return authority.list_pending(limit)
        except Exception as exc:
            _log.warning("listing pending memory failed: %s", exc)
            return []

    # ==================================================== compliance layer §3-§13
    # `launch` prepares a managed session; `shell` (the CLI) runs the agent inside
    # it; `audit` asks where the work was recorded; `complete` refuses until it was.

    def _require_session(self, session_id: str) -> ManagerSession:
        session = self.storage.get_session(session_id)
        if session is None:
            raise NotFound(f"unknown session {session_id!r}")
        return session

    def _require_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.storage.get_workspace(workspace_id)
        if workspace is None:
            raise NotFound(f"unknown workspace {workspace_id!r}")
        return workspace

    # ---------------------------------------------------------------- launch
    def launch_session(
        self,
        manager_id: str,
        task_id: Optional[str] = None,
        review_id: Optional[str] = None,
        claim: bool = False,
        readonly: bool = False,
        force_readonly: bool = False,
        repo_source: Optional[str] = None,
        repo_mode: str = "auto",
        launch_command: str = "",
        token_ttl_seconds: int = sessions_mod.DEFAULT_TOKEN_TTL_SECONDS,
        api_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Prepare a managed AgentConnect session (§3.1).

        Returns the session, the workspace, the files written, and the *plaintext*
        token — the only time it exists. Everything else is durable.
        """
        if not task_id and not review_id:
            raise InvalidRequest("launch needs either a task_id or a review_id")

        review: Optional[Review] = None
        if review_id:
            review = self._require_review(review_id)
            task_id = task_id or review.task_id
        task = self._require_task(task_id)  # verify it exists before touching disk

        mode = mode_for(review_id, readonly)
        now = self._now()
        entity_id = review_id or task_id

        # 1. Claim before building anything on disk: a refused claim must not leave
        #    a half-prepared workspace behind.
        claim_id: Optional[str] = None
        if claim and not readonly:
            try:
                if review_id:
                    self.claim_review(review_id, manager_id)
                    claim_id = review_id
                else:
                    claim_id = self.claim_task(task_id, manager_id).id
            except Conflict:
                if not force_readonly:
                    raise
                mode = SessionMode.readonly
                _log.warning("claim refused for %s; downgrading to readonly", entity_id)

        # 2. Workspace, instructions, config.
        base, repo_path, artifact_path, resolved_mode = self.workspaces.build(
            entity_id, task.title, manager_id, repo_source=repo_source, repo_mode=repo_mode,
        )
        workspace = Workspace(
            id=ids.new_id(ids.WORKSPACE), task_id=task_id, review_id=review_id,
            path=str(base), repo_path=str(repo_path), artifact_path=str(artifact_path),
            repo_mode=resolved_mode, created_at=now,
            metadata={"manager_id": manager_id, "repo_source": repo_source},
        )
        self.storage.insert_workspace(workspace)

        session = ManagerSession(
            id=ids.new_id(ids.SESSION), task_id=task_id, review_id=review_id,
            manager_id=manager_id, workspace_id=workspace.id, mode=mode,
            status=SessionStatus.prepared, claim_id=claim_id, started_at=now,
            launch_command=launch_command,
        )
        self.storage.insert_session(session)

        # 3. The one credential the agent gets. Scoped to this session, this
        #    entity, and exactly the actions its mode allows.
        token = self.mint_session_token(session, ttl_seconds=token_ttl_seconds)

        resolved_api = api_url or self.api_url
        env = sessions_mod.session_env_vars(
            api_url=resolved_api, task_id=task_id, review_id=review_id,
            manager_id=manager_id, workspace_id=workspace.id, session_id=session.id,
            token=token.plaintext, mode=mode,
        )
        files = self.workspaces.write_instructions(base, manager_id)
        self.workspaces.write_env_file(base, sessions_mod.render_env_file(env))
        self.workspaces.write_mcp_config(base, resolved_api, env)
        metadata = {
            "workspace_id": workspace.id, "task_id": task_id, "review_id": review_id,
            "manager_id": manager_id, "repo_source": repo_source,
            "repo_mode": resolved_mode.value, "repo_path": str(repo_path),
            "artifact_path": str(artifact_path), "created_at": now,
            "session_id": session.id, "mode": mode.value,
        }
        self.workspaces.write_metadata(base, metadata)
        files += [".env.agentconnect", ".mcp.json", "workspace.json"]

        self.storage.update_session(session.id, workspace_id=workspace.id)
        self.record_event(
            task_id, "session_prepared", manager_id,
            {"session_id": session.id, "workspace_id": workspace.id, "mode": mode.value,
             "review_id": review_id, "claim_id": claim_id},
        )
        shell_flag = f"--review {review_id}" if review_id else f"--task {task_id}"
        return {
            "session": session.model_copy(update={"mode": mode}),
            "workspace": workspace,
            "claim_id": claim_id,
            "token": token.plaintext,
            "files": files,
            "env": env,
            "shell_command": f"agentconnect shell {shell_flag} -- {manager_id}",
        }

    # ----------------------------------------------------------------- shell
    def start_shell(self, session_id: str, shell_command: str) -> ManagerSession:
        session = self._require_session(session_id)
        if session.status not in (SessionStatus.prepared, SessionStatus.running):
            raise Conflict(f"session {session_id} is {session.status.value}")
        self.storage.update_session(
            session_id, status=SessionStatus.running.value, shell_command=shell_command
        )
        self.record_event(
            session.task_id, "shell_started", session.manager_id,
            {"session_id": session_id, "command": shell_command},
        )
        return self._require_session(session_id)

    def end_shell(self, session_id: str, exit_code: int = 0) -> ManagerSession:
        session = self._require_session(session_id)
        now = self._now()
        status = SessionStatus.ended if exit_code == 0 else SessionStatus.failed
        self.storage.update_session(session_id, status=status.value, ended_at=now)
        # The token dies with the shell. A leaked `.env.agentconnect` is then inert.
        self.storage.revoke_tokens_for_session(session_id, now)
        self.record_event(
            session.task_id, "shell_ended", session.manager_id,
            {"session_id": session_id, "exit_code": exit_code, "status": status.value},
        )
        return self._require_session(session_id)

    def active_session_for(
        self, task_id: Optional[str] = None, review_id: Optional[str] = None
    ) -> Optional[ManagerSession]:
        return self.storage.latest_session(
            task_id=task_id, review_id=review_id,
            statuses=(SessionStatus.prepared.value, SessionStatus.running.value),
        )

    def get_session(self, session_id: str) -> ManagerSession:
        return self._require_session(session_id)

    def list_sessions(
        self, task_id: Optional[str] = None, manager_id: Optional[str] = None,
        status: Optional[str] = None, limit: int = 50,
    ) -> list[ManagerSession]:
        return self.storage.list_sessions(task_id, manager_id, status, limit)

    def get_workspace(self, workspace_id: str) -> Workspace:
        return self._require_workspace(workspace_id)

    def list_workspaces(self, include_destroyed: bool = False) -> list[Workspace]:
        return self.storage.list_workspaces(include_destroyed)

    def workspace_for(
        self, task_id: Optional[str] = None, review_id: Optional[str] = None
    ) -> Optional[Workspace]:
        return self.storage.find_workspace(task_id=task_id, review_id=review_id)

    # ---------------------------------------------------------------- tokens
    def mint_session_token(
        self, session: ManagerSession,
        ttl_seconds: int = sessions_mod.DEFAULT_TOKEN_TTL_SECONDS,
    ) -> SessionToken:
        now = self._now()
        plaintext = sessions_mod.mint_token()
        token = SessionToken(
            id=ids.new_id(ids.TOKEN), session_id=session.id,
            scope=sessions_mod.build_scope(
                session.id, session.manager_id, session.mode, session.task_id,
                session.review_id,
            ),
            expires_at=now + max(1, ttl_seconds), created_at=now, plaintext=plaintext,
        )
        self.storage.insert_token(token, sessions_mod.hash_token(plaintext))
        return token

    def authorize(self, token: str, action: str) -> dict[str, Any]:
        """Check a session token against one action. Raises, never returns False.

        A token buys exactly the actions of its mode. `promote_memory_candidate`,
        Temporal admin, and secret reads are in no mode's list, so no agent-held
        token can ever reach them — the deny is structural, not a special case.
        """
        record = self.storage.get_token_by_hash(sessions_mod.hash_token(token or ""))
        if record is None:
            raise PolicyViolation("unknown session token")
        if not record.active_at(self._now()):
            reason = "revoked" if record.revoked_at else "expired"
            raise PolicyViolation(f"session token is {reason}")
        allowed = set(record.scope.get("actions", []))
        if action in sessions_mod.FORBIDDEN_ACTIONS or action not in allowed:
            raise PolicyViolation(
                f"action {action!r} is not permitted for a "
                f"{record.scope.get('mode')} session token"
            )
        return record.scope

    def revoke_session_tokens(self, session_id: str) -> int:
        return self.storage.revoke_tokens_for_session(session_id, self._now())

    # ----------------------------------------------------------------- audit
    def audit_task(self, task_id: str) -> AuditReport:
        """§12. Reads the ledger and the worktree; **writes nothing.**

        The read-only part is load-bearing. `get_handoff_summary` persists the
        summary as a side effect, so auditing through it would repair the very
        staleness it reports: the first audit would fail and the second would
        pass. An audit that changes what it measures is not an audit.
        """
        detail = self.get_task(task_id)
        workspace = self.workspace_for(task_id=task_id)
        session = self.storage.latest_session(task_id=task_id)

        stored = detail.task.handoff_summary
        fresh = handoff_mod.build(
            detail, None, self._now(),
            running_workflows=self._running_workflows(detail),
            waiting_approvals=[
                a for a in self.storage.list_approvals(task_id)
                if a.status is ApprovalStatus.pending
            ],
        )

        linear_ref = self.get_external_ref("task", task_id, "linear")
        linear_state = None
        for event in self.storage.list_events(task_id, limit=200):
            if event.kind == "linear_status_change":
                linear_state = event.payload.get("state")
                break  # events come back newest first

        captured = any(
            e.kind == "memory_candidate_captured"
            for e in self.storage.list_events(task_id, limit=200)
        )
        return audit_mod.audit_task(
            detail, workspace, session, fresh.text, stored,
            linear_ref=linear_ref, linear_state=linear_state,
            memory_captured=captured,
            memory_enabled=self.memory_config.enabled and bool(self.memory_backends),
        )

    def audit_review(self, review_id: str) -> AuditReport:
        review = self._require_review(review_id)
        detail = self.get_task(review.task_id)
        workspace = self.workspace_for(review_id=review_id)
        session = self.storage.latest_session(review_id=review_id)
        return audit_mod.audit_review(review, detail, workspace, session)

    # -------------------------------------------------------------- complete
    def complete_task(
        self, task_id: str, completed_by: str, force: bool = False,
    ) -> dict[str, Any]:
        """§13. The audit runs first; Linear hears about it last.

        A proprietary agent cannot shortcut this: `complete_task` is not an MCP
        tool. `force` exists for a human operator with a good reason, and it is
        recorded as such.
        """
        task = self._require_task(task_id)
        if task.status is TaskStatus.succeeded:
            raise Conflict(f"task {task_id} is already succeeded")

        # Completing a task must leave a current handoff behind, so refresh it
        # first and *then* audit. This also keeps `complete` from tripping on the
        # `handoff_fresh` check for a manager who did everything else right.
        self.regenerate_handoff_summary(task_id)
        report = self.audit_task(task_id)
        if not report.passed and not force:
            self.record_event(
                task_id, "completion_refused", completed_by, {"problems": report.problems}
            )
            raise PolicyViolation(
                "audit failed; task cannot be marked complete:\n"
                + "\n".join(f"- {p}" for p in report.problems)
            )

        self._touch(task_id, status=TaskStatus.succeeded.value)
        self.record_event(
            task_id, "task_completed", completed_by,
            {"forced": force, "warnings": report.warnings,
             "problems": report.problems if force else []},
        )

        # Only now does the tracker hear about it. Linear mirrors AgentConnect;
        # it never decides completion (§13).
        mirrored = []
        for hook in self.completion_hooks:
            try:
                hook(task_id)
                mirrored.append(getattr(hook, "__name__", "hook"))
            except Exception as exc:  # a mirror outage is not a ledger failure
                _log.warning("completion hook failed for %s: %s", task_id, exc)
        return {
            "task_id": task_id, "status": TaskStatus.succeeded.value,
            "audit": report.to_dict(), "forced": force, "mirrored": mirrored,
        }

    def complete_review_audited(
        self, review_id: str, request: ReviewResultRequest, force: bool = False,
    ) -> dict[str, Any]:
        report = self.audit_review(review_id)
        if not report.passed and not force:
            raise PolicyViolation(
                "audit failed; review cannot be completed:\n"
                + "\n".join(f"- {p}" for p in report.problems)
            )
        review = self.complete_review(review_id, request)
        return {"review": review, "audit": report.to_dict(), "forced": force}

    # --------------------------------------------------------------- cleanup
    def cleanup_workspace(self, workspace_id: str, actor: str = "system") -> Workspace:
        """Mark a workspace destroyed. The directory is the CLI's to remove — the
        service does not delete a human's files."""
        workspace = self._require_workspace(workspace_id)
        self.storage.update_workspace(workspace_id, destroyed_at=self._now())
        self.record_event(
            workspace.task_id, "workspace_destroyed", actor, {"workspace_id": workspace_id}
        )
        return self._require_workspace(workspace_id)

    def abandon_stale_sessions(self, older_than_seconds: float = 24 * 3600) -> list[str]:
        """A session whose shell died without `end_shell` is abandoned, not running.

        Without this, `audit` would keep measuring attempts against a session that
        ended days ago, and the token would stay live.
        """
        cutoff = self._now() - max(0.0, older_than_seconds)
        abandoned: list[str] = []
        for session in self.storage.list_sessions(limit=1000):
            if session.status not in (SessionStatus.prepared, SessionStatus.running):
                continue
            if session.started_at > cutoff:
                continue
            self.storage.update_session(
                session.id, status=SessionStatus.abandoned.value, ended_at=self._now()
            )
            self.revoke_session_tokens(session.id)
            abandoned.append(session.id)
        return abandoned
