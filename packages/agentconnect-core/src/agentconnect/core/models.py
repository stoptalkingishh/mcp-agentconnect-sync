"""The backplane data model (spec §9) plus the request/response shapes §10 names.

Deliberately independent of ``agentconnect.common.schemas``: that module models a
*routing decision for one generation*, this one models *durable work state across
managers*. They overlap in name only. In particular the privacy tiers differ —
§9.8 defines ``public | public_redacted | repo_sensitive | secret_sensitive |
local_only`` for subtask placement, while ``common.PrivacyClass`` classifies text
for provider dispatch. Do not merge them without a migration.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now() -> float:
    return time.time()


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------- enums
class TaskStatus(str, Enum):
    queued = "queued"
    in_progress = "in_progress"
    blocked = "blocked"
    needs_review = "needs_review"
    needs_approval = "needs_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.succeeded, TaskStatus.failed, TaskStatus.cancelled}
)


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class ClaimRole(str, Enum):
    primary_manager = "primary_manager"
    reviewer = "reviewer"
    planner = "planner"
    implementer = "implementer"
    observer = "observer"
    human_owner = "human_owner"
    worker_delegate = "worker_delegate"


#: Roles allowed to supersede or unlock a locked decision (§9.4).
DECISION_AUTHORITY_ROLES = frozenset({ClaimRole.human_owner})


class ActorType(str, Enum):
    manager = "manager"
    worker = "worker"
    human = "human"
    system = "system"


class ArtifactType(str, Enum):
    summary = "summary"
    report = "report"
    patch = "patch"
    test_log = "test_log"
    file_snapshot = "file_snapshot"
    review = "review"
    worker_output = "worker_output"
    plan = "plan"
    route_explanation = "route_explanation"
    other = "other"


class ReviewStatus(str, Enum):
    open = "open"
    claimed = "claimed"
    in_progress = "in_progress"
    completed = "completed"
    rejected = "rejected"
    cancelled = "cancelled"


class SubtaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    needs_approval = "needs_approval"


class PrivacyTier(str, Enum):
    public = "public"
    public_redacted = "public_redacted"
    repo_sensitive = "repo_sensitive"
    secret_sensitive = "secret_sensitive"
    local_only = "local_only"


#: Strictness order, loosest first. Used to compute a task's effective privacy
#: (the strictest tier among its subtasks) and to decide what may leave the box.
PRIVACY_STRICTNESS: dict[PrivacyTier, int] = {
    PrivacyTier.public: 0,
    PrivacyTier.public_redacted: 1,
    PrivacyTier.repo_sensitive: 2,
    PrivacyTier.local_only: 3,
    PrivacyTier.secret_sensitive: 4,
}


def strictest(tiers: list[PrivacyTier]) -> PrivacyTier:
    if not tiers:
        return PrivacyTier.public
    return max(tiers, key=lambda t: PRIVACY_STRICTNESS[t])


class FilesystemAccess(str, Enum):
    none = "none"
    readonly = "readonly"
    workspace_write = "workspace_write"


FS_ORDER: dict[FilesystemAccess, int] = {
    FilesystemAccess.none: 0,
    FilesystemAccess.readonly: 1,
    FilesystemAccess.workspace_write: 2,
}


class WorkerLocation(str, Enum):
    local = "local"
    cloud = "cloud"
    rented = "rented"


class RunStatus(str, Enum):
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class InboxKind(str, Enum):
    review = "review"
    task = "task"
    approval = "approval"


class ApprovalStatus(str, Enum):
    pending = "pending"
    granted = "granted"
    denied = "denied"
    expired = "expired"


# ---------------------------------------------------------------- value types
class SandboxSpec(BaseModel):
    """What a subtask needs (or what a worker can offer). §21."""

    filesystem: FilesystemAccess = FilesystemAccess.none
    network: bool = False
    shell: bool = False

    def satisfied_by(self, offered: "SandboxSpec") -> bool:
        return (
            FS_ORDER[self.filesystem] <= FS_ORDER[offered.filesystem]
            and (offered.network or not self.network)
            and (offered.shell or not self.shell)
        )


# -------------------------------------------------------------------- records
class Task(BaseModel):
    id: str
    title: str
    goal: str = ""
    status: TaskStatus = TaskStatus.queued
    priority: Priority = Priority.normal
    created_by: str = "unknown"
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    current_manager: Optional[str] = None
    handoff_summary: Optional[str] = None
    linear_issue_id: Optional[str] = None
    linear_issue_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Constraint(BaseModel):
    id: str
    task_id: str
    text: str
    created_by: str = "unknown"
    created_at: float = Field(default_factory=now)


class Claim(BaseModel):
    id: str
    task_id: str
    manager_id: str
    role: ClaimRole
    expires_at: float
    created_at: float = Field(default_factory=now)
    released_at: Optional[float] = None

    def active_at(self, when: float) -> bool:
        return self.released_at is None and self.expires_at > when


class Decision(BaseModel):
    id: str
    task_id: str
    made_by: str
    decision: str
    rationale: str = ""
    locked: bool = False
    created_at: float = Field(default_factory=now)
    superseded_by: Optional[str] = None


class Attempt(BaseModel):
    id: str
    task_id: str
    actor_id: str
    actor_type: ActorType
    summary: str
    outcome: str = ""
    created_at: float = Field(default_factory=now)
    artifact_refs: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    id: str
    task_id: str
    type: ArtifactType
    path: str
    summary: str = ""
    created_by: str = "unknown"
    created_at: float = Field(default_factory=now)
    size_bytes: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactSummary(BaseModel):
    """What adapters return in lists: never a body. §13, §21."""

    id: str
    task_id: str
    type: ArtifactType
    summary: str
    size_bytes: int
    created_by: str
    created_at: float


class ArtifactChunk(BaseModel):
    artifact_id: str
    offset: int
    limit: int
    content: str
    next_offset: Optional[int] = None
    eof: bool = True
    size_bytes: int = 0


class Review(BaseModel):
    id: str
    task_id: str
    requested_by: str
    assigned_to: str
    status: ReviewStatus = ReviewStatus.open
    criteria: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    result_artifact_id: Optional[str] = None
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)


class Subtask(BaseModel):
    id: str
    parent_task_id: str
    title: str
    instructions: str
    status: SubtaskStatus = SubtaskStatus.queued
    privacy_tier: PrivacyTier = PrivacyTier.repo_sensitive
    preferred_worker: Optional[str] = None
    assigned_worker: Optional[str] = None
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    result_artifact_id: Optional[str] = None
    route_reason: dict[str, Any] = Field(default_factory=dict)
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)
    required_capabilities: list[str] = Field(default_factory=list)
    approved_by: Optional[str] = None
    approved_max_cost_usd: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerRun(BaseModel):
    id: str
    subtask_id: str
    worker_id: str
    harness: str
    model: Optional[str] = None
    status: RunStatus = RunStatus.running
    route_reason: dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=now)
    finished_at: Optional[float] = None
    input_artifact_id: Optional[str] = None
    output_artifact_id: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ApprovalRecord(BaseModel):
    """A durable human decision about spending money or leaving the box (§15).

    It outlives the workflow that waited on it: the workflow can crash and be
    replayed, and the answer is still here.
    """

    id: str
    task_id: str
    subtask_id: str
    status: ApprovalStatus = ApprovalStatus.pending
    requested_worker: Optional[str] = None
    requested_location: Optional[str] = None
    estimated_cost_usd: float = 0.0
    max_cost_usd: Optional[float] = None
    decided_by: Optional[str] = None
    reason: str = ""
    created_at: float = Field(default_factory=now)
    decided_at: Optional[float] = None


class ExternalRef(BaseModel):
    id: str
    entity_type: str  # task | subtask | review | artifact | decision
    entity_id: str
    provider: str  # linear | github | jira | local_markdown
    external_id: str
    external_url: Optional[str] = None
    sync_enabled: bool = True
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboxItem(BaseModel):
    id: str
    manager_id: str
    kind: InboxKind
    ref_id: str
    task_id: Optional[str] = None
    title: str = ""
    created_at: float = Field(default_factory=now)
    dismissed_at: Optional[float] = None


class Event(BaseModel):
    """Durable record of something that happened *to* the ledger from outside —
    a Linear approval, a webhook status change, an operator override."""

    id: str
    task_id: Optional[str]
    kind: str
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)


# ------------------------------------------------------------------- requests
class CreateTaskRequest(BaseModel):
    title: str
    goal: str = ""
    priority: Priority = Priority.normal
    created_by: str = "unknown"
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskFilters(BaseModel):
    status: Optional[TaskStatus] = None
    current_manager: Optional[str] = None
    limit: int = 50
    offset: int = 0


class TaskSummary(BaseModel):
    id: str
    title: str
    status: TaskStatus
    priority: Priority
    current_manager: Optional[str]
    updated_at: float
    linear_issue_url: Optional[str] = None


class TaskDetail(BaseModel):
    task: Task
    constraints: list[Constraint] = Field(default_factory=list)
    active_claims: list[Claim] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)

    @property
    def effective_privacy(self) -> PrivacyTier:
        """Strictest tier among the subtasks, floored by any declared task tier.

        Drives what the Linear adapter is allowed to mirror (§21).
        """
        tiers = [s.privacy_tier for s in self.subtasks]
        declared = self.task.metadata.get("privacy")
        if declared:
            try:
                tiers.append(PrivacyTier(declared))
            except ValueError:
                pass
        return strictest(tiers)


class RecordDecisionRequest(BaseModel):
    made_by: str
    decision: str
    rationale: str = ""
    locked: bool = False
    #: Decision IDs this one replaces. Superseding a *locked* decision requires
    #: the caller to hold a human_owner claim (§9.4).
    supersedes: list[str] = Field(default_factory=list)


class RecordAttemptRequest(BaseModel):
    actor_id: str
    actor_type: ActorType = ActorType.manager
    summary: str
    outcome: str = ""
    artifact_refs: list[str] = Field(default_factory=list)


class CreateArtifactRequest(BaseModel):
    type: ArtifactType = ArtifactType.other
    content: str = ""
    summary: str = ""
    created_by: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    requested_by: str
    assigned_to: str
    criteria: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class ReviewResultRequest(BaseModel):
    completed_by: str
    status: ReviewStatus = ReviewStatus.completed
    summary: str = ""
    content: str = ""
    #: Reuse an already-stored artifact instead of writing a new one.
    result_artifact_id: Optional[str] = None


class SubtaskRequest(BaseModel):
    title: str
    instructions: str
    privacy_tier: PrivacyTier = PrivacyTier.repo_sensitive
    preferred_worker: Optional[str] = None
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubtaskDetail(BaseModel):
    subtask: Subtask
    runs: list[WorkerRun] = Field(default_factory=list)


class HandoffSummary(BaseModel):
    """Structured *and* rendered. Adapters show ``text``; tests assert on fields."""

    task_id: str
    title: str
    goal: str
    status: TaskStatus
    current_manager: Optional[str]
    viewer_holds_claim: Optional[bool] = None
    linear_issue_url: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    locked_decisions: list[Decision] = Field(default_factory=list)
    recent_attempts: list[Attempt] = Field(default_factory=list)
    important_artifacts: list[ArtifactSummary] = Field(default_factory=list)
    open_reviews: list[Review] = Field(default_factory=list)
    completed_reviews: list[Review] = Field(default_factory=list)
    open_subtasks: list[Subtask] = Field(default_factory=list)
    #: Durable executions still in flight (Temporal workflow ids, or direct handles).
    running_workflows: list[dict[str, Any]] = Field(default_factory=list)
    waiting_approvals: list[ApprovalRecord] = Field(default_factory=list)
    suggested_next_step: Optional[str] = None
    text: str = ""
