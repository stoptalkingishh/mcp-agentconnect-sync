"""AgentConnect backplane core — the durable middle layer (see docs/BACKPLANE_SPEC.md).

Managers and workers are replaceable; this ledger is not. Adapters (MCP, HTTP,
CLI, Linear) call :class:`AgentConnectService` and nothing below it.
"""

from .artifacts import FilesystemArtifactStore, default_artifact_dir
from .errors import AgentConnectError, Conflict, InvalidRequest, NotFound, PolicyViolation
from .execution import (
    DirectExecutionBackend,
    ExecutionBackend,
    ExecutionHandle,
    ExecutionState,
    ExecutionStatus,
)
from .local_compute import (
    HttpLocalComputeProvider,
    LocalComputeProvider,
    LocalEstimate,
    LocalEstimateRequest,
    LocalModel,
    LocalModelManagerWorkerAdapter,
    LocalRunRequest,
    LocalRunResult,
)
from .memory import (
    CaptureRequest,
    CaptureResult,
    HttpMemoryAdapter,
    MemoryAdapter,
    MemoryFeedbackRequest,
    MemoryItem,
    MemoryScope,
    NoopMemoryAdapter,
    RecallPack,
    RecallRequest,
    StaticMemoryAdapter,
)
from .models import (
    ActorType,
    ApprovalRecord,
    ApprovalStatus,
    Artifact,
    ArtifactChunk,
    ArtifactSummary,
    ArtifactType,
    Attempt,
    Claim,
    ClaimRole,
    Constraint,
    CreateArtifactRequest,
    CreateTaskRequest,
    Decision,
    Event,
    ExternalRef,
    FilesystemAccess,
    HandoffSummary,
    InboxItem,
    InboxKind,
    Priority,
    PrivacyTier,
    RecordAttemptRequest,
    RecordDecisionRequest,
    Review,
    ReviewRequest,
    ReviewResultRequest,
    ReviewStatus,
    RunStatus,
    SandboxSpec,
    Subtask,
    SubtaskDetail,
    SubtaskRequest,
    SubtaskStatus,
    Task,
    TaskDetail,
    TaskFilters,
    TaskStatus,
    TaskSummary,
    WorkerLocation,
    WorkerRun,
)
from .routing import RouteExplanation, RoutePolicy, WorkerRegistry, route
from .service import AgentConnectService, TaskContextPack
from .storage import SqliteStorage, default_db_path
from .workers import (
    EchoWorker,
    RawModelWorker,
    WorkerAdapter,
    WorkerCapabilities,
    WorkerContext,
    WorkerEstimate,
    WorkerHealth,
    WorkerResult,
)

__all__ = [
    "AgentConnectError", "AgentConnectService", "ActorType", "ApprovalRecord", "ApprovalStatus",
    "Artifact", "ArtifactChunk", "ArtifactSummary", "ArtifactType", "Attempt", "CaptureRequest",
    "CaptureResult", "Claim", "ClaimRole", "Conflict", "Constraint", "CreateArtifactRequest",
    "CreateTaskRequest", "Decision", "DirectExecutionBackend", "EchoWorker", "Event",
    "ExecutionBackend", "ExecutionHandle", "ExecutionState", "ExecutionStatus", "ExternalRef",
    "FilesystemAccess", "FilesystemArtifactStore", "HandoffSummary", "HttpLocalComputeProvider",
    "HttpMemoryAdapter", "InboxItem", "InboxKind", "InvalidRequest", "LocalComputeProvider",
    "LocalEstimate", "LocalEstimateRequest", "LocalModel", "LocalModelManagerWorkerAdapter",
    "LocalRunRequest", "LocalRunResult", "MemoryAdapter", "MemoryFeedbackRequest", "MemoryItem",
    "MemoryScope", "NoopMemoryAdapter", "NotFound", "PolicyViolation", "Priority", "PrivacyTier",
    "RawModelWorker", "RecallPack", "RecallRequest", "RecordAttemptRequest",
    "RecordDecisionRequest", "Review", "ReviewRequest", "ReviewResultRequest", "ReviewStatus",
    "RouteExplanation", "RoutePolicy", "RunStatus", "SandboxSpec", "SqliteStorage",
    "StaticMemoryAdapter", "Subtask", "SubtaskDetail", "SubtaskRequest", "SubtaskStatus", "Task",
    "TaskContextPack", "TaskDetail", "TaskFilters", "TaskStatus", "TaskSummary", "WorkerAdapter",
    "WorkerCapabilities", "WorkerContext", "WorkerEstimate", "WorkerHealth", "WorkerLocation",
    "WorkerRegistry", "WorkerResult", "WorkerRun", "default_artifact_dir", "default_db_path",
    "route",
]
