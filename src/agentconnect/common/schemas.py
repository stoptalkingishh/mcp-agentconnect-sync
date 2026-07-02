"""Shared data contracts (pydantic models).

These mirror the JSON objects described throughout the handoff document and are
the single source of truth for the shapes exchanged between the Agent Router,
the Local Model Manager, agents, and shared memory.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class PrivacyClass(str, Enum):
    """Handoff §13."""

    public = "public"
    low_sensitive = "low_sensitive"
    repo_sensitive = "repo_sensitive"
    secret_sensitive = "secret_sensitive"
    restricted = "restricted"


class ProviderType(str, Enum):
    local = "local"
    cloud = "cloud"


class ProviderPrivacyTier(str, Enum):
    local_only = "local_only"
    external = "external"
    external_paid = "external_paid"


class TaskState(str, Enum):
    """Deterministic task state machine (handoff §19)."""

    CREATED = "CREATED"
    CLASSIFIED = "CLASSIFIED"
    PRIVACY_CHECKED = "PRIVACY_CHECKED"
    ELIGIBLE_PROVIDERS_COMPUTED = "ELIGIBLE_PROVIDERS_COMPUTED"
    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    ARTIFACTS_WRITTEN = "ARTIFACTS_WRITTEN"
    CHECKS_RUN = "CHECKS_RUN"
    REVIEW_READY = "REVIEW_READY"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETRY = "RETRY"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    urgent = "urgent"


# --------------------------------------------------------------------------- #
# Task submission / status
# --------------------------------------------------------------------------- #
class TaskConstraints(BaseModel):
    """Optional caller-supplied constraints for a task."""

    profile: Optional[str] = None
    require_exact_model: Optional[str] = None
    privacy_class: Optional[PrivacyClass] = None
    max_output_tokens: Optional[int] = None
    allow_external: bool = True
    allow_paid: bool = False
    priority: Priority = Priority.normal
    quality: str = "standard"  # standard | high | best_effort


class TaskSubmission(BaseModel):
    """Payload for submit_task (handoff §23)."""

    task: str
    agent_type: Optional[str] = None
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    refs: list[str] = Field(default_factory=list)  # artifact/memory references as input


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str  # patch | test_log | trace | code_map | summary | output
    size_chars: int
    mime: str = "text/plain"


class TaskSummary(BaseModel):
    """Compact status summary returned through MCP (handoff §8)."""

    task_id: str
    status: TaskState
    summary: Optional[str] = None
    artifacts: dict[str, str] = Field(default_factory=dict)  # kind -> artifact_id
    recommended_next_action: Optional[str] = None
    risks: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Worker structured return (handoff §21)
# --------------------------------------------------------------------------- #
class WorkerResult(BaseModel):
    status: str = "completed"
    summary: str = ""
    confidence: float = 0.0
    changed_artifacts: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_next_action: Optional[str] = None


# --------------------------------------------------------------------------- #
# Routing (handoff §11, §12)
# --------------------------------------------------------------------------- #
class RejectedOption(BaseModel):
    provider: str
    reason: str


class ScoreBreakdown(BaseModel):
    provider: str
    model: Optional[str] = None
    total: float
    terms: dict[str, float] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Explainable routing-decision record (handoff §11)."""

    task_id: str
    decision: str  # route_to_local_resident_model | route_to_cloud | queue | ...
    selected_provider: Optional[str] = None
    selected_model: Optional[str] = None
    rejected_options: list[RejectedOption] = Field(default_factory=list)
    scores: list[ScoreBreakdown] = Field(default_factory=list)
    policy_version: str = "unknown"


# --------------------------------------------------------------------------- #
# Redaction (handoff §14)
# --------------------------------------------------------------------------- #
class RedactionResult(BaseModel):
    cloud_safe: bool
    privacy_class: PrivacyClass
    redactions: list[str] = Field(default_factory=list)
    payload_ref: Optional[str] = None
    lossiness: str = "none"  # none | low | medium | high


# --------------------------------------------------------------------------- #
# Quota (handoff §15)
# --------------------------------------------------------------------------- #
class QuotaReservation(BaseModel):
    reservation_id: str
    provider: str
    task_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    requests: int = 1
    tokens: int = 0
    expires_in_seconds: int = 120
    granted: bool = True
    reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Local Model Manager status objects (handoff §5, §22)
# --------------------------------------------------------------------------- #
class GpuStatus(BaseModel):
    name: str
    vram_total_gb: float
    vram_free_gb: float
    gpu_utilization_pct: float


class LoadedModel(BaseModel):
    model_id: str
    profile: Optional[str] = None
    quantization: Optional[str] = None
    max_model_len: int = 16384
    max_active_sequences: int = 4
    active_sequences: int = 0


class AvailableModel(BaseModel):
    model_id: str
    profiles: list[str] = Field(default_factory=list)
    loadable: bool = True
    estimated_load_seconds: int = 45
    supports_tools: bool = False
    supports_vision: bool = False
    max_model_len: int = 16384
    quantization: Optional[str] = None


class QueueStatus(BaseModel):
    local_waiting: int = 0
    oldest_wait_seconds: int = 0


class ManagerStatus(BaseModel):
    """The status object the Local Model Manager publishes (handoff §5)."""

    node_id: str
    status: str = "ready"  # ready | loading | busy | offline
    backend: str = "stub"
    gpu: Optional[GpuStatus] = None
    loaded_model: Optional[LoadedModel] = None
    available_models: list[AvailableModel] = Field(default_factory=list)
    queue: QueueStatus = Field(default_factory=QueueStatus)


class CanAcceptRequest(BaseModel):
    model_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int = 0
    priority: Priority = Priority.normal


class CanAcceptResponse(BaseModel):
    can_accept: bool
    estimated_queue_wait_seconds: int = 0
    reason: str = ""


class GenerateRequest(BaseModel):
    request_id: str
    task_id: str
    model_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_output_tokens: int = 800
    temperature: float = 0.2
    priority: Priority = Priority.normal


class GenerateResponse(BaseModel):
    request_id: str
    model_id: str
    output_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"


class LoadRequest(BaseModel):
    target_model: str
    reason: str = ""
    priority: Priority = Priority.normal


class LoadResponse(BaseModel):
    accepted: bool
    loaded_model: Optional[str] = None
    estimated_load_seconds: int = 0
    reason: str = ""
