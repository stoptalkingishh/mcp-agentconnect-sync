"""Memory adapter interface (adapters spec, Part A).

**AgentConnect controls *when* memory is read or written. The backend controls
*how* it is stored, indexed, retrieved, and governed.**

Managers and workers never query a memory backend directly — they ask the
service for a bounded, scoped pack. That is what keeps a manager's context
window under our policy rather than the backend's, and it is why `recall` takes
a `profile` and a `max_items` instead of a raw search string.

Two invariants worth stating out loud, because they are load-bearing:

1. **Memory is optional and never fatal.** No configured backend means an empty
   pack with a warning — never an exception, never a failed workflow.
2. **Capture never promotes.** Anything an agent volunteers arrives as
   ``pending``. Promotion is a governance decision that happens in the backend,
   with a human in the loop; the adapter cannot shortcut it.

Recalled memory is *external context*, not ledger truth. Adapters must keep it
labeled that way wherever it is rendered.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

_log = logging.getLogger(__name__)

MemoryProfile = Literal[
    "manager_brief",
    "worker_brief",
    "reviewer_brief",
    "implementation_constraints",
    "user_preferences",
    "known_failures",
    "model_performance",
]

MemoryStatus = Literal[
    "promoted", "pending", "rejected", "superseded", "contradicted", "archived", "unknown",
]

MemoryConfidence = Literal["low", "medium", "high", "verified", "unknown"]

#: Statuses a caller may see without asking for them explicitly.
TRUSTED_STATUSES: frozenset[str] = frozenset({"promoted"})

DEFAULT_MAX_ITEMS = 8


@dataclass
class MemoryScope:
    scope_type: str  # global | user | project | repo | task | manager | worker | model | tool
    scope_id: str


@dataclass
class RecallRequest:
    query: str
    task_id: Optional[str] = None
    profile: MemoryProfile = "manager_brief"
    scopes: list[MemoryScope] = field(default_factory=list)
    max_items: int = DEFAULT_MAX_ITEMS
    trusted_only: bool = True
    include_pending: bool = False
    include_superseded: bool = False
    include_sources: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    text: str
    status: MemoryStatus
    confidence: MemoryConfidence
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    scope: Optional[MemoryScope] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallPack:
    profile: MemoryProfile
    query: str
    items: list[MemoryItem]
    warnings: list[str] = field(default_factory=list)
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureRequest:
    text: str
    task_id: Optional[str] = None
    proposed_scopes: list[MemoryScope] = field(default_factory=list)
    origin_actor_id: Optional[str] = None
    origin_actor_type: Optional[str] = None  # manager | worker | human | system
    source_ref: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureResult:
    accepted: bool
    candidate_id: Optional[str] = None
    status: MemoryStatus = "pending"
    message: Optional[str] = None
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryFeedbackRequest:
    task_id: Optional[str]
    memory_item_id: Optional[str]
    source_id: Optional[str]
    feedback: str  # useful | irrelevant | stale | wrong | too_broad | missing_context
    actor_id: Optional[str] = None
    note: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryAdapter(abc.ABC):
    """Implement this to plug WikiBrain, Cognee, Graphiti, Mem0, or your own in."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str: ...

    @abc.abstractmethod
    def recall(self, request: RecallRequest) -> RecallPack: ...

    @abc.abstractmethod
    def capture_candidate(self, request: CaptureRequest) -> CaptureResult: ...

    def record_feedback(self, request: MemoryFeedbackRequest) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "status": "unknown"}


def apply_visibility(items: list[MemoryItem], request: RecallRequest) -> list[MemoryItem]:
    """Enforce the caller-facing visibility policy, backend-independently.

    An adapter may or may not honor `trusted_only` server-side. We filter again
    here so a sloppy backend cannot smuggle a `pending` item into a manager's
    context just because it forgot the flag. Defaults are the safe ones.
    """
    kept: list[MemoryItem] = []
    for item in items:
        status = item.status
        if status == "superseded" and not request.include_superseded:
            continue
        if status == "pending" and not request.include_pending:
            continue
        if status in ("rejected", "archived", "contradicted"):
            continue
        if request.trusted_only and status not in TRUSTED_STATUSES:
            # `include_pending` is an explicit override of trusted_only for
            # pending items only — everything else still has to be promoted.
            if not (status == "pending" and request.include_pending):
                continue
        kept.append(item)
    return kept[: max(0, request.max_items)]


def label_pending(items: list[MemoryItem]) -> list[str]:
    """Warnings that force pending memory to announce itself downstream."""
    count = sum(1 for i in items if i.status == "pending")
    if not count:
        return []
    return [f"{count} unpromoted (pending) memory item(s) included at explicit request"]


class NoopMemoryAdapter(MemoryAdapter):
    """Memory disabled. Every call succeeds and does nothing."""

    @property
    def backend_name(self) -> str:
        return "none"

    def recall(self, request: RecallRequest) -> RecallPack:
        return RecallPack(
            profile=request.profile, query=request.query, items=[], backend="none",
            warnings=["memory is disabled; no context was recalled"],
        )

    def capture_candidate(self, request: CaptureRequest) -> CaptureResult:
        return CaptureResult(
            accepted=False, status="archived", backend="none",
            message="memory is disabled; candidate discarded",
        )

    def health(self) -> dict[str, Any]:
        return {"backend": "none", "status": "disabled"}


class StaticMemoryAdapter(MemoryAdapter):
    """Fixture-backed adapter for tests and offline demos.

    Matching is a case-insensitive substring scan — deliberately dumb, so a test
    asserting on visibility policy is not really asserting on a ranker.
    """

    def __init__(self, items: Optional[list[MemoryItem]] = None,
                 backend_name: str = "static") -> None:
        self._items = list(items or [])
        self._name = backend_name
        self.captured: list[CaptureRequest] = []
        self.feedback: list[MemoryFeedbackRequest] = []

    @property
    def backend_name(self) -> str:
        return self._name

    def recall(self, request: RecallRequest) -> RecallPack:
        needle = request.query.strip().lower()
        matched = [
            i for i in self._items
            if not needle or needle in i.text.lower()
            or any(needle in str(v).lower() for v in i.metadata.values())
        ]
        visible = apply_visibility(matched, request)
        return RecallPack(
            profile=request.profile, query=request.query, items=visible,
            backend=self._name, warnings=label_pending(visible),
        )

    def capture_candidate(self, request: CaptureRequest) -> CaptureResult:
        self.captured.append(request)
        candidate_id = f"candidate_{len(self.captured)}"
        # Never promoted. Governance is the backend's, with a human in the loop.
        return CaptureResult(
            accepted=True, candidate_id=candidate_id, status="pending",
            backend=self._name, message="Memory candidate captured for later review.",
        )

    def record_feedback(self, request: MemoryFeedbackRequest) -> None:
        self.feedback.append(request)

    def health(self) -> dict[str, Any]:
        return {"backend": self._name, "status": "ok", "items": len(self._items)}


class HttpMemoryAdapter(MemoryAdapter):
    """Generic HTTP memory service.

    Expects ``POST {base_url}/recall``, ``POST {base_url}/capture``,
    ``POST {base_url}/feedback``, ``GET {base_url}/health``. The transport is
    injectable so this is testable without a server; production lazily imports
    httpx.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        transport: Optional[Any] = None,
        backend_name: str = "http",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self._name = backend_name
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return self._name

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(method, f"{self.base_url}{path}", payload) or {}
        import httpx  # lazy: only the network path needs it

        headers = {"Authorization": self._api_key} if self._api_key else {}
        response = httpx.request(
            method, f"{self.base_url}{path}", json=payload, headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json() or {}

    def recall(self, request: RecallRequest) -> RecallPack:
        body = self._call("POST", "/recall", {
            "query": request.query, "task_id": request.task_id, "profile": request.profile,
            "max_items": request.max_items, "trusted_only": request.trusted_only,
            "include_pending": request.include_pending,
            "include_superseded": request.include_superseded,
            "scopes": [{"scope_type": s.scope_type, "scope_id": s.scope_id}
                       for s in request.scopes],
        })
        items = [
            MemoryItem(
                text=str(raw.get("text", "")),
                status=raw.get("status", "unknown"),
                confidence=raw.get("confidence", "unknown"),
                source_id=raw.get("source_id"), source_url=raw.get("source_url"),
                superseded_by=raw.get("superseded_by"), metadata=raw.get("metadata") or {},
            )
            for raw in body.get("items", [])
        ]
        visible = apply_visibility(items, request)
        return RecallPack(
            profile=request.profile, query=request.query, items=visible,
            backend=body.get("backend", self._name),
            warnings=list(body.get("warnings", [])) + label_pending(visible),
        )

    def capture_candidate(self, request: CaptureRequest) -> CaptureResult:
        body = self._call("POST", "/capture", {
            "text": request.text, "task_id": request.task_id,
            "origin_actor_id": request.origin_actor_id,
            "origin_actor_type": request.origin_actor_type,
            "source_ref": request.source_ref, "tags": request.tags,
            "proposed_scopes": [{"scope_type": s.scope_type, "scope_id": s.scope_id}
                                for s in request.proposed_scopes],
        })
        # A backend that claims it promoted the item is not believed: capture is
        # pending by contract, and lying about it would bypass governance.
        status = body.get("status", "pending")
        if status == "promoted":
            _log.warning(
                "memory backend %s reported an immediate promotion on capture; "
                "recording it as pending (capture must never promote)", self._name,
            )
            status = "pending"
        return CaptureResult(
            accepted=bool(body.get("accepted", True)),
            candidate_id=body.get("candidate_id"), status=status,
            message=body.get("message"), backend=body.get("backend", self._name),
        )

    def record_feedback(self, request: MemoryFeedbackRequest) -> None:
        self._call("POST", "/feedback", {
            "task_id": request.task_id, "memory_item_id": request.memory_item_id,
            "source_id": request.source_id, "feedback": request.feedback,
            "actor_id": request.actor_id, "note": request.note,
        })

    def health(self) -> dict[str, Any]:
        try:
            body = self._call("GET", "/health")
        except Exception as exc:  # an unreachable backend is not an outage for us
            return {"backend": self._name, "status": "unreachable", "detail": str(exc)}
        body.setdefault("backend", self._name)
        return body
