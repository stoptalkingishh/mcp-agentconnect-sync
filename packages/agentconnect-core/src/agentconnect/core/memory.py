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

#: What a backend *is*, not what it stores. Trust is conferred by promotion in the
#: trusted authority (WikiBrain) — never by a retrieval engine finding something.
MemoryRole = Literal["ledger", "trusted_authority", "broad_retrieval", "temporal_graph"]

LEDGER = "ledger"
TRUSTED_AUTHORITY = "trusted_authority"
BROAD_RETRIEVAL = "broad_retrieval"
TEMPORAL_GRAPH = "temporal_graph"

#: Statuses a caller may see without asking for them explicitly.
TRUSTED_STATUSES: frozenset[str] = frozenset({"promoted"})

DEFAULT_MAX_ITEMS = 8


def label(
    item: "MemoryItem", backend: str, role: MemoryRole,
    authority_trusted: Optional[bool] = None,
) -> "MemoryItem":
    """Stamp provenance onto an item so nothing downstream has to guess.

    Every consumer — the ranker, the MCP response, a human reading Linear — can
    then tell a promoted claim apart from a semantic search hit.

    `status == "promoted"` is NOT sufficient authority. The trusted authority may
    return a claim that is promoted and still not trustworthy — WikiBrain does
    exactly this for a claim in an open contradiction, because a contradiction is
    a warning, not a deletion, and the claim remains of record. When the authority
    supplies its own verdict, pass it as `authority_trusted`.

    The verdict may only ever **downgrade**:

      * a retrieval engine claiming `trusted: true` cannot grant itself authority
        (its role is not authoritative, so the conjunction is still False), and
      * a promoted-but-disputed claim is not trusted however its status reads.

    Stored under `authority_trusted` so re-labelling is idempotent: `ContextBuilder`
    re-labels every item it receives, and must not be able to resurrect trust the
    authority already withheld.
    """
    md = dict(item.metadata or {})
    if authority_trusted is not None:
        md["authority_trusted"] = bool(authority_trusted)
    md["backend"] = backend
    md["role"] = role
    trusted = role in (LEDGER, TRUSTED_AUTHORITY) and item.status == "promoted"
    verdict = md.get("authority_trusted")
    if verdict is not None:
        trusted = trusted and bool(verdict)
    md["trusted"] = trusted
    item.metadata = md
    return item


def is_disputed(item: "MemoryItem") -> bool:
    """A promoted claim the authority explicitly flagged as contradicted.

    Distinct from `is_untrusted_authority_claim`: this one the authority *told* us
    about, so we may say "disputed" out loud. A claim that merely arrived without a
    `trusted` field is unknown, not disputed, and calling it contradicted would be
    inventing a fact.
    """
    md = item.metadata or {}
    return (
        md.get("role") in (LEDGER, TRUSTED_AUTHORITY)
        and item.status == "promoted"
        and md.get("contradiction_status") == "open"
    )


def is_untrusted_authority_claim(item: "MemoryItem") -> bool:
    """A `promoted` claim from the authority that the authority did not trust.

    Covers both the disputed case and the dangerous silent one: a response with no
    `trusted` field at all. Absence is treated as untrusted — never inferred from
    `status`, which is the whole point of the boundary.
    """
    md = item.metadata or {}
    return (
        md.get("role") in (LEDGER, TRUSTED_AUTHORITY)
        and item.status == "promoted"
        and not md.get("trusted", False)
    )


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
    #: The authority's id for this item (e.g. WikiBrain `claim_4`). Feedback and
    #: supersession both need to name a claim; without it a manager can report
    #: "this was stale" about nothing in particular.
    item_id: Optional[str] = None


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

    #: What this backend contributes. Only `trusted_authority` can confer trust.
    role: MemoryRole = BROAD_RETRIEVAL

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


class TrustedMemoryAdapter(MemoryAdapter):
    """A backend that owns the pending → promoted lifecycle (WikiBrain).

    Promotion is the *only* way a fact becomes trusted, and it is never available
    to an agent — see `AgentConnectService.promote_memory_candidate`.
    """

    role: MemoryRole = TRUSTED_AUTHORITY

    @abc.abstractmethod
    def promote_candidate(self, candidate_id: str, promoted_by: str,
                          confidence: Optional[str] = None,
                          scope: Optional[str] = None) -> dict[str, Any]: ...

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return []


class IndexingMemoryAdapter(MemoryAdapter):
    """A retrieval backend that is *fed* promoted claims, never written by agents."""

    def index_claim(self, claim: dict[str, Any]) -> None:
        return None


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
        if request.trusted_only and is_untrusted_authority_claim(item):
            # Promoted, but the authority withheld trust — an open contradiction,
            # or a response that never said `trusted` at all. `status` says
            # promoted; only `trusted` is authority. Never silently upgraded.
            # Scoped to `promoted` so the explicit pending/superseded overrides
            # below still work.
            continue
        if request.trusted_only and status not in TRUSTED_STATUSES:
            # `include_pending` and `include_superseded` are explicit, per-status
            # overrides of trusted_only. Without the superseded override the
            # `project_evolution` profile could never see a superseded claim from
            # the trusted authority — the one backend that knows what superseded
            # what. Everything else still has to be promoted.
            asked_for = (
                (status == "pending" and request.include_pending)
                or (status == "superseded" and request.include_superseded)
            )
            if not asked_for:
                continue
        kept.append(item)
    return kept[: max(0, request.max_items)]


def label_disputed(items: list[MemoryItem], dropped_disputed: int = 0,
                   dropped_untrusted: int = 0) -> list[str]:
    """Warnings for claims the authority promoted but declined to trust."""
    out: list[str] = []
    if dropped_disputed:
        out.append(
            f"{dropped_disputed} promoted claim(s) withheld: the trusted authority "
            "marked them DISPUTED (open contradiction)"
        )
    if dropped_untrusted:
        out.append(
            f"{dropped_untrusted} promoted claim(s) withheld: the trusted authority "
            "did not mark them trusted"
        )
    shown = sum(1 for i in items if is_disputed(i))
    if shown:
        out.append(
            f"{shown} promoted claim(s) are DISPUTED (open contradiction) and are "
            "NOT trusted — do not treat them as established guidance"
        )
    return out


def label_pending(items: list[MemoryItem]) -> list[str]:
    """Warnings that force pending memory to announce itself downstream."""
    count = sum(1 for i in items if i.status == "pending")
    if not count:
        return []
    return [f"{count} unpromoted (pending) memory item(s) included at explicit request"]


def _http_call(
    transport: Optional[Any], base_url: str, api_key: Optional[str], timeout: float,
    method: str, path: str, payload: Optional[dict] = None,
) -> dict[str, Any]:
    if transport is not None:
        return transport(method, f"{base_url}{path}", payload) or {}
    import httpx  # lazy: only the network path needs it

    headers = {"Authorization": api_key} if api_key else {}
    response = httpx.request(
        method, f"{base_url}{path}", json=payload, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    return response.json() or {}


class WikiBrainMemoryAdapter(TrustedMemoryAdapter):
    """The trusted authority: pending candidates, promotion, supersession, provenance.

    Nothing else in the stack may declare a fact trusted. Cognee finding a
    sentence twice does not make it true; a librarian promoting it does.
    """

    role: MemoryRole = TRUSTED_AUTHORITY

    def __init__(
        self, base_url: str = "http://localhost:8787", api_key: Optional[str] = None,
        transport: Optional[Any] = None, timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return "wikibrain"

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        return _http_call(
            self._transport, self.base_url, self._api_key, self._timeout, method, path, payload
        )

    def recall(self, request: RecallRequest) -> RecallPack:
        body = self._call("POST", "/recall", {
            "query": request.query, "task_id": request.task_id, "profile": request.profile,
            "max_items": request.max_items, "trusted_only": request.trusted_only,
            "include_pending": request.include_pending,
            "include_superseded": request.include_superseded,
            "scopes": [{"scope_type": s.scope_type, "scope_id": s.scope_id}
                       for s in request.scopes],
        })
        items = []
        for raw in body.get("items", []):
            scope = raw.get("scope") or {}
            metadata = dict(raw.get("metadata") or {})
            if raw.get("tags"):
                metadata["tags"] = list(raw["tags"])
            if raw.get("sources"):
                metadata["sources"] = raw["sources"]
            contradiction = raw.get("contradiction_status") or (
                "open" if raw.get("contradicted") else None)
            if contradiction:
                metadata["contradiction_status"] = contradiction
            if raw.get("validity"):
                metadata["validity"] = raw["validity"]
            # THE trust boundary. `trusted` is the authority's verdict and the only
            # authority signal; `status` is not. A missing `trusted` means UNTRUSTED
            # — never inferred from `status == "promoted"`, because a promoted claim
            # in an open contradiction is exactly the case where the two disagree.
            authority_trusted = bool(raw.get("trusted", False))
            items.append(label(MemoryItem(
                item_id=raw.get("id"),
                text=str(raw.get("text", "")),
                status=raw.get("status", "unknown"),
                confidence=raw.get("confidence", "unknown"),
                source_id=raw.get("source_id"), source_url=raw.get("source_url"),
                superseded_by=raw.get("superseded_by"),
                valid_from=raw.get("valid_from"), valid_until=raw.get("valid_until"),
                scope=MemoryScope(scope["scope_type"], scope["scope_id"]) if scope else None,
                metadata=metadata,
            ), self.backend_name, self.role, authority_trusted=authority_trusted))
        visible = apply_visibility(items, request)
        kept = {id(i) for i in visible}
        dropped = [i for i in items if id(i) not in kept
                   and is_untrusted_authority_claim(i)]
        n_disputed = sum(1 for i in dropped if is_disputed(i))
        return RecallPack(
            profile=request.profile, query=request.query, items=visible,
            backend=self.backend_name,
            warnings=(list(body.get("warnings", []))
                      + label_pending(visible)
                      + label_disputed(visible, dropped_disputed=n_disputed,
                                       dropped_untrusted=len(dropped) - n_disputed)),
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
        status = body.get("status", "pending")
        if status == "promoted":
            _log.warning("wikibrain reported promotion on capture; recording as pending")
            status = "pending"
        return CaptureResult(
            accepted=bool(body.get("accepted", True)), candidate_id=body.get("candidate_id"),
            status=status, message=body.get("message"), backend=self.backend_name,
        )

    def promote_candidate(
        self, candidate_id: str, promoted_by: str,
        confidence: Optional[str] = None, scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Promote a pending candidate. Human/librarian only.

        `confidence` and `scope` are the authority's, not ours, and it will refuse
        to guess either: confidence is what the profile filters compare against
        (`implementation_constraints` requires `high`), and a guessed scope is how a
        repo-local fact leaks into global recall. We forward them and let WikiBrain
        raise if they are missing and cannot be inherited.
        """
        payload: dict[str, Any] = {"promoted_by": promoted_by}
        if confidence is not None:
            payload["confidence"] = confidence
        if scope is not None:
            payload["scope"] = scope
        body = self._call("POST", f"/candidates/{candidate_id}/promote", payload)
        body.setdefault("claim_id", candidate_id)
        body.setdefault("status", "promoted")
        return body

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._call("GET", f"/candidates?status=pending&limit={limit}")
                    .get("candidates", []))

    def record_feedback(self, request: MemoryFeedbackRequest) -> None:
        self._call("POST", "/feedback", {
            "task_id": request.task_id, "memory_item_id": request.memory_item_id,
            "source_id": request.source_id, "feedback": request.feedback,
            "actor_id": request.actor_id, "note": request.note,
        })

    def health(self) -> dict[str, Any]:
        try:
            body = self._call("GET", "/health")
        except Exception as exc:
            return {"backend": self.backend_name, "status": "unreachable", "detail": str(exc)}
        # `backend` means different things on each side: to us it names the adapter,
        # to WikiBrain it names its *retrieval* backend (sqlite_fts, graphiti, …).
        # Map rather than `setdefault`, which would silently keep WikiBrain's value
        # and make this adapter report itself as "sqlite_fts".
        ok = bool(body.get("ok", True))
        return {
            "backend": self.backend_name,
            "status": "ok" if ok else "degraded",
            "ok": ok,
            "role": self.role,
            "retrieval": body.get("backend"),
            "ledger": body.get("ledger", {}),
            "profiles": body.get("profiles", []),
            "schema_version": body.get("schema_version"),
        }


class CogneeMemoryAdapter(IndexingMemoryAdapter):
    """Broad semantic retrieval. Improves *breadth*, never confers trust.

    Everything it returns is ``unknown`` status: a search hit is a lead, not a
    fact. Agents cannot write here — promoted WikiBrain claims are indexed in.
    """

    role: MemoryRole = BROAD_RETRIEVAL

    def __init__(
        self, base_url: str = "http://localhost:8001", api_key: Optional[str] = None,
        transport: Optional[Any] = None, timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return "cognee"

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        return _http_call(
            self._transport, self.base_url, self._api_key, self._timeout, method, path, payload
        )

    def recall(self, request: RecallRequest) -> RecallPack:
        body = self._call("POST", "/search", {
            "query": request.query, "top_k": request.max_items,
        })
        items = [
            label(MemoryItem(
                text=str(raw.get("text", "")), status="unknown", confidence="unknown",
                source_id=raw.get("source_id"), source_url=raw.get("source_url"),
                metadata={"score": raw.get("score")},
            ), self.backend_name, self.role)
            for raw in body.get("results", [])
        ]
        return RecallPack(
            profile=request.profile, query=request.query,
            items=items[: max(0, request.max_items)], backend=self.backend_name,
            warnings=["cognee results are broad retrieval, not trusted claims"] if items else [],
        )

    def capture_candidate(self, request: CaptureRequest) -> CaptureResult:
        # The write path is one-way: candidates go to the trusted authority, and
        # only promoted claims are indexed here. Refusing is the correct answer.
        return CaptureResult(
            accepted=False, status="rejected", backend=self.backend_name,
            message="cognee is a retrieval index; capture candidates in the trusted authority",
        )

    def index_claim(self, claim: dict[str, Any]) -> None:
        self._call("POST", "/add", {
            "text": claim.get("text", ""), "source_id": claim.get("claim_id"),
            "metadata": {"scope": claim.get("scope"), "promoted_by": claim.get("promoted_by")},
        })

    def health(self) -> dict[str, Any]:
        try:
            body = self._call("GET", "/health")
        except Exception as exc:
            return {"backend": self.backend_name, "status": "unreachable", "detail": str(exc)}
        body.setdefault("backend", self.backend_name)
        return body


class GraphitiMemoryAdapter(IndexingMemoryAdapter):
    """Time-aware relationships: what superseded what, and when.

    Facts that Graphiti reports as invalidated are returned with status
    ``superseded`` — so they are excluded by default, and surface as *warnings*
    in a `project_evolution` pack rather than as advice.
    """

    role: MemoryRole = TEMPORAL_GRAPH

    def __init__(
        self, base_url: str = "http://localhost:8002", api_key: Optional[str] = None,
        transport: Optional[Any] = None, timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return "graphiti"

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        return _http_call(
            self._transport, self.base_url, self._api_key, self._timeout, method, path, payload
        )

    def recall(self, request: RecallRequest) -> RecallPack:
        body = self._call("POST", "/search", {
            "query": request.query, "top_k": request.max_items,
        })
        items, warnings = [], []
        for raw in body.get("facts", []):
            superseded_by = raw.get("superseded_by") or raw.get("invalidated_by")
            status = "superseded" if superseded_by else "unknown"
            item = label(MemoryItem(
                text=str(raw.get("fact", raw.get("text", ""))), status=status,
                confidence="unknown", source_id=raw.get("source_id"),
                superseded_by=superseded_by, valid_from=raw.get("valid_from"),
                valid_until=raw.get("valid_until"),
                metadata={"relation": raw.get("relation")},
            ), self.backend_name, self.role)
            items.append(item)
            if superseded_by:
                warnings.append(
                    f"temporal graph: {item.source_id or 'a fact'} was superseded by "
                    f"{superseded_by}"
                )
        visible = apply_visibility(items, request)
        return RecallPack(
            profile=request.profile, query=request.query, items=visible,
            backend=self.backend_name, warnings=warnings,
        )

    def capture_candidate(self, request: CaptureRequest) -> CaptureResult:
        return CaptureResult(
            accepted=False, status="rejected", backend=self.backend_name,
            message="graphiti is a temporal index; capture candidates in the trusted authority",
        )

    def index_claim(self, claim: dict[str, Any]) -> None:
        self._call("POST", "/episodes", {
            "name": claim.get("claim_id"), "body": claim.get("text", ""),
            "source_id": claim.get("claim_id"), "supersedes": claim.get("supersedes", []),
            "valid_from": claim.get("valid_from"),
        })

    def health(self) -> dict[str, Any]:
        try:
            body = self._call("GET", "/health")
        except Exception as exc:
            return {"backend": self.backend_name, "status": "unreachable", "detail": str(exc)}
        body.setdefault("backend", self.backend_name)
        return body


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
