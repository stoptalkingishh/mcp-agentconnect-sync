"""ContextBuilder, MemoryRouter, MemoryRanker (memory-stack spec §5–§10).

> AgentConnect controls access. WikiBrain controls trust. Cognee improves breadth.
> Graphiti improves temporal reasoning.

This module is where "controls access" lives. Managers and workers never touch a
memory backend; they get one bounded, ranked, source-labeled pack from here.

**How the two kinds of agent actually receive memory**

* *Managers* (Claude Code, Codex, Linear Agent — proprietary, unmodifiable) **pull**:
  they call the `get_task_context_pack` MCP tool. They cannot reach WikiBrain,
  Cognee, or Graphiti, because only the AgentConnect MCP server is mounted for
  them. This is the entire reason the MCP adapter exists.
* *Workers* (bounded, often no MCP client at all) get memory **pushed**: the
  `recall_context` activity builds a `worker_brief` pack and attaches it to the
  subtask before `run_worker` runs, so the harness reads it from
  `subtask.metadata["context_pack"]`.

Neither ever promotes a fact. Trust is conferred by a human in WikiBrain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from .memory import (
    BROAD_RETRIEVAL,
    DEFAULT_MAX_ITEMS,
    LEDGER,
    TEMPORAL_GRAPH,
    TRUSTED_AUTHORITY,
    MemoryAdapter,
    MemoryItem,
    MemoryScope,
    RecallPack,
    RecallRequest,
    label,
)

_log = logging.getLogger(__name__)

WIKIBRAIN = "wikibrain"
COGNEE = "cognee"
GRAPHITI = "graphiti"


@dataclass
class ProfileConfig:
    backends: list[str]
    max_items: int = DEFAULT_MAX_ITEMS
    #: Whether the pack carries the full deterministic handoff. A worker gets the
    #: subtask and its constraints, never the manager's debate (§10).
    include_handoff: bool = True
    #: Locked decisions and task constraints are ledger truth, always allowed.
    include_ledger: bool = True
    include_superseded: bool = False


#: Spec §7 + §14. A profile names *what the caller is for*, and that determines
#: which backends are even asked.
PROFILES: dict[str, ProfileConfig] = {
    "manager_brief": ProfileConfig([WIKIBRAIN, COGNEE, GRAPHITI], 8),
    "worker_brief": ProfileConfig([WIKIBRAIN, COGNEE], 5, include_handoff=False),
    "reviewer_brief": ProfileConfig([WIKIBRAIN, GRAPHITI], 8),
    "implementation_constraints": ProfileConfig([WIKIBRAIN], 6, include_handoff=False),
    "known_failures": ProfileConfig([WIKIBRAIN, GRAPHITI], 8),
    "model_performance": ProfileConfig([WIKIBRAIN, GRAPHITI], 8, include_handoff=False),
    # Roomier than the rest: superseded claims rank last, so at a budget of 8 the
    # ledger and the live claims would crowd out the very history this asks for.
    "project_evolution": ProfileConfig([WIKIBRAIN, GRAPHITI], 10, include_superseded=True),
    "broad_project_rag": ProfileConfig([COGNEE], 8, include_handoff=False),
    "hard_policy": ProfileConfig([WIKIBRAIN], 6, include_handoff=False),
}

DEFAULT_PROFILE = "manager_brief"


@dataclass
class MemoryDefaults:
    trusted_only: bool = True
    include_pending: bool = False
    include_superseded: bool = False
    max_items: int = DEFAULT_MAX_ITEMS


@dataclass
class MemoryConfig:
    enabled: bool = True
    trusted_authority: str = WIKIBRAIN
    defaults: MemoryDefaults = field(default_factory=MemoryDefaults)
    profiles: dict[str, ProfileConfig] = field(default_factory=lambda: dict(PROFILES))
    #: Hard preferences that affect routing/privacy/cost/safety (§3). These are
    #: policy, not memory: they are never ranked away and never expire.
    hard_policies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryConfig":
        memory = raw.get("memory", raw) or {}
        defaults_raw = memory.get("defaults") or {}
        profiles = dict(PROFILES)
        for name, spec in (memory.get("profiles") or {}).items():
            base = profiles.get(name, ProfileConfig([WIKIBRAIN]))
            profiles[name] = ProfileConfig(
                backends=list(spec.get("backends", base.backends)),
                max_items=int(spec.get("max_items", base.max_items)),
                include_handoff=bool(spec.get("include_handoff", base.include_handoff)),
                include_ledger=bool(spec.get("include_ledger", base.include_ledger)),
                include_superseded=bool(
                    spec.get("include_superseded", base.include_superseded)
                ),
            )
        return cls(
            enabled=bool(memory.get("enabled", True)),
            trusted_authority=str(memory.get("trusted_authority", WIKIBRAIN)),
            defaults=MemoryDefaults(
                trusted_only=bool(defaults_raw.get("trusted_only", True)),
                include_pending=bool(defaults_raw.get("include_pending", False)),
                include_superseded=bool(defaults_raw.get("include_superseded", False)),
                max_items=int(defaults_raw.get("max_items", DEFAULT_MAX_ITEMS)),
            ),
            profiles=profiles,
            hard_policies=list(memory.get("hard_policies", [])),
        )

    def profile(self, name: str) -> ProfileConfig:
        return self.profiles.get(name) or self.profiles[DEFAULT_PROFILE]


class ContextPack(BaseModel):
    """Ledger truth and external context, side by side and never merged.

    ``memory_is_external_context`` is the contract that stops a Cognee search hit
    from being read as a recorded decision.
    """

    model_config = {"arbitrary_types_allowed": True}

    task_id: str
    profile: str
    handoff: Optional[Any] = None
    memory: RecallPack
    backends_queried: list[str] = []
    warnings: list[str] = []
    memory_is_external_context: bool = True


#: Back-compat name from the previous handoff. Same object.
TaskContextPack = ContextPack


class MemoryRouter:
    """Which backends does *this profile* get to ask? (§7)

    `implementation_constraints` asks only the trusted authority. A worker never
    reaches the temporal graph. Narrowing the question is how the pack stays
    bounded before ranking ever runs.
    """

    def __init__(self, config: MemoryConfig, adapters: dict[str, MemoryAdapter]) -> None:
        self.config = config
        self.adapters = adapters

    def select_backends(
        self, profile: str, task_id: Optional[str] = None, query: Optional[str] = None
    ) -> list[str]:
        if not self.config.enabled:
            return []
        wanted = self.config.profile(profile).backends
        selected = [name for name in wanted if name in self.adapters]
        if selected:
            return selected
        # A deployment with a backend the profiles do not name (a StaticMemoryAdapter
        # in tests, a bespoke engine) still gets queried rather than silently ignored.
        return [n for n in sorted(self.adapters) if n != "none"]


_WORD = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return _WORD.sub(" ", text.lower()).strip()


class MemoryRanker:
    """Merge, dedupe, and order results from backends that disagree (§8).

    Authority order is fixed and boring on purpose. A retrieval engine surfacing
    a sentence three times must never outrank a librarian promoting it once.
    """

    #: Lower is more authoritative.
    LEDGER_RANK = 0
    WIKIBRAIN_VERIFIED = 1
    WIKIBRAIN_PROMOTED = 2
    GRAPHITI_TIED_TO_PROMOTED = 3
    COGNEE_BROAD = 4
    PENDING_OR_UNKNOWN = 5

    def authority(self, item: MemoryItem) -> int:
        role = (item.metadata or {}).get("role", BROAD_RETRIEVAL)
        if role == LEDGER:
            return self.LEDGER_RANK
        if item.status == "pending":
            return self.PENDING_OR_UNKNOWN
        if role == TRUSTED_AUTHORITY and item.status == "promoted":
            # `promoted` is not authority; `trusted` is. A claim the authority
            # promoted but declined to trust (an open contradiction) must not rank
            # above a search hit, let alone above an undisputed claim.
            if not (item.metadata or {}).get("trusted", False):
                return self.PENDING_OR_UNKNOWN
            return (
                self.WIKIBRAIN_VERIFIED if item.confidence == "verified"
                else self.WIKIBRAIN_PROMOTED
            )
        if role == TEMPORAL_GRAPH:
            # Only relationships anchored to a promoted claim carry weight; a bare
            # graph edge is no better than a search hit.
            return (
                self.GRAPHITI_TIED_TO_PROMOTED if item.source_id
                else self.PENDING_OR_UNKNOWN
            )
        if role == BROAD_RETRIEVAL and item.source_id:
            return self.COGNEE_BROAD
        return self.PENDING_OR_UNKNOWN

    def merge_and_rank(
        self, packs: list[RecallPack], profile: str, max_items: int
    ) -> RecallPack:
        best: dict[str, MemoryItem] = {}
        order: list[str] = []
        warnings: list[str] = []
        backends: list[str] = []

        for pack in packs:
            warnings.extend(pack.warnings)
            if pack.backend not in backends:
                backends.append(pack.backend)
            for item in pack.items:
                key = _normalize(item.text)
                if not key:
                    continue
                if key not in best:
                    best[key] = item
                    order.append(key)
                    continue
                incumbent = best[key]
                if self.authority(item) < self.authority(incumbent):
                    # The same fact from a more authoritative backend replaces the
                    # weaker copy, but keeps the corroborating source visible.
                    item.metadata = dict(item.metadata or {})
                    item.metadata["also_seen_in"] = sorted(
                        {*(incumbent.metadata or {}).get("also_seen_in", []),
                         (incumbent.metadata or {}).get("backend", "unknown")}
                    )
                    best[key] = item
                else:
                    incumbent.metadata = dict(incumbent.metadata or {})
                    incumbent.metadata["also_seen_in"] = sorted(
                        {*(incumbent.metadata or {}).get("also_seen_in", []),
                         (item.metadata or {}).get("backend", "unknown")}
                    )

        items = [best[k] for k in order]
        items.sort(key=lambda i: (self.authority(i), -float(
            (i.metadata or {}).get("score") or 0.0
        )))
        return RecallPack(
            profile=profile, query="", items=items[: max(0, max_items)],
            backend="+".join(backends) or "none",
            warnings=list(dict.fromkeys(warnings)),
        )


class ContextBuilder:
    """Builds the one bounded pack a manager or worker is allowed to see (§6)."""

    def __init__(
        self,
        service: Any,
        adapters: dict[str, MemoryAdapter],
        config: Optional[MemoryConfig] = None,
        ranker: Optional[MemoryRanker] = None,
    ) -> None:
        self.service = service
        self.adapters = adapters
        self.config = config or MemoryConfig()
        self.router = MemoryRouter(self.config, adapters)
        self.ranker = ranker or MemoryRanker()

    # ------------------------------------------------------------- ledger
    def _ledger_items(self, detail: Any) -> list[MemoryItem]:
        """Locked decisions and hard policy: the only facts that outrank memory."""
        items: list[MemoryItem] = []
        for policy in self.config.hard_policies:
            items.append(label(MemoryItem(
                text=policy, status="promoted", confidence="verified",
                source_id="agentconnect_config",
                metadata={"kind": "hard_policy"},
            ), "agentconnect", LEDGER))
        for constraint in detail.constraints:
            items.append(label(MemoryItem(
                text=constraint.text, status="promoted", confidence="verified",
                source_id=constraint.id, metadata={"kind": "constraint"},
            ), "agentconnect", LEDGER))
        for decision in detail.decisions:
            if decision.locked and decision.superseded_by is None:
                text = decision.decision
                if decision.rationale:
                    text = f"{text} ({decision.rationale})"
                items.append(label(MemoryItem(
                    text=text, status="promoted", confidence="verified",
                    source_id=decision.id, metadata={"kind": "locked_decision"},
                ), "agentconnect", LEDGER))
        return items

    # ------------------------------------------------------------- backends
    def _recall(
        self, name: str, request: RecallRequest, warnings: list[str]
    ) -> Optional[RecallPack]:
        adapter = self.adapters[name]
        try:
            pack = adapter.recall(request)
        except Exception as exc:
            # §11: a memory outage degrades the pack, it never fails the caller.
            _log.warning("memory backend %r recall failed: %s", name, exc)
            warnings.append(f"{name} recall failed: {exc}")
            return None
        role = getattr(adapter, "role", BROAD_RETRIEVAL)
        for item in pack.items:
            label(item, adapter.backend_name, role)
        return pack

    def build_context_pack(
        self,
        task_id: str,
        profile: str = DEFAULT_PROFILE,
        query: Optional[str] = None,
        max_items: Optional[int] = None,
        manager_id: Optional[str] = None,
        include_pending: bool = False,
    ) -> ContextPack:
        profile_cfg = self.config.profile(profile)
        budget = max_items or profile_cfg.max_items
        warnings: list[str] = []

        detail = self.service.get_task(task_id)
        handoff = (
            self.service.get_handoff_summary(task_id, manager_id)
            if profile_cfg.include_handoff else None
        )
        query = query or f"{detail.task.title}\n{detail.task.goal}".strip()

        packs: list[RecallPack] = []
        if profile_cfg.include_ledger:
            ledger_items = self._ledger_items(detail)
            if ledger_items:
                packs.append(RecallPack(
                    profile=profile, query=query, items=ledger_items, backend="agentconnect",
                ))

        backends = self.router.select_backends(profile, task_id, query) if self.config.enabled \
            else []
        if not self.config.enabled:
            warnings.append("memory is disabled; context is task state only")

        for name in backends:
            adapter = self.adapters[name]
            is_authority = getattr(adapter, "role", None) == TRUSTED_AUTHORITY
            request = RecallRequest(
                query=query, task_id=task_id, profile=profile,  # type: ignore[arg-type]
                scopes=[MemoryScope("task", task_id)],
                max_items=budget,
                # Only the trusted authority is asked to enforce trust: a retrieval
                # engine has no notion of promotion, and filtering it on
                # `trusted_only` would silently return nothing.
                trusted_only=is_authority and not include_pending,
                include_pending=include_pending,
                include_superseded=profile_cfg.include_superseded,
            )
            pack = self._recall(name, request, warnings)
            if pack is not None:
                packs.append(pack)

        merged = self.ranker.merge_and_rank(packs, profile, budget)
        merged.query = query
        merged.warnings = list(dict.fromkeys(merged.warnings + warnings))
        return ContextPack(
            task_id=task_id, profile=profile, handoff=handoff, memory=merged,
            backends_queried=backends, warnings=merged.warnings,
        )
