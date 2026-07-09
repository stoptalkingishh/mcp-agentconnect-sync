"""Deterministic worker routing with a persisted explanation (spec §20).

No learning, no model, no randomness: the same subtask against the same registry
always selects the same worker, and always for the same stated reason. That
reason is stored (`Subtask.route_reason`, plus a `route_explanation` artifact),
so a human reading Linear six weeks later can see *why local was rejected*
without re-running anything.

Hard gates filter; score terms rank. A worker that fails a hard gate is never
ranked — it is listed under ``rejected_workers`` with the gate it failed.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .errors import NotFound
from .models import PrivacyTier, Subtask, WorkerLocation
from .workers import WorkerAdapter, WorkerCapabilities, WorkerEstimate

#: Every hard gate, in evaluation order. A worker must pass all of them.
HARD_GATES = (
    "healthy",
    "privacy_allowed",
    "capability_match",
    "sandbox_supported",
    "budget_allowed",
    "approval_granted",
)

#: Score weights. They sum to 1.0 so ``total_score`` reads as a fraction.
WEIGHTS = {
    "privacy_fit": 0.35,
    "cost": 0.25,
    "availability": 0.15,
    "capability": 0.15,
    "preferred": 0.10,
}

#: Local-first (§21): a worker that keeps data on the box scores highest even
#: when a cloud worker is also privacy-eligible.
_LOCATION_FIT = {
    WorkerLocation.local: 1.0,
    WorkerLocation.rented: 0.6,
    WorkerLocation.cloud: 0.3,
}

#: Applied when a subtask declares no required capabilities: we cannot verify
#: fit, so we neither reward nor punish it.
_UNKNOWN_CAPABILITY_FIT = 0.7


class RoutePolicy(BaseModel):
    """Budget and posture. Defaults are the safe ones: nothing paid runs
    without an explicit human approval carrying its own ceiling."""

    max_cost_usd: float = 0.0


class RejectedWorker(BaseModel):
    worker: str
    reason: str
    gate: str


class RouteExplanation(BaseModel):
    subtask_id: str
    selected_worker: Optional[str] = None
    selected_harness: Optional[str] = None
    selected_model: Optional[str] = None
    hard_gates: list[str] = Field(default_factory=list)
    score_terms: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    estimated_cost_usd: float = 0.0
    rejected_workers: list[RejectedWorker] = Field(default_factory=list)
    #: True when the *only* thing standing between this subtask and a worker is
    #: a human approval — the signal that drives `needs_approval` (§15).
    #: For a `local_model_manager` worker: what the external manager said it
    #: would run this on. AgentConnect records the answer; it never picks it.
    worker_type: Optional[str] = None
    local_estimate: Optional[dict[str, Any]] = None
    needs_approval: bool = False
    approval_candidate: Optional[str] = None
    approval_reason: Optional[str] = None
    #: Where the approval-blocked candidate would run ("cloud" | "rented" | "local").
    #: A Linear approval names a route class (`/agentconnect approve cloud`), and
    #: this is what that name is matched against.
    approval_location: Optional[str] = None


class WorkerRegistry:
    """The set of worker tuples this deployment may route to."""

    def __init__(self, workers: Optional[Iterable[WorkerAdapter]] = None) -> None:
        self._workers: dict[str, WorkerAdapter] = {}
        for worker in workers or ():
            self.register(worker)

    def register(self, worker: WorkerAdapter) -> None:
        self._workers[worker.worker_id] = worker

    def unregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> WorkerAdapter:
        try:
            return self._workers[worker_id]
        except KeyError:
            raise NotFound(f"unknown worker {worker_id!r}") from None

    def all(self) -> list[WorkerAdapter]:
        return [self._workers[k] for k in sorted(self._workers)]

    def __len__(self) -> int:
        return len(self._workers)


def _effective_ceiling(subtask: Subtask, policy: RoutePolicy) -> float:
    """An approval may carry its own, higher, ceiling (`/agentconnect approve
    rented-gpu max_cost=3.00`); otherwise the standing policy applies."""
    if subtask.approved_max_cost_usd is not None:
        return subtask.approved_max_cost_usd
    return policy.max_cost_usd


def _preferred_match(subtask: Subtask, caps: WorkerCapabilities) -> bool:
    pref = (subtask.preferred_worker or "").strip()
    if not pref:
        return False
    return pref in (caps.worker_id, caps.harness, caps.location.value)


def _gate_failure(
    subtask: Subtask, caps: WorkerCapabilities, estimate: WorkerEstimate,
    healthy: bool, health_detail: str, ceiling: float,
) -> Optional[tuple[str, str]]:
    """First failing gate as ``(gate, human_reason)``, or None if all pass."""
    if not healthy:
        return "healthy", f"worker unavailable: {health_detail or 'health check failed'}"
    if subtask.privacy_tier not in caps.privacy_tiers:
        return (
            "privacy_allowed",
            f"{subtask.privacy_tier.value} subtask cannot use "
            f"{caps.location.value} worker (tiers: "
            f"{', '.join(t.value for t in caps.privacy_tiers) or 'none'})",
        )
    missing = sorted(set(subtask.required_capabilities) - set(caps.capability_tags))
    if missing:
        return "capability_match", f"missing capabilities: {', '.join(missing)}"
    if not subtask.sandbox.satisfied_by(caps.sandbox):
        return (
            "sandbox_supported",
            f"sandbox demand exceeds worker offer (needs filesystem="
            f"{subtask.sandbox.filesystem.value}"
            f"{', network' if subtask.sandbox.network else ''}"
            f"{', shell' if subtask.sandbox.shell else ''})",
        )
    if estimate.estimated_cost_usd > ceiling:
        return (
            "budget_allowed",
            f"estimated ${estimate.estimated_cost_usd:.4f} exceeds ceiling ${ceiling:.4f}",
        )
    if caps.requires_approval and not subtask.approved_by:
        return (
            "approval_granted",
            f"{caps.location.value} execution requires explicit human approval",
        )
    return None


def _score(subtask: Subtask, caps: WorkerCapabilities, estimate: WorkerEstimate,
           ceiling: float) -> dict[str, float]:
    cost = estimate.estimated_cost_usd
    if cost <= 0:
        cost_term = 1.0
    elif ceiling > 0:
        cost_term = max(0.0, 1.0 - cost / ceiling)
    else:
        cost_term = 0.0
    capability = (
        1.0 if subtask.required_capabilities else _UNKNOWN_CAPABILITY_FIT
    )
    return {
        "privacy_fit": _LOCATION_FIT[caps.location],
        "cost": round(cost_term, 4),
        "availability": round(max(0.0, min(1.0, caps.availability)), 4),
        "capability": capability,
        "preferred": 1.0 if _preferred_match(subtask, caps) else 0.0,
    }


def _total(terms: dict[str, float]) -> float:
    return round(sum(WEIGHTS[k] * v for k, v in terms.items()), 6)


def _local_estimate(worker: WorkerAdapter, subtask: Subtask) -> Optional[dict[str, Any]]:
    """Nest an external local model manager's own decision inside our explanation.

    Duck-typed on purpose: `core.routing` must not import `core.local_compute`,
    because a worker adapter of any provenance may want to contribute a nested
    estimate. A worker that has nothing to say returns None and costs nothing.
    """
    getter = getattr(worker, "local_estimate", None)
    if not callable(getter):
        return None
    try:
        estimate = getter(subtask)
    except Exception:  # a chatty worker must never break routing
        return None
    if estimate is None:
        return None
    if hasattr(estimate, "__dict__"):
        return {k: v for k, v in vars(estimate).items() if not k.startswith("_")}
    return dict(estimate) if isinstance(estimate, dict) else None


def route(
    subtask: Subtask, registry: WorkerRegistry, policy: Optional[RoutePolicy] = None
) -> RouteExplanation:
    """Filter, score, select — and explain, whatever the outcome."""
    policy = policy or RoutePolicy()
    ceiling = _effective_ceiling(subtask, policy)
    explanation = RouteExplanation(subtask_id=subtask.id)

    eligible: list[tuple[float, str, WorkerCapabilities, WorkerEstimate, dict[str, float]]] = []
    approval_blocked: list[tuple[WorkerCapabilities, WorkerEstimate, str]] = []

    for worker in registry.all():
        caps = worker.capabilities()
        health = worker.health()
        estimate = worker.estimate(subtask, None)
        failure = _gate_failure(
            subtask, caps, estimate, health.available, health.detail, ceiling
        )
        if failure is not None:
            gate, reason = failure
            explanation.rejected_workers.append(
                RejectedWorker(worker=caps.worker_id, reason=reason, gate=gate)
            )
            if gate == "approval_granted":
                approval_blocked.append((caps, estimate, reason))
            continue
        terms = _score(subtask, caps, estimate, ceiling)
        eligible.append((_total(terms), caps.worker_id, caps, estimate, terms))

    if eligible:
        # Deterministic: highest score, ties broken by worker_id ascending.
        eligible.sort(key=lambda e: (-e[0], e[1]))
        total, _, caps, estimate, terms = eligible[0]
        explanation.selected_worker = caps.worker_id
        explanation.selected_harness = caps.harness
        explanation.selected_model = caps.model
        explanation.worker_type = caps.harness
        explanation.hard_gates = list(HARD_GATES)
        explanation.score_terms = terms
        explanation.total_score = total
        explanation.estimated_cost_usd = estimate.estimated_cost_usd
        explanation.local_estimate = _local_estimate(registry.get(caps.worker_id), subtask)
        return explanation

    if approval_blocked:
        # Nothing is runnable, but a human could unblock it. Surface the cheapest
        # candidate so the Linear approval comment can quote a real price (§15).
        approval_blocked.sort(key=lambda e: (e[1].estimated_cost_usd, e[0].worker_id))
        caps, estimate, reason = approval_blocked[0]
        explanation.needs_approval = True
        explanation.approval_candidate = caps.worker_id
        explanation.approval_reason = reason
        explanation.approval_location = caps.location.value
        explanation.estimated_cost_usd = estimate.estimated_cost_usd
    return explanation


def local_rejection_summary(explanation: RouteExplanation) -> str:
    """Why free/local workers were rejected — the sentence a human needs before
    approving spend (§15 step 5)."""
    if not explanation.rejected_workers:
        return "No other workers are registered."
    return "; ".join(
        f"{r.worker}: {r.reason}" for r in explanation.rejected_workers
    )


def privacy_allows_external(tier: PrivacyTier) -> bool:
    return tier in (PrivacyTier.public, PrivacyTier.public_redacted)
