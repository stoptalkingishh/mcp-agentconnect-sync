"""Decision authority (spec §9.4).

A locked decision is the mechanism by which a manager binds its successors. The
spec asks us to *reject a decision that contradicts a locked one* — but semantic
contradiction is not decidable from two strings, and guessing it with a model
would make the ledger nondeterministic, which §16 forbids.

So contradiction is made **explicit**: a caller that means to overturn a prior
decision names it in ``supersedes``. Overturning an *unlocked* decision is
ordinary work; overturning a *locked* one requires a live ``human_owner`` claim.
A manager that simply records a conflicting decision without declaring the
supersession has recorded an inconsistency in the log, and the handoff summary
shows both — visible, not silently authoritative.
"""

from __future__ import annotations

from typing import Iterable

from .errors import NotFound, PolicyViolation
from .models import Claim, ClaimRole, Decision


def has_decision_authority(claims: Iterable[Claim], manager_id: str, at: float) -> bool:
    return any(
        c.manager_id == manager_id and c.role is ClaimRole.human_owner and c.active_at(at)
        for c in claims
    )


def check_supersede_allowed(
    targets: list[Decision],
    missing_ids: list[str],
    made_by: str,
    claims: Iterable[Claim],
    at: float,
) -> None:
    if missing_ids:
        raise NotFound(f"cannot supersede unknown decision(s): {', '.join(sorted(missing_ids))}")
    locked = [d for d in targets if d.locked]
    if not locked:
        return
    if has_decision_authority(claims, made_by, at):
        return
    ids = ", ".join(d.id for d in locked)
    raise PolicyViolation(
        f"{made_by!r} may not supersede locked decision(s) {ids}: a live human_owner "
        f"claim is required to overturn a locked decision"
    )


def locked_decisions(decisions: Iterable[Decision]) -> list[Decision]:
    """Locked and still standing — a superseded decision is history, not law."""
    return [d for d in decisions if d.locked and d.superseded_by is None]
