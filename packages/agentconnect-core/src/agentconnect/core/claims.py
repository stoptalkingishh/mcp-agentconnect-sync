"""Claim policy (spec §9.3).

A claim is a *lease*, not a lock: it expires on its own so a manager that dies
mid-task never strands the work. The only exclusivity rule is on
``primary_manager`` — every other role may be held concurrently by many managers
(a reviewer and a planner and three observers is a normal state).
"""

from __future__ import annotations

from typing import Optional

from .errors import Conflict, InvalidRequest
from .models import Claim, ClaimRole

MIN_TTL_SECONDS = 1
MAX_TTL_SECONDS = 24 * 3600


def validate_ttl(ttl_seconds: int) -> int:
    if ttl_seconds < MIN_TTL_SECONDS:
        raise InvalidRequest(f"ttl_seconds must be >= {MIN_TTL_SECONDS}, got {ttl_seconds}")
    return min(ttl_seconds, MAX_TTL_SECONDS)


def active_primary(claims: list[Claim], at: float) -> Optional[Claim]:
    for claim in claims:
        if claim.role is ClaimRole.primary_manager and claim.active_at(at):
            return claim
    return None


def check_primary_exclusivity(claims: list[Claim], manager_id: str, at: float) -> None:
    """Raise if a *different* manager holds a live primary claim.

    The same manager re-claiming is a renewal, not a conflict. An expired or
    released claim never blocks — that is the whole point of the lease.
    """
    holder = active_primary(claims, at)
    if holder is not None and holder.manager_id != manager_id:
        raise Conflict(
            f"task already has an active primary_manager claim held by "
            f"{holder.manager_id!r} (expires at {holder.expires_at})"
        )


def holds_role(claims: list[Claim], manager_id: str, role: ClaimRole, at: float) -> bool:
    return any(
        c.manager_id == manager_id and c.role is role and c.active_at(at) for c in claims
    )
