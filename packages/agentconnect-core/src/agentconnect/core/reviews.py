"""Review lifecycle (spec §9.7, §17).

Reviews are the manager-to-manager coordination primitive: instead of one
manager dumping its context into another, it leaves a ticket pointing at
artifacts and criteria, and the other manager answers with an artifact. The
state machine below is the entire protocol.
"""

from __future__ import annotations

from .errors import Conflict, PolicyViolation
from .models import Review, ReviewStatus

TRANSITIONS: dict[ReviewStatus, frozenset[ReviewStatus]] = {
    ReviewStatus.open: frozenset({ReviewStatus.claimed, ReviewStatus.cancelled}),
    ReviewStatus.claimed: frozenset(
        {ReviewStatus.in_progress, ReviewStatus.completed, ReviewStatus.rejected,
         ReviewStatus.cancelled, ReviewStatus.open}
    ),
    ReviewStatus.in_progress: frozenset(
        {ReviewStatus.completed, ReviewStatus.rejected, ReviewStatus.cancelled}
    ),
    ReviewStatus.completed: frozenset(),
    ReviewStatus.rejected: frozenset(),
    ReviewStatus.cancelled: frozenset(),
}

TERMINAL = frozenset({ReviewStatus.completed, ReviewStatus.rejected, ReviewStatus.cancelled})


def check_transition(review: Review, to: ReviewStatus) -> None:
    if to not in TRANSITIONS[review.status]:
        raise Conflict(
            f"review {review.id} cannot move {review.status.value} -> {to.value}"
        )


def check_claimable(review: Review, manager_id: str) -> None:
    if review.status is not ReviewStatus.open:
        raise Conflict(
            f"review {review.id} is {review.status.value}, not open"
            + (f" (held by {review.assigned_to})" if review.status is ReviewStatus.claimed else "")
        )
    if review.assigned_to != manager_id:
        raise PolicyViolation(
            f"review {review.id} is assigned to {review.assigned_to!r}, not {manager_id!r}"
        )


def check_completable(review: Review, completed_by: str) -> None:
    if review.status in TERMINAL:
        raise Conflict(f"review {review.id} is already {review.status.value}")
    if review.assigned_to != completed_by:
        raise PolicyViolation(
            f"review {review.id} is assigned to {review.assigned_to!r}; "
            f"{completed_by!r} cannot complete it"
        )
