"""Deterministic task state machine (handoff §19).

Models produce artifacts *inside* states but must not freely mutate the state
machine — all transitions go through :func:`assert_transition` / :class:`TaskFSM`,
which reject illegal moves. This keeps the control plane deterministic (§10).
"""

from __future__ import annotations

from .schemas import TaskState

# The linear happy path (§19):
#   CREATED -> CLASSIFIED -> PRIVACY_CHECKED -> ELIGIBLE_PROVIDERS_COMPUTED
#   -> QUEUED -> DISPATCHED -> RUNNING -> ARTIFACTS_WRITTEN -> CHECKS_RUN
#   -> REVIEW_READY -> APPROVED/REJECTED/RETRY -> COMPLETE
#
# Plus terminal/transverse edges: CANCELLED (from any non-terminal), FAILED,
# and RETRY looping back to QUEUED.
_ALLOWED: dict[TaskState, set[TaskState]] = {
    TaskState.CREATED: {TaskState.CLASSIFIED},
    TaskState.CLASSIFIED: {TaskState.PRIVACY_CHECKED},
    TaskState.PRIVACY_CHECKED: {TaskState.ELIGIBLE_PROVIDERS_COMPUTED, TaskState.REJECTED},
    TaskState.ELIGIBLE_PROVIDERS_COMPUTED: {TaskState.QUEUED, TaskState.REJECTED},
    TaskState.QUEUED: {TaskState.DISPATCHED},
    TaskState.DISPATCHED: {TaskState.RUNNING},
    TaskState.RUNNING: {TaskState.ARTIFACTS_WRITTEN, TaskState.FAILED},
    TaskState.ARTIFACTS_WRITTEN: {TaskState.CHECKS_RUN},
    TaskState.CHECKS_RUN: {TaskState.REVIEW_READY},
    TaskState.REVIEW_READY: {TaskState.APPROVED, TaskState.REJECTED, TaskState.RETRY},
    TaskState.APPROVED: {TaskState.COMPLETE},
    TaskState.REJECTED: {TaskState.COMPLETE, TaskState.RETRY},
    TaskState.RETRY: {TaskState.QUEUED},
    # Terminal states:
    TaskState.COMPLETE: set(),
    TaskState.CANCELLED: set(),
    TaskState.FAILED: set(),
}

TERMINAL_STATES = {TaskState.COMPLETE, TaskState.CANCELLED, TaskState.FAILED}


class IllegalTransition(ValueError):
    pass


def allowed_transitions(state: TaskState) -> set[TaskState]:
    # CANCELLED may be reached from any non-terminal state.
    base = set(_ALLOWED.get(state, set()))
    if state not in TERMINAL_STATES:
        base.add(TaskState.CANCELLED)
    return base


def can_transition(src: TaskState, dst: TaskState) -> bool:
    return dst in allowed_transitions(src)


def assert_transition(src: TaskState, dst: TaskState) -> None:
    if not can_transition(src, dst):
        raise IllegalTransition(f"Illegal task transition: {src.value} -> {dst.value}")


class TaskFSM:
    """A tiny stateful wrapper for driving one task through the machine."""

    def __init__(self, state: TaskState = TaskState.CREATED):
        self.state = state
        self.history: list[TaskState] = [state]

    def to(self, dst: TaskState) -> "TaskFSM":
        assert_transition(self.state, dst)
        self.state = dst
        self.history.append(dst)
        return self

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
